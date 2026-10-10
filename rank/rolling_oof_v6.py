"""Rolling, chronological OOF coarse scores for v6.

Put in xhs-recsys/rank/rolling_oof_v6.py, then run from project root:
    python rank/rolling_oof_v6.py

Requires recall/export_five_train.py supporting:
    --data <directory> --model all --topk 200
That exporter must ONLY train from <directory>/rank_train_history and
must export five *_train.parquet/csv files into --data.

No W_k labels or features may be used in retrieval / coarse fit for W_k.
W0 is the warm-up supervised coarse label window; W1-W4 become DeepFM
training; W5 is the independent DeepFM selection window.
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CHANNELS = {'MostPopular': 'mostpopular_decay', 'ItemCF': 'itemcf_weighted',
            'BPR-MF': 'bprmf', 'LightGCN': 'lightgcn', 'TwoTower': 'twotower'}


def read(root, stem):
    for ext in ('.parquet', '.csv'):
        p = Path(root) / (stem + ext)
        if p.exists():
            return pd.read_parquet(p) if ext == '.parquet' else pd.read_csv(p)
    raise FileNotFoundError(f'Missing {stem} under {root}')


def write(df, root, stem):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / (stem + '.parquet')
    try:
        df.to_parquet(path, index=False)
    except ImportError:
        path = root / (stem + '.csv')
        df.to_csv(path, index=False)
    return path


def prepare_windows(future, n_windows=6):
    x = future.sort_values(['timestamp', 'user_id', 'note_id']).reset_index(drop=True)
    # Quantile boundaries use event time (not per-user or random row split).
    boundaries = [int(x.timestamp.quantile(i / n_windows)) for i in range(1, n_windows)]
    if len(set(boundaries)) != len(boundaries):
        raise ValueError('Timestamp ties collapse fold boundaries; use fewer windows')
    times = x.timestamp.to_numpy('int64')
    windows = []
    for i, (left, right) in enumerate(zip([None] + boundaries, boundaries + [None])):
        mask = np.ones(len(x), dtype=bool)
        if left is not None:
            mask &= times > left
        if right is not None:
            mask &= times <= right
        w = x.loc[mask].copy()
        if w.empty:
            raise ValueError(f'Empty chronological window W{i}')
        windows.append(w)
    assert sum(len(z) for z in windows) == len(x)
    for a, b in zip(windows, windows[1:]):
        assert a.timestamp.max() < b.timestamp.min()
    return windows


def merge_recall(root, split="train"):
    base = None
    for name, prefix in CHANNELS.items():
        raw = read(root, prefix + '_' + split)
        if 'recall_score' in raw and 'score' not in raw:
            raw = raw.rename(columns={'recall_score': 'score'})
        required = {'user_id', 'note_id', 'score'}
        if not required.issubset(raw):
            raise ValueError(f'{prefix}_train missing {required - set(raw.columns)}')
        df = raw[['user_id', 'note_id', 'score']].copy()
        df = df.loc[np.isfinite(df.score.to_numpy())].copy()
        df = (df.sort_values(['user_id', 'score', 'note_id'], ascending=[True, False, True])
              .drop_duplicates(['user_id', 'note_id']))
        df['rank'] = df.groupby('user_id').cumcount().add(1).astype('int32')
        df = df.loc[df['rank'] <= 200].copy()
        ag = df.groupby('user_id').score
        mn, mx = ag.transform('min'), ag.transform('max')
        df['norm_score'] = ((df.score - mn) / (mx - mn).replace(0, np.nan)).fillna(1.0)
        df[f'{name}_hit'] = np.int8(1)
        df[f'{name}_rrf'] = 1.0 / (60.0 + df['rank'])
        df = df.rename(columns={'score': f'{name}_score', 'rank': f'{name}_rank',
                                'norm_score': f'{name}_norm_score'})
        df = df.set_index(['user_id', 'note_id'])
        base = df if base is None else base.join(df, how='outer')
    out = base.reset_index()
    for name in CHANNELS:
        for suffix in ('hit', 'rrf', 'norm_score'):
            c = f'{name}_{suffix}'
            out[c] = out[c].fillna(0).astype('float32')
        out[f'{name}_rank'] = out[f'{name}_rank'].fillna(10000).astype('float32')
    out['channel_count'] = out[[f'{n}_hit' for n in CHANNELS]].sum(axis=1).astype('float32')
    out['rrf_score'] = out[[f'{n}_rrf' for n in CHANNELS]].sum(axis=1).astype('float32')
    out['max_norm_score'] = out[[f'{n}_norm_score' for n in CHANNELS]].max(axis=1).astype('float32')
    out['mean_norm_score'] = (out[[f'{n}_norm_score' for n in CHANNELS]].sum(axis=1)
                              / out.channel_count.clip(lower=1)).astype('float32')
    return out


def make_snapshot_features(candidates, history, target, notes, cutoff):
    meta = notes[['note_id', 'category', 'publish_ts']]
    frame = candidates.merge(meta, on='note_id', how='left', validate='many_to_one')
    if frame.publish_ts.isna().any():
        raise ValueError('Unknown candidate note')
    frame = frame.loc[frame.publish_ts <= cutoff].copy()
    h = history[['user_id', 'note_id']].drop_duplicates()
    seen = pd.MultiIndex.from_frame(h)
    frame = frame.loc[~pd.MultiIndex.from_frame(frame[['user_id', 'note_id']]).isin(seen)].copy()
    h = h.merge(meta[['note_id', 'category']], on='note_id', how='left')
    user_c = h.groupby('user_id').size()
    note_c = h.groupby('note_id').size()
    user_cat = h.groupby(['user_id', 'category']).size()
    frame['user_hist_log'] = np.log1p(frame.user_id.map(user_c).fillna(0).to_numpy('float32'))
    frame['note_pop_log'] = np.log1p(frame.note_id.map(note_c).fillna(0).to_numpy('float32'))
    uc_idx = pd.MultiIndex.from_frame(frame[['user_id', 'category']])
    uc = user_cat.reindex(uc_idx, fill_value=0).to_numpy('float32')
    frame['cat_affinity'] = uc / np.maximum(np.expm1(frame.user_hist_log.to_numpy()), 1)
    frame['note_age_days'] = ((cutoff - frame.publish_ts) / 86400).astype('float32')
    truth = target[['user_id', 'note_id']].drop_duplicates().merge(
        meta[['note_id', 'publish_ts']], on='note_id', how='left')
    truth = truth.loc[truth.publish_ts <= cutoff, ['user_id', 'note_id']]
    truth = truth.loc[~pd.MultiIndex.from_frame(truth).isin(
        pd.MultiIndex.from_frame(h[['user_id', 'note_id']]))].copy()
    frame['label'] = pd.MultiIndex.from_frame(frame[['user_id', 'note_id']]).isin(
        pd.MultiIndex.from_frame(truth)).astype('int8')
    return frame.reset_index(drop=True), truth


def feature_cols(frame):
    return [c for c in frame if c not in {'user_id', 'note_id', 'publish_ts',
                                         'label', 'prediction', 'coarse_rank'}]


def sampled_coarse_frame(df, rng, negative_ratio):
    """Persist compact supervised history instead of storing millions of rows per fold."""
    pos = np.flatnonzero(df.label.to_numpy() == 1)
    neg = np.flatnonzero(df.label.to_numpy() == 0)
    if not len(pos):
        raise ValueError('No supervised positives in fold')
    n_neg = min(len(neg), len(pos)*negative_ratio)
    selected = np.r_[pos, rng.choice(neg, n_neg, replace=False)]
    return df.iloc[selected].copy().reset_index(drop=True)


def fit_coarse(historical_frames, features, cats, trees, seed):
    # All frames here are already sampled and exclusively from older windows.
    fit = pd.concat(historical_frames, ignore_index=True)
    fit['category'] = pd.Categorical(fit.category, categories=cats)
    model = lgb.LGBMClassifier(
        objective='binary', n_estimators=trees, learning_rate=.04,
        num_leaves=15, max_depth=6, min_child_samples=80,
        reg_lambda=5.0, random_state=seed, verbosity=-1, n_jobs=4)
    model.fit(fit[features], fit.label, categorical_feature=['category'])
    return model


def predict_coarse(model, frame, features, cats, batch=200_000):
    out = np.empty(len(frame), dtype='float32')
    # Chunked prediction reduces peak memory for multi-million rows.
    for a in range(0, len(frame), batch):
        part = frame.iloc[a:a+batch][features].copy()
        part['category'] = pd.Categorical(part.category, categories=cats)
        out[a:a+len(part)] = model.predict_proba(part)[:, 1].astype('float32')
    return out


def topk(df, k):
    x = (df.sort_values(['user_id', 'prediction', 'note_id'],
                        ascending=[True, False, True])
         .groupby('user_id', sort=False).head(k).copy())
    x['coarse_rank'] = x.groupby('user_id').cumcount().add(1).astype('int32')
    return x.reset_index(drop=True)


def copy_metadata_and_history(source, stage_dir, hist, cutoff, count):
    stage_dir.mkdir(parents=True, exist_ok=True)
    manifest = stage_dir / 'snapshot.json'
    signature = hashlib.sha256(f'{cutoff}-{len(hist)}-{count}'.encode()).hexdigest()
    old = json.loads(manifest.read_text()) if manifest.exists() else {}
    ready = old.get('signature') == signature
    if ready:
        ready = all(any((stage_dir / (p+'_train'+ext)).exists()
                        for ext in ('.parquet', '.csv')) for p in CHANNELS.values())
    if ready:
        return True
    for stem in ('users', 'notes'):
        for ext in ('.parquet', '.csv'):
            src = source / (stem + ext)
            if src.exists():
                shutil.copy2(src, stage_dir / src.name)
                break
        else:
            raise FileNotFoundError(f'Missing {stem} in {source}')
    write(hist, stage_dir, 'rank_train_history')
    manifest.write_text(json.dumps({'signature': signature, 'cutoff': cutoff,
                                    'history_rows': len(hist)}, indent=2))
    return False


def run():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, default=ROOT/'data'/'raw_v6')
    p.add_argument('--output', type=Path, default=ROOT/'rank'/'output_v6'/'rolling_oof')
    p.add_argument('--exporter', type=Path, default=ROOT/'recall'/'export_five_train.py')
    p.add_argument('--folds', type=int, default=6)
    p.add_argument('--coarse-trees', type=int, default=160)
    p.add_argument('--negative-ratio', type=int, default=20)
    p.add_argument('--topk', type=int, default=200)
    p.add_argument('--seed', type=int, default=42)
    a = p.parse_args()
    if a.folds != 6:
        raise ValueError('This v2 pipeline expects W0 bootstrap + W1-W4 train + W5 validation; use --folds 6')
    if a.topk != 200:
        raise ValueError('DeepFM v2 uses --topk=200')
    src = a.data.resolve()
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    notes = read(src, 'notes')
    hist0 = read(src, 'rank_train_history')
    future = read(src, 'rank_train_future')
    windows = prepare_windows(future, a.folds)
    cats = sorted(notes.category.astype(str).unique())
    initial_cutoff = int(hist0.timestamp.max())
    if initial_cutoff >= int(windows[0].timestamp.min()):
        raise ValueError('Initial history overlaps W0')
    rng = np.random.default_rng(a.seed)
    supervised = []
    common_features = None
    # Bootstrap W0: recall scores have been fitted on H0 alone.
    for i, target in enumerate(windows):
        prefix = pd.concat([hist0] + windows[:i], ignore_index=True) if i else hist0
        cutoff = int(prefix.timestamp.max())
        stage = out / f'fold_{i:02d}'
        if i == 0:
            source = src  # original *_train are models fitted only on H0
        else:
            if not a.exporter.exists():
                raise FileNotFoundError(f'Cannot find recall exporter: {a.exporter}')
            ready = copy_metadata_and_history(src, stage, prefix, cutoff, i)
            if not ready:
                cmd = [sys.executable, str(a.exporter.resolve()),
                       '--data', str(stage), '--model', 'all', '--topk', '200']
                print('Rebuilding all five recall models:', ' '.join(cmd), flush=True)
                subprocess.run(cmd, cwd=ROOT, check=True)
            source = stage
        merged = merge_recall(source)
        frame, truth = make_snapshot_features(merged, prefix, target, notes, cutoff)
        if frame.empty:
            raise ValueError(f'Fold W{i} has no candidates')
        frame['category'] = pd.Categorical(frame.category, categories=cats)
        if common_features is None:
            common_features = feature_cols(frame)
        if common_features != feature_cols(frame):
            raise ValueError(f'Feature schema changed in W{i}')
        if i == 0:
            supervised.append(sampled_coarse_frame(frame, rng, a.negative_ratio))
            print(f'W0 bootstrap coarse: {len(frame):,} candidates, {int(frame.label.sum()):,} positives', flush=True)
        else:
            model = fit_coarse(supervised, common_features, cats, a.coarse_trees, a.seed)
            frame['prediction'] = predict_coarse(model, frame, common_features, cats)
            result = topk(frame, 200)
            result_path = write(result, out, f'oof_fold_{i:02d}_top200')
            write(target[['user_id', 'note_id', 'timestamp']], out, f'target_fold_{i:02d}')
            print(f'W{i} OOF Top200: {len(result):,} candidates, '
                  f'{int(result.label.sum()):,} positives => {result_path}', flush=True)
            # Do not use W5's labels to train a coarse model or DeepFM.
            if i < 5:
                supervised.append(sampled_coarse_frame(frame.drop(columns=['prediction']), rng, a.negative_ratio))
        del merged, frame
    summary = {'windows': [{'fold': i, 'rows': int(len(w)),
                            'first_ts': int(w.timestamp.min()),
                            'last_ts': int(w.timestamp.max())} for i,w in enumerate(windows)],
               'coarse_trees': a.coarse_trees,
               'train_folds': [1,2,3,4], 'validation_fold': 5}
    (out/'manifest.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print('OOF complete:', out, flush=True)

if __name__ == '__main__':
    run()
