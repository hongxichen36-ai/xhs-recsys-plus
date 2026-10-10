"""Export DeepFM predictions from a saved prototype checkpoint.

Usage from project root:
 python rank/export_deepfm.py --data data/raw_v5 --rank-dir rank/output --split test

This reconstructs exact training-time features from the original prototype.
The checkpoint's categorical mapping was not saved. The original trained with
pd.factorize(notes.category), so re-apply that same deterministic row order.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn


class DeepFM(nn.Module):
    def __init__(self, field_sizes, dense_dim, embed_dim=16):
        super().__init__()
        self.first_order = nn.ModuleList([nn.Embedding(n, 1) for n in field_sizes])
        self.embeddings = nn.ModuleList([nn.Embedding(n, embed_dim) for n in field_sizes])
        self.linear_dense = nn.Linear(dense_dim, 1)
        input_dim = len(field_sizes) * embed_dim + dense_dim
        self.mlp = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(0.2),
                                 nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2),
                                 nn.Linear(64, 1))

    def forward(self, sparse, dense):
        x = torch.stack([emb(sparse[:, j]) for j, emb in enumerate(self.embeddings)], dim=1)
        linear = sum(emb(sparse[:, j]) for j, emb in enumerate(self.first_order)) + self.linear_dense(dense)
        fm = .5 * (x.sum(dim=1).square() - x.square().sum(dim=1)).sum(dim=1, keepdim=True)
        return (linear + fm + self.mlp(torch.cat([x.flatten(start_dim=1), dense], dim=1))).squeeze(-1)


def read_table(folder, name):
    for ext in ('.parquet', '.csv'):
        p = folder / (name + ext)
        if p.exists():
            return pd.read_parquet(p) if ext == '.parquet' else pd.read_csv(p)
    raise FileNotFoundError(f'{name}.parquet/csv missing in {folder}')


def load_checkpoint(path, device):
    # Checkpoints created by your own local training script are trusted.
    # weights_only=True avoids unpickling arbitrary objects.
    return torch.load(path, map_location=device, weights_only=True)


def make_features(candidates, notes, history, checkpoint):
    required = {'user_id', 'note_id', 'prediction', 'coarse_rank'}
    missing = required - set(candidates.columns)
    if missing:
        raise ValueError(f'Coarse candidates missing: {sorted(missing)}')
    candidates = candidates.copy()
    if candidates.duplicated(['user_id','note_id']).any():
        raise ValueError('Duplicate user-note pairs in coarse rank')

    # DO NOT sort notes: the prototype used pd.factorize(notes.category) in file order.
    cat_codes, cat_names = pd.factorize(notes['category'])
    expected_cats = int(checkpoint['field_sizes'][2])
    if len(cat_names) != expected_cats:
        raise ValueError(f'Category cardinality mismatch: {len(cat_names)} != {expected_cats}')
    category_by_note = pd.Series(cat_codes, index=notes['note_id'])
    candidates['category_id'] = candidates.note_id.map(category_by_note)
    if candidates.category_id.isna().any():
        raise ValueError('Candidates contain unknown note IDs')

    user_counts = history.groupby('user_id').size()
    item_counts = history.groupby('note_id').size()
    candidates['user_hist'] = candidates.user_id.map(user_counts).fillna(0)
    candidates['item_hist'] = candidates.note_id.map(item_counts).fillna(0)
    candidates['user_hist_log'] = np.log1p(candidates.user_hist)
    candidates['item_hist_log'] = np.log1p(candidates.item_hist)
    p = candidates.prediction.clip(1e-6, 1-1e-6)
    candidates['coarse_score_logit'] = np.log(p / (1-p)).clip(-10,10)
    candidates['rank_normalized'] = candidates.coarse_rank / 100.0

    sparse_cols = list(checkpoint['sparse_cols'])
    dense_cols = list(checkpoint['dense_cols'])
    if set(sparse_cols) - set(candidates.columns) or set(dense_cols) - set(candidates.columns):
        raise ValueError('Checkpoint features do not match the original prototype columns')
    sparse = candidates[sparse_cols].to_numpy(dtype=np.int64, copy=True)
    for j, size in enumerate(checkpoint['field_sizes']):
        if np.any(sparse[:,j] < 0) or np.any(sparse[:,j] >= size):
            raise ValueError(f'Out-of-range categorical IDs for {sparse_cols[j]}')
    mean = pd.Series(checkpoint['dense_mean']).reindex(dense_cols)
    std = pd.Series(checkpoint['dense_std']).reindex(dense_cols)
    if mean.isna().any() or std.isna().any() or (std <= 0).any():
        raise ValueError('Invalid normalization stats in checkpoint')
    dense = ((candidates[dense_cols] - mean) / std).to_numpy(dtype=np.float32)
    if not np.isfinite(dense).all():
        raise ValueError('Nonfinite dense features')
    return candidates, sparse, dense


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=Path, default=Path('data/raw_v5'))
    ap.add_argument('--rank-dir', type=Path, default=Path('rank/output'))
    ap.add_argument('--checkpoint', type=Path, default=None)
    ap.add_argument('--split', choices=['valid', 'test'], default='test')
    ap.add_argument('--batch-size', type=int, default=8192)
    ap.add_argument('--output', type=Path, default=None)
    args = ap.parse_args()
    if args.batch_size < 1:
        raise ValueError('--batch-size must be >= 1')
    checkpoint_path = args.checkpoint or args.rank_dir / 'deepfm_prototype.pt'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = load_checkpoint(checkpoint_path, device)
    model = DeepFM(checkpoint['field_sizes'], len(checkpoint['dense_cols']), 16).to(device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    model.eval()
    notes = read_table(args.data, 'notes')
    train = read_table(args.data, 'train')
    history = train if args.split == 'valid' else pd.concat([train, read_table(args.data, 'valid')], ignore_index=True)
    candidates = read_table(args.rank_dir, f'coarse_{args.split}_top100')
    candidates, sparse, dense = make_features(candidates, notes, history, checkpoint)
    results = []
    with torch.inference_mode():
        for st in range(0, len(candidates), args.batch_size):
            en = st + args.batch_size
            sparse_t = torch.from_numpy(sparse[st:en]).to(device)
            dense_t = torch.from_numpy(dense[st:en]).to(device)
            results.append(torch.sigmoid(model(sparse_t, dense_t)).cpu().numpy())
    candidates['fine_score'] = np.concatenate(results)
    candidates = candidates.sort_values(['user_id','fine_score','note_id'], ascending=[True,False,True])
    candidates['fine_rank'] = candidates.groupby('user_id').cumcount() + 1
    path = args.output or args.rank_dir / f'fine_{args.split}_top100.parquet'
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ['user_id','note_id','fine_score','fine_rank','prediction','coarse_rank']
    if path.suffix.lower() == '.csv':
        candidates[cols].to_csv(path, index=False)
    else:
        candidates[cols].to_parquet(path, index=False)
    print(f'Exported {len(candidates):,} fine scores, {candidates.user_id.nunique():,} users -> {path}')
    print('WARNING: this is the prior DeepFM prototype trained using coarse_valid_top100; not a leakage-free final offline experiment.')


if __name__ == '__main__':
    main()
