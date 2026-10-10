import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).parent))

# 复用 ItemCF 中相同的数据读取和评价函数
from itemcf_topk import load_data, evaluate
from dump_utils import dump_recall


# =====================================================
# 1. 双塔模型
# =====================================================

class TwoTower(nn.Module):

    def __init__(self, n_users, n_items, n_cats, dim,
                 user_counts, item_counts, profiles):
        super().__init__()

        self.user_id = nn.Embedding(n_users, dim)
        self.item_id = nn.Embedding(n_items, dim)
        self.category = nn.Embedding(n_cats, dim)

        self.user_features = nn.Linear(n_cats, dim, bias=False)

        nn.init.normal_(self.user_id.weight, std=0.05)
        nn.init.normal_(self.item_id.weight, std=0.05)
        nn.init.normal_(self.category.weight, std=0.05)

        self.register_buffer(
            "user_gate",
            torch.tensor(user_counts / (user_counts + 5), dtype=torch.float32)
        )
        self.register_buffer(
            "item_gate",
            torch.tensor(item_counts / (item_counts + 10), dtype=torch.float32)
        )
        self.register_buffer(
            "profiles",
            torch.tensor(profiles, dtype=torch.float32)
        )

    def user_vec(self, uid):
        id_vector = self.user_id(uid) * self.user_gate[uid, None]
        history_vector = self.user_features(self.profiles[uid])
        return F.normalize(id_vector + history_vector, dim=-1)

    def item_vec(self, iid, item_cat):
        id_vector = self.item_id(iid) * self.item_gate[iid, None]
        category_vector = self.category(item_cat[iid])
        return F.normalize(id_vector + category_vector, dim=-1)


# =====================================================
# 2. 训练与评价
# =====================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--data", type=Path, default=Path("./data/raw_v6"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--topk-dump", type=int, default=200,
                        help="导出的每用户召回条数")

    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("计算设备:", device)

    users, notes, train, valid, test = load_data(args.data)

    n_users = len(users)
    n_items = len(notes)

    cat_idx, cat_names = pd.factorize(notes.category)
    n_cats = len(cat_names)
    cat_idx = cat_idx.astype(np.int64)

    user = train.user_id.to_numpy(np.int64)
    item = train.note_id.to_numpy(np.int64)

    user_counts = np.bincount(user, minlength=n_users).astype(np.float32)
    item_counts = np.bincount(item, minlength=n_items).astype(np.float32)

    profiles = np.zeros((n_users, n_cats), dtype=np.float32)
    np.add.at(profiles, (user, cat_idx[item]), 1)
    profiles /= np.maximum(profiles.sum(axis=1, keepdims=True), 1)

    model = TwoTower(
        n_users=n_users,
        n_items=n_items,
        n_cats=n_cats,
        dim=args.dim,
        user_counts=user_counts,
        item_counts=item_counts,
        profiles=profiles
    ).to(device)

    item_cat = torch.tensor(cat_idx, dtype=torch.long, device=device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=1e-5
    )

    pos_key = np.sort(user.astype(np.int64) * n_items + item)

    rng = np.random.default_rng(args.seed)

    def sample_neg(uu):
        neg = rng.integers(n_items, size=len(uu), dtype=np.int64)
        key = uu * n_items + neg
        idx = np.searchsorted(pos_key, key)
        bad = (
            (idx < len(pos_key))
            & (pos_key[np.minimum(idx, len(pos_key) - 1)] == key)
        )
        while bad.any():
            neg[bad] = rng.integers(n_items, size=bad.sum(), dtype=np.int64)
            key = uu * n_items + neg
            idx = np.searchsorted(pos_key, key)
            bad = (
                (idx < len(pos_key))
                & (pos_key[np.minimum(idx, len(pos_key) - 1)] == key)
            )
        return neg

    @torch.no_grad()
    def score_all(uid):

        model.eval()

        if not hasattr(score_all, "item_cache"):
            ids = torch.arange(n_items, device=device)
            score_all.item_cache = torch.cat(
                [
                    model.item_vec(chunk, item_cat)
                    for chunk in ids.split(8192)
                ],
                dim=0
            )

        users_emb = model.user_vec(
            torch.as_tensor(uid, device=device)
        )

        scores = (users_emb @ score_all.item_cache.T).cpu().numpy()

        if np.any(user_counts[uid] == 0):
            scores[user_counts[uid] == 0] = (
                item_counts[None, :] / max(item_counts.max(), 1)
            )

        return scores

    # -------------------------------------------------
    # 训练
    # -------------------------------------------------
    best_ndcg = -1.0
    best_state = None

    for epoch in range(1, args.epochs + 1):

        model.train()
        losses = []

        permutation = rng.permutation(len(user))
        n_batches = max(1, int(np.ceil(len(user) / args.batch_size)))
        batches = np.array_split(permutation, n_batches)

        for ids in batches:

            uu = user[ids]
            ii = item[ids]
            jj = sample_neg(uu)

            uu = torch.as_tensor(uu, dtype=torch.long, device=device)
            ii = torch.as_tensor(ii, dtype=torch.long, device=device)
            jj = torch.as_tensor(jj, dtype=torch.long, device=device)

            uv = model.user_vec(uu)

            pos_score = (uv * model.item_vec(ii, item_cat)).sum(-1) * 10
            neg_score = (uv * model.item_vec(jj, item_cat)).sum(-1) * 10

            loss = F.softplus(neg_score - pos_score).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        # 清理旧 cache
        if hasattr(score_all, "item_cache"):
            del score_all.item_cache

        metrics = evaluate(
            score_all, train, valid, notes, n_users, args.k
        )

        print(f"Epoch {epoch} | Loss {np.mean(losses):.4f} | Valid {metrics}")

        ndcg = metrics[f"NDCG@{args.k}"]

        if ndcg > best_ndcg:
            best_ndcg = ndcg
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }

    # -------------------------------------------------
    # 测试
    # -------------------------------------------------
    model.load_state_dict(best_state)

    if hasattr(score_all, "item_cache"):
        del score_all.item_cache

    test_metrics = evaluate(
        score_all, train, test, notes, n_users, args.k
    )

    print("\n========== TwoTower ==========")
    print("最佳验证 NDCG:", best_ndcg)
    print("测试集:", test_metrics)

    # -------------------------------------------------
    # 保存模型
    # -------------------------------------------------
    torch.save(
        {
            "model": model.state_dict(),
            "categories": list(cat_names)
        },
        args.data / "twotower.pt"
    )
    print("模型保存完成")

    # -------------------------------------------------
    # 导出 valid / test 明细
    # 注意：load_state_dict 后已清过 cache，
    # 这里第一次调用会自动用 best 权重重算 item embedding
    # -------------------------------------------------
    for split_name, target in (("valid", valid), ("test", test)):
        dump_recall(
            score_all,
            split_name,
            train,
            target,
            notes,
            prefix="twotower",
            topk=args.topk_dump,
        )


if __name__ == "__main__":
    main()