import argparse
from pathlib import Path
import sys

import numpy as np
import torch

from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).parent))
from topk_common import (
    load_data,
    evaluate,
    positive_keys,
    sample_neg
)
from dump_utils import dump_recall


# =====================================================
# 1. LightGCN 模型
# =====================================================

class LightGCN(nn.Module):

    def __init__(self, n_users, n_items, dim, layers,
                 edge_users, edge_items):
        super().__init__()

        self.n_users = n_users
        self.n_items = n_items
        self.layers = layers

        self.users = nn.Embedding(n_users, dim)
        self.items = nn.Embedding(n_items, dim)

        nn.init.normal_(self.users.weight, std=0.1)
        nn.init.normal_(self.items.weight, std=0.1)

        u = torch.as_tensor(edge_users, dtype=torch.long)
        i = torch.as_tensor(edge_items, dtype=torch.long) + n_users

        degree = torch.bincount(
            torch.cat([u, i]),
            minlength=n_users + n_items
        ).float().clamp_min(1)

        src = torch.cat([u, i])
        dst = torch.cat([i, u])

        vals = degree[src].rsqrt() * degree[dst].rsqrt()

        graph = torch.sparse_coo_tensor(
            torch.stack([src, dst]),
            vals,
            (n_users + n_items, n_users + n_items)
        ).coalesce()

        self.register_buffer("graph", graph)

    def propagate(self):
        embedding = torch.cat([self.users.weight, self.items.weight])
        vectors = [embedding]

        for _ in range(self.layers):
            embedding = torch.sparse.mm(self.graph, embedding)
            vectors.append(embedding)

        combined = torch.stack(vectors, dim=0).mean(dim=0)

        return combined[:self.n_users], combined[self.n_users:]


# =====================================================
# 2. 主程序
# =====================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--data", default="./data/raw_v6")   # 修正路径
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--reg", type=float, default=1e-4)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--topk-dump", type=int, default=200,
                        help="导出的每用户召回条数")

    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    users, notes, train, valid, test = load_data(args.data)
    n_items = len(notes)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("Device:", device)

    u = train.user_id.to_numpy(np.int64)
    i = train.note_id.to_numpy(np.int64)
    times = train.timestamp.to_numpy(np.int64)

    keys = positive_keys(train, n_items)
    published = notes.publish_ts.to_numpy(np.int64)

    model = LightGCN(
        n_users=len(users),
        n_items=n_items,
        dim=args.dim,
        layers=args.layers,
        edge_users=u,
        edge_items=i
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    cache = {}

    @torch.no_grad()
    def score_fn(batch_users):
        model.eval()
        if "emb" not in cache:
            cache["emb"] = model.propagate()
        user_emb, item_emb = cache["emb"]

        batch = torch.as_tensor(batch_users, device=device)
        scores = user_emb[batch] @ item_emb.T
        return scores.cpu().numpy()

    # -------------------------------------------------
    # 训练循环
    # -------------------------------------------------
    best_ndcg = -1.0
    best_weights = None

    for epoch in range(1, args.epochs + 1):

        model.train()
        order = rng.permutation(len(u))
        total = 0.0

        for start in range(0, len(order), args.batch_size):

            ids = order[start:start + args.batch_size]

            negative = sample_neg(
                u[ids], keys, n_items,
                published, times[ids], rng
            )

            bu = torch.as_tensor(u[ids], device=device)
            bi = torch.as_tensor(i[ids], device=device)
            bj = torch.as_tensor(negative, device=device)

            user_emb, item_emb = model.propagate()

            pos = (user_emb[bu] * item_emb[bi]).sum(-1)
            neg = (user_emb[bu] * item_emb[bj]).sum(-1)

            reg = (
                model.users(bu).square().sum(-1)
                + model.items(bi).square().sum(-1)
                + model.items(bj).square().sum(-1)
            ).mean() / 2

            loss = F.softplus(neg - pos).mean() + args.reg * reg

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total += loss.item() * len(ids)

        cache.clear()

        metrics = evaluate(
            score_fn, train, valid, test, notes,
            split="valid", k=args.k
        )

        print(
            f"Epoch={epoch}, "
            f"Loss={total / len(u):.5f}, "
            f"Valid={metrics}"
        )

        current_ndcg = metrics[f"NDCG@{args.k}"]

        if current_ndcg > best_ndcg:
            best_ndcg = current_ndcg
            best_weights = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }

    # -------------------------------------------------
    # 测试集
    # -------------------------------------------------
    model.load_state_dict(best_weights)
    cache.clear()

    test_metrics = evaluate(
        score_fn, train, valid, test, notes,
        split="test", k=args.k
    )

    print("\n========== LightGCN ==========")
    print("Best Valid NDCG:", best_ndcg)
    print("Test:", test_metrics)

    # -------------------------------------------------
    # 导出 valid / test 明细
    # -------------------------------------------------
    for split_name, target in (("valid", valid), ("test", test)):
        dump_recall(
            score_fn,
            split_name,
            train,
            target,
            notes,
            prefix="lightgcn",
            topk=args.topk_dump,
        )


if __name__ == "__main__":
    main()