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
# 1. BPR 矩阵分解模型
# =====================================================

class BPRMF(nn.Module):

    def __init__(self, n_users, n_items, dim):
        super().__init__()

        self.user = nn.Embedding(n_users, dim)
        self.item = nn.Embedding(n_items, dim)

        nn.init.normal_(self.user.weight, std=0.05)
        nn.init.normal_(self.item.weight, std=0.05)

    def forward(self, u, i):
        return (self.user(u) * self.item(i)).sum(-1)


# =====================================================
# 2. 主程序
# =====================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--data", default="./data/raw_v6")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--reg", type=float, default=1e-4)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--topk-dump", type=int, default=200,
                        help="导出的每用户召回条数")

    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    # -------------------------------------------------
    # 2.1 数据加载
    # -------------------------------------------------
    users, notes, train, valid, test = load_data(args.data)
    n_items = len(notes)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("Device:", device)

    # -------------------------------------------------
    # 2.2 初始化模型
    # -------------------------------------------------
    model = BPRMF(len(users), n_items, args.dim).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    u = train.user_id.to_numpy(np.int64)
    i = train.note_id.to_numpy(np.int64)
    times = train.timestamp.to_numpy(np.int64)

    keys = positive_keys(train, n_items)
    published = notes.publish_ts.to_numpy(np.int64)

    # -------------------------------------------------
    # 2.3 全量笔记预测分数
    # -------------------------------------------------
    @torch.no_grad()
    def score_fn(batch_users):
        model.eval()
        uvec = model.user(
            torch.as_tensor(batch_users, device=device)
        )
        scores = uvec @ model.item.weight.T
        return scores.cpu().numpy()

    # -------------------------------------------------
    # 2.4 模型训练
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

            pos = model(bu, bi)
            neg = model(bu, bj)

            reg = (
                model.user(bu).square().sum(-1)
                + model.item(bi).square().sum(-1)
                + model.item(bj).square().sum(-1)
            ).mean()

            loss = F.softplus(neg - pos).mean() + args.reg * reg

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total += loss.item() * len(ids)

        # -------------------------------------------------
        # 2.5 验证集
        # -------------------------------------------------
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
    # 2.6 测试集
    # -------------------------------------------------
    model.load_state_dict(best_weights)

    test_metrics = evaluate(
        score_fn, train, valid, test, notes,
        split="test", k=args.k
    )

    print("\n========== BPR-MF ==========")
    print("Best Valid NDCG:", best_ndcg)
    print("Test:", test_metrics)

    # -------------------------------------------------
    # 2.7 导出 valid / test 明细（必须在 load_state_dict 之后）
    # -------------------------------------------------
    for split_name, target in (("valid", valid), ("test", test)):
        dump_recall(
            score_fn,
            split_name,
            train,
            target,
            notes,
            prefix="bprmf",
            topk=args.topk_dump,
        )


if __name__ == "__main__":
    main()