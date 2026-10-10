# XHS-RecSys

小红书风格推荐系统实践：多路召回 → 融合 → 粗排 → 精排 → MMR 重排 → 服务化。

在合成数据集（v6）上跑通完整 pipeline，测试集最终 **NDCG@20 = 0.342**，重排后 ILD 提升 56%、精度几乎无损。

## 整体架构

```
数据生成 → 多路召回 → 融合 → 粗排 → 精排 → 重排 → 服务化
  v6        5 路       加权   LightGBM  DeepFM  MMR   FastAPI
```

## 最终结果（v6, test split）

| 阶段 | Recall@20 | NDCG@20 | ILD@20 |
|---|---|---|---|
| Random baseline | 0.0004 | — | — |
| 最优单路召回 BPR-MF@200 | 0.103 | — | — |
| 融合覆盖率上限 Merged_All | 0.537 | — | — |
| RRF 融合@20 | 0.122 | — | — |
| LightGBM 粗排@20 | 0.136 | 0.326 | 0.015 |
| DeepFM 精排@20 | **0.141** | **0.342** | 0.015 |
| **MMR 重排@20（最终）** | **0.141** | **0.342** | **0.024** |

**MMR 重排（beta=0.4）**：用 0.03% 的 NDCG 损失，换 55.8% 的 ILD（Intra-List Diversity）提升。

## 数据集（v6）

合成数据集，模拟小红书场景：

| 维度 | 值 |
|---|---|
| 用户数 | 5,000 |
| 笔记数 | 75,000 |
| 行为事件总数 | 1,000,000 |
| 唯一点击关系 | 840,000 |
| 类别 | 10 个（美妆/穿搭/美食/旅行/职场/学习/健身/宠物/家居/数码） |
| 微主题 | 50 个（每类别 5 个标签） |
| 冷启动用户 | 10%，1~5 次点击 |
| 冷启动笔记 | 10%，0~3 个点击者 |
| 时间窗口 | 30 天 |

### v6 相比 v5 的关键改动

| 维度 | v5 | v6 |
|---|---|---|
| 主题粒度 | 10 粗类别 | **50 微主题** |
| 用户画像 | Dirichlet 兴趣向量 | **primary/secondary/tertiary 三个具体微主题** |
| i2i 信号 | 无 | **种子注入 + 主题匹配采样** |

**结果**：所有 CF 类模型从"接近随机"（0.0004）提升到 0.06~0.10。

### 数据切分

按**全局时间**切分：70% train / 15% valid / 15% test。

冷启动标签在**训练集**上重新计算（`train_is_cold`），避免用未来信息评估。

### 生成

```bash
python xhs_topk_v6.py --out data/raw_v6
```

## 项目结构

```
xhs-recsys/
├── xhs_topk_v6.py                    # 数据生成
├── requirements.txt
├── README.md
│
├── data/                             # 数据集（不入库）
│   └── raw_v6/
│
├── recall/                           # 召回
│   ├── mostpopular.py                # 时间衰减热门
│   ├── itemcf_weighted.py            # 加权共现 i2i
│   ├── bprmf.py                      # BPR 矩阵分解
│   ├── lightgcn.py                   # LightGCN
│   ├── two_tower.py                  # 双塔
│   ├── export_five_train.py          # 排序训练召回导出
│   ├── merge_recall.py               # 多路融合
│   └── evaluate_merged_recall.py     # 召回质量评估
│
├── rank/                             # 排序
│   ├── coarse_rank_lgbm.py           # LightGBM 粗排
│   ├── deepfm_residual_v2.py         # DeepFM 残差精排
│   └── evaluate_deepfm_v2.py         # 精排评估
│
├── rerank/                           # 重排
│   ├── rerank_mmr.py                 # MMR 核心实现
│   └── finalize_mmr_v6.py            # 一键：valid 选参 + test 导出
│
└── serve/                            # 服务化
    ├── api.py                        # FastAPI 后端
    └── app.py                        # Streamlit 前端
```

## 快速开始

### 1. 环境

```bash
conda create -n xhs python=3.10
conda activate xhs
pip install -r requirements.txt
```

GPU 可选（LightGCN / BPR-MF / TwoTower 会用到）。

### 2. 生成数据

```bash
python xhs_topk_v6.py --out data/raw_v6
```

### 3. 训练五路召回

```bash
python recall/mostpopular.py       --data data/raw_v6
python recall/itemcf_weighted.py   --data data/raw_v6
python recall/bprmf.py             --data data/raw_v6 --epochs 25
python recall/lightgcn.py          --data data/raw_v6 --epochs 30 --seed 42
python recall/two_tower.py         --data data/raw_v6 --epochs 50
```

**各模型训练轮次选择依据**：

| 模型 | 最佳 epoch | 说明 |
|---|---|---|
| BPR-MF | 13~16 | 之后缓慢过拟合 |
| LightGCN | 30+ | epoch 15 时未收敛 |
| TwoTower | 48 | epoch 50 收敛 |

### 4. 融合 + 粗排 + 精排

```bash
# 融合五路召回
python recall/merge_recall.py

# LightGBM 粗排
python rank/coarse_rank_lgbm.py

# DeepFM 残差精排
python rank/deepfm_residual_v2.py
```

### 5. MMR 重排（一键）

```bash
python rerank/finalize_mmr_v6.py
```

**输出**：
- `rank/output_v6/rerank_test_top20_v2.parquet`
- `rank/output_v6/rerank_test_top20_v2.metrics.json`

**参数选择协议**：用 **valid** 扫描 beta，冻结后仅在 test 上评估一次，无泄漏。

### 6. 服务化

```bash
# 终端 1：后端
uvicorn serve.api:app --port 8000

# 终端 2：前端
streamlit run serve/app.py --server.port 8501
```

浏览器打开 `http://localhost:8501`。

## 关键设计

### 召回层

五路互补：

| 通道 | test Recall@20 | 特点 |
|---|---|---|
| BPR-MF | 0.103 | ID embedding MF |
| LightGCN | 0.084 | 图传播 |
| Weighted ItemCF | 0.083 | 共现 i2i |
| TwoTower | 0.062 | 双塔 + 画像 |
| MostPopular(decay) | 0.002 | 时间衰减热度兜底 |

**融合覆盖率上限 0.537**（Merged_All），70% 候选只被 1 路命中 → 通道高度互补。

### 排序层

**LightGBM 粗排** 输入：

- 融合特征：`rrf_score`、`channel_count`、`max_norm_score`
- 各路特征：`{channel}_rank`、`{channel}_score`、`{channel}_hit`
- 用户特征：`age`、`interest_*`、`train_click_count`
- 物品特征：`item_train_count`、`publish_age`
- 交叉特征：`user_cat_interest`

相对 RRF baseline 提升 **+14% NDCG**。

**DeepFM 残差精排** 用 OOF 协议训练（无泄漏），在粗排基础上提升 **+4.7% NDCG@20**。

### 重排层

**MMR（Maximal Marginal Relevance）**：

```
value = relevance - beta * redundancy
```

- 相似度 = `0.7 * category_match + 0.3 * tag_jaccard`
- beta=0.4（valid 上选出）

**效果**：用 0.03% 的 NDCG 损失换 56% 的 ILD 提升。

## 技术栈

- **数据处理**：numpy、pandas、scipy.sparse
- **深度学习**：PyTorch 2.x（CUDA）
- **树模型**：LightGBM
- **服务化**：FastAPI + uvicorn、Streamlit
- **存储**：Parquet（pyarrow）

## 已知局限

1. **合成数据**：所有结论仅在这个数据集上成立，不能直接外推到真实业务。
2. **类别多样性受限**：数据生成时 78% 用户的 primary/secondary 主题在同一类别，所以推荐结果类别单一（`categories@20 ≈ 1.1`）。
3. **精排提升有限**：DeepFM 残差仅带来 4.7% 提升，`best epoch=1` 说明模型容量过剩，仍有优化空间。
4. **召回覆盖率天花板**：约 54% 的正反馈能被候选池覆盖，剩余 46% 无法触达。

## 后续可做

- [ ] 扩展召回覆盖率：加 UserCF、tag 召回、跨类别探索通道
- [ ] 精排特征工程：加序列特征、时间衰减、多粒度交叉
- [ ] 重排改进：用 DPP 替代 MMR，支持更复杂的多样性约束
- [ ] 上线真实数据：替换合成数据，重跑 pipeline
- [ ] AB 实验框架：接入线上流量，验证离线指标的泛化性

## 许可

MIT