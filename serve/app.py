import os

import pandas as pd
import requests
import streamlit as st


API = os.environ.get("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="小红书推荐可视化", layout="wide")
st.title("小红书推荐系统 · 重排结果可视化")

# ---------- 侧边栏 ----------
with st.sidebar:
    st.header("参数")
    api_url = st.text_input("API 地址", API)
    try:
        r = requests.get(f"{api_url}/users?limit=500", timeout=10)
        r.raise_for_status()
        meta = r.json()
        user_options = meta["users"]
        st.caption(f"总用户数: {meta['total']}，可选前 {len(user_options)} 个")
    except Exception as e:
        st.error(f"API 不可用: {e}")
        st.stop()

    uid = st.selectbox("user_id", user_options, index=0)
    topk = st.slider("Top-K", min_value=5, max_value=50, value=20, step=5)


# ---------- 拉数据 ----------
@st.cache_data(ttl=30, show_spinner=False)
def fetch(api_url: str, uid: int, topk: int):
    r = requests.get(f"{api_url}/recommend/{uid}", params={"topk": topk}, timeout=30)
    r.raise_for_status()
    return r.json()


try:
    data = fetch(api_url, uid, topk)
except Exception as e:
    st.error(f"获取推荐失败: {e}")
    st.stop()

items = pd.DataFrame(data["items"])
summary = data["summary"]

# ---------- 顶部指标 ----------
c1, c2, c3, c4 = st.columns(4)
c1.metric("用户", uid)
c2.metric("推荐条数", len(items))
c3.metric("覆盖类别", summary["n_categories"])
c4.metric("覆盖作者", summary["n_authors"])

st.divider()

# ---------- 主区：推荐列表 ----------
st.subheader(f"用户 {uid} 的 Top-{len(items)} 推荐")

for _, row in items.iterrows():
    with st.container(border=True):
        cols = st.columns([0.6, 5, 1.4, 1.4, 1.4])

        cols[0].markdown(
            f"<div style='font-size:28px;font-weight:700;color:#ff2e4d;"
            f"text-align:center;'>#{int(row['rerank_rank'])}</div>",
            unsafe_allow_html=True,
        )

        title = row.get("title") or f"笔记_{int(row['note_id'])}"
        cols[1].markdown(f"**{title}**")
        cols[1].caption(
            f"note_id={int(row['note_id'])} · "
            f"category={row.get('category','-')} · "
            f"author={int(row.get('author_id',-1))} · "
            f"tags={row.get('tags','-')}"
        )

        cols[2].metric("fine", f"{row['fine_score']:.4f}")
        cols[3].metric("rerank", f"{row['rerank_score']:.4f}" if row.get("rerank_score") is not None else "-")
        cols[4].metric("旧排名", int(row["fine_rank"]) if "fine_rank" in row and row["fine_rank"] is not None else "-")

# ---------- 底部：多样性 ----------
st.divider()
left, right = st.columns([1, 1])

with left:
    st.subheader("类别分布")
    dist = summary.get("category_dist") or {}
    if dist:
        st.bar_chart(pd.Series(dist).sort_values(ascending=False))
    else:
        st.info("无类别信息")

with right:
    st.subheader("排名变化（fine → rerank）")
    if "fine_score" in items.columns and "rerank_rank" in items.columns:
        chart_df = items[["note_id", "fine_score", "rerank_score", "rerank_rank"]].copy()
        chart_df = chart_df.set_index("note_id")
        st.line_chart(chart_df[["fine_score", "rerank_score"]])
    else:
        st.info("缺少分数列")

# ---------- 原始数据 ----------
with st.expander("查看原始数据"):
    st.dataframe(items, use_container_width=True)