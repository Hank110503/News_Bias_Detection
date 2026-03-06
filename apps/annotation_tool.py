import streamlit as st
import json
import os
import datetime
import copy
import csv

# =====================================================
# 页面配置
# =====================================================
st.set_page_config(layout="wide", page_title="数据标注可视化与人工校验工具")

# =====================================================
# 辅助函数
# =====================================================
@st.cache_data
def load_data(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None

def fix_path(path):
    if not path:
        return None
    return os.path.normpath(path)

# =====================================================
# 一致性计算
# =====================================================
def compute_consistency(original_item, manual_item, tol=0.5):
    summary = {"cues": {}}
    total, agree = 0, 0

    # overall bias
    ob_o = original_item.get("overall_bias_intensity", 0)
    ob_m = manual_item.get("overall_bias_intensity", 0)
    summary["overall_bias"] = {
        "orig": ob_o,
        "manual": ob_m,
        "agree": abs(ob_o - ob_m) <= tol
    }

    orig_cues = original_item.get("cues", {})
    manu_cues = manual_item.get("cues", {})

    for grp in manu_cues:
        summary["cues"][grp] = {}
        for key, m_val in manu_cues[grp].items():
            o_val = orig_cues.get(grp, {}).get(key, {})
            total += 1

            p_agree = o_val.get("present") == m_val.get("present")
            try:
                s_agree = abs(float(o_val.get("score", 0)) - float(m_val.get("score", 0))) <= tol
            except:
                s_agree = False

            r_agree = str(o_val.get("reason", "")).strip() == str(m_val.get("reason", "")).strip()
            is_agree = sum([p_agree, s_agree, r_agree]) >= 2

            if is_agree:
                agree += 1

            summary["cues"][grp][key] = {
                "present": {"orig": o_val.get("present"), "manual": m_val.get("present"), "agree": p_agree},
                "score": {"orig": o_val.get("score"), "manual": m_val.get("score"), "agree": s_agree},
                "reason": {"orig": o_val.get("reason"), "manual": m_val.get("reason"), "agree": r_agree},
                "agree_overall": is_agree
            }

    summary["agreement_rate"] = agree / total if total else None
    return summary

# =====================================================
# 手工标注 UI 构建
# =====================================================
def manual_annotation_ui(item):
    manual = {}

    annotator = st.text_input("Annotator", value=st.session_state.get("annotator", ""))
    st.session_state["annotator"] = annotator

    manual["meta"] = {
        "annotator": annotator,
        "timestamp": datetime.datetime.now().isoformat(),
        "news_id": item.get("news_id")
    }

    manual["overall_bias_intensity"] = st.slider(
        "Overall Bias Intensity (Manual)",
        0, 5, int(item.get("overall_bias_intensity", 0))
    )

    st.markdown("### Analysis Chain")
    oc = item.get("analysis_chain", {})
    manual["analysis_chain"] = {
        "visual_observation": st.text_area("Visual Observation", oc.get("visual_observation", ""), height=80),
        "textual_observation": st.text_area("Textual Observation", oc.get("textual_observation", ""), height=80),
        "joint_mechanism": st.text_area("Joint Mechanism", oc.get("joint_mechanism", ""), height=80)
    }

    st.markdown("### Cues")
    manual["cues"] = {}
    for grp, grp_data in item.get("cues", {}).items():

        # ===============================
        # 单独处理 consistency_dimension
        # ===============================
        if grp == "consistency_dimension":
            with st.expander("consistency_dimension", expanded=False):
                manual["cues"]["consistency_dimension"] = {}

                manual["cues"]["consistency_dimension"]["D1_Relationship_Type"] = (
                    st.text_input(
                        "D1_Relationship_Type",
                        value=grp_data.get("D1_Relationship_Type", "")
                    )
                )

                manual["cues"]["consistency_dimension"]["Coherence_Score"] = (
                    st.number_input(
                        "Coherence_Score",
                        0.0, 1.0,
                        float(grp_data.get("Coherence_Score", 0.0)),
                        step=0.05
                    )
                )

                manual["cues"]["consistency_dimension"]["reason"] = (
                    st.text_area(
                        "Reason",
                        value=grp_data.get("reason", ""),
                        height=80
                    )
                )
            continue   # ⬅️ 非常重要

        # ===============================
        # 普通 cues（统一结构）
        # ===============================
        with st.expander(grp, expanded=False):
            manual["cues"][grp] = {}
            for key, val in grp_data.items():
                if not isinstance(val, dict):
                    continue

                st.markdown(f"**{key}**")
                c1, c2, c3 = st.columns([1, 1, 2])

                with c1:
                    p = st.checkbox(
                        "present",
                        value=val.get("present", False),
                        key=f"{item['news_id']}_{grp}_{key}_p"
                    )

                with c2:
                    s = st.number_input(
                        "score",
                        0.0, 5.0,
                        float(val.get("score", 0)),
                        step=0.5,
                        key=f"{item['news_id']}_{grp}_{key}_s"
                    )

                with c3:
                    r = st.text_input(
                        "reason",
                        val.get("reason", ""),
                        key=f"{item['news_id']}_{grp}_{key}_r"
                    )

                manual["cues"][grp][key] = {
                    "present": p,
                    "score": s,
                    "reason": r
                }

    return manual


# =====================================================
# Sidebar
# =====================================================
st.sidebar.header("配置")
json_file = st.sidebar.text_input("JSON 文件路径", "label_test_3.json")
data = load_data(json_file)
if not data:
    st.error("无法加载 JSON 文件")
    st.stop()

if "idx" not in st.session_state:
    st.session_state.idx = 0

if st.sidebar.button("⬅ 上一个") and st.session_state.idx > 0:
    st.session_state.idx -= 1

if st.sidebar.button("下一个 ➡") and st.session_state.idx < len(data)-1:
    st.session_state.idx += 1

st.sidebar.info(f"{st.session_state.idx+1} / {len(data)}")

# =====================================================
# 主界面
# =====================================================
item = data[st.session_state.idx]
col_l, col_r = st.columns([1, 1.2])

# ---------- 左侧 ----------
with col_l:
    st.subheader("原始数据")
    st.markdown(f"### {item.get('title')}")
    st.caption(f"News ID: {item.get('news_id')}")

    img = fix_path(item.get("image_path"))
    if img and os.path.exists(img):
        st.image(img, use_container_width=True)
    else:
        st.warning("图片未找到")

    txt = fix_path(item.get("text_path"))
    if txt and os.path.exists(txt):
        st.text_area("文本内容", open(txt, encoding="utf-8").read(), height=200)

# ---------- 右侧 ----------
with col_r:
    st.subheader("人工标注 & 校验")

    manual = manual_annotation_ui(item)
    st.session_state["manual"] = manual

    c1, c2, c3 = st.columns(3)

    with c1:
        if st.button("💾 保存人工标注"):
            out = json_file.replace(".json", "_manual.json")
            data_copy = copy.deepcopy(data)
            data_copy[st.session_state.idx]["manual_annotation"] = manual
            with open(out, "w", encoding="utf-8") as f:
                json.dump(data_copy, f, ensure_ascii=False, indent=2)
            st.success(f"保存到 {out}")

    with c2:
        if st.button("⬇ 导出 CSV"):
            rows = []
            for grp, g in manual["cues"].items():
                for k, v in g.items():
                    orig = item.get("cues", {}).get(grp, {}).get(k, {})
                    rows.append({
                        "news_id": item["news_id"],
                        "cue_group": grp,
                        "cue": k,
                        "orig_present": orig.get("present"),
                        "manual_present": v.get("present"),
                        "orig_score": orig.get("score"),
                        "manual_score": v.get("score"),
                    })
            csv_path = json_file.replace(".json", "_manual_export.csv")
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            st.success(f"导出到 {csv_path}")

    with c3:
        if st.button("🔍 一致性分析"):
            summary = compute_consistency(item, manual)
            st.metric("Agreement Rate",
                      f"{summary['agreement_rate']:.2%}" if summary["agreement_rate"] else "N/A")
            with st.expander("查看详情"):
                st.json(summary)
