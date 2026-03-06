import streamlit as st
import json
import os

# --- 页面配置 ---
st.set_page_config(layout="wide", page_title="数据标注可视化工具 v2.0")

# --- 辅助函数：加载数据 ---
@st.cache_data
def load_data(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        st.error("JSON 文件格式错误")
        return None

# --- 辅助函数：处理路径 ---
def fix_path(path):
    """
    处理JSON中的路径分隔符，使其适应当前操作系统。
    """
    if not path:
        return None
    return os.path.normpath(path)

# --- CSS样式优化 ---
st.markdown("""
    <style>
    .reportview-container .main .block-container {
        padding-top: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f9f9f9;
        border-radius: 4px 4px 0 0;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# --- 侧边栏：文件加载与导航 ---
st.sidebar.header("配置与导航")

# 1. 指定 JSON 文件路径
default_file = "label-gemini-flash-lite-2.5.json"
json_file = st.sidebar.text_input("JSON 文件路径", value=default_file)

data = load_data(json_file)

if not data:
    st.error(f"无法找到文件: {json_file}。请确保文件在当前目录下或路径正确。")
    st.stop()

# --- Session State 初始化 ---
if 'current_index' not in st.session_state:
    st.session_state.current_index = 0

# 2. 搜索功能
search_id = st.sidebar.text_input("🔍 根据 News ID 查询")
if st.sidebar.button("查询"):
    # 兼容 news_id 可能是 int 或 string 的情况
    found_index = next((i for i, item in enumerate(data) if str(item.get("news_id")) == search_id), None)
    if found_index is not None:
        st.session_state.current_index = found_index
        st.success(f"已跳转至 ID: {search_id}")
    else:
        st.error("未找到该 ID")

st.sidebar.markdown("---")

# 3. 上一个/下一个 按钮
col_prev, col_next = st.sidebar.columns(2)

if col_prev.button("⬅️ 上一个"):
    if st.session_state.current_index > 0:
        st.session_state.current_index -= 1
    else:
        st.warning("已经是第一条数据")

if col_next.button("下一个 ➡️"):
    if st.session_state.current_index < len(data) - 1:
        st.session_state.current_index += 1
    else:
        st.warning("已经是最后一条数据")

# 显示当前进度
st.sidebar.info(f"当前进度: {st.session_state.current_index + 1} / {len(data)}")

# --- 主界面显示 ---

# 获取当前数据项
item = data[st.session_state.current_index]

# 分栏布局：左边原始数据，右边标注分析
col_left, col_right = st.columns([1, 1.3])

# ================= 左侧：原始数据 =================
with col_left:
    st.subheader("📄 原始数据 (Raw Data)")
    
    # 尝试获取基本信息，如果JSON根目录下没有这些字段，可以根据实际情况修改
    news_id = item.get('news_id', 'Unknown ID')
    title = item.get('title', 'No Title')
    
    st.caption(f"News ID: {news_id}")
    st.markdown(f"### {title}")
    
    # 图片显示
    img_path = fix_path(item.get('image_path'))
    if img_path and os.path.exists(img_path):
        st.image(img_path, caption=f"Image: {os.path.basename(img_path)}", use_container_width=True)
    else:
        # 如果找不到图片，显示占位提示
        st.warning(f"图片未找到: `{img_path}`")

    # 文本内容显示
    st.markdown("#### 文本内容")
    txt_path = fix_path(item.get('text_path'))
    
    content_displayed = False
    if txt_path and os.path.exists(txt_path):
        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                if txt_path.endswith('.json'):
                    txt_content = json.load(f)
                    st.json(txt_content, expanded=False)
                else:
                    txt_content = f.read()
                    st.text_area("Content", txt_content, height=300)
            content_displayed = True
        except Exception as e:
            st.error(f"读取文本文件失败: {e}")
    
    if not content_displayed:
        if txt_path:
             st.info(f"文本路径存在但无法读取: {txt_path}")
        else:
             st.info("未提供文本路径")


# ================= 右侧：标注指标 =================
with col_right:
    st.subheader("🏷️ 标注分析 (Analysis)")

    # 1. 总体偏差强度
    bias_score = item.get('overall_bias_intensity', 0)
    col_metric, col_bar = st.columns([1, 3])
    with col_metric:
        st.metric(label="Bias Intensity", value=f"{bias_score} / 5")
    with col_bar:
        st.write("") # Spacer
        st.write("") 
        st.progress(bias_score / 5)

    # 2. Analysis Chain (观察链)
    with st.expander("🔗 Analysis Chain (观察链)", expanded=True):
        chain = item.get('analysis_chain', {})
        st.markdown("**👁️ Visual Observation:**")
        st.info(chain.get('visual_observation', 'N/A'))
        
        st.markdown("**📝 Textual Observation:**")
        st.info(chain.get('textual_observation', 'N/A'))
        
        st.markdown("**🤝 Joint Mechanism:**")
        st.success(chain.get('joint_mechanism', 'N/A'))

    # 3. Cues (细粒度指标)
    st.markdown("### 🎯 Cues Details")
    cues = item.get('cues', {})
    
    # 根据新结构定义Tabs
    # tab_list = ["Visual", "Text", "Joint", "Theme & Affect", "Consistency"]
    tab_list = ["Visual", "Text", "Joint", "Consistency"]

    tabs = st.tabs(tab_list)

    # --- 通用渲染函数 ---
    def display_cue_item(key, value):
        """渲染单个指标卡片"""
        # 判断是否 Present
        is_present = value.get('present', False)
        emoji = "✅" if is_present else "⬜"
        bg_color = "#e6fffa" if is_present else "#ffffff"
        border_color = "#b2f5ea" if is_present else "#e2e8f0"
        
        # 使用 HTML/CSS 做一个稍微好看点的卡片容器
        with st.container():
            st.markdown(f"""
            <div style="border:1px solid {border_color}; border-radius:8px; padding:10px; margin-bottom:10px; background-color:{bg_color}">
                <h4 style="margin:0; padding-bottom:5px;">{emoji} {key}</h4>
            </div>
            """, unsafe_allow_html=True)
            
            # 1. 显示 Score (如果存在)
            if 'score' in value:
                st.write(f"**Score:** `{value['score']}`")
            
            # 2. 显示嵌套 Analysis (特用于 V2_Perspective)
            if 'analysis' in value and isinstance(value['analysis'], dict):
                sub_analysis = value['analysis']
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(f"**Subject:** `{sub_analysis.get('subject_type', 'N/A')}`")
                with c2:
                    st.markdown(f"**Angle:** `{sub_analysis.get('camera_angle', 'N/A')}`")
            
            # 3. 显示 Reason
            if 'reason' in value:
                st.markdown(f"**Reason:** {value['reason']}")
            
    # --- Tab 1: Visual Only ---
    with tabs[0]:
        visual_data = cues.get('visual_only', {})
        for k, v in visual_data.items():
            display_cue_item(k, v)

    # --- Tab 2: Text Only ---
    with tabs[1]:
        text_data = cues.get('text_only', {})
        for k, v in text_data.items():
            display_cue_item(k, v)

    # --- Tab 3: Joint Multimodal ---
    with tabs[2]:
        joint_data = cues.get('joint_multimodal', {})
        for k, v in joint_data.items():
            display_cue_item(k, v)

    # # --- Tab 4: Theme, Value, Affect (分组显示) ---
    # with tabs[3]:
    #     tva_data = cues.get('theme_value_affect', {})
        
    #     # 手动分组以提高可读性
    #     groups = {
    #         "🖼️ Framing (TV)": [k for k in tva_data.keys() if k.startswith("TV")],
    #         "💎 Values (VV)": [k for k in tva_data.keys() if k.startswith("VV")],
    #         "❤️ Affect (AV)": [k for k in tva_data.keys() if k.startswith("AV")]
    #     }
        
    #     for group_name, keys in groups.items():
    #         if keys:
    #             st.markdown(f"##### {group_name}")
    #             for k in keys:
    #                 display_cue_item(k, tva_data[k])
    #             st.markdown("---")

    # --- Tab 4: Consistency ---
    with tabs[3]:
        cons = cues.get('consistency_dimension', {})
        if cons:
            st.markdown(f"### Relationship: `{cons.get('D1_Relationship_Type', 'N/A')}`")
            
            score = cons.get('Coherence_Score', 0)
            st.write(f"**Coherence Score:** {score} / 5")
            st.progress(score/5 if isinstance(score, (int, float)) else 0)
            
            st.info(f"**Reasoning:** {cons.get('reason', 'N/A')}")
        else:
            st.warning("No consistency data found.")