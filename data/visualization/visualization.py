# ...existing code...
import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

TARGET_FILE = 'label-gemini-flash-lite-2.5.json'
OUT_DIR = 'visualization'
os.makedirs(OUT_DIR, exist_ok=True)

# -----------------------------
# 1. 加载数据
# -----------------------------
with open(TARGET_FILE, 'r', encoding='utf-8') as f:
    data = json.load(f)  # 假设是 list of dicts

# -----------------------------
# 2. 提取所有量化字段（增强：提取 V2.analysis 字段 与 Coherence_Score）
# -----------------------------
records = []

def safe_get(d, *keys):
    for k in keys:
        if isinstance(d, dict) and k in d:
            d = d[k]
        else:
            return None
    return d

for item in data:
    flat = {}
    flat['news_id'] = item.get('news_id')
    flat['overall_bias_intensity'] = item.get('overall_bias_intensity', None)

    # Visual only cues
    visual_cues = item.get('cues', {}).get('visual_only', {})
    for key, val in visual_cues.items():
        if isinstance(val, dict):
            flat[f"{key}.present"] = val.get('present')
            flat[f"{key}.score"] = val.get('score')
            # If this is V2_Perspective, also capture nested analysis fields
            if key == 'V2_Perspective':
                analysis = val.get('analysis', {}) or {}
                flat['V2_subject_type'] = analysis.get('subject_type')
                flat['V2_camera_angle'] = analysis.get('camera_angle')

    # Text only cues
    text_cues = item.get('cues', {}).get('text_only', {})
    for key, val in text_cues.items():
        if isinstance(val, dict):
            flat[f"{key}.present"] = val.get('present')
            flat[f"{key}.score"] = val.get('score')

    # Joint multimodal cues
    joint_cues = item.get('cues', {}).get('joint_multimodal', {})
    for key, val in joint_cues.items():
        if isinstance(val, dict):
            flat[f"{key}.present"] = val.get('present')
        elif isinstance(val, bool):
            flat[f"{key}.present"] = val

    # Consistency dimension: D1 type and Coherence_Score
    consistency = item.get('cues', {}).get('consistency_dimension', {}) or {}
    flat['D1_Relationship_Type'] = consistency.get('D1_Relationship_Type')
    flat['Coherence_Score'] = consistency.get('Coherence_Score')

    records.append(flat)

df = pd.DataFrame(records)

# -----------------------------
# 3. 统计与可视化
# 要求：
# - 统计所有指标的 true/false 比例
# - 统计所有存在 score 字段的指标中 score 的分布
# - 针对 V2_Perspective 中 subject_type 与 camera_angle 统计分布
# - 统计 consistency_dimension 字段的 score（Coherence_Score）和 type（D1_Relationship_Type）
# -----------------------------
plt.style.use('seaborn-v0_8-whitegrid')
figs = []

# A. overall_bias_intensity 分布
if 'overall_bias_intensity' in df.columns:
    fig, ax = plt.subplots(figsize=(6, 4))
    col = pd.to_numeric(df['overall_bias_intensity'], errors='coerce').dropna()
    if not col.empty:
        min_val = int(col.min())
        max_val = int(col.max())
        bins = range(min_val, max_val + 2)
        col.hist(bins=bins, ax=ax, color='skyblue', edgecolor='black')
        ax.set_title('Distribution of overall_bias_intensity')
        ax.set_xlabel('Bias Intensity')
        ax.set_ylabel('Frequency')
    else:
        ax.text(0.5, 0.5, 'No valid data', transform=ax.transAxes, ha='center')
        ax.set_title('Distribution of overall_bias_intensity')
    figs.append((fig, 'overall_bias_intensity'))

# B. 所有 .present 字段的 True/False 比例（并绘制条形图）
present_cols = [col for col in df.columns if col.endswith('.present')]
if present_cols:
    # Convert non-bool to boolean where possible, keep NaN
    present_df = df[present_cols].applymap(lambda x: True if x is True else (False if x is False else pd.NA))
    present_ratios = present_df.apply(lambda col: col.dropna().mean() if col.dropna().size>0 else pd.NA).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, max(4, len(present_ratios)*0.35)))
    present_ratios.fillna(0).plot(kind='barh', ax=ax, color='coral')
    ax.set_xlabel('Proportion of "True" (excluding missing)')
    ax.set_title('Proportion of Present = True for Each Cue')
    ax.set_xlim(0, 1)
    figs.append((fig, 'present_ratios'))
    # Also save CSV summary
    present_summary = pd.DataFrame({
        'present_true_ratio': present_ratios,
        'n_non_missing': present_df.apply(lambda col: col.dropna().size)
    })
    present_summary.to_csv(os.path.join(OUT_DIR, 'present_summary.csv'), encoding='utf-8-sig')

# C. 所有 .score 字段的分布（箱线图 & 直方图统计）
score_cols = [col for col in df.columns if col.endswith('.score')]
if score_cols:
    # 聚合为 long format
    score_data = df[score_cols].melt(var_name='Cue', value_name='Score').dropna(subset=['Score'])
    # 尝试数值化
    score_data['Score'] = pd.to_numeric(score_data['Score'], errors='coerce')
    score_data = score_data.dropna(subset=['Score'])
    if not score_data.empty:
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.boxplot(data=score_data, x='Cue', y='Score', ax=ax, palette='Set2')
        ax.set_title('Distribution of Scores Across Cues (boxplot)')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        figs.append((fig, 'score_boxplot'))
        # histogram per cue (saved separately)
        for cue, grp in score_data.groupby('Cue'):
            fig2, ax2 = plt.subplots(figsize=(6,3))
            ax2.hist(grp['Score'], bins=range(int(grp['Score'].min()), int(grp['Score'].max())+2), color='skyblue', edgecolor='black')
            ax2.set_title(f'Score Distribution: {cue}')
            ax2.set_xlabel('Score')
            ax2.set_ylabel('Frequency')
            figs.append((fig2, f'score_hist_{cue.replace(".", "_")}'))
        # 保存 score summary
        score_summary = score_data.groupby('Cue')['Score'].describe()
        score_summary.to_csv(os.path.join(OUT_DIR, 'score_summary.csv'), encoding='utf-8-sig')

# D. V2_Perspective 的 subject_type 与 camera_angle 分布
v2_subject_counts = df['V2_subject_type'].dropna().value_counts() if 'V2_subject_type' in df.columns else pd.Series()
v2_angle_counts = df['V2_camera_angle'].dropna().value_counts() if 'V2_camera_angle' in df.columns else pd.Series()

if not v2_subject_counts.empty:
    fig, ax = plt.subplots(figsize=(8, max(3, len(v2_subject_counts)*0.4)))
    v2_subject_counts.plot(kind='barh', ax=ax, color='lightsteelblue')
    ax.set_title('V2_Perspective: subject_type distribution')
    ax.set_xlabel('Count')
    for i, v in enumerate(v2_subject_counts.values):
        ax.text(v + 0.5, i, str(v), va='center')
    figs.append((fig, 'V2_subject_type'))

if not v2_angle_counts.empty:
    fig, ax = plt.subplots(figsize=(8, max(3, len(v2_angle_counts)*0.4)))
    v2_angle_counts.plot(kind='barh', ax=ax, color='lightcoral')
    ax.set_title('V2_Perspective: camera_angle distribution')
    ax.set_xlabel('Count')
    for i, v in enumerate(v2_angle_counts.values):
        ax.text(v + 0.5, i, str(v), va='center')
    figs.append((fig, 'V2_camera_angle'))

# E. consistency_dimension: D1_Relationship_Type 分布 与 Coherence_Score 分布
if 'D1_Relationship_Type' in df.columns:
    rel_counts = df['D1_Relationship_Type'].fillna('NULL').value_counts()
    fig, ax = plt.subplots(figsize=(8, max(3, len(rel_counts)*0.4)))
    rel_counts.plot(kind='barh', ax=ax, color='lightgreen')
    ax.set_title('Distribution of D1_Relationship_Type')
    ax.set_xlabel('Count')
    for i, v in enumerate(rel_counts.values):
        ax.text(v + 0.5, i, str(v), va='center')
    figs.append((fig, 'D1_Relationship_Type'))
    rel_counts.to_csv(os.path.join(OUT_DIR, 'D1_relationship_counts.csv'), encoding='utf-8-sig')

if 'Coherence_Score' in df.columns:
    coh = pd.to_numeric(df['Coherence_Score'], errors='coerce').dropna()
    if not coh.empty:
        fig, ax = plt.subplots(figsize=(6,4))
        bins = range(int(coh.min()), int(coh.max())+2)
        ax.hist(coh, bins=bins, color='mediumpurple', edgecolor='black')
        ax.set_title('Distribution of Coherence_Score')
        ax.set_xlabel('Coherence_Score (0-5)')
        ax.set_ylabel('Count')
        figs.append((fig, 'Coherence_Score_hist'))
        # summary
        coh.describe().to_csv(os.path.join(OUT_DIR, 'Coherence_Score_summary.csv'), header=['value'])

# -----------------------------
# 4. 显示或保存图表与表格
# -----------------------------
for i, (fig, name) in enumerate(figs):
    fig.tight_layout()
    save_path = os.path.join(OUT_DIR, f'bias_visualization_{i+1}_{name}.png')
    fig.savefig(save_path, dpi=200)
    # plt.show()  # 可根据需要打开

# 另外保存整表以便快速检查
df.to_csv(os.path.join(OUT_DIR, 'extracted_metrics_table.csv'), index=False, encoding='utf-8-sig')

print("✅ Visualization completed and saved to", OUT_DIR)
# ...existing code...