import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from transformers import CLIPProcessor, AutoTokenizer, AutoModel
from peft import PeftModel  # 新增：用于加载 LoRA

# 假设你的模型定义在 models.multimodal_model 中
# 请确保该文件中的 MultimodalBiasModel 已经按照上一轮对话修改过（返回 attn_weights）
from models.multimodal_model import MultimodalBiasModel

# -----------------------------------------------------------------------------
# 1. 全局配置
# -----------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "test/multimodal_bias_detection_lora&crossattension/best_model.pt" # 你的训练好的权重路径
CSV_PATH = "data/processed/labels/label-gemini-flash-lite-2.5.csv"
CLIP_NAME = "openai/clip-vit-large-patch14"
TEXT_MODEL_NAME = "FacebookAI/xlm-roberta-large"

# 设置 HF 缓存 (可选)
os.environ["HF_HOME"] = "D:/hf_cache"

print(f"🚀 正在初始化系统... (Device: {DEVICE})")

# -----------------------------------------------------------------------------
# 2. 数据与标签加载
# -----------------------------------------------------------------------------
# 加载关系标签
try:
    df = pd.read_csv(CSV_PATH)
    df = df.dropna(subset=["D1_Relationship_Type"])
    unique_relations = sorted(df["D1_Relationship_Type"].unique())
    id_to_relation = {idx: label for idx, label in enumerate(unique_relations)}
    print(f"✅ 加载了 {len(unique_relations)} 个关系类别")
except Exception as e:
    print(f"⚠️ 警告: 无法读取 CSV ({e})，使用默认标签。")
    id_to_relation = {0: "Unknown", 1: "Supports", 2: "Contradicts"}
    unique_relations = list(id_to_relation.values())

# 偏见指标名称 (需与训练时一致)
presence_labels_names = [
    "V1_Salience", "V2_Perspective", "V3_Color_Lighting", "V4_Symbolism",
    "T1_Loaded_Language", "T2_Moral_Judgment", 
    "J1_Role_Framing", "J2_Selective_Imbalance"#, "J3_Stereotyping"
]

# -----------------------------------------------------------------------------
# 3. 模型初始化与权重加载
# -----------------------------------------------------------------------------
print("正在构建模型架构...")
model = MultimodalBiasModel(
    clip_name=CLIP_NAME,
    text_model_name=TEXT_MODEL_NAME,
    num_presence_labels=len(presence_labels_names), # 注意这里要对应
    num_relation_classes=len(unique_relations),
    use_lora=True, # 如果你的权重包含 LoRA 结构，这里必须为 True
    lora_r=8,
    lora_alpha=16
).to(DEVICE)

# 【关键】加载权重逻辑
# 情况 A: 如果你保存的是整个模型的 state_dict (包含 backbone + LoRA + Heads)
# 情况 B: 如果你只保存了 LoRA adapter (需要使用 PeftModel.from_pretrained)
# 这里假设你保存的是整个 state_dict (常见于毕设简易保存法)
try:
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    
    # 处理可能存在的 'module.' 前缀 (如果是 DDP 训练的)
    new_state_dict = {}
    for k, v in checkpoint.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
        
    model.load_state_dict(new_state_dict, strict=False) # strict=False 防止因层名微小差异报错
    print("✅ 模型权重加载成功")
except FileNotFoundError:
    print(f"❌ 错误: 未找到模型文件 {MODEL_PATH}")
    exit()

model.eval()

# 加载处理器 (必须与训练时一致)
# 注意：CLIPProcessor 需要对应你训练用的模型版本 (large-patch14)
clip_processor = CLIPProcessor.from_pretrained(CLIP_NAME)
text_tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)

# -----------------------------------------------------------------------------
# 4. 可视化辅助函数
# -----------------------------------------------------------------------------

def generate_text_highlight(text, input_ids, attention_mask, i2t_weights, head_idx=0):
    """
    自适应处理 [B, H, L_img, L_txt] 或 [B, L_img, L_txt]
    """
    import numpy as np
    
    if i2t_weights is None:
        return text

    try:
        # 调试打印：看看接收到的真实形状
        # print(f"DEBUG App: i2t_weights shape received: {i2t_weights.shape}")
        
        w = i2t_weights[0].cpu() # 取 Batch 0
        
        # 自动判断维度
        if w.ndim == 3:
            # 情况 A: [Heads, N_img, N_txt] -> 需要平均 Heads
            # print("DEBUG App: Detected Heads dimension, averaging...")
            avg_attn = w.mean(dim=0) # [N_img, N_txt]
        elif w.ndim == 2:
            # 情况 B: [N_img, N_txt] -> 已经平均过或单头
            # print("DEBUG App: No Heads dimension, using directly.")
            avg_attn = w
        else:
            raise ValueError(f"Unexpected dims: {w.ndim}")

        # 对图片维度求和 -> 得到每个 Text Token 的得分
        token_scores = avg_attn.sum(dim=0).numpy() # [N_txt]

        # 标量保护
        if np.ndim(token_scores) == 0:
            token_scores = np.array([token_scores])
            
        # 获取 Tokens
        ids_list = input_ids[0].cpu().tolist()
        mask_arr = attention_mask[0].cpu().numpy()
        tokens = text_tokenizer.convert_ids_to_tokens(ids_list)
        
        # 长度对齐
        min_len = min(len(token_scores), len(tokens), len(mask_arr))
        if min_len == 0: return text
        
        token_scores = token_scores[:min_len]
        tokens = tokens[:min_len]
        mask_arr = mask_arr[:min_len]

        # 归一化
        valid_scores = token_scores[mask_arr == 1]
        if len(valid_scores) > 0 and valid_scores.max() > valid_scores.min():
            token_scores = (token_scores - valid_scores.min()) / (valid_scores.max() - valid_scores.min() + 1e-6)
        else:
            token_scores = np.zeros_like(token_scores)

        # 生成 HTML
        html_parts = []
        for token, m, score in zip(tokens, mask_arr, token_scores):
            if m == 0: continue
            display_token = token.replace("▁", " ").replace("<s>", "").replace("</s>", "")
            if not display_token: display_token = " "
            alpha = 0.2 + (score * 0.6)
            html_parts.append(f'<span style="background-color: rgba(66, 133, 244, {alpha:.2f}); padding: 2px 1px; border-radius: 3px;">{display_token}</span>')
        
        return "".join(html_parts) if html_parts else text

    except Exception as e:
        return f"<span style='color:red'>[Error: {str(e)}]</span> {text}"


def generate_image_heatmap(image_pil, pixel_values, t2i_weights):
    """
    自适应处理 [B, H, L_txt, L_img] 或 [B, L_txt, L_img]
    """
    import numpy as np
    from scipy.ndimage import zoom
    import matplotlib.pyplot as plt

    if t2i_weights is None:
        return image_pil

    try:
        # print(f"DEBUG App: t2i_weights shape received: {t2i_weights.shape}")
        w = t2i_weights[0].cpu()
        
        if w.ndim == 3:
            # [Heads, N_txt, N_img]
            w = w.mean(dim=0) # [N_txt, N_img]
        elif w.ndim == 2:
            # [N_txt, N_img]
            pass
        else:
            return image_pil
            
        # 对 Text 维度求和 -> 得到每个 Patch 的得分
        patch_scores = w.sum(dim=0).numpy() # [N_img]
        
        if len(patch_scores) < 2: return image_pil
        
        # 去掉 CLS token (假设第一个是 CLS)
        grid_scores = patch_scores[1:]
        
        grid_size = int(np.sqrt(len(grid_scores)))
        if grid_size * grid_size != len(grid_scores):
            return image_pil # 无法重塑为正方形网格
            
        grid_scores = grid_scores.reshape(grid_size, grid_size)
        
        # 归一化
        if grid_scores.max() > grid_scores.min():
            grid_scores = (grid_scores - grid_scores.min()) / (grid_scores.max() - grid_scores.min())
        else:
            grid_scores = np.zeros_like(grid_scores)
            
        # 插值与叠加
        img_resized = image_pil.resize((224, 224))
        img_np = np.array(img_resized)
        
        zoom_factor = img_np.shape[0] / grid_size
        heatmap = zoom(grid_scores, zoom_factor, order=1)
        
        cmap = plt.get_cmap('jet')
        heatmap_color = cmap(heatmap)[:, :, :3]
        alpha_channel = heatmap * 0.6
        
        img_float = img_np.astype(float) / 255.0
        blended = img_float * (1 - alpha_channel[..., None]) + heatmap_color * alpha_channel[..., None]
        
        return Image.fromarray((blended * 255).astype(np.uint8))
        
    except Exception as e:
        print(f"Heatmap Error: {e}")
        return image_pil


# -----------------------------------------------------------------------------
# 5. 核心预测函数
# -----------------------------------------------------------------------------
def predict(image, text):
    if image is None or not text:
        return "请提供图片和文本", {}, {}, image, text # 返回原始图占位

    try:
        # --- 预处理 ---
        # 图片 (返回 pixel_values)
        inputs_img = clip_processor(images=image, return_tensors="pt")
        pixel_values = inputs_img["pixel_values"].to(DEVICE)
        
        # 文本 (返回 input_ids, attention_mask)
        inputs_txt = text_tokenizer(
            text, padding="max_length", truncation=True, 
            max_length=512, return_tensors="pt"
        )
        input_ids = inputs_txt["input_ids"].to(DEVICE)
        attention_mask = inputs_txt["attention_mask"].to(DEVICE)
        
        # --- 推理 ---
        with torch.no_grad():
            # 注意：模型现在返回 4 个值：pres_logits, rel_logits, attn_weights
            outputs = model(pixel_values, input_ids, attention_mask)
            
            # 兼容旧版模型可能只返回 2 个值的情况
            if len(outputs) == 2:
                p_logits, r_logits = outputs
                attn_weights = None
            else:
                p_logits, r_logits, attn_weights = outputs
        
        # --- 后处理 ---
        # Presence (Sigmoid)
        p_probs = torch.sigmoid(p_logits).squeeze().cpu().numpy()
        if p_probs.ndim == 0: p_probs = np.array([p_probs]) # 标量保护
        
        # Relation (Softmax)
        r_probs = F.softmax(r_logits, dim=1).squeeze().cpu().numpy()
        if r_probs.ndim == 1: r_probs = r_probs.reshape(1, -1)

        # --- 结果格式化 ---
        # 1. 关系 Top 3
        top_k = 3
        top_indices = r_probs[0].argsort()[-top_k:][::-1]
        relation_results = {id_to_relation.get(idx, "Unknown"): float(r_probs[0][idx]) for idx in top_indices}
        
        # 2. 偏见指标
        presence_results = {name: float(prob) for name, prob in zip(presence_labels_names, p_probs)}
        
        # 3. 可视化生成
        # A. 文本高亮 (Image attends to Text)
        highlighted_html = text
        if attn_weights and 'i2t' in attn_weights:
            print(f"DEBUG: i2t_weights shape: {attn_weights['i2t'].shape}")
            print(f"DEBUG: input_ids shape: {input_ids.shape}")
            print(f"DEBUG: attention_mask shape: {attention_mask.shape}")
            highlighted_html = generate_text_highlight(text, input_ids, attention_mask, attn_weights['i2t'])
            # 包裹在 div 中以防样式冲突
            highlighted_html = f"<div style='line-height: 1.8; font-size: 16px;'>{highlighted_html}</div>"
        
        # B. 图片热力图 (Text attends to Image)
        heatmap_image = image
        if attn_weights and 't2i' in attn_weights:
            heatmap_image = generate_image_heatmap(image, pixel_values, attn_weights['t2i'])
            
        return relation_results, presence_results, highlighted_html, heatmap_image

    except Exception as e:
        import traceback
        error_msg = f"发生错误: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return {}, {}, f"<span style='color:red'>{error_msg}</span>", image

# -----------------------------------------------------------------------------
# 6. Gradio 界面构建
# -----------------------------------------------------------------------------
custom_css = """
#main-container {max_width: 1200px; margin: auto;}
.highlight-box {border: 1px solid #ddd; padding: 15px; border-radius: 8px; background: #f9f9f9;}
"""

with gr.Blocks(css=custom_css, title="Explainable News Bias Detector") as demo:
    gr.Markdown(
        """
        # 🕵️‍♂️ 可解释的多模态新闻偏见检测系统
        上传新闻图片和文本，系统不仅会预测偏见类型，还会通过**注意力热力图**展示模型是依据图片的哪个区域和文本的哪些词语做出的判断。
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📥 输入区域")
            img_input = gr.Image(type="pil", label="新闻图片")
            txt_input = gr.Textbox(lines=6, placeholder="在此输入新闻标题或正文...", label="新闻文本")
            submit_btn = gr.Button("🔍 开始深度分析", variant="primary", size="lg")
            
            gr.Examples(
                examples=[
                    # 替换为你本地的示例图片路径
                    ["apps/example.jpg", "The ruthless invaders destroyed the peaceful village, leaving children crying in the rubble."], 
                ],
                inputs=[img_input, txt_input]
            )

        with gr.Column(scale=1):
            gr.Markdown("### 📊 预测结果")
            
            # 1. 关系分类
            with gr.Accordion("🔗 图文关系推断", open=True):
                label_output_relation = gr.Label(num_top_classes=3, label="最可能的关系")
            
            # 2. 偏见指标
            with gr.Accordion("⚠️ 偏见指标检测 (概率)", open=False):
                label_output_presence = gr.Label(num_top_classes=5, label="Top 偏见类型")
                # 也可以用 BarPlot 展示所有指标
                # gr.BarPlot(...) 

    # -----------------------------------------------------------------------------
    # 7. 可解释性可视化区域 (新增核心部分)
    # -----------------------------------------------------------------------------
    gr.Markdown("---")
    gr.Markdown("### 🧠 模型注意力可视化 (Explainability)")
    gr.Markdown("*左侧展示了图片关注了哪些文字（高亮），右侧展示了文字关注了图片的哪些区域（热力图）。*")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("**📝 文本注意力高亮** (Image → Text)")
            # 使用 HTML 组件渲染带颜色的文本
            html_output = gr.HTML(label="Highlighted Text")
            
        with gr.Column(scale=1):
            gr.Markdown("**🖼️ 图像注意力热力图** (Text → Image)")
            img_output_heatmap = gr.Image(label="Attention Heatmap", type="pil")

    # 绑定事件
    # 输出顺序必须与 predict 函数返回值顺序一致
    submit_btn.click(
        fn=predict,
        inputs=[img_input, txt_input],
        outputs=[label_output_relation, label_output_presence, html_output, img_output_heatmap]
    )

if __name__ == "__main__":
    # share=True 生成公网链接，方便演示
    demo.launch(share=True, server_name="0.0.0.0")