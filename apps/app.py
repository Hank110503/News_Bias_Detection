import os
import torch
import torch.nn as nn
import pandas as pd
import gradio as gr
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, RobertaTokenizer, RobertaModel
import torch.nn.functional as F

# -----------------------------------------------------------------------------
# 1. 模型定义 (必须与训练时完全一致)
# -----------------------------------------------------------------------------
class MultimodalBiasModel(nn.Module):
    def __init__(self, clip_name, text_model_name, num_presence_labels, num_relation_classes, finetune_text=True):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(clip_name)
        self.text_encoder = RobertaModel.from_pretrained(text_model_name)

        # 这里的参数需要与加载的权重匹配，但推理时不需要调整requires_grad
        img_dim = self.clip.config.projection_dim  # 512
        txt_dim = self.text_encoder.config.hidden_size  # 1024
        joint_dim = img_dim + txt_dim

        self.presence_head = nn.Sequential(
            nn.Linear(joint_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_presence_labels)
        )

        self.relation_head = nn.Sequential(
            nn.Linear(joint_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_relation_classes)
        )

    def forward(self, pixel_values, input_ids, attention_mask):
        img_feat = self.clip.get_image_features(pixel_values=pixel_values)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)  # L2 Norm

        txt_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        txt_feat = txt_outputs.last_hidden_state[:, 0, :]  # [CLS] token

        joint = torch.cat([img_feat, txt_feat], dim=1)
        return self.presence_head(joint), self.relation_head(joint)

# -----------------------------------------------------------------------------
# 2. 全局配置与初始化
# -----------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "multimodal_bias_model.pt"
CSV_PATH = "label-gemini-flash-lite-2.5.csv"

# 环境变量 (如果你的环境需要)
os.environ["HF_HOME"] = "D:/hf_cache"

print("正在初始化资源...")

# A. 加载标签映射 (为了将预测的 ID 转回文字)
try:
    df = pd.read_csv(CSV_PATH)
    df = df.dropna(subset=["D1_Relationship_Type"])
    unique_relations = sorted(df["D1_Relationship_Type"].unique())
    id_to_relation = {idx: label for idx, label in enumerate(unique_relations)}
    print(f"✅ 加载了 {len(unique_relations)} 个关系类别")
except Exception as e:
    print(f"❌ 错误: 无法读取 CSV 文件 ({e})。请确保文件存在。")
    # 作为一个 fallback，防止程序崩溃
    id_to_relation = {0: "Unknown"} 

presence_labels_names = [
    "V1_Salience (视觉显著性)", 
    "V2_Perspective (拍摄视角)", 
    "V3_Color_Lighting (光影色彩)",
    "V4_Symbolism (符号隐喻)", 
    "T1_Loaded_Language (情感色彩词)", 
    "T2_Moral_Judgment (道德评判)",
    "J1_Role_Framing (角色框架)", 
    "J2_Selective_Imbalance (选择性失衡)", 
    "J3_Stereotyping (刻板印象)"
]

# B. 加载模型
print("正在加载模型权重...")
# 初始化模型结构
model = MultimodalBiasModel(
    clip_name="openai/clip-vit-base-patch32",
    text_model_name="roberta-large",
    num_presence_labels=9,
    num_relation_classes=len(unique_relations)
).to(DEVICE)

# 加载训练好的权重
try:
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    print("✅ 模型加载成功")
except FileNotFoundError:
    print(f"❌ 错误: 未找到模型文件 {MODEL_PATH}")
    exit()

# C. 加载处理器
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
text_tokenizer = RobertaTokenizer.from_pretrained("roberta-large")

# -----------------------------------------------------------------------------
# 3. 预测函数
# -----------------------------------------------------------------------------
def predict(image, text):
    if image is None or not text:
        return "请提供图片和文本", {}

    # 1. 预处理
    try:
        # 图片
        inputs_img = clip_processor(images=image, return_tensors="pt")
        pixel_values = inputs_img["pixel_values"].to(DEVICE)
        
        # 文本
        inputs_txt = text_tokenizer(
            text, padding="max_length", truncation=True, 
            max_length=512, return_tensors="pt"
        )
        input_ids = inputs_txt["input_ids"].to(DEVICE)
        attention_mask = inputs_txt["attention_mask"].to(DEVICE)
    except Exception as e:
        return f"预处理出错: {str(e)}", {}

    # 2. 推理
    with torch.no_grad():
        p_logits, r_logits = model(pixel_values, input_ids, attention_mask)
        
        # 处理 Presence (多标签分类 -> Sigmoid)
        p_probs = torch.sigmoid(p_logits).squeeze().cpu().numpy()
        
        # 处理 Relation (多分类 -> Softmax)
        r_probs = F.softmax(r_logits, dim=1).squeeze().cpu().numpy()

    # 3. 格式化输出结果
    
    # A. 关系判断 (取 Top 3)
    top_k = 3
    # 获取概率最高的索引
    top_indices = r_probs.argsort()[-top_k:][::-1]
    relation_results = {}
    for idx in top_indices:
        label_name = id_to_relation.get(idx, "Unknown")
        relation_results[label_name] = float(r_probs[idx])

    # B. 偏见指标 (Presence)
    # 我们将概率转换为字典返回
    presence_results = {}
    for name, prob in zip(presence_labels_names, p_probs):
        presence_results[name] = float(prob)
        
    return relation_results, presence_results

# -----------------------------------------------------------------------------
# 4. 构建 Gradio 界面
# -----------------------------------------------------------------------------
custom_css = """
#component-0 {max_width: 1000px; margin: auto;}
.output-class {font-size: 1.1em; font-weight: bold;}
"""

with gr.Blocks(css=custom_css, title="News Bias Detector") as demo:
    gr.Markdown(
        """
        # 📰 Multimodal News Bias Detection System
        上传新闻图片并输入相关文本，模型将分析其中潜在的视觉与文本偏见，并推断人物/实体间的关系。
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            # 输入区
            img_input = gr.Image(type="pil", label="News Image")
            txt_input = gr.Textbox(lines=5, placeholder="Enter news text here...", label="News Text")
            submit_btn = gr.Button("🚀 Analyze Bias", variant="primary")
            
            gr.Examples(
                examples=[
                    ["example1.jpg", "Police officers were seen blocking the protesters."], # 替换为你自己的示例文件路径
                ],
                inputs=[img_input, txt_input]
            )

        with gr.Column(scale=1):
            # 输出区
            gr.Markdown("### 🔍 Analysis Results")
            
            gr.Markdown("#### 1. Social Relationship (人物/实体关系)")
            label_output_relation = gr.Label(num_top_classes=3, label="Predicted Relationship")
            
            gr.Markdown("#### 2. Bias Indicators (偏见指标存在概率)")
            # 使用 Label 组件显示多标签概率，虽然它通常用于互斥分类，但在 Gradio 中也能很好地展示置信度列表
            label_output_presence = gr.Label(label="Bias Indicators Presence")

    # 绑定事件
    submit_btn.click(
        fn=predict,
        inputs=[img_input, txt_input],
        outputs=[label_output_relation, label_output_presence]
    )

if __name__ == "__main__":
    demo.launch(share=True) # share=True 会生成一个临时的公网链接