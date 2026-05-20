import os
import json
import time
import base64
import requests
import pandas as pd
import logging
import traceback  # 新增：用于打印完整错误堆栈
from PIL import Image
from tqdm import tqdm

# ================= 配置区 =================
CSV_PATH = r"split_csv_output/part_006.csv"
OUTPUT_PATH = r"label/label_p6.json"
LOG_FILE = "dmx_labeling.log"

# DMXAPI 专用配置
MODEL = "gemini-2.5-flash-lite"
API_KEY = "sk-VEVYOwFHerqwvowhqqMJJuY5zJXWv5KafCUaJPMqv1EAUVVQ"
BASE_URL = "https://www.dmxapi.cn/v1beta"

SYSTEM_PROMPT = """
   **Role:** You are a senior scholar specializing in communication studies and visual psychology, with a focus on researching implicit biases and multimodal manipulation techniques in global news.

   **Task:** Please conduct an in-depth analysis of the provided [news headline/body] and [news image], and annotate them across multiple dimensions according to the **V-T-J-D Bias Evaluation System**."Regardless of the input language (Arabic, Chinese, English, etc.), please perform the semantic analysis based on the original cultural and linguistic context. The final analysis and JSON keys must remain in English for consistency."

   **Scoring Criteria for 0-3 Scale:**

   - **0 (None):** No evidence of the bias indicator. The presentation is neutral and purely factual.

   - **1 (Subtle):** Minor presence of the indicator. Might be accidental or a common journalistic practice, but slightly nudges the viewer's perception.

   - **2 (Obvious):** Clear intent to influence the viewer. Uses specific techniques (e.g., loaded words, dramatic angles) to frame the story.

   - **3 (Extreme):** Strong, systematic manipulation. Highly emotionalized or distorted presentation designed to provoke a specific reaction or dehumanize subjects.

------

   ### Bias Indicators Definition System:

   #### 1. Visual-Only Cues

   - **V1. Visual Salience Manipulation:** Whether certain details (e.g., painful tears, bloodstains) are deliberately highlighted through extreme close-ups, cropping, high saturation, or shallow depth of field while ignoring the overall context?

   - **V2. Subject-Centered Perspective & Social Distance**:

     > **Trigger Condition:** This indicator is marked as **True (Score 1-3)** ONLY when the primary subject is a **Human Being** (individual or group). If the image only contains objects, maps, or scenery without human presence, score **0**.

     **Scoring & Categorization (Must specify which case applies):**

     - **Low Angle (Worm’s Eye View):** Enhances the subject's status.
       - *Psychological Effect:* **Empowerment** (making a soldier look heroic), **Imposing** (making a leader look authoritative), or **Threatening** (making an aggressor look looming).
     - **High Angle (Bird’s Eye View):** Diminishes the subject's status.
       - *Psychological Effect:* **Victimization** (making a civilian look helpless), **Weakness** (showing subjects as small/insignificant), or **Pity** (inducing a sense of looking down upon suffering).
     - **Intimate Distance (Extreme Close-up):** Regardless of angle, if the camera is uncomfortably close to a human face.
       - *Psychological Effect:* **Forced Empathy** (invading the subject’s private space to highlight tears/pain, making the viewer feel a personal connection).
     - **Clinical Distance (Long Shot at Eye-level):** - *Psychological Effect:* **De-individualization** (treating humans as a faceless mass or statistics, common in "objective" but detached reporting).

   - **V3. Color/Lighting Rendering:** Whether unnatural color tones (e.g., cold blue filters implying evil; warm light suggesting justice) or strong contrasts between light and dark are used to create specific atmospheres?

   - **V4. Symbolic Visual Signs:** Are there elements such as national flags, police lines, religious symbols, ruins, etc., that carry strong cultural implications?

   #### 2. Text-Only Cues

   - **T1. Loaded Language Manipulation:** Does the text use emotionally charged vocabulary (e.g., "rioters" vs. "protesters", "invasion" vs. "action") to pre-establish positions?
   - **T2. Moral Judgment & Evaluation:** Does the text directly moralize about the subjects without factual support?

   #### 3. Joint/Interactive Biases (Image and Text)

   - **J1. Role Framing:** Do the image and text together construct clear dichotomies like "hero/villain" or "victim/aggressor"?

   - **J2. Selective Omission & Imbalance:** set J2 true if any of these apply：

     **Omission of relevant contextual facts**: key factual context is absent while relevant to interpretation (e.g., article shows protesters burning a building but omits that protest followed a lethal police action).

     **Single-sided imagery selection**: across the article, images show only one side’s violence/anger while text implies conflict between two sides.

     **Asymmetric humanization**: one side shown as individualized humans (faces, families) while the other side shown as faceless mobs/weapons or other subjects.

   - **J3. Stereotype Reinforcement:** Do the combinations of images and texts conform to stereotypes about specific ethnicities, classes, or cultures?


   #### 4. Multimodal Relationship Dimension

   - D1. Relationship Qualification:

     Goal: Determine which modality primarily controls interpretation.
     Choose ONE category by strictly following the order below (do NOT skip).

     - Tension / Mismatch
       Use if image and text guide interpretation in conflicting directions (e.g., calm text + chaotic image, irony, contradiction).
       Test: Would viewers feel cognitive dissonance?

     - Visual Amplification
       Use if text is factual/neutral AND image is emotionally or morally intense, driving audience reaction.
       Test: Remove image → emotional impact largely disappears.

     - Anchoring
       Use if image alone is ambiguous/neutral AND text assigns strong qualitative labels (e.g., “invaders”, “heroes”).
       Test: Remove text → image meaning becomes unclear.

     - Reinforcing (NOT default)
       Use ONLY if image alone AND text alone independently express the SAME evaluative stance, and neither dominates.
       Test: Remove either modality → remaining one still conveys same judgment.

     Neutral
     Use ONLY if both image and text are descriptive, non-evaluative, and emotionally neutral (rare in conflict news).

     Output ONE label only.

     **To reduce "Reinforcing" overclassification:** apply Reinforcing only if BOTH:

     1. Text uses evaluative words or clear framing *and* image visually supports the same evaluative framing (e.g., text calls group "aggressors" and image is low-angle of armed people), 

     2. There are no more specific categories above (Anchoring, Amplification, Tension).

        D2. Coherence Score:

     **Coherence_Score mapping (0–5):**
        Calculate based on:

        - Semantic Alignment (0–2):
          0: different events / meanings
             1: same event, different emphasis
             2: same event, same interpretation

        - Emotional Alignment (0–2):
          0: conflicting emotions
             1: weak or one-sided emotion
             2: shared emotional tone

        - Direction Consistency (0–1)
          0: pull interpretation in different directions

             1: guide interpretation in same direction
          ---

          Coherence_Score = Semantic Alignment (0–2) + Emotional Alignment (0–2) + Direction Consistency (0–1)



## Output Requirements

### 1. Analysis

Briefly outline:

- Visual composition, perspective, color, and human status
- Key textual cues and tones
- How image and text interact semantically

### 2. JSON Format (STRICT)

{
  "analysis_chain": {
    "visual_observation": "string",
    "textual_observation": "string",
    "joint_mechanism": "string"
  },
  "cues": {
    "visual_only": {
      "V1_Salience": {"present": boolean, "score": 0-3, "reason": "string"},
      "V2_Perspective": {
        "present": boolean,
        "score": 0-3,
        "reason": "string",
        "analysis": {
          "subject_type": "Soldier / Civilian / Leader / Mass Crowd / Other Human / No Human",
          "camera_angle": "Low / High / Eye-level / Extreme Close-up / None"
        }
      },
      "V3_Color_Lighting": {"present": boolean, "score": 0-3, "reason": "string"},
      "V4_Symbolism": {"present": boolean, "score": 0-3, "reason": "string"}
    },
    "text_only": {
      "T1_Loaded_Language": {"present": boolean, "score": 0-3, "reason": "string"},
      "T2_Moral_Judgment": {"present": boolean, "score": 0-3, "reason": "string"}
    },
    "joint_multimodal": {
      "J1_Role_Framing": {"present": boolean, "reason": "string"},
      "J2_Selective_Imbalance": {"present": boolean, "reason": "string"},
      "J3_Stereotyping": {"present": boolean, "reason": "string"}
    },
    "consistency_dimension": {
      "D1_Relationship_Type": "Reinforcing / Tension / Neutral / Anchoring / Amplification",
      "Coherence_Score": 0-5,
      "reason": "string"
    }
  },
  "overall_bias_intensity": 0-5
}

"""

# ================= 日志配置 =================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler() # 同时输出到控制台
    ]
)
logger = logging.getLogger(__name__)

# ================= 工具函数 =================

def encode_image(image_path):
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')

def load_progress():
    """
    断点续传核心：加载已处理的数据
    """
    if os.path.exists(OUTPUT_PATH):
        try:
            with open(OUTPUT_PATH, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if not content:
                    return [], set()
                data = json.loads(content)
                # 兼容处理：确保返回的是列表
                if isinstance(data, list):
                    processed_ids = {str(item.get('news_id')) for item in data if item.get('news_id')}
                    return data, processed_ids
        except json.JSONDecodeError:
            # 如果文件损坏（例如写了一半崩溃），备份并重新开始
            bak_path = OUTPUT_PATH + ".bak"
            os.rename(OUTPUT_PATH, bak_path)
            logger.error(f"⚠️ 检测到 JSON 文件损坏，已备份至 {bak_path}，任务将重新开始。")
    return [], set()

def save_data(data):
    """
    安全保存函数
    """
    temp_path = OUTPUT_PATH + ".tmp"
    try:
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(temp_path, OUTPUT_PATH) # 原子替换，防止写入时崩溃导致文件清空
    except Exception as e:
        logger.error(f"❌ 保存文件失败: {str(e)}")

# ================= 核心请求：针对 DMXAPI 的流式死磕函数 =================
def call_dmx_api_persistent(image_path, prompt_text, pbar, news_id):
    url = f"{BASE_URL}/models/{MODEL}:streamGenerateContent?key={API_KEY}&alt=sse"
    image_base64 = encode_image(image_path)
    
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{
            "parts": [
                {"inline_data": {"mime_type": "image/jpeg", "data": image_base64}},
                {"text": f"{SYSTEM_PROMPT}\n\n{prompt_text}"}
            ]
        }]
    }

    wait_time = 10
    retry_count = 0
    max_wait_hits = 0  # 计数器：记录达到60s等待的次数

    while True:
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=60, stream=True)
            
            if response.status_code == 429:
                raise requests.exceptions.HTTPError("429 Too Many Requests")
            
            if response.status_code != 200:
                logger.error(f"❌ API 异常状态码 {response.status_code}: {response.text}")
                response.raise_for_status()

            full_text = ""
            for line in response.iter_lines():
                if not line: continue
                line_str = line.decode('utf-8')
                if line_str.startswith('data: '):
                    json_str = line_str[6:]
                    try:
                        data = json.loads(json_str)
                        if 'candidates' in data:
                            for cand in data['candidates']:
                                for part in cand.get('content', {}).get('parts', []):
                                    if 'text' in part: full_text += part['text']
                    except:
                        full_text += json_str

            # 成功解析 JSON 后返回
            final_json_str = full_text.strip().replace('```json', '').replace('```', '').strip()
            first_brace, last_brace = final_json_str.find('{'), final_json_str.rfind('}')
            
            if first_brace != -1 and last_brace != -1:
                return json.loads(final_json_str[first_brace:last_brace+1])
            else:
                raise ValueError("未找到合法的JSON结构")

        except (requests.exceptions.HTTPError, requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            retry_count += 1
            
            # --- 核心改动点：判断是否触发跳过逻辑 ---
            if wait_time >= 60:
                max_wait_hits += 1
            
            if max_wait_hits >= 3:
                # 记录详细日志并抛出特定异常，或者直接返回 None
                logger.error(f"🚨 ID:{news_id} 连续3次60s网络重试失败，决定跳过此条。最后错误: {str(e)}")
                return None # 返回 None 告知上层跳过
            
            status_desc = "429限流" if "429" in str(e) else "网络抖动"
            pbar.set_description(f"⏳ [{status_desc}] ID:{news_id} | 重试 {retry_count} | 等待 {wait_time}s")
            
            time.sleep(wait_time)
            # 步进等待时间，最高 60s
            wait_time = min(wait_time + 10, 60)
            continue

        except Exception as e:
            logger.error(f"❌ ID:{news_id} 致命错误:\n{traceback.format_exc()}")
            # 这种非网络错误通常是因为图片损坏或Prompt违规，建议也返回 None 跳过
            return None

# ================= 主流程 =================
def start_labeling():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df = pd.read_csv(CSV_PATH)
    df['news_id'] = df['news_id'].astype(str)
    
    results, processed_ids = load_progress()
    to_process = df[~df['news_id'].isin(processed_ids)]
    
    print(f"🚀 任务启动 | 待处理: {len(to_process)}")
    
    pbar = tqdm(to_process.iterrows(), total=len(to_process), ncols=120)
    success_count = 0

    for _, row in pbar:
        aid = str(row['news_id'])
        title = str(row['title'])
        image_path = row['image_path']
        pbar.set_description(f"⚡ 正在分析 [{aid}]")

        if not os.path.exists(image_path):
            continue

        try:
            # 读取文本描述
            text_content = ""
            if 'text_path' in row and os.path.exists(str(row['text_path'])):
                with open(row['text_path'], 'r', encoding='utf-8') as f: text_content = f.read()
            if not text_content: text_content = str(row.get('description', ''))

            # 执行请求
            res_data = call_dmx_api_persistent(
                image_path, 
                f"Title: {title}\nContent: {text_content}", 
                pbar, 
                aid
            )
            
            # 如果返回 None，说明达到了跳过条件
            if res_data is None:
                pbar.write(f"⏭️ [跳过] ID: {aid} | 原因: 多次网络重试失败或格式错误")
                continue

            # 正常处理结果
            res_data['news_id'] = aid
            res_data['title'] = title
            results.append(res_data)
            success_count += 1
            
            # 每成功一条保存一次进度（更稳妥）
            save_data(results)
            pbar.write(f"✅ [成功] ID: {aid}")
            time.sleep(1.2)

        except Exception:
            logger.error(f"❌ 循环内未知异常 ID:{aid}:\n{traceback.format_exc()}")
            continue

    save_data(results)
    print(f"\n🎉 运行结束，共完成: {success_count} 条")

if __name__ == "__main__":
    start_labeling()