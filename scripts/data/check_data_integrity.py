import os
import csv
import pandas as pd
from pathlib import Path
from tqdm import tqdm  # 用于显示进度条，如果没有安装可删除相关行或 pip install tqdm

def check_data_integrity():
    # ================= 配置区域 (请根据实际情况修改) =================
    CONFIG = {
        # CSV 文件路径
        'csv_path': 'data/processed/labels/test_new_label-gemini-flash-lite-2.5_cleaned.csv', 
        
        # 图片文件夹的根目录 (CSV中的路径通常是相对路径，会拼接到这里)
        # 如果CSV中已经是绝对路径，这里可以留空或填 '/'
        'img_root_dir': '',
        
        # 文本文件夹的根目录 (同上)
        'txt_root_dir': '',
        
        # CSV 中包含文件路径的列名 (请检查你的CSV表头)
        'img_col_name': 'image_path',   # 例如: 'image_path', 'img_file', 'photo'
        'txt_col_name': 'text_path'     # 例如: 'text_path', 'txt_file', 'article'
    }
    # =================================================================

    print(f"🔍 开始检查数据完整性...")
    print(f"📂 CSV 路径: {CONFIG['csv_path']}")
    
    if not os.path.exists(CONFIG['csv_path']):
        print(f"❌ 错误: 找不到 CSV 文件 '{CONFIG['csv_path']}'")
        return

    try:
        # 读取 CSV
        df = pd.read_csv(CONFIG['csv_path'])
        total_rows = len(df)
        print(f"📊 CSV 总行数: {total_rows}")
        
        if CONFIG['img_col_name'] not in df.columns:
            print(f"❌ 错误: CSV 中找不到图片列 '{CONFIG['img_col_name']}'")
            print(f"   可用列名: {list(df.columns)}")
            return
            
        if CONFIG['txt_col_name'] not in df.columns:
            print(f"❌ 错误: CSV 中找不到文本列 '{CONFIG['txt_col_name']}'")
            print(f"   可用列名: {list(df.columns)}")
            return

    except Exception as e:
        print(f"❌ 读取 CSV 失败: {e}")
        return

    missing_img_list = []
    missing_txt_list = []
    missing_both_list = []
    valid_count = 0

    print("\n⏳ 正在逐个检查文件 (可能需要几分钟)...")

    # 使用 tqdm 显示进度条 (如果未安装 tqdm，去掉 with tqdm(...) 改为直接遍历 df.iterrows())
    iterator = tqdm(df.iterrows(), total=total_rows, desc="Checking") if 'tqdm' in globals() else df.iterrows()

    for idx, row in iterator:
        img_rel_path = str(row[CONFIG['img_col_name']])
        txt_rel_path = str(row[CONFIG['txt_col_name']])

        # 构建完整路径
        # 逻辑：如果 csv 里是绝对路径，os.path.join 会忽略 root_dir (Linux/Mac 行为)，直接用 csv 路径
        # 如果 csv 里是相对路径，则拼接 root_dir
        full_img_path = os.path.join(CONFIG['img_root_dir'], img_rel_path) if CONFIG['img_root_dir'] else img_rel_path
        full_txt_path = os.path.join(CONFIG['txt_root_dir'], txt_rel_path) if CONFIG['txt_root_dir'] else txt_rel_path
        
        # 清理可能存在的多余斜杠导致的路径错误 (可选，视具体情况而定)
        # full_img_path = os.path.normpath(full_img_path) 
        # full_txt_path = os.path.normpath(full_txt_path)

        img_exists = os.path.exists(full_img_path)
        txt_exists = os.path.exists(full_txt_path)

        if not img_exists and not txt_exists:
            missing_both_list.append({
                'index': idx,
                'img_path': full_img_path,
                'txt_path': full_txt_path
            })
        elif not img_exists:
            missing_img_list.append({
                'index': idx,
                'img_path': full_img_path,
                'txt_path': full_txt_path # 记录对应的文本路径以便对照
            })
        elif not txt_exists:
            missing_txt_list.append({
                'index': idx,
                'img_path': full_img_path,
                'txt_path': full_txt_path
            })
        else:
            valid_count += 1

    # ================= 输出报告 =================
    print("\n" + "="*60)
    print("📋 数据完整性检查报告")
    print("="*60)
    print(f"✅ 有效数据 (图文均存在): {valid_count} 条 ({valid_count/total_rows*100:.2f}%)")
    print(f"⚠️  仅缺失图片: {len(missing_img_list)} 条")
    print(f"⚠️  仅缺失文本: {len(missing_txt_list)} 条")
    print(f"❌ 图文均缺失: {len(missing_both_list)} 条")
    print(f"🔥 总缺失数据量: {len(missing_img_list) + len(missing_txt_list) + len(missing_both_list)} 条")
    print("="*60)

    # 生成清洗后的 CSV 建议
    if valid_count < total_rows:
        output_clean_csv = "cleaned_train_data.csv"
        # 获取所有有效数据的索引
        valid_indices = set(range(total_rows)) - \
                        set([item['index'] for item in missing_img_list]) - \
                        set([item['index'] for item in missing_txt_list]) - \
                        set([item['index'] for item in missing_both_list])
        
        clean_df = df.loc[list(valid_indices)].reset_index(drop=True)
        clean_df.to_csv(output_clean_csv, index=False)
        print(f"\n💡 已生成清洗后的 CSV 文件: '{output_clean_csv}'")
        print(f"   你可以直接在训练代码中使用这个新文件！")

    # 打印部分缺失样本详情 (前 5 个)
    if missing_img_list:
        print("\n🚫 前 5 个缺失图片的样本详情:")
        for item in missing_img_list[:5]:
            print(f"   [Idx {item['index']}] 图片不存在: {item['img_path']}")
    
    if missing_txt_list:
        print("\n🚫 前 5 个缺失文本的样本详情:")
        for item in missing_txt_list[:5]:
            print(f"   [Idx {item['index']}] 文本不存在: {item['txt_path']}")

    if missing_both_list:
        print("\n🚫 前 5 个图文均缺失的样本详情:")
        for item in missing_both_list[:5]:
            print(f"   [Idx {item['index']}] 均不存在: {item['img_path']} & {item['txt_path']}")

    print("\n✨ 检查完成！")

if __name__ == "__main__":
    check_data_integrity()