import pandas as pd
import os
import sys

def check_and_clean_data():
    # ================= 配置区域 =================
    CSV_PATH = 'data/processed/labels/test_new_label-gemini-flash-lite-2.5_cleaned.csv'
    
    # 需要检查的列名列表
    PRESENCE_COLS = [
        "M1_Affect_Mismatch_present", 
        "M2_Binary_Roles_present", 
        "T1_Agent_Label_present", 
        "T2_Causal_Attribution_present", 
        "V1_Salience_present", 
        "V2_Color_Polarity_present", 
        "V3_Power_Angle_present", 
        "V4_Visual_Selectivity_present",
        "M3_Symbol_Decontex_present" # 注意：你列表中这个在最后，我把它加进去了
    ]
    
    OUTPUT_CLEAN_PATH = 'data/processed/labels/test_new_label-gemini-flash-lite-2.5_cleaned_FIXED.csv'
    # ===========================================

    print(f"🔍 正在加载文件: {CSV_PATH} ...")
    
    if not os.path.exists(CSV_PATH):
        print(f"❌ 错误：找不到文件 '{CSV_PATH}'")
        print("   请确认当前运行目录是否正确，或检查文件路径。")
        return

    try:
        df = pd.read_csv(CSV_PATH)
    except Exception as e:
        print(f"❌ 读取 CSV 失败: {e}")
        return

    print(f"📊 数据总行数: {len(df)}")
    
    # 检查配置的列是否都在 CSV 中
    missing_cols = [col for col in PRESENCE_COLS if col not in df.columns]
    if missing_cols:
        print(f"⚠️  警告：以下列在 CSV 中未找到，将跳过检查: {missing_cols}")
        # 只保留存在的列进行后续检查
        valid_cols = [col for col in PRESENCE_COLS if col in df.columns]
    else:
        valid_cols = PRESENCE_COLS

    if not valid_cols:
        print("❌ 没有有效的列可以检查，程序退出。")
        return

    print(f"\n🔎 开始检查 {len(valid_cols)} 个目标列的空值情况...")
    
    # 定义什么是“空值”：pd.isna 覆盖 NaN 和 None，另外我们手动检查空字符串
    def is_empty(val):
        if pd.isna(val):
            return True
        if isinstance(val, str) and val.strip() == "":
            return True
        return False

    # 找出所有包含空值的行索引
    bad_indices = []
    
    # 逐行检查 (对于大文件可能稍慢，但为了精确展示哪一列空了，这样最稳妥)
    # 优化：使用向量化操作先快速筛选，再详细展示
    mask = df[valid_cols].applymap(is_empty).any(axis=1)
    bad_df = df[mask]
    
    count_bad = len(bad_df)
    
    if count_bad == 0:
        print("\n✅ 太棒了！所有指定列都没有空值。无需操作。")
        return

    print(f"\n🚨 发现 {count_bad} 行数据存在空值！")
    print("-" * 60)
    print("以下是前 10 个有问题样本的详情 (显示缺失的具体列):")
    print("-" * 60)

    # 打印前 10 个样本的详细信息
    for idx, row in bad_df.head(10).iterrows():
        empty_cols_in_row = [col for col in valid_cols if is_empty(row[col])]
        # 尝试打印一点上下文信息 (如果有 'id' 或 'image_path' 列)
        context_info = ""
        if 'id' in row: context_info += f"ID: {row['id']} "
        if 'image_path' in row: context_info += f"Img: {row['image_path']} "
        
        print(f"📍 行索引 {idx}: {context_info}")
        print(f"   ❌ 缺失值的列: {', '.join(empty_cols_in_row)}")
        print("-" * 30)

    if count_bad > 10:
        print(f"... 还有 {count_bad - 10} 行未显示。")

    print("\n" + "="*60)
    print("🤔 如何处理这些数据？")
    print("="*60)
    print("选项 1: 删除所有包含空值的行 (推荐，保证数据质量)")
    print("选项 2: 不删除，保持原样 (仅查看报告)")
    print("选项 3: 退出")
    
    while True:
        choice = input("\n请输入选项 (1/2/3): ").strip()
        
        if choice == '1':
            print(f"\n⏳ 正在删除 {count_bad} 行问题数据...")
            # 保留那些 **不** 在 bad_indices 中的行
            clean_df = df[~mask].reset_index(drop=True)
            
            # 保存到新文件
            # 确保输出目录存在
            os.makedirs(os.path.dirname(OUTPUT_CLEAN_PATH), exist_ok=True)
            clean_df.to_csv(OUTPUT_CLEAN_PATH, index=False)
            
            print(f"✅ 成功！")
            print(f"   原始行数: {len(df)}")
            print(f"   删除行数: {count_bad}")
            print(f"   剩余行数: {len(clean_df)}")
            print(f"   💾 新文件已保存至: {OUTPUT_CLEAN_PATH}")
            print(f"\n💡 下一步：请在你的训练代码中将 CSV 路径改为上面的新文件路径。")
            break
            
        elif choice == '2':
            print("\nℹ️  已跳过删除操作。原始文件未修改。")
            print("   你可以手动检查数据或稍后再次运行此脚本。")
            break
            
        elif choice == '3':
            print("\n👋 已退出。")
            sys.exit(0)
        else:
            print("❌ 无效输入，请输入 1, 2 或 3。")

if __name__ == "__main__":
    check_and_clean_data()