import os
import json
import glob
import argparse

def load_json_file(filepath):
    """加载单个 JSON 文件，返回列表"""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
        if isinstance(data, dict):
            return [data]
        elif isinstance(data, list):
            return data
        else:
            raise ValueError(f"Unsupported data type in {filepath}: {type(data)}")

def merge_and_sort_json_files(input_pattern, output_file, numeric_news_id=False):
    all_data = []
    
    files = sorted(glob.glob(input_pattern))
    if not files:
        print(f"⚠️  警告：未找到匹配 '{input_pattern}' 的文件。")
        return

    for filepath in files:
        print(f"📄 加载 {filepath}...")
        all_data.extend(load_json_file(filepath))

    # 排序逻辑
    def get_sort_key(item):
        nid = item.get('news_id')
        if numeric_news_id:
            try:
                return int(nid)
            except (TypeError, ValueError):
                return float('inf')  # 无法转数字的放最后
        else:
            return str(nid) if nid is not None else ''

    all_data.sort(key=get_sort_key)

    # 写入结果
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 合并完成！共 {len(all_data)} 条记录，已保存至：{output_file}")

def main():
    parser = argparse.ArgumentParser(description="合并多个 JSON 文件并按 news_id 升序排序")
    parser.add_argument(
        "input_pattern",
        help="输入文件的通配符模式（例如 'label_*.json'）"
    )
    parser.add_argument(
        "-o", "--output",
        default="merged_output.json",
        help="输出文件名（默认：merged_output.json）"
    )
    parser.add_argument(
        "-n", "--numeric",
        action="store_true",
        help="将 news_id 视为数字进行排序（否则按字符串排序）"
    )

    args = parser.parse_args()

    merge_and_sort_json_files(
        input_pattern=args.input_pattern,
        output_file=args.output,
        numeric_news_id=args.numeric
    )

if __name__ == "__main__":
    main()