"""
合并多个 JSONL 文件为一个。

用法:
    python tools/merge_jsonl.py -o output.jsonl input1.jsonl input2.jsonl input3.jsonl

    # 也支持 glob 模式
    python tools/merge_jsonl.py -o output.jsonl "dataset_*.jsonl"

    # 去重（基于整行内容）
    python tools/merge_jsonl.py -o output.jsonl --dedup input1.jsonl input2.jsonl

    # 打乱顺序
    python tools/merge_jsonl.py -o output.jsonl --shuffle input1.jsonl input2.jsonl
"""

import argparse
import json
import random
from pathlib import Path

from tqdm import tqdm


def merge_jsonl(input_files: list[str], output_file: str, dedup: bool = False, shuffle: bool = False):
    """将多个 JSONL 文件合并为一个"""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 解析输入文件（支持 glob）
    resolved_files = []
    for pattern in input_files:
        if "*" in pattern or "?" in pattern:
            matched = sorted(Path(pattern).parent.glob(Path(pattern).name))
            matched = [f for f in matched if f.is_file()]
            resolved_files.extend(matched)
        else:
            p = Path(pattern)
            if p.exists() and p.is_file():
                resolved_files.append(p)
            else:
                print(f"[WARN] 文件不存在，跳过: {pattern}")

    if not resolved_files:
        print("没有找到任何输入文件！")
        return

    print(f"输入文件 ({len(resolved_files)} 个):")
    for f in resolved_files:
        print(f"  - {f}")
    print()

    # 读取所有行
    all_lines = []
    seen = set()
    total_raw = 0
    skipped_dup = 0

    for filepath in tqdm(resolved_files, desc="读取文件"):
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                total_raw += 1

                # 简单校验是否为合法 JSON
                try:
                    json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"[WARN] 非法 JSON，跳过 ({filepath}): {e}")
                    continue

                if dedup:
                    if line in seen:
                        skipped_dup += 1
                        continue
                    seen.add(line)

                all_lines.append(line)

    if shuffle:
        random.shuffle(all_lines)

    # 写入输出文件
    with open(output_path, "w", encoding="utf-8") as f:
        for line in all_lines:
            f.write(line + "\n")

    print("\n完成！")
    print(f"  总行数(原始): {total_raw}")
    if dedup:
        print(f"  去重跳过:      {skipped_dup}")
    print(f"  写入行数:      {len(all_lines)}")
    print(f"  输出文件:      {output_path}")


def main():
    parser = argparse.ArgumentParser(description="合并多个 JSONL 文件为一个")
    parser.add_argument("inputs", nargs="+", help="输入的 JSONL 文件路径（支持 glob 模式）")
    parser.add_argument("-o", "--output", required=True, help="输出的 JSONL 文件路径")
    parser.add_argument("--dedup", action="store_true", help="去除重复行（基于整行内容）")
    parser.add_argument("--shuffle", action="store_true", help="打乱输出行的顺序")
    parser.add_argument("--seed", type=int, default=42, help="shuffle 的随机种子 (默认: 42)")

    args = parser.parse_args()

    if args.shuffle:
        random.seed(args.seed)

    merge_jsonl(args.inputs, args.output, dedup=args.dedup, shuffle=args.shuffle)


if __name__ == "__main__":
    main()
