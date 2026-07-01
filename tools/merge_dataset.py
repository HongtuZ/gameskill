import shutil
from pathlib import Path

from tqdm import tqdm

# 配置
INPUT_DIRS = [
    "output/dataset_01",
    "output/dataset_02",
    "output/dataset_03",
    "output/dataset_04",
    "output/dataset_05",
    "output/dataset_06",
    "output/dataset_07",
    "output/dataset_08",
    "output/dataset_09",
    "output/dataset_10",
]  # 要合并的多个目录（也支持 glob 模式，如 "output/dataset_*"）
OUTPUT_DIR = "dataset"  # 合并后的输出目录，与原始 dataset 结构一致
EXISTING_OK = True  # 目标已存在时是否跳过（False 则覆盖）


def resolve_input_dirs(input_dirs):
    """解析输入目录列表，支持 glob 模式"""
    resolved = []
    for d in input_dirs:
        p = Path(d)
        if "*" in d or "?" in d:
            # glob 模式：匹配所有符合条件的目录
            matched = sorted(Path(d).parent.glob(Path(d).name))
            matched = [m for m in matched if m.is_dir()]
            resolved.extend(matched)
        elif p.exists() and p.is_dir():
            resolved.append(p)
        else:
            print(f"[WARN] 目录不存在，跳过: {d}")
    return resolved


def get_all_files(input_dirs):
    """从多个输入目录中递归获取所有文件"""
    all_files = []
    for d in input_dirs:
        d = Path(d)
        files = list(d.rglob("*"))
        files = [f for f in files if f.is_file()]
        all_files.extend([(d, f) for f in files])
    return all_files


def merge_datasets(input_dirs, output_dir, existing_ok=True):
    """将多个目录中的文件合并到一个输出目录，保持与原始 dataset 相同的结构"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_files = get_all_files(input_dirs)
    print(f"共发现 {len(all_files)} 个文件")

    if len(all_files) == 0:
        print("没有找到任何文件！")
        return

    skipped = 0
    copied = 0
    skipped_files = []

    for source_base, file_path in tqdm(all_files, desc="Merging", unit="file"):
        # 保持相对于源目录的结构（如 videos/捷风/xxx.mp4）
        rel_path = file_path.relative_to(source_base)

        # 跳过 annotations.jsonl，单独处理
        if rel_path.name == "annotations.jsonl":
            continue

        dst = output_dir / rel_path

        if dst.exists():
            # 目标目录中已存在同名视频，跳过
            skipped += 1
            skipped_files.append(str(rel_path))
            continue

        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(file_path), str(dst))
        copied += 1

    print(f"\n完成！复制: {copied} 个文件, 跳过(已存在): {skipped} 个文件")
    if skipped_files:
        print("\n跳过的文件列表:")
        for f in skipped_files:
            print(f"  - {f}")
    print(f"\n合并结果在: {output_dir}")


def main():
    dirs = resolve_input_dirs(INPUT_DIRS)

    print("输入目录:")
    for d in dirs:
        print(f"  [✓] {d}")
    print(f"  共 {len(dirs)} 个目录")

    print(f"\n输出目录: {OUTPUT_DIR}")
    print(f"已存在跳过: {EXISTING_OK}")
    print()

    merge_datasets(dirs, OUTPUT_DIR, existing_ok=EXISTING_OK)


if __name__ == "__main__":
    main()
