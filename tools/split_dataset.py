import shutil
from pathlib import Path

from tqdm import tqdm

# 配置
SOURCE_DIR = "dataset"  # 源目录
OUTPUT_DIR = "output"  # 输出目录
NUM_SPLITS = 10  # 分成10份


def get_all_videos(source_dir):
    """递归获取所有视频文件"""
    source_dir = Path(source_dir)
    video_paths = list(source_dir.rglob("*.mp4"))
    sorted_videos = sorted(video_paths, key=lambda x: str(x))
    return sorted_videos


def split_videos(videos, num_splits):
    """将视频列表均匀分成num_splits份"""
    # 计算每份大小
    total = len(videos)
    base_size = total // num_splits
    remainder = total % num_splits

    splits = []
    start = 0
    for i in range(num_splits):
        # 前remainder份多分配一个
        size = base_size + (1 if i < remainder else 0)
        end = start + size
        splits.append(videos[start:end])
        start = end

    return splits


def copy_videos(source_dir, output_dir, splits):
    """
    将视频分配到对应文件夹（带进度条）
    """
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)

    # 外层进度条：遍历每个 split
    for i, split_videos in enumerate(tqdm(splits, desc="Processing splits", unit="split"), 1):
        split_dir = output_dir / f"{source_dir.name}_{i:02d}"
        split_dir.mkdir(parents=True, exist_ok=True)

        # 内层进度条：复制当前 split 的视频
        for video_path in tqdm(
            split_videos,
            desc=f"Split {i:02d}",
            unit="file",
            leave=False,  # 完成后清除进度条，保持界面整洁
        ):
            dst = split_dir / video_path.relative_to(source_dir)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(video_path), str(dst))

        tqdm.write(f"Split {i:02d}: {len(split_videos)} videos -> {split_dir}")


def main():
    print(f"Scanning {SOURCE_DIR}...")
    videos = get_all_videos(SOURCE_DIR)
    print(f"Found {len(videos)} videos")

    if len(videos) == 0:
        print("No videos found!")
        return

    splits = split_videos(videos, NUM_SPLITS)

    # 验证分配
    print("\nDistribution:")
    for i, s in enumerate(splits, 1):
        print(f"  Split {i:02d}: {len(s)} videos")

    print(f"\nCreating splits in {OUTPUT_DIR}...")
    copy_videos(SOURCE_DIR, OUTPUT_DIR, splits)

    print("\nDone! Each split folder contains:")


if __name__ == "__main__":
    main()
