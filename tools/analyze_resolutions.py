"""
统计 WebDataset 中所有视频的帧数和第一帧分辨率分布。

用法:
    python tools/analyze_resolutions.py --webdataset-dir /path/to/webdataset
"""

import argparse
import io
import tarfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


def build_index(tar_dir: str) -> tuple[dict, dict]:
    """
    扫描所有 tar 文件，找到每个视频的第一帧位置和帧数。
    返回:
        first_frames: {video_prefix: (shard_path, offset, size)}
        frame_counts: {video_prefix: int}
    """
    tar_dir = Path(tar_dir)
    shards = sorted(str(p) for p in tar_dir.glob("*.tar"))
    if not shards:
        raise FileNotFoundError(f"No .tar files found in {tar_dir}")

    # 先收集每个视频的所有帧信息
    video_frames: dict[str, list] = defaultdict(list)

    for shard_path in tqdm(shards, desc="扫描 tar 文件"):
        with tarfile.open(shard_path, "r") as tar:
            for member in tar:
                if not member.name.endswith(".jpg"):
                    continue
                key = member.name[:-4]  # 去掉 .jpg
                parts = key.rsplit("_", 1)
                if len(parts) != 2:
                    continue
                video_prefix = parts[0]
                offset = member.offset_data
                size = member.size
                video_frames[video_prefix].append((shard_path, offset, size))

    # 每个视频只保留第一帧（offset 最小的）+ 帧数
    first_frames = {}
    frame_counts = {}
    for vp, frames in video_frames.items():
        frames.sort(key=lambda x: x[1])
        first_frames[vp] = frames[0]
        frame_counts[vp] = len(frames)

    print(f"共发现 {len(shards)} 个 shard, {len(first_frames)} 个视频")
    return first_frames, frame_counts


def main():
    parser = argparse.ArgumentParser(
        description="统计 WebDataset 中所有视频第一帧的分辨率及帧数分布",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--webdataset-dir",
        required=True,
        help="WebDataset tar 文件目录",
    )
    args = parser.parse_args()

    # 1. 建立索引：每个视频的第一帧位置 + 帧数
    first_frames, frame_counts = build_index(args.webdataset_dir)

    # 2. 按 shard 分组读取，减少文件开关
    by_shard: dict[str, list] = defaultdict(list)
    for vp, (shard_path, offset, size) in first_frames.items():
        by_shard[shard_path].append((vp, offset, size))

    # 3. 逐 shard 读取第一帧分辨率
    resolutions = {}
    for shard_path, entries in tqdm(by_shard.items(), desc="读取分辨率"):
        with open(shard_path, "rb") as f:
            for vp, offset, size in entries:
                try:
                    f.seek(offset)
                    jpg_data = f.read(size)
                    img = Image.open(io.BytesIO(jpg_data))
                    resolutions[vp] = img.size  # (width, height)
                except Exception as e:
                    print(f"  读取失败 {vp}: {e}")

    if not resolutions:
        print("未读取到任何分辨率数据！")
        return

    # 4. 统计分析
    widths = np.array([w for w, h in resolutions.values()])
    heights = np.array([h for w, h in resolutions.values()])
    res_counter = Counter(f"{w}x{h}" for w, h in resolutions.values())
    counts = np.array([frame_counts.get(vp, 0) for vp in resolutions])

    print("\n" + "=" * 60)
    print(f"  WebDataset 视频统计 (共 {len(resolutions)} 个视频)")
    print("=" * 60)

    # ── 帧数统计 ──
    fc_counter = Counter(frame_counts.get(vp, 0) for vp in resolutions)
    print("\n  帧数统计:")
    print(f"    最小值: {counts.min()}")
    print(f"    最大值: {counts.max()}")
    print(f"    平均值: {counts.mean():.1f}")
    print(f"    中位数: {np.median(counts):.1f}")
    print("\n  帧数分布:")
    for fc, cnt in sorted(fc_counter.items()):
        pct = cnt / len(resolutions) * 100
        bar = "█" * max(1, int(pct / 2))
        print(f"    {fc:>4} 帧  {cnt:>5} 个视频 ({pct:5.1f}%)  {bar}")

    # ── 分辨率统计 ──
    print("\n  分辨率 (第一帧):")
    print("\n  宽度 (width):")
    print(f"    最小值: {widths.min()}")
    print(f"    最大值: {widths.max()}")
    print(f"    平均值: {widths.mean():.1f}")
    print(f"    中位数: {np.median(widths):.1f}")

    print("\n  高度 (height):")
    print(f"    最小值: {heights.min()}")
    print(f"    最大值: {heights.max()}")
    print(f"    平均值: {heights.mean():.1f}")
    print(f"    中位数: {np.median(heights):.1f}")

    print("\n  分辨率分布 (Top 20):")
    for res, count in res_counter.most_common(20):
        pct = count / len(resolutions) * 100
        bar = "█" * int(pct / 2)
        print(f"    {res:>12}  {count:>5} ({pct:5.1f}%)  {bar}")

    # 按高度分组统计
    height_bins = [0, 360, 480, 540, 720, 900, 1080, 1440, 2160, float("inf")]
    print("\n  按高度区间分布:")
    for j in range(len(height_bins) - 1):
        lo, hi = height_bins[j], height_bins[j + 1]
        count = np.sum((heights > lo) & (heights <= hi))
        pct = count / len(resolutions) * 100
        bar = "█" * int(pct / 2)
        hi_label = f"{hi:>5}" if hi != float("inf") else "  ∞  "
        print(f"    ({lo:>5}, {hi_label}]  {count:>5} ({pct:5.1f}%)  {bar}")

    # 宽高比统计
    aspect_ratios = widths / heights
    print("\n  宽高比:")
    print(f"    最小值: {aspect_ratios.min():.3f}")
    print(f"    最大值: {aspect_ratios.max():.3f}")
    print(f"    平均值: {aspect_ratios.mean():.3f}")

    ar_counter = Counter()
    for w, h in resolutions.values():
        ratio = round(w / h, 2)
        ar_counter[ratio] += 1
    print("\n  常见宽高比 (Top 10):")
    for ratio, count in ar_counter.most_common(10):
        pct = count / len(resolutions) * 100
        print(f"    {ratio:>6.2f}  {count:>5} ({pct:5.1f}%)")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
