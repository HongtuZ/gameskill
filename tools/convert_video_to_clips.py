#!/usr/bin/env python3
"""
视频切片脚本：将目录下所有mp4视频（包括子目录）切分成每10s一段的视频片段

使用方法：
    python video_splitter.py <videos目录路径> [选项]

依赖：
    - ffmpeg (必须安装并添加到系统PATH)
    - Python 3.7+
    - tqdm (pip install tqdm)

输出：
    所有切片输出到指定的 --output 目录下，保持原目录结构
"""

import argparse
import subprocess
import sys
from pathlib import Path

from tqdm import tqdm


def get_video_duration(video_path: Path) -> float | None:
    """获取视频总时长（秒）"""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        duration = float(result.stdout.strip())
        return duration
    except Exception as e:
        print(f"  错误：无法获取视频时长 - {e}")
        return None


def split_video(video_path: Path, output_dir: Path, segment_duration: int = 10) -> bool:
    """
    将视频切分成指定时长的片段

    Args:
        video_path: 输入视频路径
        output_dir: 输出目录
        segment_duration: 每段时长（秒），默认10秒
    """
    video_name = video_path.stem
    duration = get_video_duration(video_path)

    if duration is None:
        return False

    total_segments = int(duration // segment_duration) + (1 if duration % segment_duration > 5 else 0)
    print(f"  视频时长: {duration:.2f}s, 预计生成 {total_segments} 个片段")

    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)

    # 使用tqdm显示切片进度
    for segment_idx in tqdm(range(total_segments), desc="  切片", unit="seg", leave=False):
        start_time = segment_idx * segment_duration
        end_time = min(start_time + segment_duration, duration)
        actual_duration = end_time - start_time

        # 生成输出文件名：原视频名_0001.mp4
        segment_num = segment_idx + 1
        output_file = output_dir / f"{video_name}_{segment_num:04d}.mp4"

        cmd_reencode = [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-ss",
            str(start_time),
            "-t",
            str(actual_duration),
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "23",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-reset_timestamps",
            "1",
            str(output_file),
        ]
        try:
            subprocess.run(cmd_reencode, capture_output=True, check=True)
        except subprocess.CalledProcessError as e:
            tqdm.write(f"    ✗ 片段 {segment_num:03d} 生成失败: {e}")
            return False

    return True


def find_mp4_files(videos_dir: Path) -> list[Path]:
    """递归查找所有mp4文件，排除output目录"""
    mp4_files = []
    for path in videos_dir.rglob("*.mp4"):
        mp4_files.append(path)
    return sorted(mp4_files)


def main():
    parser = argparse.ArgumentParser(description="将videos目录下的所有mp4视频切分成每10s一段的视频片段")
    parser.add_argument("videos_dir", type=Path, help="videos目录路径（包含mp4视频的根目录）")
    parser.add_argument("-o", "--output", type=Path, default=Path("output"), help="输出目录路径（默认: ./output）")
    parser.add_argument("-d", "--duration", type=int, default=10, help="每段视频时长，单位秒（默认: 10）")
    args = parser.parse_args()

    videos_dir = args.videos_dir.resolve()
    output_dir = args.output.resolve()
    segment_duration = args.duration

    # 检查输入目录
    if not videos_dir.exists():
        print(f"错误：目录不存在 - {videos_dir}")
        sys.exit(1)
    if not videos_dir.is_dir():
        print(f"错误：不是目录 - {videos_dir}")
        sys.exit(1)

    # 检查ffmpeg
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("错误：ffmpeg 未安装或未添加到PATH")
        print("安装方法：")
        print("  macOS:   brew install ffmpeg")
        print("  Ubuntu:  sudo apt install ffmpeg")
        print("  Windows: 下载ffmpeg并添加到PATH")
        sys.exit(1)

    # 查找所有mp4文件
    mp4_files = find_mp4_files(videos_dir)

    if not mp4_files:
        print(f"在 {videos_dir} 中未找到mp4视频文件")
        sys.exit(0)

    print(f"找到 {len(mp4_files)} 个mp4视频文件")
    print(f"输出目录: {output_dir}")
    print(f"切片时长: {segment_duration}s")
    print("=" * 50)

    # 使用tqdm显示总进度
    success_count = 0
    fail_count = 0

    for video_path in tqdm(mp4_files, desc="总进度", unit="video"):
        # 计算相对路径，保持目录结构
        relative_path = video_path.relative_to(videos_dir)
        # 输出路径：output_dir / 相对目录 / 视频名（不含扩展名）
        video_output_dir = output_dir / relative_path.parent

        tqdm.write(f"\n处理: {relative_path}")
        tqdm.write(f"  输出到: {video_output_dir}")

        if split_video(video_path, video_output_dir, segment_duration):
            success_count += 1
        else:
            fail_count += 1

    print("\n" + "=" * 50)
    print(f"处理完成！成功: {success_count}, 失败: {fail_count}")
    print(f"所有切片保存在: {output_dir}")


if __name__ == "__main__":
    main()
