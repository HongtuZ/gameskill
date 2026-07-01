#!/usr/bin/env python3
import argparse
import multiprocessing
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
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
    except Exception:
        return None


def split_video_reencode(
    video_path: Path, output_file: Path, start_time: float, duration: float, crf: int = 23, preset: str = "fast"
) -> bool:
    """使用重新编码模式切片（精确到帧，无黑屏）"""
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-ss",
        str(start_time),
        "-t",
        str(duration),
        "-c:v",
        "h264_nvenc",
        "-preset",
        preset,
        "-cq",
        str(crf),
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-movflags",
        "+faststart",
        str(output_file),
    ]
    try:
        subprocess.run(cmd, capture_output=True, check=True)
        return True
    except subprocess.CalledProcessError:
        return False


def split_single_video(
    video_path: Path, videos_dir: Path, output_dir: Path, segment_duration: int, crf: int, preset: str
) -> tuple[Path, bool, str]:
    """
    处理单个视频的完整切片任务（用于多进程并行）

    Returns:
        (video_path, success, message)
    """
    video_name = video_path.stem
    duration = get_video_duration(video_path)

    if duration is None:
        return (video_path, False, "无法获取视频时长")

    relative_path = video_path.relative_to(videos_dir)
    video_output_dir = output_dir / relative_path.parent / video_name
    video_output_dir.mkdir(parents=True, exist_ok=True)

    total_segments = int(duration // segment_duration) + (1 if duration % segment_duration > 5 else 0)

    for segment_idx in range(total_segments):
        start_time = segment_idx * segment_duration
        end_time = min(start_time + segment_duration, duration)
        actual_duration = end_time - start_time

        segment_num = segment_idx + 1
        output_file = video_output_dir / f"{segment_num:03d}.mp4"

        success = split_video_reencode(video_path, output_file, start_time, actual_duration, crf, preset)

        if not success:
            return (video_path, False, f"片段 {segment_num:03d} 生成失败")

    return (video_path, True, f"生成 {total_segments} 个片段")


def find_mp4_files(videos_dir: Path) -> list[Path]:
    """递归查找所有mp4文件，排除output目录"""
    mp4_files = []
    for path in videos_dir.rglob("*.mp4"):
        mp4_files.append(path)
    return sorted(mp4_files)


def main():
    parser = argparse.ArgumentParser(description="并行视频切片脚本：将videos目录下的所有mp4视频切分成指定时长")

    parser.add_argument("videos_dir", type=Path, help="videos目录路径（包含mp4视频的根目录）")

    parser.add_argument("-o", "--output", type=Path, default=Path("output"), help="输出目录路径（默认: ./output）")

    parser.add_argument("-d", "--duration", type=int, default=10, help="每段视频时长，单位秒（默认: 10）")

    parser.add_argument("-j", "--jobs", type=int, default=None, help="并行进程数（默认: CPU核心数）")

    parser.add_argument("--crf", type=int, default=23, help="重新编码时的 CRF 质量值（默认: 23）")

    parser.add_argument(
        "--preset",
        type=str,
        default="fast",
        choices=["ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow"],
        help="重新编码时的预设速度（默认: fast）",
    )

    args = parser.parse_args()

    videos_dir = args.videos_dir.resolve()
    output_dir = args.output.resolve()
    segment_duration = args.duration

    # 自动确定并行进程数
    if args.jobs is None:
        num_workers = multiprocessing.cpu_count()
    else:
        num_workers = args.jobs

    # 检查输入目录
    if not videos_dir.exists() or not videos_dir.is_dir():
        print(f"错误：目录不存在或不是目录 - {videos_dir}")
        sys.exit(1)

    # 检查ffmpeg
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("错误：ffmpeg 未安装或未添加到PATH")
        sys.exit(1)

    # 查找所有mp4文件
    mp4_files = find_mp4_files(videos_dir)

    if not mp4_files:
        print(f"在 {videos_dir} 中未找到mp4视频文件")
        sys.exit(0)

    print(f"找到 {len(mp4_files)} 个mp4视频文件")
    print(f"输出目录: {output_dir}")
    print(f"切片时长: {segment_duration}s")
    print(f"并行进程: {num_workers} 个")
    print(f"编码预设: {args.preset}, CRF: {args.crf}")
    print("=" * 50)

    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)

    # 准备任务函数（固定部分参数）
    task_func = partial(
        split_single_video,
        videos_dir=videos_dir,
        output_dir=output_dir,
        segment_duration=segment_duration,
        crf=args.crf,
        preset=args.preset,
    )

    success_count = 0
    fail_count = 0

    # 使用进程池并行处理
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # 提交所有任务
        future_to_video = {executor.submit(task_func, video_path): video_path for video_path in mp4_files}

        # 使用tqdm显示进度
        for future in tqdm(as_completed(future_to_video), total=len(mp4_files), desc="总进度", unit="video"):
            video_path, success, message = future.result()
            relative_path = video_path.relative_to(videos_dir)

            if success:
                success_count += 1
                tqdm.write(f"✓ {relative_path} → {message}")
            else:
                fail_count += 1
                tqdm.write(f"✗ {relative_path} → {message}")

    print("\n" + "=" * 50)
    print(f"处理完成！成功: {success_count}, 失败: {fail_count}")
    print(f"所有切片保存在: {output_dir}")


if __name__ == "__main__":
    main()
