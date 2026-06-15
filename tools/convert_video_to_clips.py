#!/usr/bin/env python3
"""
视频切片脚本：将目录下所有mp4视频切分成每10s一段的视频片段

使用方法：
    python video_splitter.py <videos目录路径>

依赖：
    - ffmpeg (必须安装并添加到系统PATH)
    - Python 3.6+

输出：
    在视频同级目录创建 output_原视频名/ 文件夹，存放切片后的片段
"""

import os
import sys
import subprocess
import glob
from pathlib import Path


def get_video_duration(video_path):
    """获取视频总时长（秒）"""
    cmd = [
        'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1', video_path
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        duration = float(result.stdout.strip())
        return duration
    except Exception as e:
        print(f"  错误：无法获取视频时长 - {e}")
        return None


def split_video(video_path, output_dir, segment_duration=10):
    """
    将视频切分成指定时长的片段

    Args:
        video_path: 输入视频路径
        output_dir: 输出目录
        segment_duration: 每段时长（秒），默认10秒
    """
    video_name = Path(video_path).stem
    duration = get_video_duration(video_path)

    if duration is None:
        return False

    print(f"  视频时长: {duration:.2f}s, 预计生成 {int(duration // segment_duration) + (1 if duration % segment_duration > 0 else 0)} 个片段")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 使用ffmpeg的segment muxer进行切片（更精确，避免重新编码）
    # 或者使用 -ss 和 -t 参数进行切片（兼容性更好）

    segment_count = 0
    for start_time in range(0, int(duration), segment_duration):
        end_time = min(start_time + segment_duration, duration)
        actual_duration = end_time - start_time

        # 生成输出文件名：原视频名_001.mp4, 原视频名_002.mp4, ...
        segment_count += 1
        output_file = os.path.join(output_dir, f"{video_name}_{segment_count:03d}.mp4")

        # 使用ffmpeg切片（关键帧精确切割，避免黑屏/花屏）
        # -ss 起始时间 -t 持续时间 -c copy 直接复制流（不重新编码，速度快）
        # 如果需要精确到帧，去掉 -c copy 进行重新编码（速度慢但精度高）
        cmd = [
            'ffmpeg', '-y', '-i', video_path,
            '-ss', str(start_time),
            '-t', str(actual_duration),
            '-c', 'copy',           # 直接复制，不重新编码（速度快）
            '-avoid_negative_ts', 'make_zero',
            '-reset_timestamps', '1',
            output_file
        ]

        try:
            subprocess.run(cmd, capture_output=True, check=True)
            print(f"    ✓ 生成片段 {segment_count:03d}: {start_time:.1f}s ~ {end_time:.1f}s")
        except subprocess.CalledProcessError as e:
            print(f"    ✗ 片段 {segment_count:03d} 生成失败: {e}")
            # 如果copy模式失败，尝试重新编码模式
            print(f"    尝试重新编码模式...")
            cmd_reencode = [
                'ffmpeg', '-y', '-i', video_path,
                '-ss', str(start_time),
                '-t', str(actual_duration),
                '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
                '-c:a', 'aac', '-b:a', '128k',
                '-reset_timestamps', '1',
                output_file
            ]
            try:
                subprocess.run(cmd_reencode, capture_output=True, check=True)
                print(f"    ✓ 生成片段 {segment_count:03d} (重新编码)")
            except subprocess.CalledProcessError as e2:
                print(f"    ✗ 重新编码也失败: {e2}")
                return False

    return True


def main():
    # 检查命令行参数
    if len(sys.argv) < 2:
        print("用法: python video_splitter.py <videos目录路径>")
        print("示例: python video_splitter.py ./videos")
        sys.exit(1)

    videos_dir = sys.argv[1]

    # 检查目录是否存在
    if not os.path.isdir(videos_dir):
        print(f"错误：目录不存在 - {videos_dir}")
        sys.exit(1)

    # 检查ffmpeg是否可用
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("错误：ffmpeg 未安装或未添加到PATH，请先安装ffmpeg")
        print("安装方法：")
        print("  - macOS: brew install ffmpeg")
        print("  - Ubuntu/Debian: sudo apt install ffmpeg")
        print("  - Windows: 下载ffmpeg并添加到PATH")
        sys.exit(1)

    # 查找所有mp4文件（包括子目录，如果不需要子目录可以去掉 recursive=True）
    mp4_files = glob.glob(os.path.join(videos_dir, "**/*.mp4"), recursive=True)
    # 排除已经生成的输出目录中的文件
    mp4_files = [f for f in mp4_files if not f.startswith(os.path.join(videos_dir, "output_"))]

    if not mp4_files:
        print(f"在 {videos_dir} 中未找到mp4视频文件")
        sys.exit(0)

    print(f"找到 {len(mp4_files)} 个mp4视频文件")
    print("=" * 50)

    success_count = 0
    fail_count = 0

    for video_path in sorted(mp4_files):
        video_name = Path(video_path).stem
        # 输出目录放在视频同级目录下
        output_dir = os.path.join(os.path.dirname(video_path), f"output_{video_name}")

        print(f"\n处理: {video_name}.mp4")
        print(f"  输出目录: {output_dir}")

        if split_video(video_path, output_dir, segment_duration=10):
            success_count += 1
        else:
            fail_count += 1

    print("\n" + "=" * 50)
    print(f"处理完成！成功: {success_count}, 失败: {fail_count}")
    print(f"切片后的视频保存在各视频对应的 output_xxx 目录中")


if __name__ == "__main__":
    main()
