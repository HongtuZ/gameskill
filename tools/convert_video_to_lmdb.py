import argparse
import os
import pickle
import shutil
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import lmdb
import numpy as np
import tqdm

NUM_FRAMES = 20  # 每条视频抽取帧数
MAP_SIZE = 25 * 1024 * 1024 * 1024  # 80GB，必须大于总数据量


def _check_gpu() -> bool:
    """检测是否有可用的 NVIDIA GPU"""
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True, timeout=5
        )
        if r.returncode == 0 and r.stdout.strip():
            name = r.stdout.strip().split("\n")[0]
            print(f"检测到 GPU: {name}")
            return True
    except Exception:
        pass
    return False


def _probe_video(video_path: Path) -> tuple:
    """用 ffprobe 获取视频总帧数和分辨率（读容器元数据，不解码）"""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=nb_frames,width,height,duration,r_frame_rate",
        "-of",
        "default=noprint_wrappers=1",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
    info = {}
    for line in result.stdout.strip().split("\n"):
        if "=" in line:
            k, v = line.split("=", 1)
            info[k.strip()] = v.strip()

    width = int(info.get("width", 0))
    height = int(info.get("height", 0))

    # 优先用 nb_frames
    total_frames = 0
    try:
        total_frames = int(info.get("nb_frames", 0))
    except (ValueError, TypeError):
        pass

    # 回退到 duration * fps
    if total_frames <= 0:
        try:
            duration = float(info.get("duration", 0))
            frac = info.get("r_frame_rate", "")
            if duration > 0 and "/" in frac:
                num, den = frac.split("/")
                fps = int(num) / int(den)
                total_frames = int(duration * fps)
        except Exception:
            pass

    return total_frames, width, height


def _worker_extract(videos_batch, shard_dir, num_frames, use_gpu=False):
    """
    Worker 进程：用 ffmpeg 管道抽帧，直接写入独立 LMDB 分片。
    返回 (stats_dict, video_stems_set)
    """
    env = lmdb.open(shard_dir, map_size=MAP_SIZE, writemap=True)
    stats = {"success": 0, "empty": 0, "failed": 0}
    video_stems = set()

    for video_path in videos_batch:
        try:
            total_frames, width, height = _probe_video(video_path)
            if total_frames == 0 or width == 0 or height == 0:
                stats["empty"] += 1
                continue

            indices = [int(i * total_frames / num_frames) for i in range(num_frames)]
            # select 过滤器：只输出指定序号的帧
            select_expr = "+".join(f"eq(n\\,{idx})" for idx in indices)
            prefix = f"{video_path.parent.name}/{video_path.stem}"
            video_stems.add(prefix)

            # ffmpeg 解码 → rawvideo 管道输出，纯内存无磁盘 I/O
            cmd = ["ffmpeg", "-v", "error"]
            if use_gpu:
                cmd += ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]
            cmd += ["-i", str(video_path)]
            if use_gpu:
                # GPU 解码后先下载到 CPU，再做 select 过滤
                vf = f"hwdownload,format=nv12,format=rgb24,select='{select_expr}'"
            else:
                vf = f"select='{select_expr}'"
            cmd += [
                "-vf",
                vf,
                "-vsync",
                "0",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "pipe:1",
            ]
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            raw, _ = proc.communicate(timeout=120)

            if proc.returncode != 0 or len(raw) == 0:
                stats["empty"] += 1
                continue

            frame_size = width * height * 3  # 单帧字节数 (RGB)
            actual_frames = len(raw) // frame_size
            if actual_frames == 0:
                stats["empty"] += 1
                continue

            # 写入 LMDB
            txn = env.begin(write=True)
            frame_count = 0
            try:
                for i in range(actual_frames):
                    offset = i * frame_size
                    chunk = raw[offset : offset + frame_size]
                    frame_rgb = np.frombuffer(chunk, dtype=np.uint8).reshape(height, width, 3)
                    # RGB → BGR → JPEG 编码
                    frame_bgr = frame_rgb[:, :, ::-1]
                    _, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                    txn.put(f"{prefix}/{frame_count:02d}".encode(), buf.tobytes())
                    frame_count += 1
                txn.commit()
            except Exception:
                txn.abort()
                raise

            stats["success" if frame_count > 0 else "empty"] += 1

        except Exception as e:
            print(f"[错误] {video_path.name}: {e}")
            stats["failed"] += 1

    env.close()
    return stats, video_stems


def _merge_shards(shard_dirs, lmdb_path, map_size):
    """将多个 LMDB 分片合并为最终 LMDB"""
    final_env = lmdb.open(str(lmdb_path), map_size=map_size, writemap=True)

    for shard_dir in shard_dirs:
        shard_env = lmdb.open(shard_dir, readonly=True, lock=False)
        with shard_env.begin() as src_txn:
            with final_env.begin(write=True) as dst_txn:
                cursor = src_txn.cursor()
                while cursor.next():
                    dst_txn.put(cursor.key(), cursor.value())
        shard_env.close()

    final_env.close()


def main():
    parser = argparse.ArgumentParser(description="视频并行抽帧写入 LMDB（ffmpeg 管道 + 多进程分片）")
    parser.add_argument("--video-dir", required=True, help="视频根目录，如 ./dataset/videos")
    parser.add_argument("--lmdb-path", required=True, help="输出 LMDB 路径，如 ./game_frames.lmdb")
    parser.add_argument("--map-size", type=int, default=MAP_SIZE, help="最终 LMDB map_size 上限(字节)")
    parser.add_argument("--workers", "-w", type=int, default=8, help="并行进程数")
    parser.add_argument("--num-frames", type=int, default=NUM_FRAMES, help="每条视频抽取帧数")
    parser.add_argument("--gpu", action="store_true", help="使用 NVIDIA GPU 硬件解码加速")
    args = parser.parse_args()

    # GPU 检测
    use_gpu = args.gpu
    if use_gpu and not _check_gpu():
        print("未检测到 NVIDIA GPU，回退到 CPU 模式")
        use_gpu = False

    video_root = Path(args.video_dir)
    lmdb_path = Path(args.lmdb_path)
    lmdb_path.parent.mkdir(parents=True, exist_ok=True)

    all_videos = sorted(video_root.rglob("*.mp4"))
    print(f"共发现 {len(all_videos)} 个视频，开始并行抽帧...")

    # 将视频均分给各 worker，每个 worker 写独立临时 LMDB
    tmp_root = Path(tempfile.mkdtemp(prefix="lmdb_shards_"))
    batches = [[] for _ in range(args.workers)]
    for i, v in enumerate(all_videos):
        batches[i % args.workers].append(v)

    shard_dirs = [str(tmp_root / f"shard_{i}") for i in range(args.workers)]

    total_stats = {"success": 0, "empty": 0, "failed": 0}
    all_video_stems = set()

    # ── 阶段 1：多进程并行 ffmpeg 管道抽帧 → 各自 LMDB 分片（无锁，真正并行） ──
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(_worker_extract, batch, shard_dirs[i], args.num_frames, use_gpu)
            for i, batch in enumerate(batches)
            if batch
        ]
        for future in tqdm.tqdm(as_completed(futures), total=len(futures), desc="抽帧"):
            try:
                stats, stems = future.result()
                for k in total_stats:
                    total_stats[k] += stats.get(k, 0)
                all_video_stems.update(stems)
            except Exception as e:
                total_stats["failed"] += 1
                print(f"[错误] {e}")

    # ── 阶段 2：合并所有分片到最终 LMDB ──
    print("合并 LMDB 分片...")
    _merge_shards([d for d in shard_dirs if Path(d).exists()], lmdb_path, args.map_size)

    # 写入元信息
    meta = {
        "total_frames": total_stats["success"] * args.num_frames,
        "video_stems": sorted(all_video_stems),
        "frames_per_video": args.num_frames,
    }
    final_env = lmdb.open(str(lmdb_path), map_size=args.map_size, writemap=True)
    with final_env.begin(write=True) as txn:
        txn.put(b"__meta__", pickle.dumps(meta))
    final_env.close()

    # 清理临时分片
    shutil.rmtree(tmp_root, ignore_errors=True)

    print(f"\n处理完成: {total_stats}")
    print(f"LMDB 保存至: {lmdb_path}")
    print(f"大小约: {os.path.getsize(str(lmdb_path)) / 1024**3:.1f} GB")


if __name__ == "__main__":
    main()
