import argparse
import glob
import os
import pickle
import shutil
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import lmdb
import tqdm

NUM_FRAMES = 20  # 每条视频抽取帧数
MAP_SIZE = 10 * 1024 * 1024 * 1024  # 10GB，必须大于总数据量


def _list_gpus() -> list:
    """检测所有可用的 NVIDIA GPU，返回 GPU ID 列表"""
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if r.returncode == 0 and r.stdout.strip():
            gpus = []
            for line in r.stdout.strip().split("\n"):
                parts = line.split(",", 1)  # 限制分割次数，GPU 名称可能含逗号
                if len(parts) >= 2:
                    idx, name = int(parts[0].strip()), parts[1].strip()
                    gpus.append((idx, name))
            for idx, name in gpus:
                print(f"  GPU {idx}: {name}")
            return [idx for idx, _ in gpus]
    except Exception:
        pass
    return []


def _probe_video(video_path: Path) -> tuple:
    """用 ffprobe 获取视频总帧数（读容器元数据，不解码）"""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=nb_frames,duration,r_frame_rate",
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

    total_frames = 0
    try:
        total_frames = int(info.get("nb_frames", 0))
    except (ValueError, TypeError):
        pass

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

    return total_frames


def _worker_extract(videos_batch, shard_dir, num_frames, gpu_id=-1):
    """
    Worker 进程：ffmpeg 抽帧输出 JPEG 临时文件 → 读入 LMDB。
    gpu_id >= 0 使用指定 GPU 硬件解码，-1 使用 CPU。
    GPU 失败自动回退 CPU。
    返回 (stats_dict, video_stems_set)
    """
    env = lmdb.open(shard_dir, map_size=MAP_SIZE)
    stats = {"success": 0, "empty": 0, "failed": 0}
    video_stems = set()

    # 每个 worker 独享临时目录
    tmp_dir = tempfile.mkdtemp(prefix=f"worker_frames_{gpu_id}_")

    for video_path in videos_batch:
        try:
            total_frames = _probe_video(video_path)
            if total_frames == 0:
                stats["empty"] += 1
                continue

            indices = [int(i * total_frames / num_frames) for i in range(num_frames)]
            select_expr = "+".join(f"eq(n\\,{idx})" for idx in indices)
            prefix = f"{video_path.parent.name}/{video_path.stem}"
            video_stems.add(prefix)

            def _build_cmd(use_gpu):
                c = ["ffmpeg", "-y", "-v", "error"]
                if use_gpu:
                    c += ["-hwaccel", "cuda", "-hwaccel_device", str(gpu_id)]
                c += ["-i", str(video_path)]
                if use_gpu:
                    vf = f"hwdownload,format=nv12,select='{select_expr}'"
                else:
                    vf = f"select='{select_expr}'"
                c += [
                    "-vf",
                    vf,
                    "-vsync",
                    "0",
                    "-frames:v",
                    str(num_frames),
                    "-q:v",
                    "2",
                    str(Path(tmp_dir) / "%04d.jpg"),
                ]
                return c

            # 尝试 GPU → 失败则回退 CPU
            use_gpu_flag = gpu_id >= 0
            ok = False
            for _ in range(2):
                # 清理上次尝试的残留文件
                for f in glob.glob(os.path.join(tmp_dir, "*.jpg")):
                    os.remove(f)

                cmd = _build_cmd(use_gpu_flag)
                proc = subprocess.run(cmd, capture_output=True, timeout=120)

                if proc.returncode == 0:
                    ok = True
                    break
                if use_gpu_flag:
                    use_gpu_flag = False  # GPU 失败，回退 CPU
                    continue
                # CPU 也失败
                err = proc.stderr.decode(errors="replace")[:300]
                print(f"[错误] {video_path.name}: {err}")
                break

            if not ok:
                stats["empty"] += 1
                continue

            # 读取 JPEG 文件写入 LMDB
            jpg_files = sorted(glob.glob(os.path.join(tmp_dir, "*.jpg")))
            if not jpg_files:
                stats["empty"] += 1
                continue

            txn = env.begin(write=True)
            frame_count = 0
            try:
                for jpg_path in jpg_files:
                    with open(jpg_path, "rb") as f:
                        data = f.read()
                    txn.put(f"{prefix}/{frame_count:02d}".encode(), data)
                    frame_count += 1
                txn.commit()
            except Exception:
                txn.abort()
                raise

            # 清理本轮临时文件
            for f in jpg_files:
                os.remove(f)

            stats["success" if frame_count > 0 else "empty"] += 1

        except Exception as e:
            print(f"[错误] {video_path.name}: {e}")
            stats["failed"] += 1

    # 清理临时目录
    shutil.rmtree(tmp_dir, ignore_errors=True)
    env.close()
    return stats, video_stems


def _merge_shards(shard_dirs, lmdb_path, map_size):
    """将多个 LMDB 分片合并为最终 LMDB"""
    final_env = lmdb.open(str(lmdb_path), map_size=map_size)

    for shard_dir in shard_dirs:
        if not Path(shard_dir).exists():
            continue
        shard_env = lmdb.open(shard_dir, readonly=True, lock=False)
        with shard_env.begin() as src_txn:
            with final_env.begin(write=True) as dst_txn:
                cursor = src_txn.cursor()
                while cursor.next():
                    dst_txn.put(cursor.key(), cursor.value())
        shard_env.close()

    final_env.close()


def main():
    parser = argparse.ArgumentParser(description="视频并行抽帧写入 LMDB（ffmpeg + 多进程分片）")
    parser.add_argument("--video-dir", required=True, help="视频根目录，如 ./dataset/videos")
    parser.add_argument("--lmdb-path", required=True, help="输出 LMDB 路径，如 ./game_frames.lmdb")
    parser.add_argument("--map-size", type=int, default=MAP_SIZE, help="最终 LMDB map_size 上限(字节)")
    parser.add_argument("--workers", "-w", type=int, default=8, help="并行进程数")
    parser.add_argument("--num-frames", type=int, default=NUM_FRAMES, help="每条视频抽取帧数")
    parser.add_argument("--gpu", action="store_true", help="使用 NVIDIA GPU 硬件解码加速（自动分配多 GPU）")
    args = parser.parse_args()

    # GPU 检测与分配
    gpu_ids = []
    if args.gpu:
        print("检测 GPU...")
        gpu_ids = _list_gpus()
        if not gpu_ids:
            print("未检测到 NVIDIA GPU，回退到 CPU 模式")
        else:
            print(f"共 {len(gpu_ids)} 个 GPU 可用")

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

    # ── 阶段 1：多进程并行抽帧 → 各自 LMDB 分片 ──
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = []
        for i, batch in enumerate(batches):
            if not batch:
                continue
            if gpu_ids:
                assigned_gpu = gpu_ids[i % len(gpu_ids)]
            else:
                assigned_gpu = -1
            tag = f"GPU{assigned_gpu}" if assigned_gpu >= 0 else "CPU"
            print(f"  Worker {i} ({len(batch)} 条视频) → {tag}")
            futures.append(executor.submit(_worker_extract, batch, shard_dirs[i], args.num_frames, assigned_gpu))
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
    final_env = lmdb.open(str(lmdb_path), map_size=args.map_size)
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
