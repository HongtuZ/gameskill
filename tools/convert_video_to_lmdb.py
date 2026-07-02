import argparse
import os
import pickle
import shutil
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import lmdb
import tqdm

NUM_FRAMES = 20  # 每条视频抽取帧数
MAP_SIZE = 25 * 1024 * 1024 * 1024  # 80GB，必须大于总数据量


def _worker_extract(videos_batch, shard_dir, num_frames):
    """
    Worker 进程：为每个进程分配独立的临时 LMDB 分片，完全并行写入，无锁竞争。
    返回 (stats_dict, video_stems_set)
    """
    env = lmdb.open(shard_dir, map_size=MAP_SIZE, writemap=True)
    stats = {"success": 0, "empty": 0, "failed": 0}
    video_stems = set()

    for video_path in videos_batch:
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames == 0:
            cap.release()
            stats["empty"] += 1
            continue

        indices = [int(i * total_frames / num_frames) for i in range(num_frames)]
        prefix = f"{video_path.parent.name}/{video_path.stem}"
        video_stems.add(prefix)

        txn = env.begin(write=True)
        frame_count = 0
        try:
            for i, idx in enumerate(indices):
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret:
                    continue
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                _, buf = cv2.imencode(".jpg", frame_rgb, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                txn.put(f"{prefix}/{i:02d}".encode(), buf.tobytes())
                frame_count += 1
            txn.commit()
        except Exception:
            txn.abort()
            raise
        finally:
            cap.release()

        stats["success" if frame_count > 0 else "empty"] += 1

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
    parser = argparse.ArgumentParser(description="视频并行抽帧写入 LMDB（每进程独立分片，最后合并）")
    parser.add_argument("--video-dir", required=True, help="视频根目录，如 ./dataset/videos")
    parser.add_argument("--lmdb-path", required=True, help="输出 LMDB 路径，如 ./game_frames.lmdb")
    parser.add_argument("--map-size", type=int, default=MAP_SIZE, help="最终 LMDB map_size 上限(字节)")
    parser.add_argument("--workers", "-w", type=int, default=8, help="并行进程数")
    parser.add_argument("--num-frames", type=int, default=NUM_FRAMES, help="每条视频抽取帧数")
    args = parser.parse_args()

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

    # ── 阶段 1：多进程并行写各自 LMDB 分片（无锁，真正并行） ──
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(_worker_extract, batch, shard_dirs[i], args.num_frames)
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
