import os
import pickle
from pathlib import Path

import lmdb
from tqdm import tqdm

FRAMES_ROOT = "/root/autodl-tmp/dataset"  # 你的预抽帧目录
LMDB_PATH = "/root/autodl-tmp/game_frames.lmdb"  # 输出文件
MAP_SIZE = 80 * 1024 * 1024 * 1024  # 80GB，必须大于总数据量


def build_lmdb():
    frames_root = Path(FRAMES_ROOT)
    env = lmdb.open(LMDB_PATH, map_size=MAP_SIZE, writemap=True)

    # 收集所有图片
    all_frames = sorted(frames_root.rglob("frame_*.jpg"))
    print(f"共发现 {len(all_frames)} 张图片，开始打包到 LMDB...")

    with env.begin(write=True) as txn:
        for frame_path in tqdm(all_frames):
            # 键格式: delta/1101076112_001/00
            relative = frame_path.relative_to(frames_root)
            key = str(relative.with_suffix("")).replace("\\", "/")  # 去掉 .jpg
            # frame_00 -> 00
            key = key.replace("/frame_", "/")

            # 读取二进制并写入
            with open(frame_path, "rb") as f:
                img_bytes = f.read()

            txn.put(key.encode(), img_bytes)

        # 写入元信息：视频列表
        video_stems = set()
        for f in all_frames:
            parts = f.relative_to(frames_root).parts
            if len(parts) >= 2:
                video_stems.add(f"{parts[0]}/{parts[1]}")  # game/video_stem

        meta = {
            "total_frames": len(all_frames),
            "video_stems": sorted(list(video_stems)),
            "frames_per_video": 20,
        }
        txn.put(b"__meta__", pickle.dumps(meta))

    env.close()
    print(f"LMDB 构建完成: {LMDB_PATH}")
    print(f"大小约: {os.path.getsize(LMDB_PATH) / 1024**3:.1f} GB")


if __name__ == "__main__":
    build_lmdb()
