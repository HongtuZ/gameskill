"""从 LMDB 中取出第一个视频样本的 20 帧，保存为 jpg 到当前目录"""
import io
import pickle
import lmdb
from PIL import Image

LMDB_PATH = "/root/autodl-tmp/game_frames.lmdb"
OUTPUT_DIR = "frames"  # 当前目录

env = lmdb.open(LMDB_PATH, readonly=True, lock=False, readahead=False)

with env.begin() as txn:
    meta_raw = txn.get(b"__meta__")
    meta = pickle.loads(meta_raw) if meta_raw else {}
    frames_per_video = meta.get("frames_per_video", 20)
    video_stems = meta.get("video_stems", [])

print(f"Meta: frames_per_video={frames_per_video}, video_count={len(video_stems)}")

if not video_stems:
    # 没有 video_stems 元数据，扫描所有 key 找第一个视频
    print("video_stems 为空，扫描 key 查找第一个视频...")
    with env.begin() as txn:
        cursor = txn.cursor()
        first_key = None
        for key, _ in cursor:
            k = key.decode("utf-8")
            if k == "__meta__":
                continue
            # key 格式: game/video_stem/00
            parts = k.rsplit("/", 1)
            if len(parts) == 2 and parts[1] == "00":
                first_key = parts[0]
                break
    if first_key is None:
        print("LMDB 中未找到任何帧数据！")
    else:
        video_key = first_key
else:
    video_key = video_stems[0]

print(f"\n提取视频: {video_key}")

for i in range(frames_per_video):
    frame_key = f"{video_key}/{i:02d}".encode("utf-8")
    with env.begin() as txn:
        img_data = txn.get(frame_key)
    if img_data is None:
        print(f"  帧 {i:02d} 不存在，停止")
        break
    img = Image.open(io.BytesIO(img_data)).convert("RGB")
    out_path = f"{OUTPUT_DIR}/frame_{i:02d}.jpg"
    img.save(out_path, quality=95)
    print(f"  帧 {i:02d} -> {out_path}  ({img.size[0]}x{img.size[1]})")

env.close()
print("\n完成！")
