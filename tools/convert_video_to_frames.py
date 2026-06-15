import cv2
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import tqdm

FPS = 2
NUM_FRAMES = 20  # 10s × 2fps

def extract_and_save(video_path: Path, output_dir: Path):
    """将单条视频抽帧保存为图片序列"""
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        cap.release()
        return video_path.name, "empty"
    
    # 均匀采样索引
    indices = [int(i * total_frames / NUM_FRAMES) for i in range(NUM_FRAMES)]
    
    # 创建输出目录：frames/delta/xxx_001/
    frame_dir = output_dir / video_path.parent.name / video_path.stem
    frame_dir.mkdir(parents=True, exist_ok=True)
    
    # 检查是否已处理（断点续传）
    if (frame_dir / f"frame_{NUM_FRAMES-1:02d}.jpg").exists():
        cap.release()
        return video_path.name, "skipped"
    
    for i, idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        
        # BGR -> RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        save_path = frame_dir / f"frame_{i:02d}.jpg"
        cv2.imwrite(str(save_path), frame_rgb, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    
    cap.release()
    return video_path.name, "success"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-dir", required=True, help="视频根目录，如 ./dataset")
    parser.add_argument("--output-dir", default="./frames", help="帧输出目录")
    parser.add_argument("--workers", "-w", type=int, default=8, help="并行进程数")
    args = parser.parse_args()
    
    video_root = Path(args.video_dir)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    
    # 收集所有视频
    all_videos = list(video_root.rglob("*.mp4"))
    print(f"共发现 {len(all_videos)} 个视频，开始预抽帧...")
    
    # 多进程并行（CPU 密集型，用进程池）
    results = {"success": 0, "skipped": 0, "empty": 0, "failed": 0}
    
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(extract_and_save, v, output_root): v 
            for v in all_videos
        }
        
        for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
            try:
                name, status = future.result()
                results[status] = results.get(status, 0) + 1
            except Exception as e:
                results["failed"] += 1
                print(f"[错误] {e}")
    
    print(f"\n处理完成: {results}")
    print(f"帧数据保存至: {output_root}")


if __name__ == "__main__":
    main()