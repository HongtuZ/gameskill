import argparse
import glob
import io
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import tqdm

NUM_FRAMES = 20  # 每条视频抽取帧数
MAX_TAR_SIZE = 1 * 1024 * 1024 * 1024  # 每个 tar 包最大 1GB
JPEG_QUALITY = 3  # ffmpeg -q:v 值，2=最高质量，5=较小文件


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
                parts = line.split(",", 1)
                if len(parts) >= 2:
                    idx, name = int(parts[0].strip()), parts[1].strip()
                    gpus.append((idx, name))
            for idx, name in gpus:
                print(f"  GPU {idx}: {name}")
            return [idx for idx, _ in gpus]
    except Exception:
        pass
    return []


def _probe_video(video_path: Path) -> int:
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


class _TarWriter:
    """
    保持 tar 文件句柄常驻，避免每个样本都 open/close。
    自动在达到大小上限时切换到新 shard。
    """

    def __init__(self, output_dir: str, max_size: int, shard_start_idx: int = 0):
        self.output_dir = output_dir
        self.max_size = max_size
        self._shard_idx = shard_start_idx
        self._current_size = 0
        self._file_handle = None
        self._tar = None
        self._sample_count = 0
        self._open_next()

    def _open_next(self):
        """关闭当前 shard，打开新 shard"""
        if self._tar is not None:
            self._tar.close()
        self._file_path = os.path.join(self.output_dir, f"shard-{self._shard_idx:06d}.tar")
        self._file_handle = open(self._file_path, "wb")
        self._tar = tarfile.open(fileobj=self._file_handle, mode="w")
        self._current_size = 0
        self._shard_idx += 1

    def write_sample(self, key: str, jpg_bytes: bytes, json_bytes: bytes | None = None):
        """写入一个样本（jpg + 可选 json），自动处理 shard 切换"""
        # 估算写入大小：数据 + 每个文件 512B header + 对齐填充
        jpg_padded = (len(jpg_bytes) + 511) & ~511
        json_size = len(json_bytes) if json_bytes else 0
        json_padded = (json_size + 511) & ~511
        entry_size = jpg_padded + 512 + json_padded + 512 if json_bytes else jpg_padded + 512

        # shard 已满则切换
        if self._current_size + entry_size > self.max_size and self._current_size > 0:
            self._open_next()

        # 写入 jpg
        info = tarfile.TarInfo(name=f"{key}.jpg")
        info.size = len(jpg_bytes)
        self._tar.addfile(info, io.BytesIO(jpg_bytes))

        # 写入 json
        if json_bytes is not None:
            info = tarfile.TarInfo(name=f"{key}.json")
            info.size = json_size
            self._tar.addfile(info, io.BytesIO(json_bytes))

        self._current_size += entry_size
        self._sample_count += 1

    def close(self):
        """关闭当前 tar 并返回统计信息"""
        if self._tar is not None:
            self._tar.close()
            self._tar = None
        if self._file_handle is not None:
            self._file_handle.close()
            self._file_handle = None
        return {"shards": self._shard_idx, "samples": self._sample_count}

    @property
    def current_path(self) -> str:
        return self._file_path


def _worker_extract(
    videos_batch,
    worker_id,
    num_frames,
    gpu_id=-1,
    tmp_dir_base=None,
    max_tar_size=MAX_TAR_SIZE,
    jpeg_quality=JPEG_QUALITY,
    max_resolution=1080,
    staging_dir=None,
):
    """
    Worker 进程：ffmpeg 抽帧 → 写入 WebDataset tar 文件。
    gpu_id >= 0 使用指定 GPU 硬件解码，-1 使用 CPU。
    GPU 失败自动回退 CPU。
    """
    stats = {"success": 0, "empty": 0, "failed": 0}

    # 每个 worker 独享临时目录
    tmp_dir = tempfile.mkdtemp(prefix=f"wdt_w{worker_id}_", dir=tmp_dir_base)
    frames_dir = os.path.join(tmp_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    # 使用 TarWriter 保持 tar 句柄常驻（关键优化：避免每样本 open/close）
    writer = _TarWriter(tmp_dir, max_tar_size, shard_start_idx=0)

    # 全局样本计数器（跨视频递增）
    global_sample_idx = 0

    for video_path in videos_batch:
        try:
            total_frames = _probe_video(video_path)
            if total_frames == 0:
                stats["empty"] += 1
                continue

            indices = [int(i * total_frames / num_frames) for i in range(num_frames)]
            select_expr = "+".join(f"eq(n\\,{idx})" for idx in indices)
            video_prefix = f"{video_path.parent.name}_{video_path.stem}"

            # 缩放 filter：仅缩小超过 max_resolution 的视频，保持宽高比，输出偶数尺寸
            scale_expr = f"scale='min(iw,{max_resolution * 16 // 9})':'min(ih,{max_resolution})':force_original_aspect_ratio=decrease"

            def _build_cmd(use_gpu):
                c = ["ffmpeg", "-y", "-v", "error"]
                if use_gpu:
                    c += ["-hwaccel", "cuda", "-hwaccel_device", str(gpu_id)]
                c += ["-i", str(video_path)]
                if use_gpu:
                    vf = f"hwdownload,format=nv12,{scale_expr},select='{select_expr}'"
                else:
                    vf = f"{scale_expr},select='{select_expr}'"
                c += [
                    "-vf",
                    vf,
                    "-vsync",
                    "0",
                    "-frames:v",
                    str(num_frames),
                    "-q:v",
                    str(jpeg_quality),
                    str(Path(frames_dir) / "%04d.jpg"),
                ]
                return c

            # 尝试 GPU → 失败则回退 CPU
            use_gpu_flag = gpu_id >= 0
            ok = False
            for _ in range(2):
                for f in glob.glob(os.path.join(frames_dir, "*.jpg")):
                    os.remove(f)

                cmd = _build_cmd(use_gpu_flag)
                proc = subprocess.run(cmd, capture_output=True, timeout=120)

                if proc.returncode == 0:
                    ok = True
                    break
                if use_gpu_flag:
                    use_gpu_flag = False
                    continue
                err = proc.stderr.decode(errors="replace")[:300]
                print(f"[错误] {video_path.name}: {err}")
                break

            if not ok:
                stats["empty"] += 1
                continue

            jpg_files = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
            if not jpg_files:
                stats["empty"] += 1
                continue

            # 读取 JPEG 并写入 tar（句柄保持打开）
            frame_count = 0
            for jpg_path in jpg_files:
                with open(jpg_path, "rb") as f:
                    jpg_data = f.read()

                # key 格式: {parent}_{stem}_{global_idx:08d}，与训练代码的 video_key 一致
                sample_key = f"{video_prefix}_{global_sample_idx:08d}"

                # 精简 JSON：仅保留必要元信息
                meta = {"v": video_prefix, "f": frame_count}
                json_bytes = json.dumps(meta, separators=(",", ":")).encode("utf-8")

                writer.write_sample(sample_key, jpg_data, json_bytes)
                frame_count += 1
                global_sample_idx += 1

            # 清理本轮临时帧文件
            for f in jpg_files:
                os.remove(f)

            stats["success" if frame_count > 0 else "empty"] += 1

        except Exception as e:
            print(f"[错误] {video_path.name}: {e}")
            stats["failed"] += 1

    # 关闭 tar writer（确保数据 flush）
    writer_info = writer.close()

    # 将 tar 文件移动到 staging 目录（带 worker_id 前缀避免冲突）
    tar_files = sorted(glob.glob(os.path.join(tmp_dir, "*.tar")))
    moved_tars = []
    for tf in tar_files:
        dest = os.path.join(staging_dir, f"w{worker_id}_" + os.path.basename(tf))
        shutil.move(tf, dest)
        moved_tars.append(dest)

    # 清理临时目录
    shutil.rmtree(tmp_dir, ignore_errors=True)

    return stats, moved_tars, writer_info


def main():
    parser = argparse.ArgumentParser(
        description="视频并行抽帧写入 WebDataset（ffmpeg + 多进程，每 tar ≤ 1GB）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--video-dir", required=True, help="视频根目录，如 ./dataset/videos")
    parser.add_argument("--output-dir", required=True, help="输出 WebDataset tar 目录，如 ./webdataset_out")
    parser.add_argument("--workers", "-w", type=int, default=8, help="并行进程数")
    parser.add_argument("--num-frames", type=int, default=NUM_FRAMES, help="每条视频抽取帧数")
    parser.add_argument("--max-tar-size", type=int, default=MAX_TAR_SIZE, help="每个 tar 包最大字节数（默认 1GB）")
    parser.add_argument(
        "--jpeg-quality", type=int, default=JPEG_QUALITY, help="JPEG 质量 (ffmpeg -q:v)，2=最高质量，5=较小文件"
    )
    parser.add_argument("--max-resolution", type=int, default=1080, help="最大分辨率高度（像素），超过此值会等比缩小")
    parser.add_argument("--gpu", action="store_true", help="使用 NVIDIA GPU 硬件解码加速（自动分配多 GPU）")
    parser.add_argument("--tmp-dir", default=None, help="临时文件目录（默认 /tmp，磁盘不足时可指定其他路径）")
    args = parser.parse_args()

    # 临时目录
    tmp_dir_base = args.tmp_dir
    if tmp_dir_base:
        Path(tmp_dir_base).mkdir(parents=True, exist_ok=True)
        print(f"临时文件目录: {tmp_dir_base}")

    # 输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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
    all_videos = sorted(video_root.rglob("*.mp4"))
    print(f"共发现 {len(all_videos)} 个视频，开始并行抽帧...")
    print(
        f"参数: workers={args.workers}, num_frames={args.num_frames}, jpeg_quality={args.jpeg_quality}, "
        f"max_resolution={args.max_resolution}p, max_tar_size={args.max_tar_size / 1024**3:.1f}GB"
    )

    # 将视频均分给各 worker
    batches = [[] for _ in range(args.workers)]
    for i, v in enumerate(all_videos):
        batches[i % args.workers].append(v)

    total_stats = {"success": 0, "empty": 0, "failed": 0}

    # staging 目录：worker 产出的 tar 先放这里，最后统一整理到 output_dir
    staging_dir = tempfile.mkdtemp(prefix="wdt_staging_", dir=tmp_dir_base)

    # ── 阶段 1：多进程并行抽帧 → 各自 tar 文件 ──
    all_tar_files = []
    total_writer_samples = 0
    total_writer_shards = 0

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
            futures.append(
                executor.submit(
                    _worker_extract,
                    batch,
                    i,
                    args.num_frames,
                    assigned_gpu,
                    tmp_dir_base,
                    args.max_tar_size,
                    args.jpeg_quality,
                    args.max_resolution,
                    staging_dir,
                )
            )
        for future in tqdm.tqdm(as_completed(futures), total=len(futures), desc="抽帧"):
            try:
                stats, tar_files, writer_info = future.result()
                for k in total_stats:
                    total_stats[k] += stats.get(k, 0)
                all_tar_files.extend(tar_files)
                total_writer_samples += writer_info["samples"]
                total_writer_shards += writer_info["shards"]
            except Exception as e:
                total_stats["failed"] += 1
                print(f"[错误] {e}")

    # ── 阶段 2：将所有 worker 的 tar 文件重命名到输出目录 ──
    print("整理 tar 文件...")
    all_tar_files_sorted = sorted(all_tar_files)
    final_tars = []

    for idx, src_tar_path in enumerate(tqdm.tqdm(all_tar_files_sorted, desc="整理tar")):
        if not Path(src_tar_path).exists():
            continue
        tar_size = os.path.getsize(src_tar_path)
        if tar_size == 0:
            continue
        dest_name = f"shard-{idx:06d}.tar"
        dest_path = output_dir / dest_name
        shutil.move(src_tar_path, str(dest_path))
        final_tars.append(dest_path)

    # 写入索引文件
    index = {
        "total_samples": total_writer_samples,
        "total_tars": len(final_tars),
        "frames_per_video": args.num_frames,
        "jpeg_quality": args.jpeg_quality,
        "max_resolution": args.max_resolution,
        "max_tar_size": args.max_tar_size,
        "tar_files": [t.name for t in final_tars],
    }
    with open(output_dir / "index.json", "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

    # 清理 staging 目录
    shutil.rmtree(staging_dir, ignore_errors=True)

    print(f"\n处理完成: {total_stats}")
    print(f"WebDataset 保存至: {output_dir}")
    print(f"总样本数: {total_writer_samples}")
    print(f"tar 文件数: {len(final_tars)}")
    total_size = sum(p.stat().st_size for p in final_tars)
    print(f"总大小: {total_size / 1024**3:.2f} GB")


if __name__ == "__main__":
    main()
