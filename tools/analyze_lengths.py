"""统计训练数据集中每个样本编码后的 token 长度分布（双GPU加速版，WebDataset + 原始视频回退）"""

import argparse
import os
import tempfile

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["IMAGE_MAX_TOKEN_NUM"] = "2048"
os.environ["VIDEO_MAX_TOKEN_NUM"] = "2048"
os.environ["FPS_MAX_FRAMES"] = "20"

import copy
import io
import json
import tarfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image
from tqdm import tqdm

# ========================== 默认配置区 ==========================
DEFAULT_WEBDATASET_DIR = "/mnt/data/1/user_workspace/zhouhongtu/gameskill_webdataset"
DEFAULT_JSONL_PATH = "dataset/dataset.jsonl"
PROMPTS_DIR = "prompts"
MODEL_ID = "Qwen/Qwen3.5-4B"
NUM_FRAMES = 20  # 每条视频抽取帧数
BATCH_SIZE = 100  # 每批处理的样本数
NUM_LOAD_WORKERS = 16  # 并行加载帧的线程数
NUM_ENCODE_WORKERS = 8  # 并行编码的线程数


# ========================== WebDataset 帧读取器 ==========================
class WebDatasetFrameReader:
    """
    从 WebDataset tar 文件读取视频帧。
    初始化时扫描所有 tar 建立索引（video_prefix → 帧位置），
    读取时按需 seek 到对应 offset，利用 OS page cache 加速。
    """

    def __init__(self, tar_dir: str, num_frames: int = 20):
        self.tar_dir = Path(tar_dir)
        self.num_frames = num_frames
        # 索引: video_prefix -> [(frame_idx, shard_path, offset, size), ...]
        self.index: Dict[str, list] = {}
        self.shards: List[str] = []
        self._build_index()

    def _build_index(self):
        """扫描所有 tar 文件，建立 video_prefix → 帧位置索引"""
        self.shards = sorted(str(p) for p in self.tar_dir.glob("*.tar"))
        if not self.shards:
            raise FileNotFoundError(f"No .tar files found in {self.tar_dir}")

        total_entries = 0
        for shard_path in tqdm(self.shards, desc="[扫描 tar 文件]"):
            with tarfile.open(shard_path, "r") as tar:
                for member in tar:
                    if not member.name.endswith(".jpg"):
                        continue
                    # key 格式: {video_prefix}_{global_idx:08d}
                    # 分离: video_prefix = key 去掉最后的 _NNNNNNNN
                    key = member.name[:-4]  # 去掉 .jpg
                    parts = key.rsplit("_", 1)
                    if len(parts) != 2:
                        continue
                    video_prefix = parts[0]
                    # member.offset 是 tar 内数据区的起始偏移
                    offset = member.offset_data
                    size = member.size
                    if video_prefix not in self.index:
                        self.index[video_prefix] = []
                    self.index[video_prefix].append((shard_path, offset, size))
                    total_entries += 1

        # 对每个视频的帧按 offset 排序（保证帧顺序）
        for vp in self.index:
            self.index[vp].sort(key=lambda x: x[1])

        print(
            f"WebDataset index built: {len(self.shards)} shards, {len(self.index)} videos, {total_entries} total frames"
        )

    def get_frames(self, video_key: str) -> List[Image.Image]:
        """
        video_key 格式: "parent_stem"  例: "valorant_001"
        对应 WebDataset key: "valorant_001_00000000", "valorant_001_00000001", ...
        """
        if video_key not in self.index:
            raise KeyError(f"Video key not found in WebDataset: {video_key}")

        frames = []
        # 按 shard 分组，减少 tar 文件开关次数
        by_shard: Dict[str, list] = {}
        for shard_path, offset, size in self.index[video_key]:
            by_shard.setdefault(shard_path, []).append((offset, size))

        for shard_path, locations in by_shard.items():
            with open(shard_path, "rb") as f:
                for offset, size in locations:
                    f.seek(offset)
                    jpg_data = f.read(size)
                    img = Image.open(io.BytesIO(jpg_data)).convert("RGB")
                    frames.append(img)

        return frames


# ========================== 帧加载（原始视频优先，WebDataset 回退） ==========================
def _load_sample(reader, sample):
    """加载单个样本的帧并构建编码输入。"""
    # 没有视频字段的样本直接跳过
    if "videos" not in sample or not sample["videos"]:
        return None

    video_path_str = sample["videos"][0]
    path_obj = Path(video_path_str)
    messages = copy.deepcopy(sample["messages"])
    video_key = f"{path_obj.parent.name}_{path_obj.stem}"

    frames = None

    try:
        frames = reader.get_frames(video_key)
    except KeyError:
        pass

    if frames is None or len(frames) == 0:
        return None  # 都失败，跳过

    return {"messages": messages, "videos": [frames]}


def gpu_worker(
    gpu_id: int,
    indices: List[int],
    result_path: str,
    total_samples: int,
    webdataset_dir: str,
    jsonl_path: str,
):
    """在指定 GPU 上编码指定索引范围的样本，结果写入 npy 文件"""
    try:
        _gpu_worker_impl(gpu_id, indices, result_path, total_samples, webdataset_dir, jsonl_path)
    except Exception as e:
        import traceback

        print(f"[FATAL] GPU {gpu_id} worker crashed: {e}")
        traceback.print_exc()


def _gpu_worker_impl(
    gpu_id: int,
    indices: List[int],
    result_path: str,
    total_samples: int,
    webdataset_dir: str,
    jsonl_path: str,
):
    """在指定 GPU 上编码指定索引范围的样本，结果写入 npy 文件。
    流式处理：逐个样本「读帧 → 编码 → 释放」，避免 OOM。
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    from swift import get_model_processor, get_template
    from swift.utils import get_logger, seed_everything

    logger = get_logger()
    seed_everything(42)

    # 1. 加载 prompts 和 samples
    all_samples = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                all_samples.append(json.loads(line))

    # 2. 初始化数据源
    reader = WebDatasetFrameReader(webdataset_dir, num_frames=NUM_FRAMES)
    logger.info(f"[GPU {gpu_id}] 数据源: {webdataset_dir}")

    # 3. 加载模型 & Template
    model, processor = get_model_processor(
        MODEL_ID,
        torch_dtype="bfloat16",
    )

    template = get_template(
        processor,
        default_system=None,
        max_length=None,
        loss_scale="default+ignore_empty_think",
        add_non_thinking_prefix=True,
        is_binary_loss_scale=False,
    )
    template.set_mode("train")
    if getattr(template, "use_model", False):
        template.model = model

    # 4. 分批流式处理：每批「并行读帧 → 并行编码 → 释放」，避免 OOM
    lengths = []
    skipped = 0
    errors = 0

    with tqdm(total=len(indices), desc=f"GPU {gpu_id}", position=gpu_id) as pbar:
        for batch_start in range(0, len(indices), BATCH_SIZE):
            batch_indices = indices[batch_start : batch_start + BATCH_SIZE]

            # 4a. 并行加载本批帧
            encode_inputs = {}
            with ThreadPoolExecutor(max_workers=NUM_LOAD_WORKERS) as loader:
                futures = {loader.submit(_load_sample, reader, all_samples[gidx]): gidx for gidx in batch_indices}
                for future in as_completed(futures):
                    gidx = futures[future]
                    try:
                        result = future.result()
                        if result is None:
                            skipped += 1
                        else:
                            encode_inputs[gidx] = result
                    except Exception as e:
                        errors += 1
                        if errors <= 5:
                            logger.warning(f"[GPU {gpu_id}] Sample {gidx} load failed: {e}")

            # 4b. 并行编码本批样本
            batch_lengths = []
            with ThreadPoolExecutor(max_workers=NUM_ENCODE_WORKERS) as encoder:
                enc_futures = {encoder.submit(template.encode, inp): gidx for gidx, inp in encode_inputs.items()}
                for future in as_completed(enc_futures):
                    gidx = enc_futures[future]
                    try:
                        encoded = future.result()
                        n = len(encoded["input_ids"])
                        lengths.append(n)
                        batch_lengths.append(n)
                    except Exception as e:
                        errors += 1
                        if errors <= 5:
                            logger.warning(f"[GPU {gpu_id}] Sample {gidx} encode failed: {e}")

            # 4c. 输出本批统计
            if batch_lengths:
                bl = np.array(batch_lengths)
                batch_id = batch_start // BATCH_SIZE + 1
                tqdm.write(
                    f"  [Batch {batch_id}] samples={len(bl)}, min={bl.min()}, max={bl.max()}, mean={bl.mean():.0f}"
                )

            pbar.update(len(batch_indices))

    # 保存结果
    lengths_arr = np.array(lengths, dtype=np.int64)
    np.save(result_path, lengths_arr)
    logger.info(f"[GPU {gpu_id}] Done: {len(lengths)} encoded, {skipped} skipped, {errors} errors")
    return len(lengths), errors, skipped


# ========================== 主函数 ==========================
def main():
    from swift.utils import get_logger

    logger = get_logger()

    num_gpus = args.num_gpus
    webdataset_dir = args.webdataset_dir
    jsonl_path = args.jsonl

    logger.info(f"WebDataset dir: {webdataset_dir}")
    logger.info(f"JSONL path: {jsonl_path}")
    logger.info(f"Num GPUs: {num_gpus}")

    # 统计总样本数
    total = 0
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                total += 1
    logger.info(f"Total samples: {total}, splitting across {num_gpus} GPUs")

    # 均分索引
    chunk_size = total // num_gpus
    splits = []
    for g in range(num_gpus):
        start = g * chunk_size
        end = start + chunk_size if g < num_gpus - 1 else total
        splits.append(list(range(start, end)))

    gpu_worker(
        gpu_id=0,
        indices=splits[0],
        result_path="result_gpu0.npy",
        total_samples=total,
        webdataset_dir=webdataset_dir,
        jsonl_path=jsonl_path,
    )
    return
    # 启动多进程
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    result_paths = []
    processes = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for gpu_id in range(num_gpus):
            rpath = os.path.join(tmpdir, f"result_gpu{gpu_id}.npy")
            result_paths.append(rpath)
            p = ctx.Process(
                target=gpu_worker,
                args=(gpu_id, splits[gpu_id], rpath, total, webdataset_dir, jsonl_path),
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        # 合并结果
        all_lengths = []
        for rpath in result_paths:
            if os.path.exists(rpath):
                all_lengths.append(np.load(rpath))

    if not all_lengths:
        logger.error("No results collected!")
        return

    lengths = np.concatenate(all_lengths)
    errors = total - len(lengths)

    # 输出统计
    print("\n" + "=" * 60)
    print(f"  数据集 Token 长度统计 (共 {len(lengths)} 个样本, {errors} 个编码失败)")
    print("=" * 60)
    print(f"  最小值:   {lengths.min():>10,}")
    print(f"  最大值:   {lengths.max():>10,}")
    print(f"  平均值:   {lengths.mean():>10,.1f}")
    print(f"  中位数:   {np.median(lengths):>10,.1f}")
    print(f"  标准差:   {lengths.std():>10,.1f}")
    for p in [50, 75, 90, 95, 99, 99.5, 99.9]:
        print(f"  P{p:<5}   {np.percentile(lengths, p):>10,.1f}")
    print("=" * 60)

    # 直方图
    print("\n  长度分布直方图:")
    bins = [0, 2048, 4096, 8192, 12288, 16384, 20480, 24576, 32768, 40960, 65536, float("inf")]
    for j in range(len(bins) - 1):
        lo, hi = bins[j], bins[j + 1]
        count = np.sum((lengths > lo) & (lengths <= hi))
        pct = count / len(lengths) * 100
        bar = "█" * int(pct / 2)
        hi_label = f"{hi:>7,}" if hi != float("inf") else "    ∞  "
        print(f"  ({lo:>7,}, {hi_label}]  {count:>5} ({pct:5.1f}%)  {bar}")

    # 建议
    print("\n  MAX_LENGTH 建议:")
    for threshold in [8192, 12288, 16384, 20480, 24576, 32768]:
        covered = np.sum(lengths <= threshold)
        pct = covered / len(lengths) * 100
        print(f"    MAX_LENGTH={threshold:>6}  可覆盖 {covered}/{len(lengths)} 样本 ({pct:.1f}%)")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="统计训练数据集中每个样本编码后的 token 长度分布（WebDataset + 原始视频回退）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--webdataset-dir",
        default=DEFAULT_WEBDATASET_DIR,
        help="WebDataset tar 文件目录（当未指定 --videos-dir 时使用）",
    )
    parser.add_argument(
        "--videos-dir",
        default=None,
        help="原始视频根目录，使用 ffmpeg 从原始视频抽帧。不指定则使用 WebDataset",
    )
    parser.add_argument(
        "--jsonl",
        default=DEFAULT_JSONL_PATH,
        help="训练标注 JSONL 文件路径",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=2,
        help="并行 GPU 数量",
    )
    args = parser.parse_args()

    # reader = WebDatasetFrameReader(args.webdataset_dir, num_frames=NUM_FRAMES)
    # import time

    # for key in tqdm(reader.index):
    #     start = time.perf_counter()
    #     frames = reader.get_frames(key)
    #     tqdm.write(f"{key}: {time.perf_counter() - start:.2f}s")
    main()
