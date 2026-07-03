"""统计训练数据集中每个样本编码后的 token 长度分布（双GPU加速版）"""
import os
import tempfile

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["IMAGE_MAX_TOKEN_NUM"] = "2048"
os.environ["VIDEO_MAX_TOKEN_NUM"] = "2048"
os.environ["FPS_MAX_FRAMES"] = "20"

import io
import json
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
from pathlib import Path
from typing import List

import lmdb
import numpy as np
from PIL import Image
from tqdm import tqdm

# ========================== 配置区 ==========================
LMDB_PATH = "/root/autodl-tmp/game_frames.lmdb"
JSONL_PATH = "dataset.jsonl"
PROMPTS_DIR = "prompts"
MODEL_ID = "Qwen/Qwen3.5-4B"
NUM_GPUS = 2
NUM_THREADS_PER_GPU = 16  # 96GB显存充裕，每GPU可用更多线程
NUM_PRELOAD_WORKERS = 32  # 预加载帧的线程数


# ========================== LMDB 帧读取器 ==========================
class LMDBFrameReader:
    def __init__(self, lmdb_path: str):
        self.env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, max_readers=64)
        with self.env.begin() as txn:
            meta_raw = txn.get(b"__meta__")
            self.meta = pickle.loads(meta_raw) if meta_raw else {}
            self.frames_per_video = self.meta.get("frames_per_video", 20)

    def get_frames(self, video_key: str) -> List[Image.Image]:
        frames = []
        with self.env.begin() as txn:
            for i in range(self.frames_per_video):
                frame_key = f"{video_key}/{i:02d}".encode("utf-8")
                img_jpg = txn.get(frame_key)
                if img_jpg is None:
                    break
                img = Image.open(io.BytesIO(img_jpg)).convert("RGB")
                frames.append(img)
        return frames

    def close(self):
        self.env.close()


# ========================== 单 GPU Worker ==========================
def _load_sample(lmdb_reader, sample, prompts):
    """加载单个样本的帧并构建编码输入"""
    video_path = sample["videos"][0]
    path_obj = Path(video_path)
    prompt = prompts.get(path_obj.parent.name)
    messages = copy.deepcopy(sample["messages"])
    messages[0]["content"] += "以下游戏知识供参考：\n\n" + prompt + "\n\n" if prompt else ""
    video_key = f"{path_obj.parent.name}/{path_obj.stem}"
    frames = lmdb_reader.get_frames(video_key)
    if len(frames) == 0:
        raise ValueError(f"Empty frames for key={video_key}, path={video_path}")
    return {"messages": messages, "videos": [frames]}


def gpu_worker(gpu_id: int, indices: List[int], result_path: str, total_samples: int):
    """在指定 GPU 上编码指定索引范围的样本，结果写入 npy 文件"""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    import torch
    from swift import get_model_processor, get_template
    from swift.utils import get_logger, seed_everything

    logger = get_logger()
    seed_everything(42)

    # 1. 加载 prompts 和 samples
    prompts = {}
    for prompt_file in Path(PROMPTS_DIR).glob("*.md"):
        prompts[prompt_file.stem] = prompt_file.read_text(encoding="utf-8")
    all_samples = []
    with open(JSONL_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                all_samples.append(json.loads(line))

    # 2. 并行预加载所有帧到内存（消除编码阶段的 I/O 瓶颈）
    lmdb_reader = LMDBFrameReader(LMDB_PATH)
    logger.info(f"[GPU {gpu_id}] Preloading {len(indices)} samples' frames into memory ...")
    encode_inputs = [None] * len(indices)
    with ThreadPoolExecutor(max_workers=NUM_PRELOAD_WORKERS) as loader:
        futures = {}
        for local_idx, global_idx in enumerate(indices):
            futures[loader.submit(_load_sample, lmdb_reader, all_samples[global_idx], prompts)] = local_idx
        for future in as_completed(futures):
            local_idx = futures[future]
            encode_inputs[local_idx] = future.result()
    lmdb_reader.close()
    logger.info(f"[GPU {gpu_id}] Preload done, {len(encode_inputs)} samples ready")

    # 3. 加载模型 & Template
    model, processor = get_model_processor(
        MODEL_ID,
        torch_dtype="bfloat16",
        attn_impl="flash_attention_2",
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

    # 4. 并行编码（纯 CPU 计算，无 I/O）
    lengths = []
    errors = 0

    def _encode_one(local_idx: int):
        encoded = template.encode(encode_inputs[local_idx])
        return len(encoded["input_ids"])

    with ThreadPoolExecutor(max_workers=NUM_THREADS_PER_GPU) as executor:
        futures = {executor.submit(_encode_one, i): i for i in range(len(indices))}
        with tqdm(total=len(indices), desc=f"GPU {gpu_id}", position=gpu_id) as pbar:
            for future in as_completed(futures):
                local_idx = futures[future]
                try:
                    n_tokens = future.result()
                    lengths.append(n_tokens)
                except Exception as e:
                    errors += 1
                    if errors <= 5:
                        logger.warning(f"[GPU {gpu_id}] Sample {indices[local_idx]} encode failed: {e}")
                pbar.update(1)

    # 保存结果
    lengths_arr = np.array(lengths, dtype=np.int64)
    np.save(result_path, lengths_arr)
    return len(lengths), errors


# ========================== 主函数 ==========================
def main():
    import torch
    from swift.utils import get_logger
    logger = get_logger()

    # 统计总样本数
    total = 0
    with open(JSONL_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                total += 1
    logger.info(f"Total samples: {total}, splitting across {NUM_GPUS} GPUs")

    # 均分索引
    chunk_size = total // NUM_GPUS
    splits = []
    for g in range(NUM_GPUS):
        start = g * chunk_size
        end = start + chunk_size if g < NUM_GPUS - 1 else total
        splits.append(list(range(start, end)))

    # 启动多进程
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    result_paths = []
    processes = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for gpu_id in range(NUM_GPUS):
            rpath = os.path.join(tmpdir, f"result_gpu{gpu_id}.npy")
            result_paths.append(rpath)
            p = ctx.Process(target=gpu_worker, args=(gpu_id, splits[gpu_id], rpath, total))
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
    bins = [0, 2048, 4096, 8192, 12288, 16384, 20480, 24576, 32768, 40960, 65536, float('inf')]
    for j in range(len(bins) - 1):
        lo, hi = bins[j], bins[j + 1]
        count = np.sum((lengths > lo) & (lengths <= hi))
        pct = count / len(lengths) * 100
        bar = "█" * int(pct / 2)
        hi_label = f"{hi:>7,}" if hi != float('inf') else "    ∞  "
        print(f"  ({lo:>7,}, {hi_label}]  {count:>5} ({pct:5.1f}%)  {bar}")

    # 建议
    print("\n  MAX_LENGTH 建议:")
    for threshold in [8192, 12288, 16384, 20480, 24576, 32768]:
        covered = np.sum(lengths <= threshold)
        pct = covered / len(lengths) * 100
        print(f"    MAX_LENGTH={threshold:>6}  可覆盖 {covered}/{len(lengths)} 样本 ({pct:.1f}%)")
    print()


if __name__ == "__main__":
    main()
