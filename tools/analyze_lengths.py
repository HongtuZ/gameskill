"""统计训练数据集中每个样本编码后的 token 长度分布（双GPU加速版，WebDataset + 原始视频回退）"""

import argparse
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
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

    def get_first_frame(self, video_key: str) -> Image.Image:
        """只加载第一帧，用于轻量级分辨率扫描"""
        if video_key not in self.index:
            raise KeyError(f"Video key not found in WebDataset: {video_key}")

        shard_path, offset, size = self.index[video_key][0]
        with open(shard_path, "rb") as f:
            f.seek(offset)
            jpg_data = f.read(size)
        return Image.open(io.BytesIO(jpg_data)).convert("RGB")


def _get_video_key(sample) -> str:
    """从样本中提取 video_key"""
    video_path_str = sample["videos"][0]
    path_obj = Path(video_path_str)
    return f"{path_obj.parent.name}_{path_obj.stem}"


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
    except Exception:
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
    text_token_count_override: int = None,
    probe_index: int = None,
):
    """在指定 GPU 上编码指定索引范围的样本，结果写入 npy 文件"""
    try:
        _gpu_worker_impl(
            gpu_id,
            indices,
            result_path,
            total_samples,
            webdataset_dir,
            jsonl_path,
            text_token_count_override,
            probe_index,
        )
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
    text_token_count_override: int = None,
    probe_index: int = None,
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

    # 2. 加载模型 & Template
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

    # 3. Probe: 确定视频 token 数
    video_token_count = None
    text_token_count = text_token_count_override
    reader = WebDatasetFrameReader(webdataset_dir, num_frames=NUM_FRAMES)
    logger.info(f"[GPU {gpu_id}] 数据源: {webdataset_dir}")
    if text_token_count is not None:
        logger.info(f"[GPU {gpu_id}] Using provided text_token_count={text_token_count}")

    # 3a. 确定 probe 样本
    best_gidx = None
    best_resolution = None
    if probe_index is not None:
        # 指定了 probe 样本索引，直接使用
        if probe_index < 0:
            # -1 表示找第一个可用样本
            logger.info(f"[GPU {gpu_id}] Probe: finding first available sample")
            for gidx in indices:
                sample = all_samples[gidx]
                if "videos" not in sample or not sample["videos"]:
                    continue
                video_key = _get_video_key(sample)
                if video_key in reader.index:
                    best_gidx = gidx
                    break
        else:
            best_gidx = probe_index
        logger.info(f"[GPU {gpu_id}] Probe: using sample={best_gidx} (no scan)")
    else:
        # 扫描所有样本的首帧，找最高分辨率的视频
        best_height = 0
        scanned = 0
        for gidx in indices:
            sample = all_samples[gidx]
            if "videos" not in sample or not sample["videos"]:
                continue
            video_key = _get_video_key(sample)
            if video_key not in reader.index:
                continue
            try:
                first_frame = reader.get_first_frame(video_key)
            except Exception:
                continue
            w, h = first_frame.size
            scanned += 1
            if h > best_height:
                best_height = h
                best_gidx = gidx
                best_resolution = (w, h)
            if h >= 1080:
                logger.info(f"[GPU {gpu_id}] Found 1080p at sample={gidx}, resolution={w}x{h}")
                break
        logger.info(f"[GPU {gpu_id}] Probe scan: scanned={scanned}, best={best_resolution}, gidx={best_gidx}")

    # 3b. 只加载最高分辨率的样本（全部帧）做 probe 编码
    if best_gidx is not None:
        probe_result = _load_sample(reader, all_samples[best_gidx])
        if probe_result is not None:
            try:
                enc_with_video = template.encode(probe_result)
                total_n = len(enc_with_video["input_ids"])
                if text_token_count is None:
                    text_only_inp = {k: v for k, v in probe_result.items() if k != "videos"}
                    enc_text_only = template.encode(text_only_inp)
                    text_token_count = len(enc_text_only["input_ids"])
                video_token_count = total_n - text_token_count
                w, h = best_resolution
                logger.info(
                    f"[GPU {gpu_id}] Probe: sample={best_gidx}, resolution={w}x{h}, total={total_n}, "
                    f"text={text_token_count}, video_tokens={video_token_count}"
                )
            except Exception as e:
                logger.warning(f"[GPU {gpu_id}] Probe encode failed: {e}")
        del probe_result  # 释放帧内存

    if video_token_count is None:
        logger.error(f"[GPU {gpu_id}] Failed to probe video token count, using 0")
        video_token_count = 0

    # 确定是否用固定文本 token（跳过逐样本文本编码）
    use_fixed_text = text_token_count_override is not None
    if use_fixed_text:
        logger.info(f"[GPU {gpu_id}] Fixed text mode: text_token_count={text_token_count}")

    # 4. 分批处理：不加载帧，只检查索引存在性 → 纯文本编码 + 固定视频 token
    lengths = []
    visual_lengths = []
    text_lengths = []
    skipped = 0
    errors = 0

    with tqdm(total=len(indices), desc=f"GPU {gpu_id}", position=gpu_id) as pbar:
        for batch_start in range(0, len(indices), BATCH_SIZE):
            batch_indices = indices[batch_start : batch_start + BATCH_SIZE]

            # 4a. 轻量级验证：只检查 video_key 是否在索引中，不加载帧
            valid_indices = []
            for gidx in batch_indices:
                sample = all_samples[gidx]
                if "videos" not in sample or not sample["videos"]:
                    skipped += 1
                    continue
                video_key = _get_video_key(sample)
                if video_key not in reader.index:
                    skipped += 1
                    continue
                valid_indices.append(gidx)

            # 4b. 编码：固定文本模式直接用固定值，否则逐样本纯文本编码
            batch_lengths = []
            batch_visual = []
            batch_text = []

            if use_fixed_text:
                # 固定文本模式：不需要编码，直接用固定值
                for gidx in valid_indices:
                    txt_n = text_token_count
                    total_n = txt_n + video_token_count
                    lengths.append(total_n)
                    visual_lengths.append(video_token_count)
                    text_lengths.append(txt_n)
                    batch_lengths.append(total_n)
                    batch_visual.append(video_token_count)
                    batch_text.append(txt_n)
            else:
                # 逐样本纯文本编码模式
                with ThreadPoolExecutor(max_workers=NUM_ENCODE_WORKERS) as encoder:
                    text_inputs = {
                        gidx: {"messages": copy.deepcopy(all_samples[gidx]["messages"])} for gidx in valid_indices
                    }
                    enc_futures = {encoder.submit(template.encode, inp): gidx for gidx, inp in text_inputs.items()}
                    for future in as_completed(enc_futures):
                        gidx = enc_futures[future]
                        try:
                            encoded = future.result()
                            txt_n = len(encoded["input_ids"])
                            total_n = txt_n + video_token_count
                            lengths.append(total_n)
                            visual_lengths.append(video_token_count)
                            text_lengths.append(txt_n)
                            batch_lengths.append(total_n)
                            batch_visual.append(video_token_count)
                            batch_text.append(txt_n)
                        except Exception as e:
                            errors += 1
                            if errors <= 5:
                                logger.warning(f"[GPU {gpu_id}] Sample {gidx} encode failed: {e}")

            # 4c. 输出本批统计
            if batch_lengths:
                bl = np.array(batch_lengths)
                bv = np.array(batch_visual)
                bt = np.array(batch_text)
                batch_id = batch_start // BATCH_SIZE + 1
                tqdm.write(
                    f"  [Batch {batch_id}] samples={len(bl)}"
                    f" | total: min={bl.min()}, max={bl.max()}, mean={bl.mean():.0f}"
                    f" | video: min={bv.min()}, max={bv.max()}, mean={bv.mean():.0f}"
                    f" | text:  min={bt.min()}, max={bt.max()}, mean={bt.mean():.0f}"
                )

            pbar.update(len(batch_indices))

    # 保存结果：用 pickle 保存，彻底避免 np.save 后缀问题
    import pickle

    result_dir = os.path.dirname(result_path)
    os.makedirs(result_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(result_path))[0]  # e.g. "result_gpu0"
    save_path = os.path.join(result_dir, f"{stem}_result.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(
            {
                "total": np.array(lengths, dtype=np.int64),
                "visual": np.array(visual_lengths, dtype=np.int64),
                "text": np.array(text_lengths, dtype=np.int64),
            },
            f,
        )
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

    # 启动多进程并行处理
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    result_dir = os.path.join("output", "analyze_tmp")
    # 清理旧结果，避免残留文件干扰
    import shutil

    if os.path.exists(result_dir):
        shutil.rmtree(result_dir)
    os.makedirs(result_dir, exist_ok=True)

    result_paths = []
    processes = []
    for gpu_id in range(num_gpus):
        rpath = os.path.join(result_dir, f"result_gpu{gpu_id}.npy")
        result_paths.append(rpath)
        p = ctx.Process(
            target=gpu_worker,
            args=(
                gpu_id,
                splits[gpu_id],
                rpath,
                total,
                webdataset_dir,
                jsonl_path,
                args.text_token_count,
                args.probe_index,
            ),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # 合并各 GPU 的结果
    import pickle

    all_total = []
    all_visual = []
    all_text = []
    for rpath in result_paths:
        result_dir = os.path.dirname(rpath)
        stem = os.path.splitext(os.path.basename(rpath))[0]
        pkl_path = os.path.join(result_dir, f"{stem}_result.pkl")
        if os.path.exists(pkl_path):
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)
            all_total.append(data["total"])
            all_visual.append(data["visual"])
            all_text.append(data["text"])

    # 清理临时文件
    shutil.rmtree(result_dir, ignore_errors=True)

    if not all_total:
        logger.error("No results collected!")
        return

    lengths = np.concatenate(all_total)
    visual_lengths = np.concatenate(all_visual) if all_visual else None
    text_lengths = np.concatenate(all_text) if all_text else None
    errors = total - len(lengths)
    _print_statistics(lengths, visual_lengths, text_lengths, errors)


def _print_statistics(lengths, visual_lengths, text_lengths, errors):
    """打印 token 长度统计（支持分拆视觉/文本/总计）"""

    def _print_array_stats(name, arr):
        print(f"\n  --- {name} (共 {len(arr)} 个样本) ---")
        print(f"  最小值:   {arr.min():>10,}")
        print(f"  最大值:   {arr.max():>10,}")
        print(f"  平均值:   {arr.mean():>10,.1f}")
        print(f"  中位数:   {np.median(arr):>10,.1f}")
        print(f"  标准差:   {arr.std():>10,.1f}")
        for p in [50, 75, 90, 95, 99, 99.5, 99.9]:
            print(f"  P{p:<5}   {np.percentile(arr, p):>10,.1f}")

    print("\n" + "=" * 70)
    print(f"  数据集 Token 长度统计 (共 {len(lengths)} 个样本, {errors} 个编码失败)")
    print("=" * 70)

    # 总长度
    _print_array_stats("总 Token 长度", lengths)

    # 视觉 token 长度
    if visual_lengths is not None and len(visual_lengths) > 0:
        _print_array_stats("视频/图像 Token 长度", visual_lengths)

    # 文本 token 长度
    if text_lengths is not None and len(text_lengths) > 0:
        _print_array_stats("文本 Token 长度", text_lengths)

    print("\n" + "=" * 70)
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
    parser.add_argument(
        "--text-token-count",
        type=int,
        default=None,
        help="固定文本 token 数量，提供后跳过逐样本文本编码，直接使用该值",
    )
    parser.add_argument(
        "--probe-index",
        type=int,
        default=None,
        help="指定 probe 样本索引，跳过首帧扫描。-1 表示用第一个可用样本",
    )
    args = parser.parse_args()

    # reader = WebDatasetFrameReader(args.webdataset_dir, num_frames=NUM_FRAMES)
    # import time

    # for key in tqdm(reader.index):
    #     start = time.perf_counter()
    #     frames = reader.get_frames(key)
    #     tqdm.write(f"{key}: {time.perf_counter() - start:.2f}s")
    main()
