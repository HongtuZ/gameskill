import os

# ========== 必须在 import torch 前设置 ==========
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# 以下环境变量控制多模态预处理时的 token / 帧数上限，与命令行等价
os.environ["IMAGE_MAX_TOKEN_NUM"] = "2048"
os.environ["VIDEO_MAX_TOKEN_NUM"] = "2048"
os.environ["FPS_MAX_FRAMES"] = "20"

import io
import json
import tarfile
from pathlib import Path
from typing import Any, Dict, List

import torch

# 多 GPU 训练：根据 LOCAL_RANK 设置当前进程使用的设备，消除 init_process_group 警告
if torch.cuda.is_available():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

import numpy as np
from PIL import Image
from swift import get_model_processor, get_template
from swift.dataset import LazyLLMDataset
from swift.trainers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from swift.tuners import LoraConfig, Swift
from swift.utils import get_logger, get_model_parameter_info, get_multimodal_target_regex, seed_everything
from torch.utils.data import Dataset, Subset
from tqdm import tqdm

logger = get_logger()
seed_everything(42)


# ========================== 配置区 ==========================
WEBDATASET_DIR = "../gameskill_webdataset"  # WebDataset tar 目录
JSONL_PATH = "dataset/dataset.jsonl"  # 训练标注 jsonl
OUTPUT_DIR = "output/Qwen3.5-4B-webdataset"

MODEL_ID = "Qwen/Qwen3.5-4B"

MAX_LENGTH = 12 * 1024  # 12K
NUM_FRAMES = 20  # 与 LMDB 中的 frames_per_video 一致

# LoRA 配置
LORA_RANK = 8
LORA_ALPHA = 32
FREEZE_LLM = False
FREEZE_VIT = True
FREEZE_ALIGNER = True


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

        logger.info(
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


# ========================== 自定义数据集 ==========================
class WebDatasetVideoDataset(Dataset):
    """
    从 JSONL 读取文本标注（messages / loss_scale），从 WebDataset tar 读取视频帧。
    输出格式严格符合 ms-swift 多模态标准：
        {
            "messages": [...],
            "videos": [[PIL.Image, PIL.Image, ...]]   # 每个视频是一个帧列表
        }
    """

    def __init__(self, jsonl_path: str, frame_reader: WebDatasetFrameReader):
        self.samples = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.samples.append(json.loads(line))
        self.frame_reader = frame_reader
        logger.info(f"Loaded {len(self.samples)} samples from {jsonl_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        if "videos" not in sample:
            return sample

        # 原始 videos 字段是文件路径，如 videos/valorant/捷风/.../001.mp4
        video_path = sample["videos"][0]
        path_obj = Path(video_path)

        # 映射为 WebDataset video_key: parent_stem（与转换脚本的 video_prefix 一致）
        video_key = f"{path_obj.parent.name}_{path_obj.stem}"

        # 从 WebDataset 读取帧
        frames = self.frame_reader.get_frames(video_key)
        if len(frames) == 0:
            raise ValueError(f"Empty frames for key={video_key}, path={video_path}")

        return {
            "messages": sample["messages"],
            "videos": [
                frames,
            ],  # 嵌套列表：每个元素是一个视频的所有帧
        }


# ========================== 训练主流程 ==========================
def main():
    # 1. 初始化 WebDataset 帧读取器（扫描 tar 建立索引）
    frame_reader = WebDatasetFrameReader(WEBDATASET_DIR, num_frames=NUM_FRAMES)

    # 2. 构建完整数据集
    full_dataset = WebDatasetVideoDataset(JSONL_PATH, frame_reader)

    # 3. 划分训练/验证集（split_dataset_ratio=0.01）
    dataset_size = len(full_dataset)
    val_size = max(1, int(dataset_size * 0.01))
    train_size = dataset_size - val_size

    indices = np.random.RandomState(42).permutation(dataset_size)
    train_indices = indices[:train_size].tolist()
    val_indices = indices[train_size:].tolist()

    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    logger.info(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")

    # 4. 加载模型 & Processor
    #    对应命令行: --torch_dtype bfloat16 --attn_impl flash_attention_2
    model, processor = get_model_processor(
        MODEL_ID,
        torch_dtype="bfloat16",
        attn_impl="flash_attention_2",
    )
    logger.info(f"Model info: {model.model_info}")

    # 5. 构建 Template
    #    对应命令行:
    #      --loss_scale ignore_empty_think
    #      --add_non_thinking_prefix true
    #    注意: 你的 answer loss_scale 有 2.0，必须设置 is_binary_loss_scale=False
    template = get_template(
        processor,
        default_system=None,
        max_length=MAX_LENGTH,
        loss_scale="default+ignore_empty_think",
        add_non_thinking_prefix=True,
        is_binary_loss_scale=False,  # 支持 1.0 / 2.0 等非 0/1 的 loss_scale
    )
    template.set_mode("train")
    if getattr(template, "use_model", False):
        template.model = model

    # 6. 包装为 LazyLLMDataset（延迟 tokenize，出错时自动换样本）
    #    n_try_fetch 增大到 50，避免连续多个坏样本导致放弃
    #    traceback_limit=20 打印前 20 次错误的详细信息，方便定位真正原因
    train_dataset = LazyLLMDataset(
        train_dataset, template.encode, random_state=42, n_try_fetch=50, strict=False, traceback_limit=20
    )
    val_dataset = LazyLLMDataset(
        val_dataset, template.encode, random_state=42, n_try_fetch=50, strict=False, traceback_limit=20
    )

    # 快速测试一个样本，确认编码正常
    logger.info("Testing encode on first sample...")
    test_data = train_dataset[0]
    logger.info(f"Encoded keys: {list(test_data.keys())}")
    # template.print_inputs(test_data)

    # 7. LoRA 配置
    #    对应命令行: --target_modules all-linear（由 get_multimodal_target_regex 自动生成）
    target_modules = get_multimodal_target_regex(
        model,
        freeze_llm=FREEZE_LLM,
        freeze_vit=FREEZE_VIT,
        freeze_aligner=FREEZE_ALIGNER,
    )
    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=target_modules,
    )
    model = Swift.prepare_model(model, lora_config)  # ms-swift 4.x 推荐方式
    logger.info(f"LoRA target_modules: {target_modules}")

    model_parameter_info = get_model_parameter_info(model)
    logger.info(f"Trainable parameters info:\n{model_parameter_info}")

    # 8. 训练参数（与命令行一一对应）
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        learning_rate=1e-4,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_checkpointing=True,
        weight_decay=0.1,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        report_to=["tensorboard"],
        logging_first_step=True,
        save_strategy="steps",
        save_steps=50,
        eval_strategy="steps",
        eval_steps=50,
        gradient_accumulation_steps=1,
        num_train_epochs=2,
        metric_for_best_model="loss",
        save_total_limit=2,
        logging_steps=5,
        dataloader_num_workers=32,
        data_seed=42,
        remove_unused_columns=False,  # 必须保留，否则 videos/messages 会被 HF Trainer 过滤掉
        group_by_length=False,  # 对应 --group_by_length true
    )

    # 9. 初始化 Trainer
    model.enable_input_require_grads()  # 兼容 gradient_checkpointing
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        template=template,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    # 10. 训练
    trainer.train()

    last_checkpoint = trainer.state.last_model_checkpoint
    logger.info(f"Last checkpoint: {last_checkpoint}")

    # 清理
    logger.info("Training completed!")


if __name__ == "__main__":
    main()
