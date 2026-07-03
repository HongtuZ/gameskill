import os

# ========== 必须在 import torch 前设置 ==========
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# 以下环境变量控制多模态预处理时的 token / 帧数上限，与命令行等价
os.environ["IMAGE_MAX_TOKEN_NUM"] = "2048"
os.environ["VIDEO_MAX_TOKEN_NUM"] = "2048"
os.environ["FPS_MAX_FRAMES"] = "20"

import io
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List

import lmdb
import numpy as np
from PIL import Image
from swift import get_model_processor, get_template
from swift.dataset import LazyLLMDataset
from swift.trainers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from swift.tuners import LoraConfig, Swift
from swift.utils import get_logger, get_model_parameter_info, get_multimodal_target_regex, seed_everything
from torch.utils.data import Dataset, Subset

logger = get_logger()
seed_everything(42)


# ========================== 配置区 ==========================
LMDB_PATH = "/root/autodl-tmp/game_frames.lmdb"  # 你的 LMDB 文件
JSONL_PATH = "dataset.jsonl"  # 你之前生成的 jsonl
PROMPTS_DIR = "prompts"  # 你之前生成的 jsonl
OUTPUT_DIR = "output/Qwen3.5-4B-lmdb"

MODEL_ID = "Qwen/Qwen3.5-4B"

MAX_LENGTH = 24 * 1024 # 24K
NUM_FRAMES = 20  # 与 LMDB 中的 frames_per_video 一致

# LoRA 配置
LORA_RANK = 8
LORA_ALPHA = 32
FREEZE_LLM = False
FREEZE_VIT = True
FREEZE_ALIGNER = True


# ========================== LMDB 帧读取器 ==========================
class LMDBFrameReader:
    """从 LMDB 读取视频帧，返回 List[base64]"""

    def __init__(self, lmdb_path: str):
        self.env = lmdb.open(
            lmdb_path,
            readonly=True,
            lock=False,
            readahead=False,  # 避免顺序读取大文件时的预读开销
            max_readers=64,
        )
        with self.env.begin() as txn:
            meta_raw = txn.get(b"__meta__")
            self.meta = pickle.loads(meta_raw) if meta_raw else {}
            self.frames_per_video = self.meta.get("frames_per_video", 20)
            logger.info(f"LMDB loaded: {lmdb_path}")
            logger.info(
                f"Meta: total_frames={self.meta.get('total_frames')}, "
                f"video_count={len(self.meta.get('video_stems', []))}, "
                f"frames_per_video={self.frames_per_video}"
            )

    def get_frames(self, video_key: str) -> List[Image.Image]:
        """
        video_key 格式: "game/video_stem"  例: "delta/1101076112_001"
        LMDB 内键格式: "game/video_stem/00", "game/video_stem/01", ...
        """
        frames = []
        for i in range(self.frames_per_video):
            frame_key = f"{video_key}/{i:02d}".encode("utf-8")
            with self.env.begin() as txn:
                img_jpg = txn.get(frame_key)
                if img_jpg is None:
                    if i == 0:
                        logger.warning(f"[LMDB] Video key not found: {video_key}")
                    break
                img = Image.open(io.BytesIO(img_jpg)).convert("RGB")
                frames.append(img)
        return frames

    def close(self):
        self.env.close()


# ========================== 自定义数据集 ==========================
class LMDBVideoDataset(Dataset):
    """
    从 JSONL 读取文本标注（messages / loss_scale），从 LMDB 读取视频帧。
    输出格式严格符合 ms-swift 多模态标准：
        {
            "messages": [...],
            "videos": [[PIL.Image, PIL.Image, ...]]   # 每个视频是一个帧列表
        }
    """

    def __init__(self, jsonl_path: str, prompts_dir: str, lmdb_reader: LMDBFrameReader):
        self.prompts = {}
        for prompt_file in Path(prompts_dir).glob("*.md"):
            self.prompts[prompt_file.stem] = prompt_file.read_text(encoding="utf-8")
        self.samples = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.samples.append(json.loads(line))
                else:
                    print("Warning: Skipping empty line in JSONL")
        self.lmdb_reader = lmdb_reader
        logger.info(f"Loaded {list(self.prompts.keys())} prompts from {prompts_dir}")
        logger.info(f"Loaded {len(self.samples)} samples from {jsonl_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]

        # 原始 videos 字段是文件路径，如 /root/autodl-tmp/dataset/delta/1101076112_001.mp4
        video_path = sample["videos"][0]
        path_obj = Path(video_path)
        prompt = self.prompts.get(path_obj.parent.name)
        sample["messages"][0]["content"] += "以下游戏知识供参考：\n\n" + prompt + "\n\n" if prompt else ""

        # 映射为 LMDB key: game/video_stem
        video_key = f"{path_obj.parent.name}/{path_obj.stem}"

        # 从 LMDB 读取帧
        frames = self.lmdb_reader.get_frames(video_key)
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
    # 1. 初始化 LMDB
    lmdb_reader = LMDBFrameReader(LMDB_PATH)

    # 2. 构建完整数据集
    full_dataset = LMDBVideoDataset(JSONL_PATH, PROMPTS_DIR, lmdb_reader)

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
    train_dataset = LazyLLMDataset(train_dataset, template.encode, random_state=42, n_try_fetch=50, strict=False, traceback_limit=20)
    val_dataset = LazyLLMDataset(val_dataset, template.encode, random_state=42, n_try_fetch=50, strict=False, traceback_limit=20)

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
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
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
        dataloader_num_workers=8,
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
    lmdb_reader.close()
    logger.info("Training completed!")


if __name__ == "__main__":
    main()
