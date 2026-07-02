import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

# ==================== 配置区 ====================

PROMPTS_DIR = Path("prompts")
ANNOTATION_ROOT = Path("annotations")
VIDEO_ROOT = Path("/root/autodl-tmp/dataset")
OUTPUT_PATH = Path("dataset.jsonl")

# 默认 loss_scale：think 低权重，answer 高权重
DEFAULT_THINK_LOSS_SCALE = 1.0
DEFAULT_ANSWER_LOSS_SCALE = 2.0

# ==================== 工具函数 ====================


def load_game_prompt(game: str) -> str:
    """从markdown文件加载游戏专属提示词"""
    prompt_path = PROMPTS_DIR / f"{game}.md"
    if not prompt_path.exists():
        raise FileNotFoundError(f"未找到游戏提示词文件: {prompt_path}")
    return prompt_path.read_text(encoding="utf-8")


def build_think_content(annotation: Dict[str, Any]) -> str:
    parts = []
    game_view = annotation.get("game_view")
    map_name = annotation.get("map_name")
    ego_name = annotation.get("ego_name")
    stage = annotation.get("stage")
    team = annotation.get("team")
    teammates = annotation.get("teammates")
    enemies = annotation.get("enemies")
    kills = annotation.get("kills")
    base_info = ""
    if map_name:
        base_info += f"当前地图：{map_name}\n"
    if game_view:
        base_info += f"当前游戏视角：{game_view}\n"
    if ego_name:
        base_info += f"当前角色：{ego_name}\n"
    if stage:
        base_info += f"当前游戏阶段：{stage}\n"
    if team:
        base_info += f"当前阵营：{team}\n"
    if teammates:
        base_info += f"队友列表：{teammates}\n"
    if enemies:
        base_info += f"敌人列表：{enemies}\n"
    if kills:
        base_info += "击杀信息:\n"
        for kill in kills:
            method = kill.get("method")
            if "复活" in method:
                method = "复活"
            elif "技能" in method:
                method = "使用技能击杀"
            elif "击杀" in method:
                method = "击杀"
            else:
                method = f"使用{method}击杀"
            base_info += f"{kill.get('side1')}{kill.get('hero1')}{method}{kill.get('side2')}{kill.get('hero2')}\n"
    parts.append(f"【基本信息】:\n{base_info}")

    description = annotation.get("reasons", {}).get("description").strip()
    situation = annotation.get("reasons", {}).get("situation").strip()
    guidance_reason = annotation.get("reasons", {}).get("guidance_reason").strip()

    if description:
        parts.append(f"【局势分析】{description + situation}")

    if guidance_reason:
        parts.append(f"【指导原因】{guidance_reason}")

    think_body = "\n".join(parts)
    return f"<think>\n{think_body}\n</think>\n"


def build_answer_content(annotation: Dict[str, Any]) -> str:
    """构造  /think  之后的最终答案（不含 think 标签）"""
    need_guidance = annotation.get("guide", {}).get("need", False)
    guidance = annotation.get("guide", {}).get("advice")

    guidance_flag = "是" if need_guidance else "否"
    if guidance and str(guidance).strip() and str(guidance).lower() != "null":
        guidance_text = guidance.strip()
    else:
        guidance_text = "无"

    return f"【是否需要指导】{guidance_flag}\n【指导内容】{guidance_text}"


def build_sample(
    game_name: str,
    annotation: Dict[str, Any],
    video_path: Path,
    think_loss_scale: float = DEFAULT_THINK_LOSS_SCALE,
    answer_loss_scale: float = DEFAULT_ANSWER_LOSS_SCALE,
) -> Dict[str, Any]:
    think_content = build_think_content(annotation)
    answer_content = build_answer_content(annotation)

    SYSTEM_PROMPT = "你是一个专业的FPS游戏实时教练。请根据视频片段分析玩家操作并结合具体游戏机制给出建议。"

    USER_PROMPT = "<video>\n请分析这段游戏视频片段，判断是否需要立即给出实时指导。"

    sample = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT},
            {"role": "assistant", "content": think_content, "loss_scale": think_loss_scale},
            {"role": "assistant", "content": answer_content, "loss_scale": answer_loss_scale},
        ],
        "videos": [str(video_path.resolve())],
    }

    return sample


def process_annotation_file(
    json_path: Path, annotation_root: Path, video_root: Path, think_loss_scale: float, answer_loss_scale: float
) -> Optional[Dict[str, Any]]:
    """处理单个注释文件"""
    try:
        with json_path.open("r", encoding="utf-8") as f:
            annotation = json.load(f)
    except Exception as e:
        print(f"[错误] JSON 解析失败: {json_path} | {e}")
        return None

    rel_json_path = json_path.relative_to(annotation_root)

    video_path = video_root / rel_json_path.with_suffix(".mp4")
    if not video_path.exists():
        print(f"[WARNNING] 视频文件不存在: {str(video_path)}")
        return None
    video_path = video_path.resolve()

    game = rel_json_path.parts[0]

    return build_sample(
        game, annotation, video_path, think_loss_scale=think_loss_scale, answer_loss_scale=answer_loss_scale
    )


def build_dataset(
    annotation_root: Path,
    video_root: Path,
    output_path: Path,
    think_loss_scale: float,
    answer_loss_scale: float,
) -> None:
    """主流程：扫描注释目录，生成 JSONL"""
    json_files = sorted(annotation_root.rglob("*.json"))
    print(f"[信息] 发现 {len(json_files)} 个注释文件")

    samples = []
    skip_count = 0

    for jpath in json_files:
        sample = process_annotation_file(jpath, annotation_root, video_root, think_loss_scale, answer_loss_scale)
        if sample:
            samples.append(sample)
        else:
            skip_count += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    pos = sum(1 for s in samples if "【是否需要指导】是" in s["messages"][-1]["content"])
    neg = len(samples) - pos

    print("\n" + "=" * 50)
    print("数据集构造完成")
    print(f"输出路径 : {output_path.resolve()}")
    print(f"成功样本 : {len(samples)}")
    print(f"跳过样本 : {skip_count}")
    print(f"正样本   : {pos} (需指导)")
    print(f"负样本   : {neg} (无需指导)")
    print(f"think loss_scale : {think_loss_scale}")
    print(f"answer loss_scale: {answer_loss_scale}")
    print("=" * 50)


def validate_dataset(dataset_path: Path) -> None:
    """验证数据集格式是否符合 ms-swift 要求"""
    print(f"\n[验证] 开始检查: {dataset_path.resolve()}")
    errors = []
    total = 0

    with dataset_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            total += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                errors.append(f"行 {i}: JSON 解析失败")
                continue

            msgs = obj.get("messages")
            if not isinstance(msgs, list) or len(msgs) < 4:
                errors.append(f"行 {i}: messages 长度不足（需至少 system + user + 2 assistant）")
                continue

            if msgs[0].get("role") != "system":
                errors.append(f"行 {i}: 第1条应为 system")
            if msgs[1].get("role") != "user":
                errors.append(f"行 {i}: 第2条应为 user")
            if msgs[2].get("role") != "assistant":
                errors.append(f"行 {i}: 第3条应为 assistant (think)")
            if msgs[3].get("role") != "assistant":
                errors.append(f"行 {i}: 第4条应为 assistant (answer)")

            user_content = msgs[1].get("content", "")
            if "<video>" not in user_content:
                errors.append(f"行 {i}: user 消息缺少 <video> 标签")

            think_content = msgs[2].get("content", "")
            if "<think>" not in think_content or "</think>" not in think_content:
                errors.append(f"行 {i}: 第3条 assistant 缺少  <think>  标签")

            if "loss_scale" not in msgs[2]:
                errors.append(f"行 {i}: 第3条 assistant 缺少 loss_scale")
            if "loss_scale" not in msgs[3]:
                errors.append(f"行 {i}: 第4条 assistant 缺少 loss_scale")

            videos = obj.get("videos")
            if not videos or not isinstance(videos, list):
                errors.append(f"行 {i}: 缺少 videos 字段")
            else:
                for v in videos:
                    if not Path(v).exists():
                        errors.append(f"行 {i}: 视频文件不存在: {v}")

    if errors:
        print(f"[验证] 发现 {len(errors)} 个问题 (共 {total} 条):")
        for e in errors[:15]:
            print(f"  - {e}")
        if len(errors) > 15:
            print(f"  ... 还有 {len(errors) - 15} 个问题")
    else:
        print(f"[验证] 通过！共 {total} 条样本，格式正确。")


def split_train_val(dataset_path: Path, train_ratio: float = 0.9, seed: int = 42) -> None:
    """划分训练集和验证集"""
    base = dataset_path.with_suffix("")
    train_path = Path(f"{base}_train.jsonl")
    val_path = Path(f"{base}_val.jsonl")

    with dataset_path.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    import random

    random.seed(seed)
    random.shuffle(lines)

    split_idx = int(len(lines) * train_ratio)
    train_lines = lines[:split_idx]
    val_lines = lines[split_idx:]

    with train_path.open("w", encoding="utf-8") as f:
        f.writelines(train_lines)
    with val_path.open("w", encoding="utf-8") as f:
        f.writelines(val_lines)

    print("\n[划分] 完成")
    print(f"  训练集: {train_path.resolve()} ({len(train_lines)} 条)")
    print(f"  验证集: {val_path.resolve()} ({len(val_lines)} 条)")


# ==================== 命令行入口 ====================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="构造 ms-swift 游戏教练数据集（Pathlib + 双 assistant）")
    parser.add_argument("--annotation-root", type=Path, default=ANNOTATION_ROOT, help="注释 JSON 根目录")
    parser.add_argument("--video-root", type=Path, default=VIDEO_ROOT, help="视频文件根目录")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH, help="输出 JSONL 路径")
    parser.add_argument(
        "--think-loss-scale", type=float, default=DEFAULT_THINK_LOSS_SCALE, help="think 部分 loss_scale（默认 1.0）"
    )
    parser.add_argument(
        "--answer-loss-scale",
        type=float,
        default=DEFAULT_ANSWER_LOSS_SCALE,
        help="answer 部分 loss_scale（默认 2.0，权重更高）",
    )
    parser.add_argument("--validate", action="store_true", help="验证数据集格式")
    parser.add_argument("--split", action="store_true", help="划分训练/验证集")
    parser.add_argument("--train-ratio", type=float, default=0.9, help="训练集比例")

    args = parser.parse_args()

    build_dataset(
        annotation_root=args.annotation_root,
        video_root=args.video_root,
        output_path=args.output,
        think_loss_scale=args.think_loss_scale,
        answer_loss_scale=args.answer_loss_scale,
    )

    if args.validate:
        validate_dataset(args.output)

    if args.split:
        split_train_val(args.output, args.train_ratio)
