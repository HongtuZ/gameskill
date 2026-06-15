import argparse
import base64
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI

# ====================== 配置 ======================

PROMPTS_DIR = Path(__file__).parent / "prompts"
SUMMARY_BASE_PROMPT = (
    (PROMPTS_DIR / "summary_base.md").read_text(encoding="utf-8") if (PROMPTS_DIR / "summary_base.md").exists() else ""
)


def load_game_prompt(game: str) -> str:
    """从markdown文件加载游戏专属提示词"""
    prompt_path = PROMPTS_DIR / f"{game}.md"
    if not prompt_path.exists():
        raise FileNotFoundError(f"未找到游戏提示词文件: {prompt_path}")
    return prompt_path.read_text(encoding="utf-8")


def encode_video(video_path):
    with open(video_path, "rb") as video_file:
        return base64.b64encode(video_file.read()).decode("utf-8")


def detect_game_type(path: str) -> str:
    """从路径识别游戏类型"""
    lower = Path(path).as_posix().lower()
    if "cs2" in lower or "csgo" in lower:
        return "cs2"
    elif "valorant" in lower or "无畏" in lower:
        return "valorant"
    elif "delta" in lower or "三角洲" in lower:
        return "delta"
    # 尝试从上级目录名判断
    parent = Path(path).parent.name.lower()
    if parent in ("cs2", "valorant", "delta"):
        return parent
    raise ValueError(f"无法从路径识别游戏类型: {path}")


def get_next_clip_path(current_path: str) -> str | None:
    """xxx_001.mp4 -> xxx_002.mp4"""
    path = Path(current_path)
    match = re.match(r"(.+)_(\d{3,})$", path.stem)
    if not match:
        return None
    prefix, num_str = match.groups()
    next_num = int(num_str) + 1
    next_path = path.parent / f"{prefix}_{next_num:03d}{path.suffix}"
    return str(next_path) if next_path.exists() else None


def parse_json(text: str) -> dict:
    """从模型输出提取 JSON"""
    text = text.strip()
    if text.startswith("{"):
        return json.loads(text)
    # 尝试提取 markdown 代码块
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        return json.loads(m.group(1))
    # 兜底：提取第一个 { 到最后一个 }
    m = re.search(r"(\{.*\})", text, re.DOTALL)
    if m:
        return json.loads(m.group(1))
    raise ValueError(f"无法解析: {text[:200]}")


# ====================== 第一阶段：生成后10s摘要 ======================


def generate_summary(
    clip_path: str,
    client: OpenAI,
    model: str,
    fps: float,
    max_tokens: int,
    timeout: float,
    output_dir: Path,
    enable_thinking: bool,
) -> dict:
    """为单个片段生成后10s操作摘要"""
    clip_name = Path(clip_path).stem
    game = detect_game_type(clip_path)
    game_prompt = load_game_prompt(game)

    # 按游戏类型分目录保存
    game_summary_dir = output_dir / game
    game_summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = game_summary_dir / f"{clip_name}_summary.json"

    if summary_path.exists():
        return {"clip": clip_name, "status": "skipped"}

    # 读取视频
    video_b64 = encode_video(clip_path)

    # 构建system prompt：基础摘要要求 + 游戏专属知识
    system_prompt = f"""{SUMMARY_BASE_PROMPT}
【游戏专属知识 - {game.upper()}】
{game_prompt}
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {
                    "type": "video_url",
                    "video_url": {"url": f"data:video/mp4;base64,{video_b64}"},
                    "fps": fps,
                },
                {"type": "text", "text": "请分析这段10秒视频，输出操作摘要JSON。仅记录客观事实。使用中文。"},
            ],
        },
    ]

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.2,
            top_p=0.8,
            max_tokens=max_tokens,
            timeout=timeout,
            extra_body={"enable_thinking": enable_thinking},
        )
        content = resp.choices[0].message.content.strip()

        summary = parse_json(content)

        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"clip": clip_name, "status": "success"}

    except Exception as e:
        return {"clip": clip_name, "status": "failed", "error": str(e)}


# ====================== 第二阶段：生成教练标注 ======================


def generate_annotation(
    clip_path: str,
    client: OpenAI,
    model: str,
    fps: float,
    max_tokens: int,
    timeout: float,
    summary_dir: Path,
    output_dir: Path,
    enable_thinking: bool,
) -> dict:
    """基于前10s视频 + 后10s摘要，生成教练指导"""
    clip_name = Path(clip_path).stem
    next_clip_path = get_next_clip_path(clip_path)
    if next_clip_path is None:
        return {"clip": clip_name, "status": "no_summary"}
    next_clip_name = Path(next_clip_path).stem
    game = detect_game_type(clip_path)

    # 读取后10s摘要
    summary_path = summary_dir / game / f"{next_clip_name}_summary.json"
    if not summary_path.exists():
        return {"clip": clip_name, "status": "no_summary"}

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary_text = json.dumps(summary, ensure_ascii=False, indent=2)

    game_prompt = load_game_prompt(game)

    # 按游戏类型分目录
    game_anno_dir = output_dir / game
    game_anno_dir.mkdir(parents=True, exist_ok=True)
    anno_path = game_anno_dir / f"{clip_name}_annotation.json"

    if anno_path.exists():
        return {"clip": clip_name, "status": "skipped"}

    # 读取前10s视频
    video_b64 = encode_video(clip_path)


    # 构建教练system prompt
    system_prompt = f"""你是一位{game.upper()}职业教练，精通该游戏的战术体系。

【游戏专属知识】
{game_prompt}

【核心任务 - Hindsight Coaching】
你会看到玩家"前10秒"的处境视频，以及"后10秒"实际发生了什么（结果已知）。
你的职责：判断前10秒内是否存在值得干预的决策点，并给出如果当时做出更优决策，后10秒结果会更好的具体建议。

【指导原则】
1. 沉默是金：常规移动、搜点、默认架枪、信息不足时的保守选择 → need_guidance=false
2. 值得指导：明显决策失误、次优选择、关键抉择点（1vN残局、转点、经济局起枪、回防路线）→ need_guidance=true
3. 必须具体可执行：禁止"打准一点"、"注意身位"等废话
4. 必须口语化：适合语音播报，15-40字
5. 必须基于证据：建议必须能从后10秒结果中得到验证

【输出格式】
严格JSON格式：
{{
  "situation_analysis": "一句话概括前10秒局势",
  "need_guidance": true/false,
  "guidance_urgency": "low/medium/high",
  "silence_reason": "如果need_guidance=false，说明原因。否则null",
  "reasoning": "详细思维链：1)局势拆解 2)实际决策分析 3)后10秒结果验证 4)最优决策推演 5)指导必要性判断",
  "guidance_content": "直接语音指导（15-40字，口语化）。如果need_guidance=false填null",
  "mistake_category": "null/positioning/timing/mechanics/decision/utility/economy/communication/rotation",
  "alternative_action": "建议的具体操作。如果need_guidance=false填null",
  "confidence": 0.0-1.0
}}
"""

    # 构建user消息：视频 + 摘要 + 元信息
    context_text = f"""【对局上下文】
游戏：{game.upper()}
【后10秒实际结果摘要】
{summary_text}

请基于以上 hindsight 信息，对前10秒画面中的玩家决策进行教学指导。"""

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {
                    "type": "video_url",
                    "video_url": {"url": f"data:video/mp4;base64,{video_b64}"},
                    "fps": fps,
                },
                {"type": "text", "text": context_text},
            ],
        },
    ]

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.7,
            top_p=0.8,
            max_tokens=max_tokens,
            timeout=timeout,
            extra_body={"enable_thinking": enable_thinking},
        )
        content = resp.choices[0].message.content.strip()

        # 解析JSON
        annotation = parse_json(content)

        annotation["meta"] = {
            "source_clip": clip_name,
            "game": game,
            "model": model,
            "fps": fps,
        }
        anno_path.write_text(json.dumps(annotation, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"clip": clip_name, "status": "success"}

    except Exception as e:
        return {"clip": clip_name, "status": "failed", "error": str(e)}


# ====================== 命令行参数 ======================


def get_args():
    parser = argparse.ArgumentParser(description="游戏视频 hindsight coaching 数据标注流水线")
    parser.add_argument(
        "--stage",
        choices=["summary", "annotation", "both"],
        default="both",
        help="执行阶段：summary=仅生成摘要, annotation=仅生成标注, both=完整流水线",
    )
    parser.add_argument("--clips-dir", "-d", required=True, help="视频片段目录")
    parser.add_argument("--summary-dir", "-s", default="./summaries", help="摘要输出目录")
    parser.add_argument("--output-dir", "-o", default="./annotations", help="标注输出目录")
    parser.add_argument("--model-summary", default="qwen3.6-plus", help="摘要模型")
    parser.add_argument("--model-annotation", default="qwen3.6-plus", help="标注模型")
    parser.add_argument("--fps-summary", type=float, default=2, help="摘要抽帧率")
    parser.add_argument("--fps-annotation", type=float, default=2, help="标注抽帧率")
    parser.add_argument("--max-tokens", type=int, default=2000)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--workers", "-w", type=int, default=1)
    parser.add_argument("--enable-thinking", action="store_true")
    return parser.parse_args()


# ====================== 主逻辑 ======================

if __name__ == "__main__":
    args = get_args()

    api_key = "sk-5bb8f23bd9a649b19403a48dee738bdf"

    client = OpenAI(api_key=api_key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")

    clips_dir = Path(args.clips_dir)
    summary_dir = Path(args.summary_dir)
    output_dir = Path(args.output_dir)

    # 收集所有视频片段（递归查找所有子目录中的mp4）
    all_clips = sorted(clips_dir.rglob("*.mp4"))
    print(f"发现 {len(all_clips)} 个片段")

    # Stage 1: 生成摘要
    if args.stage in ("summary", "both"):
        print("=== Stage 1: 生成后10s摘要 ===")
        results = {"success": 0, "failed": 0, "skipped": 0, "last_segment": 0}

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    generate_summary,
                    str(c),
                    client,
                    args.model_summary,
                    args.fps_summary,
                    args.max_tokens,
                    args.timeout,
                    summary_dir,
                    args.enable_thinking,
                ): c
                for c in all_clips
            }
            for future in as_completed(futures):
                result = future.result()
                results[result.get("status", "unknown")] = results.get(result.get("status"), 0) + 1
                print(f"[{result.get('status')}] {result['clip']} {result.get('error')}")

        print(f"摘要完成: {results}")

    # Stage 2: 生成标注
    if args.stage in ("annotation", "both"):
        print("=== Stage 2: 生成教练标注 ===")
        results = {"success": 0, "failed": 0, "skipped": 0, "no_summary": 0}

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    generate_annotation,
                    str(c),
                    client,
                    args.model_annotation,
                    args.fps_annotation,
                    args.max_tokens,
                    args.timeout,
                    summary_dir,
                    output_dir,
                    args.enable_thinking,
                ): c
                for c in all_clips
            }
            for future in as_completed(futures):
                result = future.result()
                results[result.get("status", "unknown")] = results.get(result.get("status"), 0) + 1
                print(f"[{result.get('status')}] {result['clip']} {result.get('error')}")

        print(f"标注完成: {results}")
