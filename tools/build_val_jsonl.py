#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

SYSTEM_PROMPT = "你是一个专业的FPS游戏实时教练。"


class ValData:
    def __init__(self, json_path: Path, video_path: Path):
        self.json_path = json_path
        self.video_path = video_path
        with open(json_path, "r", encoding="utf-8") as f:
            annotation = json.load(f)
        self.game_view = annotation.get("game_view")
        self.map_name = annotation.get("map_name")
        self.ego_name = annotation.get("ego_name")
        self.stage = annotation.get("stage")
        self.team = annotation.get("team")
        self.teammates = annotation.get("teammates")
        self.enemies = annotation.get("enemies")
        self.kills = annotation.get("kills")
        self.description = annotation.get("reasons", {}).get("description")
        self.situation = annotation.get("reasons", {}).get("situation")
        self.guidance_reason = annotation.get("reasons", {}).get("guidance_reason")
        guide = annotation.get("guide", {})
        self.need_guidance = guide.get("need", False)
        self.guidance = guide.get("advice", "")

    def is_valid(self) -> bool:
        if self.game_view != "非局内" and not self.description:
            return False
        if self.need_guidance and (not self.guidance or not self.description or not self.guidance_reason):
            return False
        return True

    def format_kills(self) -> str:
        if not self.kills:
            return "暂无击杀信息"
        lines = []
        for kill in self.kills:
            method = kill.get("method") or ""
            if "复活" in method:
                method = "复活"
            elif "技能" in method:
                method = "使用技能击杀"
            elif "击杀" in method:
                method = "击杀"
            else:
                method = f"使用{method}击杀"
            lines.append(f"{kill.get('side1')}{kill.get('hero1')}{method}{kill.get('side2')}{kill.get('hero2')}")
        return "\n".join(lines)

    def build_think_content(self) -> str:
        parts = []
        base_info = ""
        base_info += f"当前地图：{self.map_name}\n" if self.map_name else ""
        base_info += f"当前游戏视角：{self.game_view}\n" if self.game_view else ""
        base_info += f"当前角色：{self.ego_name}\n" if self.ego_name else ""
        base_info += f"当前游戏阶段：{self.stage}\n" if self.stage else ""
        base_info += f"当前阵营：{self.team}\n" if self.team else ""
        base_info += f"队友列表：{self.teammates}\n" if self.teammates else ""
        base_info += f"敌人列表：{self.enemies}\n" if self.enemies else ""
        base_info += f"击杀信息:\n{self.format_kills()}\n" if self.kills else ""

        parts.append(f"【基本信息】:\n{base_info}")
        if self.description:
            parts.append(f"【局势分析】{(self.description or '') + (self.situation or '')}")
        if self.guidance_reason:
            parts.append(f"【指导原因】{self.guidance_reason}")
        think_body = "\n".join(parts)
        return think_body

    def build_guidance_content(self) -> str:
        """构造  /think  之后的最终答案（不含 think 标签）"""
        guidance_flag = "是" if self.need_guidance else "否"
        if self.guidance and str(self.guidance).strip() and str(self.guidance).lower() != "null":
            guidance_text = self.guidance.strip()
        else:
            guidance_text = "无"
        return f"【是否需要指导】{guidance_flag}\n【指导内容】{guidance_text}"


def resolve_video_path(json_root: Path, json_file: Path, video_root: Path) -> Optional[Path]:
    rel_json_file = json_file.relative_to(json_root)
    # 尝试匹配同名视频
    for ext in [".mp4", ".avi", ".mov", ".mkv", ".webm"]:
        candidate = video_root / rel_json_file.with_suffix(ext)
        if candidate.exists():
            return candidate
    return None


def build_think_guidance_sample(val_data: ValData) -> Dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "<video>\n请分析这段游戏视频片段，判断是否需要立即给出实时指导。"},
            {"role": "assistant", "content": f"think>\n{val_data.build_think_content()}\n</think>", "loss_scale": 1.0},
            {"role": "assistant", "content": val_data.build_guidance_content(), "loss_scale": 2.0},
        ],
        "videos": [str(val_data.video_path)],
    }


def build_guidance_sample(val_data: ValData) -> Dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "<video>\n请分析这段游戏视频片段，判断是否需要立即给出实时指导。"},
            {"role": "assistant", "content": val_data.build_guidance_content(), "loss_scale": 1.0},
        ],
        "videos": [str(val_data.video_path)],
    }


def build_gameview_sample(val_data: ValData) -> Dict:
    if not val_data.game_view:
        return None
    answer = val_data.game_view
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "<video>\n当前游戏视角是什么?"},
            {"role": "assistant", "content": answer, "loss_scale": 1.0},
        ],
        "videos": [str(val_data.video_path)],
    }


def build_mapname_sample(val_data: ValData) -> Dict:
    if not val_data.map_name:
        return None
    answer = val_data.map_name
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "<video>\n当前游戏地图是什么?"},
            {"role": "assistant", "content": answer, "loss_scale": 1.0},
        ],
        "videos": [str(val_data.video_path)],
    }


def build_ego_name_sample(val_data: ValData) -> Dict:
    if not val_data.ego_name:
        return None
    answer = val_data.ego_name
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "<video>\n当前游戏角色是什么?"},
            {"role": "assistant", "content": answer, "loss_scale": 1.0},
        ],
        "videos": [str(val_data.video_path)],
    }


def build_stage_sample(val_data: ValData) -> Dict:
    if not val_data.stage:
        return None
    answer = val_data.stage
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "<video>\n当前游戏阶段是什么?"},
            {"role": "assistant", "content": answer, "loss_scale": 1.0},
        ],
        "videos": [str(val_data.video_path)],
    }


def build_team_sample(val_data: ValData) -> Dict:
    if not val_data.team:
        return None
    answer = val_data.team
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "<video>\n当前我方阵营是什么?"},
            {"role": "assistant", "content": answer, "loss_scale": 1.0},
        ],
        "videos": [str(val_data.video_path)],
    }


def build_description_sample(val_data: ValData) -> Dict:
    if not val_data.description:
        return None
    answer = val_data.description
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "<video>\n描述游戏画面:"},
            {"role": "assistant", "content": answer, "loss_scale": 1.0},
        ],
        "videos": [str(val_data.video_path)],
    }


def build_situation_sample(val_data: ValData) -> Dict:
    if not val_data.situation:
        return None
    answer = val_data.situation
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "<video>\n分析游戏局势:"},
            {"role": "assistant", "content": answer, "loss_scale": 1.0},
        ],
        "videos": [str(val_data.video_path)],
    }


def build_teammates_sample(val_data: ValData) -> Dict:
    if not val_data.teammates:
        return None
    answer = ", ".join(val_data.teammates)
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "<video>\n我的队友有哪些?"},
            {"role": "assistant", "content": answer, "loss_scale": 1.0},
        ],
        "videos": [str(val_data.video_path)],
    }


def build_enemies_sample(val_data: ValData) -> Dict:
    if not val_data.enemies:
        return None
    answer = ", ".join(val_data.enemies)
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "<video>\n我的敌人有哪些?"},
            {"role": "assistant", "content": answer, "loss_scale": 1.0},
        ],
        "videos": [str(val_data.video_path)],
    }


def build_kills_sample(val_data: ValData) -> Dict:
    if not val_data.kills:
        return None
    answer = val_data.format_kills()
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "<video>\n最近的击杀信息有哪些?"},
            {"role": "assistant", "content": answer, "loss_scale": 1.0},
        ],
        "videos": [str(val_data.video_path)],
    }


def build_gamename_sample(val_data: ValData) -> Dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "<video>\n当前是什么游戏?"},
            {"role": "assistant", "content": "无畏契约", "loss_scale": 1.0},
        ],
        "videos": [str(val_data.video_path)],
    }


def process_json_file(json_path: Path, video_path: Path) -> List[Dict]:
    """处理单个 JSON 文件，返回训练样本列表"""
    val_data = ValData(json_path, video_path)
    samples = []
    if not val_data.is_valid():
        return samples
    think_guidance_sample = build_think_guidance_sample(val_data)
    if think_guidance_sample:
        samples.append(think_guidance_sample)
    return samples


def main():
    parser = argparse.ArgumentParser(description="标注 JSON → 训练 JSONL")
    parser.add_argument("--json_root", "-j", required=True, help="标注 JSON 文件目录或单个文件路径")
    parser.add_argument("--output", "-o", default="./train.jsonl", help="输出 JSONL 路径")
    parser.add_argument("--video_root", "-v", default="", help="视频根目录，用于自动映射路径")

    args = parser.parse_args()

    video_root = Path(args.video_root)
    json_root = Path(args.json_root)
    json_files = sorted(json_root.rglob("*.json"))

    print(f"[INFO] 发现 {len(json_files)} 个 JSON 文件")
    print("[INFO] 检查json对应的视频文件是否存在")
    valid_json_files = []
    valid_video_files = []
    for jf in json_files:
        video_path = resolve_video_path(json_root, jf, video_root)
        if video_path:
            valid_json_files.append(jf)
            valid_video_files.append(video_path)
        else:
            print(f"[WARN] 视频文件不存在 {jf}")
    print(f"[INFO] 缺少视频文件: {len(json_files) - len(valid_json_files)}个，这些json标注数据将跳过")

    all_samples = []
    for jf, vf in zip(valid_json_files, valid_video_files):
        samples = process_json_file(jf, vf)
        all_samples.extend(samples)
        # print(f"[INFO] {jf.name} → {len(samples)} 条样本")

    # 保存
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for s in all_samples:
            if s is None:
                continue
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"[INFO] 总计 {len(all_samples)} 条样本，已保存到 {out}")


if __name__ == "__main__":
    main()
