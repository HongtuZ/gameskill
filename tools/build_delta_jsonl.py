#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

SYSTEM_PROMPT = "你是一个专业的FPS游戏实时教练。"


def str_to_guide(s: str) -> bool:
    if s == "是" or s is True:
        return True
    else:
        return False


class DeltaData:
    def __init__(self, json_path: Path, video_path: Path):
        self.json_path = json_path
        self.video_path = video_path
        with open(json_path, "r", encoding="utf-8") as f:
            try:
                annotation = json.load(f)
            except json.JSONDecodeError:
                print(f"[WARN] JSON 解码错误 {json_path}")
                raise
        self.time = annotation.get("time")
        self.map_name = annotation.get("map_name")
        self.map_difficulty = annotation.get("map_difficulty")
        self.ego_hero_name = annotation.get("ego_hero_name")
        self.skill1 = annotation.get("skill_1")
        self.skill2 = annotation.get("skill_2")
        self.skill3 = annotation.get("skill_3")
        self.ego_hero_state = annotation.get("ego_hero_state")
        self.helmet_state = annotation.get("helmet_state")
        self.armor_state = annotation.get("armor_state")
        self.need_guidance = str_to_guide(annotation.get("need_guide"))
        self.guidance = annotation.get("guide_text", "")

        self.description = annotation.get("reasons", {}).get("description")
        self.situation = annotation.get("reasons", {}).get("situation")
        self.guidance_reason = annotation.get("reasons", {}).get("guidance_reason")

        self.teammate_info = annotation.get("teammate_info", [])
        self.kill_info = annotation.get("kill_info", [])

    def is_valid(self) -> bool:
        if not self.description:
            return False
        if self.need_guidance and (not self.guidance or not self.description or not self.guidance_reason):
            return False
        return True

    def format_kills(self) -> str:
        if not self.kill_info:
            return "暂无击杀信息"
        lines = []
        for kill in self.kill_info:
            lines.append(f"{kill.get('kill_name')}: 使用{kill.get('gun')}")
        return "\n".join(lines)

    def format_teammate_info(self) -> str:
        if not self.teammate_info:
            return "暂无队友信息"
        lines = []
        for teammate in self.teammate_info:
            lines.append(f"{teammate.get('tm_name')}: {teammate.get('state')}")
        return "\n".join(lines)

    def build_think_content(self) -> str:
        parts = []
        base_info = ""
        base_info += f"时间：{self.time}\n" if self.time else ""
        base_info += f"当前地图：{self.map_name}\n" if self.map_name else ""
        base_info += f"地图难度：{self.map_difficulty}\n" if self.map_difficulty else ""
        base_info += f"当前角色：{self.ego_hero_name}\n" if self.ego_hero_name else ""
        base_info += f"当前技能1状态：{self.skill1}\n" if self.skill1 else ""
        base_info += f"当前技能2状态：{self.skill2}\n" if self.skill2 else ""
        base_info += f"当前技能3状态：{self.skill3}\n" if self.skill3 else ""
        base_info += f"当前角色状态：{self.ego_hero_state}\n" if self.ego_hero_state else ""
        base_info += f"当前头盔状态：{self.helmet_state}\n" if self.helmet_state else ""
        base_info += f"当前护甲状态：{self.armor_state}\n" if self.armor_state else ""

        base_info += f"队友信息：\n{self.format_teammate_info()}\n" if self.teammate_info else ""
        base_info += f"击杀信息:\n{self.format_kills()}\n" if self.kill_info else ""

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


def build_think_guidance_sample(delta_data: DeltaData) -> Dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "<video>\n请分析这段游戏视频片段，判断是否需要立即给出实时指导。"},
            {
                "role": "assistant",
                "content": f"<think>\n{delta_data.build_think_content()}\n</think>",
                "loss_scale": 1.0,
            },
            {"role": "assistant", "content": delta_data.build_guidance_content(), "loss_scale": 2.0},
        ],
        "videos": [str(delta_data.video_path)],
    }


def process_json_file(json_path: Path, video_path: Path) -> List[Dict]:
    """处理单个 JSON 文件，返回训练样本列表"""
    dleta_data = DeltaData(json_path, video_path)
    samples = []
    if not dleta_data.is_valid():
        return samples
    think_guidance_sample = build_think_guidance_sample(dleta_data)
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
