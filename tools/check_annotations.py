#!/usr/bin/env python3
"""
标注JSON文件内容合理性检查脚本。

用法:
    python tools/check_annotations.py <json文件路径>
    python tools/check_annotations.py <目录路径>          # 递归检查所有叶子目录下的json
    python tools/check_annotations.py <目录路径> --verbose # 显示详细信息

检查规则:
    1. description 是必填项（不能为空）
    2. 当 need_guide 为 "是" 时，必须包含 guide_text, situation, guidance_reason, ego_hero_name
    3. 同一叶子目录下连续编号的json文件中，若 map_name 或 ego_hero_name 发生变化，给出 warning 提示
"""

import json
import re
import sys
from pathlib import Path


def load_json(filepath):
    """加载JSON文件，返回解析后的字典和可能的错误信息。filepath 可以是 str 或 Path。"""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data, None
    except json.JSONDecodeError as e:
        return None, f"JSON解析失败: {e}"
    except Exception as e:
        return None, f"文件读取失败: {e}"


def safe_str(val):
    """安全地将值转为字符串，None返回空字符串。"""
    if val is None:
        return ""
    return str(val)


def check_single_file(filepath, data):
    """
    检查单个JSON文件的内容合理性。
    返回 (errors, warnings) 列表。
    """
    errors = []
    warnings = []

    # --- 必填字段检查 ---
    reasons = data.get("reasons", {}) or {}

    # 1. description 是必填项
    description = safe_str(reasons.get("description", ""))
    if not description.strip():
        errors.append("reasons.description 为空（必填项）")

    # 2. need_guide 为 "是" 时的必填字段检查
    need_guide = safe_str(data.get("need_guide", "")).strip()
    if need_guide == "是":
        # guide_text
        guide_text = safe_str(data.get("guide_text", ""))
        if not guide_text.strip():
            errors.append("need_guide=是 时 guide_text 不能为空")

        # situation
        situation = safe_str(reasons.get("situation", ""))
        if not situation.strip():
            errors.append("need_guide=是 时 reasons.situation 不能为空")

        # guidance_reason
        guidance_reason = safe_str(reasons.get("guidance_reason", ""))
        if not guidance_reason.strip():
            errors.append("need_guide=是 时 reasons.guidance_reason 不能为空")

        # ego_hero_name
        ego_hero_name = safe_str(data.get("ego_hero_name", ""))
        if not ego_hero_name.strip():
            errors.append("need_guide=是 时 ego_hero_name 不能为空")

    # --- 额外合理性检查（warning级别）---
    # time 为空
    time_val = safe_str(data.get("time", ""))
    if not time_val.strip():
        warnings.append("time 为空")

    # map_name 为空
    map_name = safe_str(data.get("map_name", ""))
    if not map_name.strip():
        warnings.append("map_name 为空")

    # ego_hero_name 为空（非 need_guide 场景）
    ego_hero_name = safe_str(data.get("ego_hero_name", ""))
    if (not ego_hero_name.strip()) and need_guide != "是":
        warnings.append("ego_hero_name 为空")

    return errors, warnings


def extract_index(filename):
    """从文件名中提取编号，如 '059.json' -> 59。"""
    stem = Path(filename).stem
    match = re.match(r"^(\d+)$", stem)
    if match:
        return int(match.group(1))
    return None


def check_transitions(sorted_files):
    """
    检查连续编号文件中 map_name 和 ego_hero_name 的变化。
    返回 {(file_pair): [(field, old_val, new_val), ...]} 的字典。
    """
    transitions = {}
    prev_data = None
    prev_file = None

    for filepath in sorted_files:
        data, err = load_json(filepath)
        if err or data is None:
            continue

        if prev_data is not None:
            changes = []
            for field in ["map_name", "ego_hero_name"]:
                old_val = safe_str(prev_data.get(field, ""))
                new_val = safe_str(data.get(field, ""))
                if old_val != new_val:
                    changes.append((field, old_val, new_val))
            if changes:
                transitions[(prev_file, filepath)] = changes

        prev_data = data
        prev_file = filepath

    return transitions


def find_leaf_dirs(root):
    """
    递归查找所有叶子目录（包含JSON文件且不再包含子目录的目录）。
    返回按路径排序的叶子目录 Path 列表。
    """
    root = Path(root)
    leaf_dirs = []
    for dirpath in sorted(root.rglob("*")):
        if not dirpath.is_dir():
            continue
        # 有子目录则不是叶子
        subdirs = [p for p in dirpath.iterdir() if p.is_dir()]
        if subdirs:
            continue
        json_files = list(dirpath.glob("*.json"))
        if json_files:
            leaf_dirs.append(dirpath)
    return leaf_dirs


def check_directory(dirpath, verbose=False):
    """
    递归检查目录树：找到所有叶子目录，每个叶子目录独立检查。
    返回 (grand_total_errors, grand_total_warnings, dir_count) 用于汇总。
    """
    dirpath = Path(dirpath)
    leaf_dirs = find_leaf_dirs(dirpath)
    if not leaf_dirs:
        # 当前目录本身可能直接包含json（非递归场景兜底）
        json_files = list(dirpath.glob("*.json"))
        if json_files:
            leaf_dirs = [dirpath]
        else:
            print(f"目录 {dirpath} 及其子目录下未找到JSON文件")
            return 0, 0, 0

    grand_total_errors = 0
    grand_total_warnings = 0

    print(f"\n{'=' * 70}")
    print(f"递归扫描目录: {dirpath}")
    print(f"发现 {len(leaf_dirs)} 个叶子目录")
    print(f"{'=' * 70}")

    for leaf in leaf_dirs:
        dir_errors, dir_warnings = _check_single_leaf(leaf, verbose=verbose)
        grand_total_errors += dir_errors
        grand_total_warnings += dir_warnings

    # 全局汇总
    print(f"\n{'=' * 70}")
    print("全局汇总:")
    print(f"  叶子目录数: {len(leaf_dirs)}")
    print(f"  总错误数:   {grand_total_errors}")
    print(f"  总警告数:   {grand_total_warnings}")
    if grand_total_errors == 0 and grand_total_warnings == 0:
        print("  结果: ✅ 全部通过")
    elif grand_total_errors == 0:
        print("  结果: ⚠️  有警告，无错误")
    else:
        print("  结果: ❌ 存在错误，请修复")
    print(f"{'=' * 70}")

    return grand_total_errors, grand_total_warnings, len(leaf_dirs)


def _check_single_leaf(dirpath, verbose=False):
    """检查单个叶子目录下所有按编号排列的JSON文件。返回 (errors, warnings) 计数。"""
    dirpath = Path(dirpath)
    json_files = sorted(dirpath.glob("*.json"), key=lambda f: extract_index(f) or float("inf"))
    if not json_files:
        return 0, 0

    # 检查编号连续性
    indices = [extract_index(f) for f in json_files]
    seq_warnings = []
    for i in range(1, len(indices)):
        if indices[i] is not None and indices[i - 1] is not None:
            if indices[i] != indices[i - 1] + 1 and indices[i] != indices[i - 1]:
                seq_warnings.append(f"编号不连续: {json_files[i - 1].name} -> {json_files[i].name}")

    # 逐文件检查
    dir_errors = 0
    dir_warnings = 0
    file_issues = []

    for filepath in json_files:
        data, err = load_json(filepath)

        if err:
            file_issues.append((filepath.name, [f"ERROR: {err}"], []))
            dir_errors += 1
            continue

        errors, warnings = check_single_file(filepath, data)
        if errors or warnings or verbose:
            file_issues.append((filepath.name, errors, warnings))
        dir_errors += len(errors)
        dir_warnings += len(warnings)

    # 检查连续文件间的字段变化
    transitions = check_transitions(json_files)
    transition_msgs = []
    for (f1, f2), changes in transitions.items():
        for field, old_val, new_val in changes:
            transition_msgs.append(f'{Path(f1).name} -> {Path(f2).name}: {field} "{old_val}" -> "{new_val}"')
            dir_warnings += 1

    # 只有存在问题（或verbose）时才打印该目录的详情
    has_issues = dir_errors > 0 or dir_warnings > 0 or seq_warnings
    if has_issues or verbose:
        print(f"\n  [{dirpath}] ({len(json_files)} 个文件)")
        for w in seq_warnings:
            print(f"    ⚠️  WARNING: {w}")
        for fname, errors, warnings in file_issues:
            print(f"    [{fname}]")
            for e in errors:
                print(f"      ❌ ERROR: {e}")
            for w in warnings:
                print(f"      ⚠️  WARNING: {w}")
            if not errors and not warnings and verbose:
                print("      ✅ OK")
        for msg in transition_msgs:
            print(f"    ⚠️  WARNING: {msg}")

    return dir_errors, dir_warnings


def main():
    if len(sys.argv) < 2:
        print("用法:")
        print("  python tools/check_annotations.py <json文件路径>")
        print("  python tools/check_annotations.py <目录路径> [--verbose]")
        sys.exit(1)

    target = Path(sys.argv[1])
    verbose = "--verbose" in sys.argv or "-v" in sys.argv

    if target.is_file():
        # 单文件检查
        data, err = load_json(target)
        if err:
            print(f"[{target.name}] ERROR: {err}")
            sys.exit(1)

        errors, warnings = check_single_file(target, data)

        print(f"\n[{target.name}]")
        for e in errors:
            print(f"  ❌ ERROR: {e}")
        for w in warnings:
            print(f"  ⚠️  WARNING: {w}")

        if not errors and not warnings:
            print("  ✅ 检查通过")

        sys.exit(1 if errors else 0)

    elif target.is_dir():
        total_e, total_w, _ = check_directory(target, verbose=verbose)
        sys.exit(1 if total_e > 0 else 0)
    else:
        print(f"路径不存在: {target}")
        sys.exit(1)


if __name__ == "__main__":
    main()
