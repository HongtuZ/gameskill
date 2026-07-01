from pathlib import Path

# 配置
ANNOTATIONS_DIR = "annotations"  # 存放 .json 标注文件的目录
VIDEOS_DIR = "dataset_videos"  # 存放 .mp4 视频文件的目录


def main():
    annotations_dir = Path(ANNOTATIONS_DIR)
    videos_dir = Path(VIDEOS_DIR)

    if not annotations_dir.exists():
        print(f"错误: 标注目录不存在: {annotations_dir}")
        return
    if not videos_dir.exists():
        print(f"错误: 视频目录不存在: {videos_dir}")
        return

    # 遍历所有 json 文件
    json_files = sorted(annotations_dir.rglob("*.json"))
    mp4_files = list(videos_dir.rglob("*.mp4"))
    print(f"在 {ANNOTATIONS_DIR}/ 下找到 {len(json_files)} 个 .json 文件")
    print(f"在 {VIDEOS_DIR}/ 下查找对应的 .mp4 文件")
    print("-" * 60)

    missing = []
    found = []

    for json_file in json_files:
        rel_json = json_file.relative_to(annotations_dir)
        dst_mp4 = videos_dir / rel_json.with_suffix(".mp4")
        if dst_mp4.exists():
            found.append((rel_json, dst_mp4.relative_to(videos_dir)))
        else:
            missing.append(rel_json)

    # 输出结果
    print(f"有效json文件 {len(found)} 个")
    print(f"无效json文件列表 ({len(missing)} 个):")
    for i, rel_json in enumerate(missing, 1):
        print(f"{i:3d}. {rel_json}")
    print(f"还剩 {len(mp4_files) - len(found)} 个待标注视频")


if __name__ == "__main__":
    main()
