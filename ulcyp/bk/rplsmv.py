import torch
import os
from collections import OrderedDict


def deep_rename_and_save(input_path, output_path, rename_map, dry_run=False):
    """
    深度遍历 Key，替换路径中任意位置的模块名称。
    """
    print(f"正在加载模型: {input_path} ...")
    checkpoint = torch.load(input_path, map_location='cpu')

    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        is_wrapper = True
    else:
        state_dict = checkpoint
        is_wrapper = False

    new_state_dict = OrderedDict()
    changed_count = 0

    print("\n--- 深度重命名预览 ---")

    # 遍历每一个参数 Key
    for key, value in state_dict.items():
        # 1. 把 key 按点拆分成列表
        # 例如: "task.m4.bias" -> ["task", "m4", "bias"]
        parts = key.split('.')

        modified = False
        new_parts = []

        # 2. 检查每一段是否在 rename_map 里
        for part in parts:
            if part in rename_map:
                # 如果这一段也就是 "m4"，就换成 "multi_net3"
                new_parts.append(rename_map[part])
                modified = True
            else:
                new_parts.append(part)

        # 3. 重新拼回去
        new_key = ".".join(new_parts)

        if modified:
            print(f"🔄 {key}\n   ↳ {new_key}")
            changed_count += 1

        new_state_dict[new_key] = value

    if changed_count == 0:
        print("⚠️  没有发现需要重命名的参数。")
        return

    print(f"\n共 {changed_count} 个参数将被重命名。")

    if dry_run:
        print("\n[Dry Run] 模式开启，未执行保存。")
    else:
        if is_wrapper:
            checkpoint['state_dict'] = new_state_dict
            data_to_save = checkpoint
        else:
            data_to_save = new_state_dict

        print(f"正在保存新模型到: {output_path} ...")
        torch.save(data_to_save, output_path)
        print("✅ 完成！")


# ==========================================
# 👇 配置区域
# ==========================================

if __name__ == "__main__":
    OLD_PATH = "../model_results/clean_model.pth"  # 输入文件
    NEW_PATH = "../model_results/final_renamed_model.pth"  # 输出文件

    # 你的完整映射表
    RENAME_MAP = {
        "gt": "gate_fusion",
        "gud2": "fusion",
        "m4": "multi_net3",
        "m3": "multi_net2",
        "m2": "multi_net1",
        "s1": "single_net"
    }

    # 先开启 Dry Run 确认所有的层是不是都变对了
    DRY_RUN = False

    if os.path.exists(OLD_PATH):
        deep_rename_and_save(OLD_PATH, NEW_PATH, RENAME_MAP, dry_run=DRY_RUN)
    else:
        print(f"❌ 找不到文件: {OLD_PATH}")