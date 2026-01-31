import torch
import os


def clean_and_save_model(input_path, output_path, attrs_to_remove, dry_run=False):
    """
    加载模型权重，删除指定前缀的参数，然后保存。

    Args:
        input_path (str): 旧模型路径 (.pth)
        output_path (str): 新模型保存路径
        attrs_to_remove (list): 要删除的属性名称列表 (例如 ['aux_head', 'layer4'])
        dry_run (bool): 如果为 True，只打印要删除的键，不进行实际保存。
    """

    print(f"正在加载模型: {input_path} ...")
    # map_location='cpu' 保证即使没有 GPU 也能处理，且节省显存
    checkpoint = torch.load(input_path, map_location='cpu')

    # 处理 checkpoint 可能是个字典包含 'state_dict' 的情况，也可以直接是 state_dict
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        is_wrapper = True  # 标记这是一个包含 epoch 等信息的完整 checkpoint
    else:
        state_dict = checkpoint
        is_wrapper = False

    # 找出所有需要删除的 key
    keys_to_delete = []
    all_keys = list(state_dict.keys())

    for key in all_keys:
        for attr in attrs_to_remove:
            # 匹配逻辑：
            # 1. 完全匹配 (例如删除名为 'sigma' 的参数)
            # 2. 前缀匹配 (例如 'aux_head.' 开头的所有参数)
            if key == attr or key.startswith(f"{attr}."):
                keys_to_delete.append(key)
                break  # 找到一个匹配的 attr 就可以跳出内层循环了

    # 打印预览
    if not keys_to_delete:
        print("\n⚠️  未找到任何匹配 '{attrs_to_remove}' 的参数。模型未发生变化。")
        return

    print(f"\n发现 {len(keys_to_delete)} 个参数将被删除:")
    for i, k in enumerate(keys_to_delete):
        if i < 5: print(f"  - {k}")  # 只打印前5个，避免刷屏
    if len(keys_to_delete) > 5: print(f"  - ... 等 (共 {len(keys_to_delete)} 个)")

    # 执行删除或空跑
    if dry_run:
        print("\n[Dry Run] 模式开启，未执行实际删除和保存。")
    else:
        for k in keys_to_delete:
            del state_dict[k]

        # 如果原始文件是 wrapper 结构，我们需要把修改后的 state_dict 放回去
        if is_wrapper:
            checkpoint['state_dict'] = state_dict
            data_to_save = checkpoint
        else:
            data_to_save = state_dict

        print(f"\n正在保存清洗后的模型到: {output_path} ...")
        torch.save(data_to_save, output_path)
        print("✅ 完成！")


if __name__ == "__main__":
    # 1. 你的旧模型文件路径
    OLD_MODEL_PATH = "../model_results/model.pt"

    # 2. 你想保存的新文件名
    NEW_MODEL_PATH = "../model_results/clean_model.pth"

    # 3. 你在代码里删掉的那些模块的名字 (比如 self.aux_head, self.loss_layer)
    #    只需要写名字字符串即可
    ATTRS_TO_DELETE = [
        "molecule_graph",
        "uma_tools",
        "aa_attn",
        "cross",
        "gate_layer_thr",
        "fd_tools.fingerprint_mlp",
        "fd_tools.our_mlp",
        "down_stream_task.sub_model",
        "down_stream_task.inh_model",
        "down_stream_task.ahr_model.norm_task.cross_attention",
        "down_stream_task.car_model.norm_task.cross_attention",
        "down_stream_task.pxr_model.norm_task.cross_attention",
        "down_stream_task.ahr_model.norm_task.self_attention",
        "down_stream_task.car_model.norm_task.self_attention",
        "down_stream_task.pxr_model.norm_task.self_attention",
        "down_stream_task.ahr_model",
        "down_stream_task.car_model",
        "down_stream_task.pxr_model",
        "combine_tools",
        "gud1"

        # 例如：删除了 self.aux_head
    ]

    A1 = [f"down_stream_task.inducer_model_seq.inducer_model_{i}.pu_task.cross_attention" for i in range(5)]
    A2 = [f"down_stream_task.inducer_model_seq.inducer_model_{i}.pu_task.self_attention" for i in range(5)]
    A3 = [f"down_stream_task.inducer_model_seq.inducer_model_{i}.pu_task.protein_projection" for i in range(5)]
    A4 = [f"down_stream_task.inducer_model_seq.inducer_model_{i}.pu_task.output_projection_protein" for i in range(5)]
    ATTRS_TO_DELETE += A1
    ATTRS_TO_DELETE += A2
    ATTRS_TO_DELETE += A3
    ATTRS_TO_DELETE += A4

    # 4. 建议先设为 True 看看打印出来的 key 对不对，确认无误后改为 False
    DRY_RUN = False

    # 运行工具
    if os.path.exists(OLD_MODEL_PATH):
        clean_and_save_model(OLD_MODEL_PATH, NEW_MODEL_PATH, ATTRS_TO_DELETE, dry_run=DRY_RUN)
    else:
        print(f"❌ 找不到文件: {OLD_MODEL_PATH}")