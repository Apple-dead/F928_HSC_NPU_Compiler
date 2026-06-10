import os
import sys
import csv
import importlib
import importlib.util
import torch
import numpy as np


def import_model_definition(model_py_path):
    """
    导入模型定义文件，用于兼容 torch.save(model, "model.pth") 保存的完整模型对象。

    注意：如果模型定义文件内部还 import 了其他工程文件，请尽量在项目根目录下运行本脚本。
    例如：
        cd your_project_root
        python ./python/extract_pth_paramss_v2.py ./model/yolov2_14layer_quantized.py ./model/yolov2_fixed.pth ./data/model_params/
    """
    model_py_path = os.path.abspath(model_py_path)

    if not os.path.isfile(model_py_path):
        raise FileNotFoundError(f"模型定义文件不存在：{model_py_path}")

    cwd = os.getcwd()
    model_dir = os.path.dirname(model_py_path)
    model_parent_dir = os.path.dirname(model_dir)

    # 尽量把常见 import 路径都加入 sys.path，方便模型文件继续导入它依赖的其他 .py 文件
    for path in [cwd, model_dir, model_parent_dir]:
        if path and path not in sys.path:
            sys.path.insert(0, path)

    imported_modules = []

    # 1. 优先尝试按“相对于当前运行目录的 dotted module 名”导入
    #    例如 ./model/yolov2_14layer_quantized.py -> model.yolov2_14layer_quantized
    try:
        rel_path = os.path.relpath(model_py_path, cwd)
        if not rel_path.startswith(".."):
            dotted_name = os.path.splitext(rel_path)[0].replace(os.sep, ".")
            if dotted_name and dotted_name.endswith(".__init__"):
                dotted_name = dotted_name[: -len(".__init__")]
            if dotted_name:
                module = importlib.import_module(dotted_name)
                imported_modules.append(module)
                print(f"[INFO] 已按模块路径导入模型定义：{dotted_name}")
    except Exception as e:
        print(f"[WARN] 按模块路径导入模型定义失败，将尝试按文件路径导入：{e}")

    # 2. 再按文件路径导入一次，兼容普通单文件 model.py
    module_name = os.path.splitext(os.path.basename(model_py_path))[0]
    try:
        if module_name in sys.modules:
            module = sys.modules[module_name]
        else:
            spec = importlib.util.spec_from_file_location(module_name, model_py_path)
            if spec is None or spec.loader is None:
                raise ImportError(f"无法从文件创建 module spec：{model_py_path}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
        imported_modules.append(module)
        print(f"[INFO] 已按文件路径导入模型定义：{model_py_path}")
    except Exception as e:
        print(f"[WARN] 按文件路径导入模型定义失败：{e}")

    if not imported_modules:
        raise ImportError("模型定义文件导入失败，请检查该 .py 文件及其依赖是否完整")

    # 3. 兼容一种特殊情况：当初保存模型时，模型类在 __main__ 里
    #    这里把导入模块中的类/函数也挂到当前 __main__ 上，增加 torch.load 成功概率。
    main_module = sys.modules.get("__main__")
    if main_module is not None:
        for module in imported_modules:
            for name, value in module.__dict__.items():
                if not name.startswith("__"):
                    setattr(main_module, name, value)

    return imported_modules[-1]


def torch_load_compat(pth_path, allow_full_model=False):
    """
    兼容不同 PyTorch 版本的 torch.load。

    allow_full_model=False：优先按安全/普通参数方式加载，适合 state_dict/checkpoint。
    allow_full_model=True ：允许加载完整 model 对象，适合 torch.save(model, "model.pth")。
    """
    if allow_full_model:
        # 新版 PyTorch 加载完整对象通常需要 weights_only=False；老版没有该参数，所以做兼容。
        try:
            return torch.load(pth_path, map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(pth_path, map_location="cpu")
    else:
        # 尽量保持原脚本行为；如果 PyTorch 新版本支持 weights_only=True，就先走更适合参数字典的加载。
        try:
            return torch.load(pth_path, map_location="cpu", weights_only=True)
        except TypeError:
            return torch.load(pth_path, map_location="cpu")
        except Exception as e:
            print(f"[WARN] 按 weights_only=True 加载失败，尝试按原 torch.load 行为加载：{e}")
            try:
                return torch.load(pth_path, map_location="cpu")
            except Exception:
                raise e


def get_state_dict(ckpt):
    """
    兼容不同 .pth 保存格式：
    1. 直接保存 state_dict
    2. 保存 checkpoint 字典，里面包含 model_state_dict / state_dict / model
    3. 保存整个 model 对象
    """
    if isinstance(ckpt, dict):
        # 常见 checkpoint 写法
        for key in ["model_state_dict", "state_dict", "model", "net"]:
            if key in ckpt:
                if isinstance(ckpt[key], dict):
                    print(f"[INFO] 检测到 checkpoint，使用 ckpt['{key}'] 作为模型参数")
                    return ckpt[key]
                if hasattr(ckpt[key], "state_dict"):
                    print(f"[INFO] 检测到 checkpoint，使用 ckpt['{key}'].state_dict() 作为模型参数")
                    return ckpt[key].state_dict()

        # 如果这个 dict 本身就是 state_dict
        # 一般 state_dict 的 value 是 Tensor
        tensor_values = [v for v in ckpt.values() if torch.is_tensor(v)]
        if len(tensor_values) > 0:
            print("[INFO] 检测到当前 .pth 本身就是 state_dict")
            return ckpt

    # 如果保存的是整个 model 对象
    if hasattr(ckpt, "state_dict"):
        print("[INFO] 检测到保存的是整个 model 对象，使用 ckpt.state_dict()")
        return ckpt.state_dict()

    raise TypeError("无法识别这个 .pth 文件的格式")


def tensor_to_numpy(tensor):
    """
    把 PyTorch Tensor 转成 numpy array。
    """
    return tensor.detach().cpu().numpy()


def print_usage():
    print("用法1: python extract_pth_paramss_v2.py <输入模型.pth> <输出文件夹路径>")
    print("示例1: python extract_pth_paramss_v2.py your_model.pth ./params")
    print("")
    print("用法2: python extract_pth_paramss_v2.py <模型定义.py> <输入模型.pth> <输出文件夹路径>")
    print("示例2: python extract_pth_paramss_v2.py ./model/yolov2_14layer_quantized.py ./model/yolov2_fixed.pth ./data/model_params/")


def main():
    model_py_path = None

    if len(sys.argv) == 3:
        # 兼容旧用法：python extract_pth_paramss.py <输入模型> <输出文件夹路径>
        pth_path = sys.argv[1]
        output_dir = sys.argv[2]
        allow_full_model = False
    elif len(sys.argv) == 4:
        # 新用法：python extract_pth_paramss.py <模型定义.py> <输入模型> <输出文件夹路径>
        model_py_path = sys.argv[1]
        pth_path = sys.argv[2]
        output_dir = sys.argv[3]
        allow_full_model = True
    else:
        print_usage()
        return

    if not os.path.isfile(pth_path):
        raise FileNotFoundError(f"输入模型文件不存在：{pth_path}")

    os.makedirs(output_dir, exist_ok=True)

    # 如果用户提供了模型定义文件，先导入它，再加载 .pth
    # 这样可以兼容 torch.save(model, "model.pth") 保存的完整模型对象。
    if model_py_path is not None:
        print(f"[INFO] 检测到三参数模式，先导入模型定义文件：{model_py_path}")
        import_model_definition(model_py_path)

    # 1. 加载 .pth
    ckpt = torch_load_compat(pth_path, allow_full_model=allow_full_model)

    # 2. 提取 state_dict
    state_dict = get_state_dict(ckpt)

    # 3. 保存参数统计信息
    summary_path = os.path.join(output_dir, "params_summary.csv")

    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow([
            "name",
            "shape",
            "dtype",
            "numel",
            "min",
            "max",
            "mean",
            "std"
        ])

        print("=" * 100)
        print("模型参数列表")
        print("=" * 100)

        for name, tensor in state_dict.items():
            if not torch.is_tensor(tensor):
                print(f"[SKIP] {name}: 不是 Tensor，类型是 {type(tensor)}")
                continue

            arr = tensor_to_numpy(tensor)

            # 打印基本信息
            print(f"{name:30s} shape={list(arr.shape)} dtype={arr.dtype} numel={arr.size}")

            # 不再保存 .npy 文件

            # 如果你想看文本，也可以保存成 .txt
            # 注意：大权重保存成 txt 会很大
            safe_name = name.replace(".", "_")
            txt_path = os.path.join(output_dir, safe_name + ".txt")
            np.savetxt(txt_path, arr.reshape(-1), fmt="%.8f")

            # 统计信息
            if arr.size > 0:
                writer.writerow([
                    name,
                    list(arr.shape),
                    str(arr.dtype),
                    arr.size,
                    float(arr.min()),
                    float(arr.max()),
                    float(arr.mean()),
                    float(arr.std())
                ])
            else:
                writer.writerow([
                    name,
                    list(arr.shape),
                    str(arr.dtype),
                    arr.size,
                    "",
                    "",
                    "",
                    ""
                ])

    print("=" * 100)
    print(f"参数已经提取到目录：{output_dir}")
    print(f"参数统计表：{summary_path}")
    print("=" * 100)


if __name__ == "__main__":
    main()
