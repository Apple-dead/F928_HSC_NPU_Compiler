# coe_to_bin 说明

## 1. 目的

`python/coe_to_bin/export_npu_bins.py` 是 DDR 运行场景的附加导出阶段。它不改变原有 COE 生成、合并和回归逻辑，只在 `target/` 下额外生成裸二进制文件：

```text
target/npu_params.bin
target/npu_instr.bin
```

## 2. 构建位置

`build.sh` / `build.bat` 在 `python/merge.py` 生成 `target/all.coe` 之后调用本脚本：

```text
data/memory_plan.json + coe/*.coe
  -> python/coe_to_bin/export_npu_bins.py
  -> target/npu_params.bin
  -> target/npu_instr.bin
```

`clean` 会删除这两个 bin 产物。

## 3. 导出规则

`npu_params.bin` 按 `data/memory_plan.json` 的 `init_regions` 地址规划导出：

```text
从 INIT_BASE_ADDR 开始
包含 instr 之前的所有 init region
遇到 instr region 停止
region 间如有地址空洞则补 0
```

因此：

```text
IMAGE_SOURCE="coe"      -> image + layer/linear params
IMAGE_SOURCE="external" -> layer/linear params
```

该文件应烧录到 `INIT_BASE_ADDR`。

`npu_instr.bin` 只包含 `instr` region 对应的 `coe/instr.coe`。指令 bin 的烧录地址可由运行系统单独指定，只要 C 程序把同一个地址写入 NPU 指令地址寄存器。

## 4. 端序约定

参数和 image COE 按现有 `merge.py` 规则导出为 little-endian byte：

```text
00000010 -> 10 00 00 00
```

指令 COE 按指令 word 显示顺序导出：

```text
04200000 -> 04 20 00 00
```
