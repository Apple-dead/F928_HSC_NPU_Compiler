"""Repack existing Conv weights in groups of eight without changing other bytes."""
import csv
import json
from pathlib import Path

import numpy as np

import generate_conv_relu_examples as g
import merge
from export_output_ddr import refresh_package
from weight_to_bram_coe import convert_weights
from params_to_bram_coe import build_layer_bytes


LAYOUT = 'output_channel_then_input_channel_group_then_kernel_hw_then_8_channels'


def refresh(root):
    summaries = json.loads((root/'summary.json').read_text(encoding='utf-8'))
    for m in summaries:
        case = root/m['case']
        before = {p.name:p.read_bytes() for p in case.iterdir() if p.is_file()}
        height,width,cin = m['input_shape_hwc']
        cout = m['output_shape_hwc'][2]
        aligned = (cin+7)//8*8
        tensor = np.zeros((cout,3,3,aligned),dtype=np.int64)
        with (case/'weights_ohwi.csv').open(encoding='utf-8',newline='') as f:
            for row in csv.DictReader(f):
                tensor[int(row['out_ch']),int(row['kh']),int(row['kw'])] = [int(row[f'in_ch{c}']) for c in range(aligned)]
        raw = tensor[:,:,:,:cin].transpose(0,3,1,2).reshape(-1)
        weights,_,_ = convert_weights(raw,cin,cout,3,3)
        # A direct coordinate walk is independent of the compiler's OIHW conversion.
        expected = bytes(int(tensor[o,h,w,group+lane]) & 255
                         for o in range(cout) for group in range(0,aligned,8)
                         for h in range(3) for w in range(3) for lane in range(8))
        assert bytes(weights) == expected
        image = bytearray(before['all.bin'])
        base = int(m['parameter_addr'],16)
        params = bytearray(image[base:base+m['parameter_size_bytes']])
        weight_size = 9*aligned
        stride = m.get('parameter_record_stride_bytes',weight_size)
        allowed = set()
        for oc in range(cout):
            offset = oc*stride
            params[offset:offset+weight_size] = expected[oc*weight_size:(oc+1)*weight_size]
            allowed.update(range(base+offset,base+offset+weight_size))
        plan = json.loads((case/'memory_plan.json').read_text(encoding='utf-8'))
        if m.get('parameter_record_stride_bytes'):
            with (case/'bias.csv').open(encoding='utf-8',newline='') as f:
                biases = [int(row['signed_int8_value']) for row in csv.DictReader(f)][:cout]
            assert bytes(build_layer_bytes(plan['execution_plan'][0],plan['tensors']['layer1_weight'],raw,biases)) == params
        image[base:base+len(params)] = params
        assert all(image[i] == before['all.bin'][i] for i in range(len(image)) if i not in allowed)
        merge.write_word_coe(case/'params.coe',list(params),16,4,'little',1)
        merge.write_word_coe(case/'all.coe',list(image),16,64,'little',1)
        g.write_mem(case/'all.mem',bytes(image),64)
        (case/'all.bin').write_bytes(image)
        decoded,_,_ = merge.coe_values_to_bytes(case/'all.coe',64,'little')
        assert bytes(decoded) == image
        m['weight_layout'] = LAYOUT
        plan['tensors']['layer1_weight']['layout'] = LAYOUT
        g.write_json(case/'memory_plan.json',plan)
        g.write_json(case/'memory_map.json',m)
        text = before['memory_map.txt'].decode('utf-8').split('\nWEIGHT_LAYOUT =')[0]
        (case/'memory_map.txt').write_text(text+'\nWEIGHT_LAYOUT = '+LAYOUT+'\n',encoding='utf-8')
        mutable = {'all.bin','all.coe','all.mem','params.coe','memory_plan.json','memory_map.json','memory_map.txt'}
        assert all((case/name).read_bytes() == data for name,data in before.items() if name not in mutable)
        print(root.name,m['case'],'PASS: weights repacked; input/bias/instructions/scale/output/DDR reference unchanged')
    g.write_json(root/'summary.json',summaries)
    reference = root/'reference.txt'
    text = reference.read_text(encoding='utf-8')
    marker = '权重内存：输出通道→输入通道组→核行→核列→组内8通道；本TXT仍按逻辑通道显示矩阵。'
    if marker not in text:
        text = text.replace('Conv + LeakyReLU 仿真参考矩阵\n','Conv + LeakyReLU 仿真参考矩阵\n'+marker+'\n',1)
    reference.write_text(text,encoding='utf-8')
    readme = root/'README.md'
    text = readme.read_text(encoding='utf-8').split('\n## Weight layout update')[0]
    text = text.replace('Weights: kh -> kw -> input channel (input channels padded to a multiple of 8).',
                        'Weights: output channel -> input channel group -> kh -> kw -> eight channel lanes (zero padded).')
    text += '\n## Weight layout update\n\nWeight order: output channel -> input channel group -> kernel row -> kernel column -> eight channel lanes. For each output, all 3x3 positions of channels 1..8 precede all 3x3 positions of channels 9..16. Zero padding, bias bytes and placement, addresses, input, instructions, scale and expected output are unchanged.\n'
    readme.write_text(text,encoding='utf-8')
    refresh_package(root)


if __name__ == '__main__':
    for summary in sorted((g.PROJECT_ROOT/'examples').rglob('summary.json')):
        refresh(summary.parent)
