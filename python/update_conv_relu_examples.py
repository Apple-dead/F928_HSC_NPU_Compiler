"""Refresh existing examples without changing their parameter bytes or addresses."""
from pathlib import Path
import csv
import hashlib
import json
import re
import zipfile

import numpy as np
import assembler
import generate_conv_relu_examples as g
import merge


def rows(path):
    with path.open(encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))


def refresh(root):
    summaries = json.loads((root / 'summary.json').read_text(encoding='utf-8'))
    first = root / summaries[0]['case']
    h, w, cin = summaries[0]['input_shape_hwc']
    cout = summaries[0]['output_shape_hwc'][2]
    padded_cin = (cin + 7) // 8 * 8
    padded_cout = (cout + 7) // 8 * 8
    x = np.zeros((h, w, padded_cin), dtype=np.int8)
    weights = np.zeros((cout, 3, 3, padded_cin), dtype=np.int8)
    bias = np.zeros(cout, dtype=np.int64)
    for row in rows(first / 'input_hwc.csv'):
        x[int(row['h']), int(row['w'])] = [int(row[f'ch{c}']) for c in range(padded_cin)]
    for row in rows(first / 'weights_ohwi.csv'):
        weights[int(row['out_ch']), int(row['kh']), int(row['kw'])] = [int(row[f'in_ch{c}']) for c in range(padded_cin)]
    for row in rows(first / 'bias.csv'):
        if int(row['out_ch']) < cout:
            bias[int(row['out_ch'])] = int(row['signed_int8_value'])
    stages = g.reference(x[:, :, :cin], weights[:, :, :, :cin], bias)
    acc, scaled, output, scale = stages
    packed = g.pack_input_hwc8(x)
    # Independent explicit indexing verifies both group boundaries and endian lanes.
    assert packed == bytes(int(x[i, j, group + lane]) & 255
                           for group in range(0, padded_cin, 8)
                           for i in range(h) for j in range(w) for lane in range(8))
    stored_output = np.zeros((h, w, padded_cout), dtype=np.int8)
    stored_output[:, :, :cout] = output
    expected = stored_output.tobytes()
    for m in summaries:
        case = root / m['case']
        untouched_names = ('params.coe', 'weights_ohwi.csv', 'bias.csv', 'input_hwc.csv', 'model_ir.json')
        before = {name: (case / name).read_bytes() for name in untouched_names}
        image = bytearray((case / 'all.bin').read_bytes())
        original = bytes(image)
        a = int(m['input_addr'], 16)
        pc = int(m['instruction_entry_addr'], 16)
        p = int(m['parameter_addr'], 16)
        params = original[p:p + m['parameter_size_bytes']]
        m.update(start_bit=scale,
                 input_layout='channel_group_then_h_then_w_then_8_channels',
                 accumulator_range=[int(acc.min()), int(acc.max())],
                 scaled_range=[int(scaled.min()), int(scaled.max())],
                 compute_order='int32_sum_plus_bias -> arithmetic_shift_right_scale -> signed_int8 -> negative_arithmetic_shift_right_3',
                 expected_output_sha256=hashlib.sha256(expected).hexdigest())
        reg = (int(m['registers']['RCONV2'], 16) & 0xFFFF) | (scale << 16)
        m['registers']['RCONV2'] = f'0x{reg:08X}'
        asm = (case / 'instr.asm').read_text(encoding='utf-8')
        asm = re.sub(r'(CFG_REGISTER CONV_P_4,\s*)0x[0-9A-Fa-f]+', lambda match: match[1] + f'0x{scale:04X}', asm)
        asm = re.sub(r'RCONV2=0x[0-9A-Fa-f]+', f'RCONV2=0x{reg:08X}', asm)
        words = assembler.assemble_lines(asm.splitlines())
        old_words = [int.from_bytes(original[pc+i*4:pc+i*4+4], 'little') for i in range(len(words))]
        assert len(words) == m['instruction_count']
        for old, new in zip(old_words, words):
            assert old == new or (old >> 26 == 1 and (old >> 16) & 0x3FF == 5 and old & 0xFFFF == 0)
        instructions = b''.join(word.to_bytes(4, 'little') for word in words)
        image[a:a+len(packed)] = packed
        image[pc:pc+len(instructions)] = instructions
        assert bytes(image[p:p+len(params)]) == params
        assert all(original[i] == image[i] for i in range(len(image))
                   if not a <= i < a+len(packed) and not pc <= i < pc+len(instructions))
        merge.write_word_coe(case / 'input.coe', list(packed), 16, 4, 'little', 1)
        merge.write_word_coe(case / 'instr.coe', list(instructions), 16, 4, 'little', 1)
        assembler.write_instr_txt(case / 'instr.txt', words)
        (case / 'instr.asm').write_text(asm, encoding='utf-8')
        g.write_csv(case / 'instructions.csv', ['index','byte_addr','machine_word','assembly'],
                    ((i, f'0x{pc+i*4:08X}', f'{word:08X}', line)
                     for i,(word,line) in enumerate(zip(words,[s for s in asm.splitlines() if assembler.assemble_line(s) is not None]))))
        merge.write_word_coe(case / 'all.coe', list(image), 16, 64, 'little', 1)
        g.write_mem(case / 'all.mem', bytes(image), 64)
        (case / 'all.bin').write_bytes(image)
        (case / 'expected_output.bin').write_bytes(expected)
        g.write_mem(case / 'expected_output.mem', expected, 1)
        merge.write_word_coe(case / 'expected_output.coe', list(expected)+[0]*((-len(expected))%64), 16,64,'little',1)
        g.write_csv(case / 'expected_hwc.csv', ['h','w']+[f'ch{c}' for c in range(padded_cout)],
                    ([i,j]+[int(v) for v in stored_output[i,j]] for i in range(h) for j in range(w)))
        ref = rows(case / 'reference.csv')
        for row in ref:
            i,j,c = (int(row[k]) for k in ('h','w','out_ch'))
            val = int(scaled[i,j,c]) if c < cout else 0
            y = val if val >= 0 else val >> 3
            assert y == int(stored_output[i,j,c])
            assert (int(row['accumulator_with_bias']) >> scale) == val
            row.pop('low8_signed',None)
            row.update(scale=scale, scaled_int8_signed=val, expected_signed=y,expected_hex=f'{y&255:02X}')
        g.write_csv(case / 'reference.csv', list(ref[0]), ([r[k] for k in ref[0]] for r in ref))
        g.write_json(case / 'memory_map.json', m)
        moves = json.loads((case / 'intr_move.json').read_text())
        moves['CONV_MOVE_BY_INDEX']['1'] = 1 << scale
        g.write_json(case / 'intr_move.json', moves)
        plan = json.loads((case / 'memory_plan.json').read_text())
        plan['tensors']['image']['layout'] = m['input_layout']
        plan['image']['layout'] = m['input_layout']
        g.write_json(case / 'memory_plan.json', plan)
        text = (case / 'memory_map.txt').read_text(encoding='utf-8')
        text = re.sub(r'RCONV2=0x[0-9A-Fa-f]+', f'RCONV2=0x{reg:08X}', text)
        text = text.split('\nINPUT_LAYOUT =')[0]
        text += f'\nINPUT_LAYOUT = group -> H -> W -> 8 channels\nCONV_SCALE = {scale}\nACCUMULATOR_RANGE = {m["accumulator_range"]}\nSCALED_RANGE = {m["scaled_range"]}\n'
        (case / 'memory_map.txt').write_text(text,encoding='utf-8')
        assert all((case/name).read_bytes() == data for name,data in before.items())
        decoded,_,_ = merge.coe_values_to_bytes(case/'all.coe',64,'little')
        assert bytes(decoded) == image
        assert b''.join(int(s,16).to_bytes(64,'little') for s in (case/'all.mem').read_text().splitlines()) == image
        from export_output_ddr import export_case
        export_case(case, m)
        print(root.name, m['case'], 'scale',scale,'acc',m['accumulator_range'],'scaled',m['scaled_range'],'parameters unchanged')
    g.write_json(root / 'summary.json', summaries)
    g.write_matrix_reference(root / 'reference.txt', x[:,:,:cin], weights[:,:,:,:cin], bias, output, summaries)
    readme = (root/'README.md').read_text(encoding='utf-8')
    readme = readme.split('\n## Input layout and scale update')[0]
    readme = re.sub(r'起始 bit=0',f'起始 bit={scale}',readme)
    readme = readme.replace('取低 8 位解释为 signed int8','按scale算术右移后取8位解释为 signed int8')
    readme = readme.replace('实际内存仍按 HWC 排列','输入内存按通道组、行、列、组内8通道排列')
    readme = readme.replace('interpret low 8 bits as int8','arithmetic-shift right by scale, then take 8 bits as int8')
    readme = readme.replace('q = ((acc + 128) % 256) - 128', f'q = (((acc >> {scale}) + 128) % 256) - 128')
    readme = re.sub(r'^- input：.*$', '- input：通道组 → h → w → 组内8通道，先排ch1～ch8的所有像素，再排ch9～ch16。', readme, flags=re.M)
    readme += f'\n## Input layout and scale update\n\nInput bytes: channel group -> row -> column -> eight channels. Channels 1..8 for all pixels precede channels 9..16. Little-endian storage is unchanged.\n\nConv scale={scale}; accumulator range={summaries[0]["accumulator_range"]}; shifted range={summaries[0]["scaled_range"]}. Apply this arithmetic shift before taking 8 bits and applying LeakyReLU. Parameter bytes, addresses, weights, bias, seed and all other operator settings are unchanged.\n'
    (root/'README.md').write_text(readme,encoding='utf-8')
    files=sorted(p for p in root.rglob('*') if p.is_file() and p.name != 'SHA256SUMS.txt')
    (root/'SHA256SUMS.txt').write_text(''.join(f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(root).as_posix()}\n' for p in files),encoding='ascii')
    archive=g.PROJECT_ROOT/'examples'/f'{root.name}.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as bundle:
        for p in sorted(root.rglob('*')):
            if p.is_file(): bundle.write(p,p.relative_to(root.parent))


if __name__ == '__main__':
    for path in sorted((g.PROJECT_ROOT/'examples').rglob('summary.json')):
        refresh(path.parent)
