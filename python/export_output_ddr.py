"""Address-aware views of existing reference output; does not change output bytes."""
from pathlib import Path
import hashlib
import json
import zipfile


def export_case(case: Path, summary: dict) -> None:
    data = (case / 'expected_output.bin').read_bytes()
    base = int(summary['output_addr'], 16)
    end = base + len(data)
    assert len(data) == summary['output_size_bytes']
    height, width, valid_channels = summary['output_shape_hwc']
    channels = len(data) // (height * width)
    assert channels == 8
    lines = [
        'DDR output reference - absolute BYTE addresses',
        f'Output range: [0x{base:08X}, 0x{end:08X})  ({len(data)} bytes)',
        f'Shape: H={height}, W={width}, valid C={valid_channels}, stored C={channels}',
        'Layout: row -> column -> channels 1..8. H/W/channel numbers below start at 1.',
        'Each channel is signed int8 stored as one two-complement byte.',
        'Byte dump reads LEFT TO RIGHT in increasing address order.',
        '-- means outside the output region: unknown / do not compare, NOT zero.',
        '',
        'ADDRESS     +00 +01 +02 +03 +04 +05 +06 +07 +08 +09 +0A +0B +0C +0D +0E +0F',
    ]
    for addr in range(base // 16 * 16, (end + 15) // 16 * 16, 16):
        values = [f'{data[a-base]:02X}' if base <= a < end else '--' for a in range(addr, addr+16)]
        lines.append(f'0x{addr:08X}  ' + '  '.join(values))
    lines += ['', 'Pixel view: first byte is channel 1; LE64 hex has channel 1 at the RIGHT.',
              'H   W   ADDRESS     BYTES ch1..ch8 (increasing addresses)   LE64_HEX          SIGNED ch1..ch8']
    for i in range(height):
        for j in range(width):
            offset = (i * width + j) * channels
            chunk = data[offset:offset+channels]
            signed = [v if v < 128 else v - 256 for v in chunk]
            lines.append(f'{i+1:02d}  {j+1:02d}  0x{base+offset:08X}  '
                         + ' '.join(f'{v:02X}' for v in chunk)
                         + f'          {int.from_bytes(chunk, "little"):016X}  '
                         + ' '.join(f'{v:4d}' for v in signed))
    (case / 'expected_output_ddr.txt').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    wide = ['DDR output reference - 512-bit little-endian word view',
            'Line label = absolute aligned BYTE address, not word index.',
            'Word text: [511:504] at LEFT, [7:0] at RIGHT (lowest byte address).',
            '-- marks bytes outside the output region; ignore those lanes when comparing.',
            f'Compare only [0x{base:08X}, 0x{end:08X}).', '']
    for addr in range(base // 64 * 64, (end + 63) // 64 * 64, 64):
        lanes = [f'{data[a-base]:02X}' if base <= a < end else '--' for a in range(addr,addr+64)]
        text = ''.join(reversed(lanes))
        # Decode every valid lane in the formatted DDR word back to the binary.
        for lane in range(64):
            token = text[2*(63-lane):2*(64-lane)]
            if base <= addr+lane < end:
                assert int(token,16) == data[addr+lane-base]
            else:
                assert token == '--'
        wide.append(f'0x{addr:08X}  {text}')
    (case / 'expected_output_ddr512.txt').write_text('\n'.join(wide)+'\n', encoding='utf-8')


def refresh_package(root: Path) -> None:
    files = sorted(p for p in root.rglob('*') if p.is_file() and p.name != 'SHA256SUMS.txt')
    (root/'SHA256SUMS.txt').write_text(''.join(
        f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(root).as_posix()}\n'
        for p in files),encoding='ascii')
    examples = Path(__file__).resolve().parents[1]/'examples'
    with zipfile.ZipFile(examples/f'{root.name}.zip','w',zipfile.ZIP_DEFLATED) as bundle:
        for p in sorted(root.rglob('*')):
            if p.is_file(): bundle.write(p,p.relative_to(root.parent))


if __name__ == '__main__':
    examples = Path(__file__).resolve().parents[1]/'examples'
    for summary_path in sorted(examples.rglob('summary.json')):
        root = summary_path.parent
        for summary in json.loads(summary_path.read_text(encoding='utf-8')):
            export_case(root/summary['case'],summary)
            print(root.name,summary['case'],summary['output_addr'],'OK')
        refresh_package(root)
