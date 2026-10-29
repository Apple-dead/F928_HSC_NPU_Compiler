from pathlib import Path
from copy import deepcopy
from docx import Document
from docx.shared import Pt, Cm
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT

source = next(p for p in Path('C:/Users/60195/Desktop').rglob('*.docx') if '（编译器章节结构修订稿）' in p.name and not p.name.startswith('~'))
doc = Document(source)
original = list(doc.paragraphs)
anchor = original[80]
body_template = original[81]

def replace(p, text):
    rpr = deepcopy(p.runs[0]._r.rPr) if p.runs and p.runs[0]._r.rPr is not None else None
    p.clear()
    r = p.add_run(text)
    if rpr is not None:
        r._r.insert(0, rpr)

for p in original:
    t = p.text
    if t.startswith(('3.1.2 ', '3.1.3 ', '3.1.4 ')):
        replace(p, t[:4] + str(int(t[4]) + 1) + t[5:])
    for old, new in [('表4','表6'), ('表3','表5'), ('表2','表4')]:
        if old in p.text:
            replace(p, p.text.replace(old,new))
            break

def para(text, template=body_template, bold=False):
    p = anchor.insert_paragraph_before()
    p.style = template.style
    if template._p.pPr is not None:
        p._p.insert(0, deepcopy(template._p.pPr))
    r = p.add_run(text)
    if template.runs and template.runs[0]._r.rPr is not None:
        r._r.insert(0,deepcopy(template.runs[0]._r.rPr))
    if bold:
        r.bold = True
        p.paragraph_format.keep_with_next = True
    return p

def table(caption, headers, rows, widths):
    para(caption, original[70])
    t = doc.add_table(rows=1, cols=len(headers))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    for c,w in zip(t.columns,widths): c.width = Cm(w)
    for vals in [headers]+rows:
        cells = t.rows[0].cells if vals is headers else t.add_row().cells
        for c,v,w in zip(cells,vals,widths):
            c.width = Cm(w)
            c.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            c.text = v
            p=c.paragraphs[0]
            p.paragraph_format.first_line_indent = Pt(0)
            p.paragraph_format.space_before = Pt(4)
            p.paragraph_format.space_after = Pt(4)
            p.paragraph_format.line_spacing = 1.05
            for r in p.runs:
                r.font.size = Pt(10)
                r.font.name = 'Times New Roman'
                r._r.get_or_add_rPr().get_or_add_rFonts().set(qn('w:eastAsia'),'宋体')
                r.bold = vals is headers
            if vals is headers:
                shd=OxmlElement('w:shd'); shd.set(qn('w:fill'),'E7E6E6'); c._tc.get_or_add_tcPr().append(shd)
        trpr=cells[0]._tc.getparent().get_or_add_trPr()
        no=OxmlElement('w:cantSplit'); trpr.append(no)
    repeat=OxmlElement('w:tblHeader'); t.rows[0]._tr.get_or_add_trPr().append(repeat)
    borders=OxmlElement('w:tblBorders')
    for name in ('top','left','bottom','right','insideH','insideV'):
        e=OxmlElement('w:'+name);e.set(qn('w:val'),'single');e.set(qn('w:sz'),'4');e.set(qn('w:color'),'D9D9D9');borders.append(e)
    t._tbl.tblPr.append(borders)
    anchor._p.addprevious(t._tbl)

para('3.1.2 NPU指令集设计', original[80])
para('目标NPU将卷积、池化等张量计算封装为专用宏指令，编译器通过配置计算参数和数据地址，将模型中的算子转换为硬件执行过程。指令集采用配置与执行相结合的组织方式：配置指令写入地址寄存器和算子参数寄存器，计算指令选择相应操作并引用地址寄存器，结束指令标识当前指令序列完成。该接口使模型分析与硬件控制分别在前端和后端完成，后端依据执行与存储计划生成具体配置。')
para('（1）指令分类与寄存器组织', bold=True)
para('当前汇编器采用32位定长编码，指令按功能划分为配置、计算和结束三类，如表2所示。地址寄存器R1至R12保存设备可访问的数据地址；算子专用寄存器保存卷积核尺寸、步长、填充、通道数和数值处理等属性。计算指令引用已经配置的寄存器，因此一条计算指令可以描述一次完整的算子执行。')
table('表2 NPU指令分类与作用', ['类别','指令或配置对象','作用'], [
 ['配置指令','CFG_REGISTER','设置地址寄存器的高低16位，或写入算子参数寄存器。'],
 ['计算指令','CONV、FULL、MADD','分别描述卷积、全连接和逐元素加法，使用三个地址寄存器操作数。'],
 ['计算指令','DSMP、AVGPOOL、MAXPOOL、USMP','分别描述降采样、平均池化、最大池化和上采样，使用两个地址寄存器操作数。'],
 ['结束指令','END','标识当前指令序列结束，机器码为0xFC000000。']], [2.1,5.3,7.6])
para('（2）基本编码格式', bold=True)
para('32位指令的高6位为操作码。配置指令的操作码为000001，低16位保存立即数。地址配置使用寄存器编号和高低半字选择字段，两条配置指令共同构成一个32位地址；算子参数配置则使用配置目标编码，选择对应参数寄存器的半字。计算指令通过三个5位字段引用地址寄存器，数据类型字段占3位，当前编译流程使用INT8编码。典型字段划分见表3，未使用的位在当前编码中置零。')
table('表3 典型32位指令字段划分', ['格式','位段与含义'], [
 ['地址配置','[31:26] 操作码；[25:21] 地址寄存器；[20:16] 半字选择（0为低16位，1为高16位）；[15:0] 立即数。'],
 ['算子参数配置','[31:26] 操作码；[25:21] 置零；[20:16] 配置目标编码；[15:0] 立即数。'],
 ['计算指令','[31:26] 操作码；[25:21]、[20:16]、[15:11] 地址寄存器编号；[10:8] 数据类型；[7:0] 置零。双操作数指令的第三个寄存器字段置零。']], [3.1,11.9])
para('以CONV为例，三个寄存器操作数依次提供输入、参数和输出地址，卷积参数由RCONV1、RCONV2等专用配置保存，包括卷积核尺寸、填充、步长、输入输出通道数、偏置使能、截位起始位置及激活选择。激活参数通过相应专用寄存器配置。编码字段提供的取值范围与当前硬件、编译器实际支持的配置应共同检查，预留编码不能直接视为已经实现的功能。')
para('（3）指令寻址与数据布局约定', bold=True)
para('指令中的地址采用字节地址，512位BRAM的一项对应64字节。因此，字节地址A对应第⌊A/64⌋个存储字，其字内偏移为A mod 64；指令入口地址也采用同一约定。32位指令按小端字节序存放，硬件读取时需要保留字内偏移，不能直接将字节地址作为512位存储字的索引。')
para('在当前参数接口中，特征图按空间位置和通道组织，通道数补齐到8的倍数，补齐位置填零。卷积参数按有效输出通道逐一存放：先排列该输出通道的全部权重，权重内部依次遍历卷积核行、卷积核列和补齐后的输入通道；随后写入该输出通道的32位小端偏置，再补32位零，形成8字节偏置槽。只保存有效输出通道的参数记录；无偏置时省略偏置槽。偏置槽大小与整个参数区的起始地址对齐是不同约束，均需由存储计划明确。')
para('以15输入通道、5输出通道、3×3卷积为例，输入通道补齐为16，每个输出通道包含3×3×16＝144字节权重，以及4字节偏置和4字节零填充，参数记录步长为152字节，5个输出通道共占760字节。若参数区起始地址为P，则第c个输出通道的权重地址为P＋152c，偏置地址为P＋152c＋144，其中c从0开始。上述地址关系由参数导出与指令生成共同遵守。')
para('（4）典型执行序列', bold=True)
para('对于卷积后接LeakyReLU的典型计算，指令序列先用6条地址配置指令设置输入、参数和输出地址，再用4条配置指令写入两个32位卷积参数寄存器，用2条配置指令写入激活参数，最后执行CONV并以END结束，共14条32位指令。该序列对应融合激活接口：激活参数在卷积执行前完成配置，由卷积输出路径执行激活处理，不额外发出独立激活计算指令。')
para('执行流程可概括为：配置数据地址→配置卷积属性→配置激活参数→执行CONV→执行END。对于负半轴斜率为1/8、截位起始位置为0的示例，先进行卷积累加并加偏置，再将结果低8位解释为有符号int8，最后对负值算术右移3位。编译器应按这一顺序建立参考结果；融合接口的完整硬件验证及性能评价仍需结合后续实验完成。')
para('本节明确指令与存储接口，后续模型解析、算子映射和指令生成均依据该接口展开。其中，图IR保存模型语义，执行与存储计划确定地址、参数步长和算子配置，后端再将计划转换为上述指令序列。')

replace(original[99], '在确定数据布局和存储地址后，还需要将模型参数转换为NPU能够读取的格式。卷积参数采用3.1.2节所述的逐输出通道排列方式，输入通道补齐到8的倍数，权重内部按卷积核行、卷积核列和输入通道顺序存储。存在偏置时，每个有效输出通道的权重后紧接32位小端偏置和32位零填充；不存在偏置时，仅保存权重数据。规划模块记录参数区大小及通道记录步长，参数导出与指令生成共同使用这些信息。')

# Insert the new entry into the cached TOC; Word refreshes page numbers later.
toc=original[27].insert_paragraph_before('3.1.2 NPU指令集设计\t9',style=original[27].style)
if original[27]._p.pPr is not None: toc._p.insert(0,deepcopy(original[27]._p.pPr))
out=source.with_name(source.stem.replace('编译器章节结构修订稿','NPU指令设计补充稿')+'.docx')
doc.save(out)
Path('report_output_path.txt').write_text(str(out),encoding='utf-8')
print(out)
