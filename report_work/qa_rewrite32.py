from pathlib import Path
from pdf2image import convert_from_path
from pypdf import PdfReader
from docx import Document
import re,hashlib
root=Path(__file__).resolve().parent
out=root/'rewrite32_render'
pop=r'C:/Users/60195/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/poppler/Library/bin'
ims=convert_from_path(str(out/'midterm_revision.pdf'),dpi=110,poppler_path=pop,fmt='png',output_folder=str(out),output_file='page',paths_only=True,thread_count=2)
for i,f in enumerate(ims,1):Path(f).replace(out/f'p{i:02d}.png')
pdf=PdfReader(str(out/'midterm_revision.pdf'))
for i,p in enumerate(pdf.pages,1):
 s=p.extract_text();print(i,s.replace('\n',' ')[:110])
doc=Document(root.parent/'论文中期材料/蒋一平-硕士论文中期检查报告（3.2重写稿）.docx')
text='\n'.join(p.text for p in doc.paragraphs)
assert '目录将在排版阶段更新' not in text
assert not re.search(r'[\u4e00-\u9fff][ \t]+[A-Za-z]|[A-Za-z][ \t]+[\u4e00-\u9fff]',text)
print('Chinese-English whitespace check passed; pages:',len(pdf.pages))

