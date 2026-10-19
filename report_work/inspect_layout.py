from pathlib import Path
from docx import Document
from zipfile import ZipFile
import json
R=Path(__file__).resolve().parent.parent
for name in ['蒋一平-开题报告.docx','蒋一平-文献综述.docx']:
 d=Document(R/name)
 print(name)
 for s in d.sections:
  print('SECTION', {k:round(getattr(s,k).cm,2) for k in ['page_width','page_height','top_margin','bottom_margin','left_margin','right_margin']})
 for p in d.paragraphs:
  if not p.text.strip():continue
  if len(p.text)<60 or '近年来' in p.text[:20]:
   print('PARA',p.text[:60],p.style.name,'size',[(r.font.size.pt if r.font.size else None,r.font.name, r._r.rPr.rFonts.xml if r._r.rPr is not None and r._r.rPr.rFonts is not None else '') for r in p.runs[:1]],'line',p.paragraph_format.line_spacing,'first',p.paragraph_format.first_line_indent)
 with ZipFile(R/name) as z:
  for f in z.namelist():
   if f.startswith('word/media/'):
    p=R/'report_work'/'original_media'/Path(name).stem/Path(f).name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(z.read(f))
