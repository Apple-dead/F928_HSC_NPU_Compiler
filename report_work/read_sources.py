from pathlib import Path
import json,sys,importlib.util
from docx import Document

root=Path(__file__).resolve().parent.parent
out=root/'report_work'/'source_extract'
out.mkdir(parents=True,exist_ok=True)
for path in root.glob('*.docx'):
    d=Document(path)
    text=[]
    for p in d.paragraphs:
        if p.text.strip():text.append(p.text)
    for i,t in enumerate(d.tables):
        text.append(f'\nTABLE {i+1}')
        for row in t.rows:text.append(' | '.join(c.text for c in row.cells))
    (out/(path.stem+'.txt')).write_text('\n'.join(text),encoding='utf-8')
    info={'sections':[{k:float(getattr(s,k).pt) if getattr(s,k) is not None else None for k in ['page_width','page_height','top_margin','bottom_margin','left_margin','right_margin']} for s in d.sections], 'paragraphs':[]}
    for p in d.paragraphs:
        if p.text.strip():
            info['paragraphs'].append({'text':p.text,'style':p.style.name,'pPr':p._p.pPr.xml if p._p.pPr is not None else '', 'runs':[{'text':r.text,'rPr':r._r.rPr.xml if r._r.rPr is not None else ''} for r in p.runs]})
    (out/(path.stem+'.format.json')).write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')
    print(path.name, len(text), 'paragraphs', len(d.tables),'tables')
for m in ['PIL','fitz','pdf2image','olefile','win32com','reportlab']:
    print(m,importlib.util.find_spec(m) is not None)
