from pathlib import Path
from pdf2image import convert_from_path
import importlib.util,hashlib
root=Path(__file__).resolve().parent
pop=r'C:/Users/60195/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/poppler/Library/bin'
for p in (root/'render').glob('*.pdf'):
 k='literature' if '综述' in p.name else 'midterm'
 out=root/'render'/k;out.mkdir(exist_ok=True)
 old={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in out.glob('p*.png')}
 ims=convert_from_path(str(p),dpi=110,poppler_path=pop,fmt='png',output_folder=str(out),output_file='page',paths_only=True,thread_count=2)
 changed=[]
 for i,f in enumerate(ims,1):
  final=out/f'p{i:02d}.png';Path(f).replace(final)
  if old.get(final.name)!=hashlib.sha256(final.read_bytes()).hexdigest():changed.append(i)
 print(k,len(ims),'changed',changed)
print('pypdf:',bool(importlib.util.find_spec('pypdf')))
