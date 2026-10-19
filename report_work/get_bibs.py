import urllib.request,re,json
from pathlib import Path
urls=[('mcunet','https://proceedings.neurips.cc/paper/2020/hash/86c51678350f656dcc7f490a43946ee5-Abstract.html'),('mcunet2','https://proceedings.neurips.cc/paper/2021/hash/1371bccec2447b5aa6d96d2a540fb401-Abstract.html'),('stream','https://proceedings.neurips.cc/paper_files/paper/2023/hash/7526508f11bbe0a123af62b9dab1fbe1-Abstract.html')]
from urllib.parse import urljoin
for key,url in urls:
 try:
  s=urllib.request.urlopen(url,timeout=25).read().decode()
  link=re.search(r'href="([^"]+/bibtex)"',s).group(1)
  b=urllib.request.urlopen(urljoin(url,link),timeout=25).read().decode()
  Path('report_work/research/'+key+'.bib').write_text(b,encoding='utf8')
  print(key,b)
 except Exception as e: print(key,str(e))
