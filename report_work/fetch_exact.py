import json,urllib.request,concurrent.futures,time
from pathlib import Path
dois=['10.1145/2491956.2462176','10.1145/3575693.3576933','10.1145/3341301.3359630','10.1109/TC.2021.3066883','10.1109/TCAD.2024.3443718']
def get(d):
 for i in range(3):
  try:
   req=urllib.request.Request('https://api.crossref.org/works/'+d,headers={'User-Agent':'AcademicBibliography/1.0'})
   x=json.load(urllib.request.urlopen(req,timeout=35))['message']
   return {k:x.get(k) for k in ['title','author','DOI','container-title','volume','issue','page','published','publisher','URL','abstract']}
  except Exception as e: time.sleep(1)
 return {'DOI':d,'error':str(e)}
res=list(concurrent.futures.ThreadPoolExecutor(3).map(get,dois))
Path('report_work/research/exact.json').write_text(json.dumps(res,ensure_ascii=False,indent=2),encoding='utf8')
old=json.loads(Path('report_work/research/crossref.json').read_text(encoding='utf8'))
allr=[r['matches'][0] for r in old if r.get('matches')]+res
for x in allr:
 print(x.get('title'),x.get('DOI'),x.get('volume'),x.get('issue'),x.get('page'),x.get('published'),'; '.join(a.get('family','')+' '+a.get('given','') for a in x.get('author',[])[:3]))
