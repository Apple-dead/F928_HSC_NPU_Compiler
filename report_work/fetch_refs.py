import json,urllib.request,urllib.parse,concurrent.futures,time
from pathlib import Path
titles=[
'The Deep Learning Compiler: A Comprehensive Survey',
'MLIR: Scaling Compiler Infrastructure for Domain Specific Computation',
'Halide: a language and compiler for optimizing parallelism locality and recomputation in image processing pipelines',
'TensorIR: An Abstraction for Automatic Tensorized Program Optimization',
'TASO: optimizing deep learning computation with automatic generation of graph substitutions',
'DNNFusion: accelerating deep neural networks execution with advanced operator fusion',
'DORY: Automatic End-to-End Deployment of Real-World DNNs on Low-Cost IoT MCUs',
'ZigZag: A Memory-Centric Rapid DNN Accelerator Design Space Exploration Framework',
'DeFiNES: Enabling Fast Exploration of the Depth-first Scheduling Space for DNN Accelerators through Analytical Modeling',
'Pex: Memory-efficient Microcontroller Deep Learning through Partial Execution',
'MATCH: Model-Aware TVM-Based Compilation for Heterogeneous Edge Devices',
'Deeploy: Enabling Energy-Efficient Deployment of Small Language Models on Heterogeneous Microcontrollers',
'MAGIS: Memory Optimization via Coordinated Graph Transformation and Scheduling for DNN',
'Understanding Reuse Performance and Hardware Cost of DNN Dataflow: A Data-Centric Approach',
'Interstellar: Using Halide for Spatially Mapping DNN Accelerators',
'Fused-layer CNN accelerators',
'Gemmini: Enabling Systematic Deep-Learning Architecture Evaluation via Full-Stack Integration',
'Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference',
'A Multi-level Compiler Backend for Accelerated Micro-kernels Targeting RISC-V ISA Extensions',
]
out=Path(__file__).parent/'research';out.mkdir(exist_ok=True)
def fetch(t):
 u='https://api.crossref.org/works?'+urllib.parse.urlencode({'query.title':t,'rows':2})
 try:
  req=urllib.request.Request(u,headers={'User-Agent':'AcademicBibliographyVerifier/1.0'})
  data=json.load(urllib.request.urlopen(req,timeout=45))['message']['items']
  return {'query':t,'matches':[{k:x.get(k) for k in ['title','author','DOI','container-title','volume','issue','page','published','publisher','URL','abstract']} for x in data]}
 except Exception as e:return {'query':t,'error':str(e)}
with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
 results=list(pool.map(fetch,titles))
(out/'crossref.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
for r in results:
 print(r['query'])
 for x in r.get('matches',[])[:1]:print(x['title'],x['DOI'],x['container-title'],x['volume'],x['issue'],x['page'],x['published'])
