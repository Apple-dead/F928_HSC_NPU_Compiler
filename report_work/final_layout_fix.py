from pathlib import Path
from docx import Document
p=Path('论文中期材料/蒋一平-AI编译器文献综述.docx')
d=Document(p)
for para in d.paragraphs:
 if para.text=='4.结论':para.paragraph_format.page_break_before=True
d.save(p)
