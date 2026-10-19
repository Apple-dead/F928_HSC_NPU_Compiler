from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
import math

OUT=Path(__file__).resolve().parent/'new_figures';OUT.mkdir(exist_ok=True)
W=1800;INK='#202A34';BLUE='#E9F0F7';GRAY='#F5F5F5'
def canvas(h):
 global im,dr
 im=Image.new('RGB',(W,h),'white');dr=ImageDraw.Draw(im)
def text(x,y,s,n=42,anchor='mm',color=INK):
 font=ImageFont.truetype('C:/Windows/Fonts/simsun.ttc',n)
 dr.multiline_text((x,y),s,font=font,fill=color,anchor=anchor,align='center',spacing=10)
def box(x,y,w,h,s,n=42,fill=GRAY):
 dr.rectangle((x,y,x+w,y+h),fill=fill,outline=INK,width=3)
 text(x+w/2,y+h/2,s,n)
def arrow(points,color=INK,width=4):
 dr.line(points,fill=color,width=width)
 a,b=points[-2:];ang=math.atan2(b[1]-a[1],b[0]-a[0]);l=18
 dr.polygon([b,(b[0]-l*math.cos(ang-.5),b[1]-l*math.sin(ang-.5)),(b[0]-l*math.cos(ang+.5),b[1]-l*math.sin(ang+.5))],fill=color)
def save(name):
 im.save(OUT/(name+'.png'),dpi=(300,300))

# Linux composition: all four independently built inputs are coordinated by Makefile.
canvas(950)
cols=[50,495,940,1385]
for x,top,bot in zip(cols,['引导程序源码\n与平台配置','Linux内核源码\n与裁剪配置','设备树源码\n硬件资源描述','BusyBox、运行库\n与用户应用'],['引导程序','内核镜像','设备树文件','根文件系统']):
 box(x,40,365,150,top,39,BLUE);arrow([(x+182,190),(x+182,255)])
 box(x,255,365,105,bot,41)
 arrow([(x+182,360),(x+182,425),(900,425),(900,490)])
box(460,490,880,110,'统一Makefile：依赖管理、编译与打包',42,BLUE)
arrow([(900,600),(900,660)])
box(460,660,880,100,'系统部署镜像及配套加载文件',42)
arrow([(900,760),(900,835)])
text(1210,800,'JTAG下载',38)
box(270,835,1260,90,'FPGA上的RISC-VSoC：Linux启动与应用运行',42,BLUE)
save('linux_build')

# Boot flow and resident SBI service are distinct.
canvas(1080)
left=100;ww=770
labels=['处理器复位与基础运行环境设置','异常入口及上下文处理准备','UART、中断控制器与定时器初始化','内核映像、设备树与启动参数准备','异常委托及运行状态设置','进入Linux内核，启动用户空间']
for i,s in enumerate(labels):
 y=45+i*166;box(left,y,ww,115,s,38,BLUE if i in [0,5] else GRAY)
 if i<5:arrow([(left+ww/2,y+115),(left+ww/2,y+166)])
box(1100,325,590,160,'机器态固件\n保留必要SBI服务',44,BLUE)
box(1100,600,590,150,'底层服务处理\n控制台、定时器等',42)
arrow([(1395,485),(1395,600)])
arrow([(870,932),(1010,932),(1010,405),(1100,405)])
text(1300,900,'Linux发起SBI调用',38)
arrow([(1690,675),(1740,675),(1740,1010),(485,1010),(485,990)])
text(1350,1045,'处理完成后返回Linux',37)
text(1380,145,'启动交接后，固件继续提供\n所需的特权级服务',40)
save('linux_boot')

# Driver collaboration: control and data directions differentiated by labels.
canvas(1040)
box(370,35,1360,110,'用户态：图形交互与加速器测试应用',45,BLUE)
box(40,265,250,220,'设备树\n寄存器范围\n中断资源',39)
for x,s in zip([370,850,1330],['显示驱动','触摸驱动','NPU驱动']):
 box(x,285,400,125,s,44,BLUE)
 arrow([(x+200,145),(x+200,285)])
dr.rectangle((735,192,1345,248),fill='white')
text(1040,220,'设备接口调用、状态与事件返回',38)
arrow([(290,375),(370,375)])
box(370,515,1360,110,'内核支撑：资源映射、中断处理与缓冲区管理',42)
for x in [570,1050,1530]:arrow([(x,410),(x,515)])
for x,s in zip([370,850,1330],['显示控制器\nDMA→LCD','触摸模块\n输入事件','NPU\n计算与状态']):
 box(x,730,400,150,s,42,BLUE);arrow([(x+200,625),(x+200,730)])
box(370,950,1360,75,'存储空间：图像缓冲、输入数据、模型参数与计算结果',38)
arrow([(570,880),(570,950)]);arrow([(1530,880),(1530,950)])
text(155,690,'硬件交互',37)
text(155,975,'数据存储',37)
save('linux_drivers')

# Memory reuse: x's last use ends after B; z can share x's storage during C.
canvas(1020)
text(900,45,'示例计算链：各节点只消费前一节点的输出',43)
for x,s in [(140,'算子A\n产生x'),(740,'算子B\n消费x，产生y'),(1340,'算子C\n消费y，产生z')]:box(x,110,320,135,s,42,BLUE)
arrow([(460,178),(740,178)]);arrow([(1060,178),(1340,178)])
text(600,135,'x',40);text(1200,135,'y',40)
text(900,310,'执行B后，x已完成全部消费；执行C时，z可复用x的空间',40)
box(70,390,780,400,'',42,'white');box(950,390,780,400,'',42,'white')
text(460,435,'顺序分配',45);text(1340,435,'生命周期复用',45)
for i,s in enumerate(['x的空间','y的空间','z的空间']):box(120+i*235,530,210,135,s,37,GRAY)
box(1010,530,310,135,'x→z共用空间',38,BLUE);box(1360,530,310,135,'y的空间',38,GRAY)
text(460,730,'已失效张量的空间仍被保留',37)
text(1340,730,'仅在生命周期不重叠时复用',37)
arrow([(850,590),(950,590)])
box(120,855,1560,125,'合法性条件：物理大小与对齐匹配；无读写冲突\n拼接张量的共享区域，须等待相关消费者全部完成',40,BLUE)
save('memory_reuse')

# Fusion is a proposed datapath extension, not claimed implemented.
canvas(920)
text(900,45,'现有方式：卷积与激活分别执行',43)
for x,w,s in [(50,220,'输入'),(350,300,'卷积'),(730,340,'中间结果\n写回并再读取'),(1150,300,'激活'),(1530,220,'输出')]:box(x,125,w,150,s,39,GRAY)
for a,b in [(270,350),(650,730),(1070,1150),(1450,1530)]:arrow([(a,200),(b,200)])
text(900,345,'拟实现方式：在卷积输出数据通路中完成激活',43)
box(50,440,220,150,'输入',40)
dr.rectangle((350,415,1450,625),outline=INK,width=3)
box(390,450,430,130,'卷积计算',43,BLUE)
box(990,450,420,130,'激活处理',43,BLUE)
arrow([(270,515),(390,515)]);arrow([(820,515),(990,515)])
text(905,480,'内部传递',33)
box(1530,440,220,150,'输出',40)
arrow([(1410,515),(1530,515)])
text(900,675,'融合配置／指令控制：卷积结果直接进入激活单元',40)
box(100,760,1600,125,'适用条件：硬件提供融合路径；量化、移位和饱和顺序一致\n激活前数据若仍被其他节点使用，保留原有执行方式',40,BLUE)
save('operator_fusion')
print('Generated:',*[p.name for p in OUT.glob('*.png')])
