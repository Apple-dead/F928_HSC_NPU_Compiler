import torch.nn as nn
import torch
from torch.autograd import Function
from loss import build_target, yolo_loss
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import time
import pynvml



class Floor(nn.Module):
    def forward(self, x):
        return torch.floor(x)


class Yolo(nn.Module):
    def __init__(self, num_classes=20,
        anchors=[(1.3221, 1.73145), (3.19275, 4.00944), (5.05587, 8.09892), (9.47112, 4.84053),(11.2364, 10.0071)]):
        super(Yolo, self).__init__()

        self.num_classes = num_classes
        self.num_anchors = 5
        self.anchors = anchors
        self.timings = []
        # 3,256 → 3,256
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
        )
        # 3,256 →
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=12, kernel_size=2, stride=2, padding=0)
        )

        # 12,128 → 12,128
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=1, padding=1)
        )
        # 12,128 → 48,64
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=48, kernel_size=2, stride=2, padding=0)
        )
        # 48,64 → 48,64
        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=48, kernel_size=3, stride=1, padding=1)
        )
        # 48,64 → 48,64
        self.layer6 = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=48, kernel_size=3, stride=1, padding=1)
        )
        # 48,64 → 48,32
        self.layer7 = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=48, kernel_size=2, stride=2, padding=0)
        )
        # 48,32 → 64,32
        self.layer8 = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=64, kernel_size=3, stride=1, padding=1)
        )
        # 64,32 → 64,16
        self.layer9 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=2, stride=2, padding=0)
        )
        # 64,16 → 96,16
        self.layer10 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=96, kernel_size=3, stride=1, padding=1)
        )
        # 96,16 → 96,16
        self.layer11 = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1)
        )
        # 96,16 → 96,16
        self.layer12 = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1)
        )
        # 96,16 → 96,16
        self.layer13 = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1)
        )
        # 96,16 → 125,16
        self.layer14 = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=125, kernel_size=3, stride=1, padding=1)
        )
        self.layer = nn.Sequential(
            nn.LeakyReLU(negative_slope=0.125),
            Floor()  # 如果有 nn.Floor 模块的话
        )

    def forward(self, input, gt_boxes=None, gt_classes=None, num_boxes=None, training=False, eval_flag=False, bit_width=8):
        int_blob = {'input': 0.0, 'conv1': 5.0, 'conv2': 6.0, 'conv3': 5.0, 'conv4': 6.0, 'conv5': 5.0,
                    'conv6': 5.0, 'conv7': 5.0, 'conv8': 5.0, 'conv9': 5.0, 'conv10': 6.0, 'conv11': 5.0,
                    'conv12': 5.0, 'conv13': 5.0, 'conv14': 4.0, 'conv15': 5.0, 'conv16': 5.0, 'conv17': 5.0,
                    'conv18': 6.0, 'conv19': 4.0, 'conv20': 4.0}
        featuremap = {}
        timings = []
        # 在 try 块外定义变量
        handle = None
        start_clocks = 0
        start_mem_clocks = 0


        in_bit = 8
        move = 128
        # out = torch.round(input * (2 ** (in_bit - 1) - 1)) # 这个意思是压到127，硬件里是把rgb565都补末尾0补到rgb777，保证是7-bit绝对值
        out = torch.clamp(input,0,127).float()  # 只保留0到127的内容
        # out = input

        featuremap['layer1_in'] = out.detach()
        out = self.layer1(out)
        featuremap['layer1'] = out.detach()

        start = time.time()
        # # 修改 try-except 结构
        # try:
        #     pynvml.nvmlInit()
        #     handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # 这里定义 handle
        #     # 使用整数常量，而不是 pynvml.NVML_CLOCK_GRAPHICS
        #     start_clocks = pynvml.nvmlDeviceGetClockInfo(handle, 0)  # 图形时钟
        #     start_mem_clocks = pynvml.nvmlDeviceGetClockInfo(handle, 2)  # 内存时钟
        # except Exception as e:
        #     print(f"获取初始GPU频率失败: {e}")
        #     # 确保 handle 被关闭
        #     if handle is not None:
        #         try:
        #             pynvml.nvmlShutdown()
        #         except:
        #             pass
        out = self.layer(out)
        out = out / move  # 统一截掉2的7次方
        out = torch.floor(out)  # 感觉这里改成floor更好
        out = out.float()

        featuremap['layer2_in'] = out.detach()
        out = self.layer2(out)
        featuremap['layer2'] = out.detach()

        end = time.time()
        # # 获取结束时的频率
        # end_clocks = 0
        # end_mem_clocks = 0
        # try:
        #     if handle is not None:
        #         end_clocks = pynvml.nvmlDeviceGetClockInfo(handle, 0)
        #         end_mem_clocks = pynvml.nvmlDeviceGetClockInfo(handle, 2)
        # except Exception as e:
        #     print(f"获取结束GPU频率失败: {e}")
        # finally:
        #     # 关闭 NVML
        #     try:
        #         pynvml.nvmlShutdown()
        #     except:
        #         pass
        #
        # # 计算和打印
        # if start_clocks > 0 and end_clocks > 0:
        #     avg_clocks = (start_clocks + end_clocks) / 2
        #     avg_mem_clocks = (start_mem_clocks + end_mem_clocks) / 2
        #     print(f"[Layer2] GPU平均主频: {avg_clocks:.0f} MHz, 内存频率: {avg_mem_clocks:.0f} MHz")

        timings.append(("layer2", (end-start)*1000))

        out = self.layer(out)
        out = out / move  # 统一截掉2的7次方
        out = torch.floor(out)  # 感觉这里改成floor更好
        out = out.float()

        featuremap['layer3_in'] = out.detach()
        out = self.layer3(out)
        featuremap['layer3'] = out.detach()

        out = self.layer(out)
        out = out / move  # 统一截掉2的7次方
        out = torch.floor(out)  # 感觉这里改成floor更好
        out = out.float()

        featuremap['layer4_in'] = out.detach()
        out = self.layer4(out)
        featuremap['layer4'] = out.detach()

        out = self.layer(out)
        out = out / move  # 统一截掉2的7次方
        out = torch.floor(out)  # 感觉这里改成floor更好
        out = out.float()

        featuremap['layer5_in'] = out.detach()
        out = self.layer5(out)
        featuremap['layer5'] = out.detach()

        out = self.layer(out)
        out = out / move  # 统一截掉2的7次方
        out = torch.floor(out)  # 感觉这里改成floor更好
        out = out.float()

        featuremap['layer6_in'] = out.detach()
        out = self.layer6(out)
        featuremap['layer6'] = out.detach()

        out = self.layer(out)
        out = out / move  # 统一截掉2的7次方
        out = torch.floor(out)  # 感觉这里改成floor更好
        out = out.float()

        featuremap['layer7_in'] = out.detach()
        out = self.layer7(out)
        featuremap['layer7'] = out.detach()

        out = self.layer(out)
        out = out / move  # 统一截掉2的7次方
        out = torch.floor(out)  # 感觉这里改成floor更好
        out = out.float()

        featuremap['layer8_in'] = out.detach()
        out = self.layer8(out)
        featuremap['layer8'] = out.detach()

        out = self.layer(out)
        out = out / move  # 统一截掉2的7次方
        out = torch.floor(out)  # 感觉这里改成floor更好
        out = out.float()

        featuremap['layer9_in'] = out.detach()
        out = self.layer9(out)
        featuremap['layer9'] = out.detach()

        out = self.layer(out)
        out = out / move  # 统一截掉2的7次方
        out = torch.floor(out)  # 感觉这里改成floor更好
        out = out.float()

        featuremap['layer10_in'] = out.detach()
        out = self.layer10(out)
        featuremap['layer10'] = out.detach()

        out = self.layer(out)
        out = out / move  # 统一截掉2的7次方
        out = torch.floor(out)  # 感觉这里改成floor更好
        out = out.float()

        featuremap['layer11_in'] = out.detach()
        out = self.layer11(out)
        featuremap['layer11'] = out.detach()

        out = self.layer(out)
        out = out / move  # 统一截掉2的7次方
        out = torch.floor(out)  # 感觉这里改成floor更好
        out = out.float()

        featuremap['layer12_in'] = out.detach()
        out = self.layer12(out)
        featuremap['layer12'] = out.detach()

        out = self.layer(out)
        out = out / move  # 统一截掉2的7次方
        out = torch.floor(out)  # 感觉这里改成floor更好
        out = out.float()

        featuremap['layer13_in'] = out.detach()
        out = self.layer13(out)
        featuremap['layer13'] = out.detach()

        out = self.layer(out)
        out = out / move  # 统一截掉2的7次方
        out = torch.floor(out)  # 感觉这里改成floor更好
        out = out.float()

        featuremap['layer14_in'] = out.detach()
        out = self.layer14(out)
        featuremap['layer14'] = out.detach()

        self.timings = timings


        bsize, _, h, w = out.size()

        # 5 + num_class tensor represents (t_x, t_y, t_h, t_w, t_c) and (class1_score, class2_score, ...)
        # reorganize the output tensor to shape (B, H * W * num_anchors, 5 + num_classes)
        out = out.permute(0, 2, 3, 1).contiguous().view(bsize, h * w * self.num_anchors, 5 + self.num_classes)

        # activate the output tensor
        # `sigmoid` for t_x, t_y, t_c; `exp` for t_h, t_w;
        # `softmax` for (class1_score, class2_score, ...)

        # out = out/4264.13 # 这是为啥要除以这个数字，还是得回到硬件里，看硬件是怎么做的这块儿  5.19的回归python里是6240.030；量化.ipynb里是还有5434.6118，具体不清楚到底是多少放大倍数
        # 9_22_daban里的版本是除除以放大倍数5642.0967，不是，这数都哪儿来的，一直在换小批量校准集应该是
        print(out.max())
        out = out/5642.0967

        xy_pred = torch.sigmoid(out[:, :, 0:2])
        conf_pred = torch.sigmoid(out[:, :, 4:5])
        hw_pred = torch.exp(out[:, :, 2:4])
        class_score = out[:, :, 5:]
        class_pred = F.softmax(class_score, dim=-1)
        delta_pred = torch.cat([xy_pred, hw_pred], dim=-1)

        if training:
            output_variable = (delta_pred, conf_pred, class_score)
            output_data = [v.data for v in output_variable]
            gt_data = (gt_boxes, gt_classes, num_boxes)
            target_data = build_target(output_data, gt_data, h, w)

            target_variable = [Variable(v) for v in target_data]
            box_loss, iou_loss, class_loss = yolo_loss(output_variable, target_variable)

            return box_loss, iou_loss, class_loss


        return delta_pred, conf_pred, class_pred ,featuremap

if __name__ == '__main__':
    net = Yolo()
    print(net)
    # 计算参数
    total_params = 0
    weight_params = 0
    bias_params = 0

    print("=== 网络参数统计 ===")
    print("各层参数详情:")

    for name, param in net.named_parameters():
        param_count = param.numel()
        total_params += param_count

        if 'weight' in name:
            weight_params += param_count
        elif 'bias' in name:
            bias_params += param_count

        print(f"{name:30} | 形状: {str(param.shape):15} | 参数个数: {param_count:6}")

    print(f"\n总参数个数: {total_params}")
    print(f"权重参数个数: {weight_params}")
    print(f"偏置参数个数: {bias_params}")

    # 测试前向传播
    im = np.random.randn(1, 3, 256, 256)
    im_variable = torch.from_numpy(im).float()
    out = net(im_variable)
    delta_pred, conf_pred, class_pred, featuremap = out
    print('\ndelta_pred size:', delta_pred.size())
    print('conf_pred size:', conf_pred.size())
    print('class_pred size:', class_pred.size())