import sys
print(sys.version)

import requests
import torch
# load a simple face detector
from retinaface import RetinaFace
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np

import cv2

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device: " + device)
print("retinaface version: " + RetinaFace.__version__)
model, transform = torch.hub.load('fkryan/gazelle', 'gazelle_dinov2_vitl14_inout')
# inference model
model.eval()
# load the model to GPU if available
model = model.to(device)

def visualize_heatmap(pil_image,heatmap,bbox=None,inout_score=None):
    # 检测heatmap是否是torch.Tensor类型(PyTorch张量)
    if isinstance(heatmap,torch.Tensor):
        # detach()：从计算图中分离出来，不再追踪梯度。这样做的目的是为了在不影响反向传播的情况下获取数据。
        # cpu()：如果张量是在 GPU 上的，先把它移到 CPU 上，因为 NumPy 不支持 GPU 张量。
        # numpy()：最后把这个张量转换为 NumPy 数组。以便进行后续处理和可视化。
        heatmap = heatmap.detach().cpu().numpy()

    # heatmap * 255：因为热力图的数据通常是 0 到 1 之间的小数，把它乘 255 变成 0 到 255 的灰度图。
        # .astype(np.uint8)：转换成 uint8 类型（8位无符号整数），这是图像常用的数据格式。
        # Image.fromarray(...)：将 NumPy 数组转换为 PIL 图像对象（PIL.Image.Image）。
        # .resize(pil_image.size, Image.Resampling.BILINEAR) ：将热力图调整为与原始图像相同的大小，使用双线性插值法进行缩放。
    heatmap=Image.fromarray((heatmap*255).astype(np.uint8)).resize(pil_image.size, Image.Resampling.BILINEAR)
    # 用 jet 颜色映射把灰度图变成彩色热图（matplotlib 的 colormap）。
    heatmap=plt.cm.jet(np.array(heatmap)/255.)
    # 去掉 alpha-透明度 通道，只保留 RGB
    heatmap=(heatmap[:,:,:3]*255).astype(np.uint8)

    heatmap=Image.fromarray(heatmap).convert('RGBA')
    heatmap.putalpha(90)
    # 原始图像和热力图叠加
    overlay_image=Image.alpha_composite(pil_image.convert('RGBA'),heatmap)

    if bbox is not None:
        width,height=pil_image.size
        xmin,ymin,xmax,ymax=bbox
        draw=ImageDraw.Draw(overlay_image)
        draw.rectangle([xmin*width,ymin*height,xmax*width,ymax*height],outline='lime',width=int(min(width,height)*0.01))

        if inout_score is not None:
            print('Inout score:', inout_score)
            # TODO:这里代码直接复制的
            text = f"in-frame: {inout_score:.2f}"
            text_width = draw.textlength(text)
            text_height = int(height * 0.01)
            text_x = xmin * width
            text_y = ymax * height + text_height
            draw.text((text_x, text_y), text, fill="lime", font=ImageFont.load_default(size=int(min(width, height) * 0.05)))

    return overlay_image

# 修改代码开始，添加视频流处理
# 打开视频文件（0 是摄像头，也可以写 'xxx.mp4'）
cap=cv2.VideoCapture('testvideo.mp4')
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    #OpenCV读取的是BGR，需要转成RGB
    frame_rgb= cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_iamge = Image.fromarray(frame_rgb)
    width, height = pil_iamge.size

    # 检测人脸
    resp=RetinaFace.detect_faces(np.array(pil_iamge))
    if not resp:
        continue
    # 人脸框坐标 bboxes-边界框
    bboxes=[resp[key]['facial_area'] for key in resp.keys()]
    print("bboxes")
    print(bboxes)
    # 归一化，[np.int64(1031), np.int64(188), np.int64(1704), np.int64(999)] 相当于边界框的坐标点/ 除以边界框的总长高，就能归一化
    norm_bboxes=[[np.array(bbox)/np.array([width, height, width, height]) for bbox in bboxes]]

    # 转换图形格式(适应模型输入要求)
        # unsqueeze(0) 是一个 PyTorch 张量操作，作用是 增加一个维度，把张量的形状从 [C, H, W]（通道数、图像高度、图像宽度）变为 [1, C, H, W]。
        # 这个操作通常用于将单张图片转换为 批处理（batch） 的格式。
        # 深度学习模型通常期望输入是一个批次（batch）而不是单张图片。
        # unsqueeze(0) 就是将图像的维度从 (C, H, W) 变为 (1, C, H, W)，表示 一个批次（batch size 为 1）。
    img_tensor=transform(pil_iamge).unsqueeze(0).to(device)
    # 设置输入，gazelle的输入要求是整个图像，和人脸边界框，key是 'images' 和 'bboxes' 千万别写错
    input={
        'images':img_tensor,
        'bboxes':norm_bboxes,
    }

    # torch.no_grad() 在不计算梯度的情况下运行模型（节省内存&加快推理）
    with torch.no_grad():
        # 进行推理，得到输出
        output=model(input)

    # 可视化处理
    for i in range(len(bboxes)):
        heatmap=output['heatmap'][0][i]
        inout_score=output['inout'][0][i] if output['inout'] is not None else None
        overlay_img=visualize_heatmap(pil_iamge,heatmap,norm_bboxes[0][i],inout_score)

        # 可视化图像转回OpenCV格式显示
        vis_frame=cv2.cvtColor(np.array(overlay_img),cv2.COLOR_RGB2BGR)
        cv2.imshow('Face Detection', vis_frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
