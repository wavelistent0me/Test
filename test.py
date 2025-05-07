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

# load the image
image_url='https://i.kym-cdn.com/entries/icons/original/000/045/575/blackcatzoningout_meme.jpg'

try:
    response = requests.get(image_url, stream=True)
    response.raise_for_status()
    image= Image.open(BytesIO(response.content))
    width, height = image.size
    # plt.imshow(image)
    # plt.axis('off')
    # plt.show()
except requests.exceptions.RequestException as e:
    print(f"An error occurred: {e}")
    
# 抓取脸部，然后打出坐标

print("这一步开始计算，请稍等...")

# 图片中的人脸数据
resp = RetinaFace.detect_faces(np.array(image))
print(resp)
# {
#   "face_1": {
#     "score": 0.9742743968963623,
#     "facial_area": [1031, 188, 1704, 999],
#     "landmarks": {
#       "right_eye": [1273.9718, 560.84827],
#       "left_eye": [1557.5142, 502.53882],
#       "nose": [1497.9034, 673.1349],
#       "mouth_right": [1380.0884, 841.3296],
#       "mouth_left": [1586.5767, 793.3872]
#     }
#   }
# }

# 人脸框坐标 bboxes-边界框
bboxes = [resp[key]['facial_area'] for key in resp.keys()]
print(bboxes)
#[[np.int64(1031), np.int64(188), np.int64(1704), np.int64(999)]]

# 转换图形格式(适应模型输入要求)
    # unsqueeze(0) 是一个 PyTorch 张量操作，作用是 增加一个维度，把张量的形状从 [C, H, W]（通道数、图像高度、图像宽度）变为 [1, C, H, W]。
    # 这个操作通常用于将单张图片转换为 批处理（batch） 的格式。
    # 深度学习模型通常期望输入是一个批次（batch）而不是单张图片。
    # unsqueeze(0) 就是将图像的维度从 (C, H, W) 变为 (1, C, H, W)，表示 一个批次（batch size 为 1）。
img_tensor=transform(image).unsqueeze(0).to(device)
# 将人脸边界框坐标归一化
norm_bboxes = [[np.array(bbox)/np.array([width, height, width, height]) for bbox in bboxes]]

# print(img_tensor)
# print(norm_bboxes)

input = {
    'images': img_tensor,
    'bboxes': norm_bboxes,
}

print('input')
print(input)

# exit()

# model(input)运行模型，输入人脸框bboxes和图片,就能计算出注释heatmap
# torch.no_grad() 在不计算梯度的情况下运行模型（节省内存&加快推理）
with torch.no_grad():
    output = model(input)

    # 这output数据看不懂
    print('output')
    print(output)

img1_person1_heatmap = output['heatmap'][0][0] # torch.Size([64, 64]) heatmap
print('img1_person1_heatmap')
print(img1_person1_heatmap.shape)

if model.inout:
    img1_person1_inout= output['inout'][0][0]
    print('img1_person1_inout')
    print(img1_person1_inout.item())

# 上面已经计算出了注释热图和人脸框，下面是将组数形式转化成图片
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

# for i in range(len(bboxes)):
#   plt.figure()
#   plt.imshow(visualize_heatmap(image, output['heatmap'][0][i], norm_bboxes[0][i], inout_score=output['inout'][0][i] if output['inout'] is not None else None))
#   plt.axis('off')
#   plt.show()

