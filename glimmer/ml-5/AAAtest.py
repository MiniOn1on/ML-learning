import torch
from PIL import Image
import torchvision
from torch import nn


img_path="./img/cat-1.jpg"
img=Image.open(img_path)
class_names_cn = [
    '飞机',
    '汽车',
    '鸟',
    '猫',
    '鹿',
    '狗',
    '青蛙',
    '马',
    '船',
    '卡车'
]


img=img.convert("RGB")# 有的png格式的图片会自带第四个通道透明度，无法适配模型三通道，需convert成rgb三通道的

transform=torchvision.transforms.Compose([torchvision.transforms.Resize((32,32)),
                                          torchvision.transforms.ToTensor()])
image=transform(img)
print(image.shape)


class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.seq=nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),nn.Conv2d(32, 64, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_features=64,out_features=10)
        )
    def forward(self,x):
        return self.seq(x)

model=Model()
model.load_state_dict(torch.load("./train_model/trainmodel_20.pth",map_location=torch.device("cuda")))
# print(model)
image=torch.reshape(image,(-1,3,32,32))

model.eval()
with torch.no_grad():
    output=model(image)

print(output)
output=torch.reshape(output,(-1,10))
pred=output.argmax(1)
pred_class = class_names_cn[pred[0]]
print(pred_class)