import os
import torch
import pandas as pd
from model import FC, MyResNet18, MobileNetV3_Small
from torch import nn
from PIL import Image
from torchvision import models
from torchsummary import summary
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms




class MyDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = Image.open(img_path).convert('RGB')
        label = int(self.img_labels.iloc[idx, 1])
        image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.img_labels)

train_transforms = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):  #@save
    net.load_state_dict(torch.load('v4.pth'))
    net.to('cuda:0')
    for epoch in range(num_epochs):
        print('epoch:', epoch)
        train_epoch_ch3(net, train_iter, loss, updater)
    torch.save(net.state_dict(), 'v4.pth')



def train_epoch_ch3(net, train_iter, loss, updater):  #@save
    # 将模型设置为训练模式
    h = 0
    net.train()
    loss_num = 0
    t = 0
    # 训练损失总和、训练准确度总和、样本数
    for x, y in train_iter:
        t = t+1
        # 计算梯度并更新参数
        x, y = x.to('cuda:0'), y.to('cuda:0')
        y_hat = net(x.float())
        l = loss(y_hat.float(), y.long())
        l.mean().backward()
        loss_num = loss_num + l.mean().item()
        updater.step()
        y_hat1 = y_hat
        y_hat1 = y_hat1.argmax(axis=1)
        for i in range(len(y)):
            if y_hat1[i].long() == y[i].long():
                h = h + 1
    print('train_accuracy:', h / 3858, 'loss:', loss_num/t)


def predict(net, text_data1):
    h1 = 0
    for x1, y1 in text_data1:
        x1, y1 = x1.to('cuda:0'), y1.to('cuda:0')
        y_hat = net(x1.float())
        y_hat1 = torch.argmax(y_hat, dim=1)
        for i in range(len(y1)):
            if y_hat1[i].long() == y1[i].long():
                h1 = h1 + 1
    print('text_accuracy:', h1 / 965)




path = "C:\\Users\\Di Yang\\Desktop\\data\\furie_net\\test"
#path1 = "C:\\Users\\Di Yang\\Desktop\\data\\furie_net\\mix"
data = MyDataset('test.csv', path, train_transforms)
#data1 = MyDataset('test.csv', path1, train_transforms)

train_size = int(0.8 * len(data))
test_size = len(data) - train_size
train_data1, test_data1 = random_split(data, [train_size, test_size])
train_data = DataLoader(train_data1, batch_size=32, shuffle=True, num_workers=0)
test_data = DataLoader(test_data1, batch_size=64, shuffle=True, num_workers=0)



#net = nn.Sequential(models.vgg16(), nn.Linear(1000, 6))
net = nn.Sequential(*list(models.mobilenet_v3_small().children())[:-2],
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Conv2d(576, 288, 1, 1, 0),
                    nn.BatchNorm2d(288),
                    nn.Conv2d(288, 144, 1, 1, 0),
                    nn.SiLU(),
                    nn.Conv2d(144, 36, 1, 1, 0),
                    nn.BatchNorm2d(36),
                    nn.Conv2d(36, 6, 1, 1, 0),
                    FC()
                    )

'''net = nn.Sequential(MobileNetV3_Small(),
                    nn.Conv2d(576, 128, 3, 2, 1),
                    nn.BatchNorm2d(128),
                    nn.Conv2d(128, 64, 3, 2, 1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    FC(),
                    nn.Linear(256, 6),
                    nn.Sigmoid())'''

#net = MyResNet18()
#net = models.mobilenet_v3_small()
loss = nn.CrossEntropyLoss()
opt = torch.optim.SGD(net.parameters(), lr=0.0000015)
#train_ch3(net, train_data, test_data, loss, 5, opt)
#predict(net, test_data)
#summary(net, (3, 224, 224))


net.load_state_dict(torch.load('v4.pth'))
net.eval()
'''image = Image.open("D:\QQ\\2663258279\FileRecv\\test.jpg").convert('RGB')
image = train_transforms(image).reshape(1, 3, 224, 224)
y = net(image)
y = torch.argmax(y, dim=1)
print(y)'''



batch_size = 1  # 批处理大小
input_shape = (3, 224, 224)  # 输入数据

x = torch.randn(batch_size, *input_shape)  # 生成张量
export_onnx_file = "test1.onnx"  # 目的ONNX文件名
torch.onnx.export(net,
                  x,
                  export_onnx_file,
                  opset_version=10,
                  do_constant_folding=True,  # 是否执行常量折叠优化
                  input_names=["input"],  # 输入名
                  output_names=["output"],  # 输出名
                  dynamic_axes={"input": {0: "batch_size"},  # 批处理变量
                                "output": {0: "batch_size"}})



























'''path = "C:\\Users\\Di Yang\\Desktop\\data\\furie_net\mix\\"
dirs = os.listdir(path)
data_dir = []
data_dir1 = []
for i in range(6):
    data_dir.append(os.listdir(os.path.join(path, dirs[i])))

label = []
#print(len(data_dir[0])+len(data_dir[1])+len(data_dir[2])+len(data_dir[3])+len(data_dir[4])+len(data_dir[5]))

for i in range(6):
    for j in range(len(data_dir[i])):
        label.append(i)

for i in range(6):
    for j in range(len(data_dir[i])):
        data_dir1.append([data_dir[i][j]])


print(label)
print(data_dir1)'''
'''data2 = pd.DataFrame(data=data_dir1, index=None, columns=['data_dir'])
# PATH为导出文件的路径和文件名
data2.to_csv('test.csv')'''

'''df = pd.read_csv('test.csv')
df['label'] = label
df.to_csv('test.csv')'''
