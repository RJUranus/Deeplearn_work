import os, sys
sys.path.append(os.getcwd())
import onnxruntime
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split
import numpy as np
from collections import Counter
import time

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



path = "C:\\Users\\Di Yang\\Desktop\\data\\furie_net\\test\\"
data = MyDataset('test.csv', path, train_transforms)
print(len(data))
test_data = DataLoader(data, batch_size=1200, shuffle=True, num_workers=0)

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
'''
# 自定义的数据增强
def get_test_transform():
    return transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

# 推理的图片路径
image = Image.open('D:\QQ\\2663258279\FileRecv\\test.jpg').convert('RGB')

img = get_test_transform()(image)
img = img.unsqueeze_(0)  # -> NCHW, 1,3,224,224
# 模型加载
onnx_model_path = "test.onnx"
resnet_session = onnxruntime.InferenceSession(onnx_model_path)
inputs = {resnet_session.get_inputs()[0].name: to_numpy(img)}
outs = resnet_session.run(None, inputs)[0]'''


dic = {'apple': 0, 'banana': 1, 'Guava': 2, 'Lime': 3, 'Orange': 4, 'Pomegranate': 5}
Fruit_label = ['apple', 'banana', 'Guava', 'Lime', 'Orange', 'Pomegranate']
num_to_class = {v : k for k, v in dic.items()}
onnx_model_path = "test1.onnx"
resnet_session = onnxruntime.InferenceSession(onnx_model_path)
h = 0
label = []
true_label = []
predict_label = []
start_time = time.time()
for img, y in test_data:
    inputs = {resnet_session.get_inputs()[0].name: to_numpy(img)}
    outs = resnet_session.run(None, inputs)[0]
    predict = outs.argmax(axis=1)
    end_time = time.time()
    for i in y:
        label.append(num_to_class[i.item()])
    for i in predict:
        predict_label.append(num_to_class[i.item()])
    for i in range(len(y)):
        if predict[i] == y[i]:
            true_label.append(num_to_class[predict[i]])
            h = h + 1
execution_time = end_time - start_time
print('ture:', true_label)
print("onnx prediction", predict_label)
actual_dict = Counter(true_label)
predict_dict = Counter(predict_label)
label_dict = Counter(label)
for i in Fruit_label:
    P = actual_dict[i]/predict_dict[i]
    R = actual_dict[i]/label_dict[i]
    print(f'{i} Precision:', P, f'{i} recall:', R, f'{i} F1-score:', 2*P*R/(P+R))
print('text_accuracy:', h / 4823)
print(f'time to one: {int(execution_time/4.823)} ms')


