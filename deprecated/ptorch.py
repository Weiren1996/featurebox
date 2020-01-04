# -*- coding: utf-8 -*-

# @Time    : 2019/12/19 13:50
# @Email   : 986798607@qq.com
# @Software: PyCharm
# @License: BSD 3-Clause
from __future__ import print_function
from __future__ import print_function, division

import torch
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from matplotlib import pyplot as plt
from torch import optim
from torch.autograd import Variable as V


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, (2, 2))
        # If the size is a square you can only specify a single number
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Net()

input = torch.randn(1, 1, 32, 32)

target = torch.randn(10)  # 随机值作为样例
target = target.view(1, -1)  # 使target和output的shape相同
criterion = nn.MSELoss()

out = net(input)
out.backward(torch.randn(1, 10))
# create your optimizer
optimizer = optim.SGD(net.parameters(), lr=0.01)
for i in range(60):
    # in your training loop:
    optimizer.zero_grad()  # zero the gradient buffers
    output = net(input)
    loss = criterion(output, target)

    loss.backward()
    optimizer.step()

from tensorboardX import SummaryWriter

with SummaryWriter(log_dir=r'C:\Users\Administrator\Desktop/logs', comment='vgg161') as writer:
    writer.add_graph(net, input)

# from torchvision import transforms
# from tensorboardX import SummaryWriter
# # from torch.utils.tensorboard import SummaryWriter
# cat_img = Image.open(r'C:\Users\Administrator\Desktop\图片1.png')
# cat_img.size
#
# transform_224 = transforms.Compose([
#         transforms.Resize(224), # 这里要说明下 Scale 已经过期了，使用Resize
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#     ])
# cat_img_224=transform_224(cat_img)
# writer = SummaryWriter(log_dir=r'C:\Users\Administrator\Desktop/logs', comment='cat image') # 这里的logs要与--logdir的参数一样
# writer.add_image("cat",cat_img_224)
# writer.close()# 执行close立即刷新，否则将每120秒自动刷新


# x = torch.FloatTensor([100])
# y = torch.FloatTensor([500])

# import numpy as np
# for epoch in range(30):
#     x = x * 1.2
#     y = y / 1.1
#     loss = np.random.random()
#     with SummaryWriter(log_dir=r'C:\Users\Administrator\Desktop/logs', comment='train') as writer: #可以直接使用python的with语法，自动调用close方法
#         writer.add_histogram('his/x', x, epoch)
#         writer.add_histogram('his/y', y, epoch)
#         writer.add_scalar('data/x', x, epoch)
#         writer.add_scalar('data/y', y, epoch)
#         writer.add_scalar('data/loss', loss, epoch)
#         writer.add_scalars('data/data_group', {'x': x,

net2 = torch.nn.Sequential(
    torch.nn.Linear(2, 10),
    torch.nn.ReLU(),
    torch.nn.Linear(10, 2),
)
print('方法2：\n', net2)
with SummaryWriter(log_dir=r'C:\Users\Administrator\Desktop/logs', comment='seq') as writer:
    input = torch.randn(100, 2)
    writer.add_graph(net2, input)
