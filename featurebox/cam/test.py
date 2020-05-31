from torch import nn
from torchvision.models.vgg import make_layers

from featurebox.cam.cam import GradCAM, GradCAMpp, SmoothGradCAMpp, GradCRM
from featurebox.cam.fig import normalize, trans_to_numpy, visualize
from featurebox.data.datasets import CAMData
import numpy as np
import torch
from skimage import io
from matplotlib.pyplot import imshow
from torchvision import models, transforms
import os
import torch.nn.functional as F


class VGG(nn.Module):

    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((3, 3))
        self.classifier = nn.Sequential(
            nn.Linear(256 * 3 * 3, 1000),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1000, 1000),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1000, 1),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


if __name__ == "__main__":

    sample = 100
    torch.set_default_tensor_type(torch.FloatTensor)
    camdata = CAMData(20, 20, sample, random_state=1, question="reg", inter=False, re_range=(-0.5, 1))
    data = camdata.gen_data(featurize_x=True)
    x = data.x
    y = data.y
    x = x.astype(np.float32)
    y = y.astype(np.float32)
    x = x[np.newaxis, :, :, :]
    x = np.repeat(x, 3, axis=0)

    x = x.transpose((3, 0, 1, 2))

    t_x = torch.from_numpy(x)
    t_y = torch.from_numpy(y)
    x1 = trans_to_numpy(t_x[3])
    # image = io.imshow(x1)

    # Max池化层,conv2d3kernel  输出通道,
    cfgs = {
        'A': [64, 'M', 128, 256],
    }


    def vgg_self(cfg="A", batch_norm=False):

        model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), num_classes=100)

        return model


    model = vgg_self(cfg="A", batch_norm=False)
    target_layer = model.features[-1]
    opitmizer = torch.optim.SGD(model.parameters(), lr=0.03)
    loss_fun = nn.MSELoss()
    for i in range(10):
        predict = model(t_x)
        # print(predict)
        loss = loss_fun(predict, t_y)
        loss = torch.as_tensor(loss, dtype=torch.float)

        print(float(loss))

        opitmizer.zero_grad()
        loss.backward()
        opitmizer.step()
    # # In [7]:
    # # # the target layer you want to visualize
    # target_layer = model.features[-1].conv2
    # # #
    # # # wrapper for class activation mapping. Choose one of the following.
    # wrapped_model = CAM(model, target_layer)
    wrapped_model = GradCRM(model, target_layer)
    # wrapped_model = GradCAMpp(model, target_layer)
    # wrapped_model = SmoothGradCAMpp(model, target_layer, n_samples=5, stdev_spread=0.15)
    # In [8]:
    tensor = t_x[5].unsqueeze(0)
    cam, idx = wrapped_model(tensor, idx=5)

    # # visualize only cam
    imshow(cam.squeeze().numpy(), alpha=0.5, cmap='jet')

    # In [11]:
    # # reverse normalization for display
    img = normalize.inverse_transform(tensor)
    # In [12]:
    heatmap = visualize(img, cam)
    # In [13]:
    # # save image
    ##save_image(heatmap, './sample/{}_cam.png'.format(idx2label[idx]).replace(" ", "_").replace(",", ""))
    # # save_image(heatmap, './sample/{}_gradcam.png'.format(idx2label[idx]).replace(" ", "_").replace(",", ""))
    # # save_image(heatmap, './sample/{}_gradcampp.png'.format(idx2label[idx]).replace(" ", "_").replace(",", ""))
    # save_image(heatmap, './sample/{}_smoothgradcampp.png'.format(idx2label[idx]).replace(" ", "_").replace(",", ""))
    # In [14]:
    # # or visualize on Jupyter
    hm = (heatmap.squeeze().numpy().transpose(1, 2, 0))
    imshow(hm)
