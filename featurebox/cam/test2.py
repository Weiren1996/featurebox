import numpy as np
import torch
from matplotlib.pyplot import imshow
from skimage import io
from torch import nn

from featurebox.cam.cam import GradCRM
from featurebox.cam.fig import trans_to_numpy, visualize, trans_to_tensor, trans_to_tensors
from featurebox.data.datasets import CAMData


class VGG(nn.Module):

    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((3, 3))
        self.classifier = nn.Sequential(
            nn.Linear(57600
                      , 1000),
            nn.ReLU(True),
            nn.Dropout(),
            # nn.Linear(1000, 1000),
            # nn.ReLU(True),
            # nn.Dropout(),
            nn.Linear(1000, 1),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        # x = self.avgpool(x)
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


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 1
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


if __name__ == "__main__":

    sample = 100
    torch.set_default_tensor_type(torch.FloatTensor)
    camdata = CAMData(100, (30, 30), random_state=1, question="reg", inter=False, re_range=(-0.5, 1))
    data = camdata.gen_data(featurize_x="total", self_func="de_2Dsite")
    data.x = camdata.cp_channel_to1(data.x)[0]
    x = data.x

    # y = data.y
    feature = data.feature
    y = torch.tensor([f[1] for f in feature]).reshape((-1, 1))

    t_x = trans_to_tensors(x)
    t_y = trans_to_tensor(y, channel=False)


    def showsite(t_x, feature, i):
        x = t_x[i]
        sitei = feature[i][0]
        x = trans_to_numpy(x)
        if x.ndim == 2:
            x = np.expand_dims(x, axis=-1)
            x = np.repeat(x, 3, axis=-1) / 2
        elif x.ndim == 3:
            if x.shape[-1] == 1:
                x = np.repeat(x, 3, axis=-1) / 2
        assert x.shape[-1] == 3
        x[..., 0][sitei[0]] = 1
        x[..., 0][sitei[1]] = 1
        x[..., 0][sitei[2]] = 1

        return x


    #

    # Max池化层,conv2d3kernel  输出通道,
    cfgs = {
        'A': [64, 64],
        'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
        'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    }


    def vgg_self(cfg="A", batch_norm=False):

        model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), num_classes=100)

        return model


    model = vgg_self(cfg="A", batch_norm=False)
    target_layer = model.features[-1]
    opitmizer = torch.optim.SGD(model.parameters(), lr=0.03)
    loss_fun = nn.MSELoss()
    for i in range(5):
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
    img = t_x[7].unsqueeze(0)
    cam, idx = wrapped_model(img, idx=7)
    heatmap = visualize(img, cam)
    hm = (heatmap.squeeze().numpy().transpose(1, 2, 0))
    xp = showsite(t_x, feature, 7)
    imshow(xp)

    cam=cam.squeeze(0).numpy().transpose(1, 2, 0)
    cam = np.repeat(cam, 3, axis=2)
    imshow(cam)
