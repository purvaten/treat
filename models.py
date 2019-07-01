"""Define dataset model classes and the learning model class of AlexNet-based encoder and ResNet-based decoder."""

import numpy as np
import torch
from torch import nn
from torchvision import transforms
from torch.utils.data import Dataset
import scipy.ndimage as ndimage
from scipy.misc import imresize
import torch.nn.functional as F
from string import ascii_lowercase

import glob
import math
import re
import cv2
import os

from logger import get_logger


logger = get_logger(__name__)

letters = ['lower' + a for a in ascii_lowercase]

# ===================transforms & preprocessing=====================
img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


def mean_subtract(dataset):
    """Subtract mean of all images in dataset from each image."""
    data = [dataset[i] for i in range(len(dataset))]
    data_numpy = [dataset[i].numpy() for i in range(len(dataset))]

    # mean
    mean = np.mean(data_numpy)

    # standard deviation
    std = np.std(data_numpy)

    # perform mean subtract
    new_dataset = []
    for i in range(len(dataset)):
        data[i] -= mean
        data[i] /= std
        new_dataset.append(data[i])
    return new_dataset, mean


# ===================Dataset Classes=====================

class MyData(Dataset):
    """Alphabet, Noun Project & Theme Clipart Data."""

    def __init__(self, args, img_size):
        """Intialize data items."""
        data = []
        self.labels = []
        self.args = args
        self.filenames = []
        weights = []
        identity = []

        logger.info("Getting MyData Data")

        if 'cliparts' in args.data:
            logger.info("Getting clipart data ...")
            for i, filename in enumerate(os.path.join(args.cliparts_dir, '*.png')):
                if i > args.datalimit:
                    logger.info("Setting a limit of %d for cliparts" % args.datalimit)
                    break
                img = ndimage.imread(filename)[:, :, 3]
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                img = cv2.bitwise_not(img)
                res = imresize(img, size=(img_size, img_size))
                res = res / 255.0
                data.append(res)
                weights.append(1)
                identity.append(1)
                self.filenames.append(filename)
                self.labels.append(letters.index('lowera'))    # dummy label

        if 'letters' in args.data:
            logger.info("Getting letter data ...")
            for i, filename in enumerate(os.path.join(args.letters_dir, '*.png')):
                if i > args.datalimit:
                    logger.info("Setting a limit of %d for letters" % args.datalimit)
                    break
                img = ndimage.imread(filename)[:, :, :3]
                res = imresize(img, size=(img_size, img_size))  # numpy array of dimensions (s,s,3)
                res = res / 255.0
                data.append(res)
                label = ''.join([i for i in filename.split('/')[-1].split('.png')[0] if not i.isdigit()])
                self.labels.append(letters.index(label))
                identity.append(2)
                self.filenames.append(filename)
                weights.append(float(args.alpha))

        self.mydata = data
        self.transform = img_transform
        self.weights = weights
        self.identity = identity

    def __getitem__(self, index):
        """Return data items."""
        if self.transform is not None:
            x = np.transpose(self.mydata[index], (2, 0, 1))
            # x = self.transform(x)
            x = torch.FloatTensor(x)
            x -= 0.5
            x /= 0.5
        else:
            x = self.mydata[index]
        return x, self.labels[index], self.weights[index], self.identity[index]    # return (img, label, w, identity)

    def __len__(self):
        """Return numner of data items."""
        return len(self.mydata)


# ===================Model Classes=====================

class Bottleneck(nn.Module):
    """Bottleneck function for ResNet-based encoder."""

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)

        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResnetEncoder(nn.Module):
    """ResNet-based Encoder."""

    def __init__(self, block, layers, args, num_classes=23):
        self.args = args
        self.inplanes = 64
        super(ResnetEncoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # return_indices = True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, args.zsize)
    # self.fc = nn.Linear(num_classes,16)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)

        x = self.bn1(x)
        x = self.relu(x)

        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class AlexnetEncoder(nn.Module):
    """AlexNet-based Encoder."""

    def __init__(self, args):
        super(AlexnetEncoder, self).__init__()
        self.args = args
        self.conv1 = nn.Conv2d(3, 64, 11, stride=4, padding=2)
        self.conv2 = nn.Conv2d(64, 192, 5, padding=2)
        self.conv3 = nn.Conv2d(192, 384, 3, padding=1)
        self.conv4 = nn.Conv2d(384, 256, 3, padding=1)
        self.conv5 = nn.Conv2d(256, 256, 3, padding=1)
        self.fc1 = nn.Linear(256 * 6 * 6, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, args.zsize)
        self.drop_layer = nn.Dropout(p=0.5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x, indices1 = F.max_pool2d(x, (3, 3), (2, 2), return_indices=True)

        x = F.relu(self.conv2(x))
        x, indices2 = F.max_pool2d(x, (3, 3), (2, 2), return_indices=True)

        x = F.relu(self.conv3(x))

        x = F.relu(self.conv4(x))

        x = F.relu(self.conv5(x))
        x, indices3 = F.max_pool2d(x, (3, 3), (2, 2), return_indices=True)

        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.drop_layer(x)

        x = F.relu(self.fc1(x))
        x = self.drop_layer(x)

        x = F.relu(self.fc2(x))

        x = self.fc3(x)

        return x


class Decoder(nn.Module):
    """ResNet-based Decoder."""

    def __init__(self, args):
        super(Decoder, self).__init__()
        self.dfc3 = nn.Linear(args.zsize, 4096)
        self.bn3 = nn.BatchNorm1d(4096)
        self.dfc2 = nn.Linear(4096, 4096)
        self.bn2 = nn.BatchNorm1d(4096)
        self.dfc1 = nn.Linear(4096, 256 * 6 * 6)
        self.bn1 = nn.BatchNorm1d(256 * 6 * 6)
        self.upsample1 = nn.Upsample(scale_factor=2)
        self.dconv5 = nn.ConvTranspose2d(256, 256, 3, padding=0)
        self.dconv4 = nn.ConvTranspose2d(256, 384, 3, padding=1)
        self.dconv3 = nn.ConvTranspose2d(384, 192, 3, padding=1)
        self.dconv2 = nn.ConvTranspose2d(192, 64, 5, padding=2)
        self.dconv1 = nn.ConvTranspose2d(64, 3, 12, stride=4, padding=4)

    def forward(self, x):  # ,i1,i2,i3):
        batch_size = x.shape[0]
        x = self.dfc3(x)
        # x = F.relu(x)
        # x = x.view(100, 16, 16, 16)
        x = F.relu(self.bn3(x))
        x = self.dfc2(x)
        x = F.relu(self.bn2(x))
        # x = F.relu(x)
        x = self.dfc1(x)
        x = F.relu(self.bn1(x))
        # x = F.relu(x)
        # logger.info(x.size())
        x = x.view(batch_size, 256, 6, 6)
        # logger.info (x.size())
        x = self.upsample1(x)
        # logger.info x.size()
        x = self.dconv5(x)
        # logger.info x.size()
        x = F.relu(x)
        # logger.info x.size()
        x = F.relu(self.dconv4(x))
        # logger.info x.size()
        x = F.relu(self.dconv3(x))
        # logger.info x.size()
        x = self.upsample1(x)
        # logger.info x.size()
        x = self.dconv2(x)
        # logger.info x.size()
        x = F.relu(x)
        x = self.upsample1(x)
        # logger.info x.size()
        x = self.dconv1(x)
        # logger.info x.size()
        # x = F.sigmoid(x) - purva
        x = torch.tanh(x)
        # logger.info x
        return x


class MultiTask(nn.Module):
    """Main multitask module."""

    def __init__(self, args):
        super(MultiTask, self).__init__()
        self.args = args
        self.fc = nn.Linear(args.zsize, 52)    # for classification loss
        self.sm = nn.Softmax()    # for classification loss

        if args.model == 'alexnet':
            self.encoder = AlexnetEncoder(args=args)
        elif args.model == 'bigresnet':
            self.encoder = ResnetEncoder(block=Bottleneck, layers=[3, 4, 6, 3], args=args)
        elif args.model == 'smallresnet':
            self.encoder = ResnetEncoder(block=Bottleneck, layers=[1, 1, 1, 1], args=args)

        self.decoder = Decoder(args)

    def forward(self, x):
        x = self.encoder(x)
        self.representation = x
        y = self.fc(x)
        z = self.sm(y)
        x = self.decoder(x)
        return x, y, z
