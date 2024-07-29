# Torch
import torch.nn as nn
import torch
import torch.optim as optim
from torch.nn import init
from torchvision.models.resnet import resnet18
# utils
import math
import os
import joblib
from tqdm import tqdm
from SimClr import SimClr
from SSFT import SSFTransformer, MODEL


def get_model(name, **kwargs):
    """
    Instantiate and obtain a model with adequate hyperparameters

    Args:
        name: string of the model name
        kwargs: hyperparameters
    Returns:
        models: PyTorch network
        optimizer: PyTorch optimizer
        criterion: PyTorch loss Function
        kwargs: hyperparameters with sane defaults
    """
    device = kwargs.setdefault('device', torch.device('cuda'))  # 给字典添加键值
    n_classes = kwargs['n_classes']
    n_bands = kwargs['n_bands']
    weights = torch.ones(n_classes)
    weights[torch.LongTensor(kwargs['ignored_labels'])] = 0.
    weights = weights.to(device)
    weights = kwargs.setdefault('weights', weights)

    if name == 'CNN':
        patch_size = kwargs.setdefault('patch_size', 21)     # 如果'patch_size'键不存在于字典中，将会添加键并将值设为默认值,5
        center_pixel = True
        model = MODEL(n_bands, n_classes=128, patch_size=patch_size)   # 模型
        lr = kwargs.setdefault('lr', 0.001)  # LearningRate = 0.075 ×BatchSize的根号
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0005)   # 调换优化器测试,adamw一般 , weight_decay=0.0005
        # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        # criterion = nn.CrossEntropyLoss(weight=kwargs['weights'])   # loss function     # weight参数分别代表n类的权重
        criterion = SimClr(batch_size=kwargs.setdefault('batch_size', 256),
                           temperature=0.5)  # device=device
        model = model.to(device)
        epoch = kwargs.setdefault('epoch', 500)
        kwargs.setdefault('scheduler', optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=epoch // 4,
                                                                            verbose=True))
        # 学习率调整，该方法提供了一些基于训练过程中的某些测量值对学习率进行动态的下降，之前的问题可能就是出在这
        # kwargs.setdefault('batch_size', 512)  # （原100）测试一下
        kwargs.setdefault('supervision', 'simclr')
        kwargs.setdefault('flip_augmentation', False)
        kwargs.setdefault('radiation_augmentation', False)
        kwargs.setdefault('mixture_augmentation', False)
        kwargs['center_pixel'] = center_pixel

    elif name == '2D_CNN':
        patch_size = kwargs.setdefault('patch_size', 21)     # 如果'patch_size'键不存在于字典中，将会添加键并将值设为默认值,5
        center_pixel = True
        # model = MODEL(n_bands, n_classes, patch_size=patch_size)   # 模型
        model = CNN_2dMODEL(n_bands, n_classes, patch_size=patch_size)
        lr = kwargs.setdefault('lr', 0.001)  # LearningRate = 0.075 ×BatchSize的根号
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0005)   # 优化器，调换优化器测试
        # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        # criterion = nn.CrossEntropyLoss(weight=kwargs['weights'])   # loss function     # weight参数分别代表n类的权重
        criterion = SimClr(batch_size=kwargs.setdefault('batch_size', 256), A=A, super_pixel=super_pixel,
                           temperature=0.5)  # device=device
        model = model.to(device)
        epoch = kwargs.setdefault('epoch', 500)
        kwargs.setdefault('scheduler', optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=epoch // 4,
                                                                            verbose=True))
        # 学习率调整，该方法提供了一些基于训练过程中的某些测量值对学习率进行动态的下降
        kwargs.setdefault('batch_size', 512)  # （原100）测试一下
        kwargs.setdefault('supervision', 'simclr')
        kwargs.setdefault('flip_augmentation', False)
        kwargs.setdefault('radiation_augmentation', False)
        kwargs.setdefault('mixture_augmentation', False)
        kwargs['center_pixel'] = center_pixel
    else:
        raise KeyError("{} model is unknown.".format(name))

    return model, optimizer, criterion, kwargs


class CNN_2dMODEL(nn.Module):  # 二维卷积
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
            init.kaiming_normal_(m.weight)      # weight kaiming函数初始化
            init.zeros_(m.bias)

    def __init__(self, input_channels, n_classes, patch_size=21):
        super(CNN_2dMODEL, self).__init__()
        self.patch_size = patch_size
        self.input_channels = input_channels
        kernel_size = 3
        nb_filter = 16
        # self.abw = nn.Sequential(
        #     LBW_BasicBlock(self.input_channels, stride=1, padding=1, patch_size=patch_size),
        # )
        # self.conv1x1 = nn.Conv2d(self.input_channels, self.input_channels, 1, stride=1, padding=0)
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.input_channels, nb_filter*4, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(nb_filter*4),
            nn.LeakyReLU(),
        )
        self.maxpool = nn.MaxPool2d((2, 2), 2, 1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(nb_filter*4, nb_filter*8, kernel_size, padding=1),
            nn.BatchNorm2d(nb_filter*8),
            nn.LeakyReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(nb_filter * 8, nb_filter * 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(nb_filter*16),
            nn.LeakyReLU()
        )
        self.flattened_size = self.flattened()
        self.fc_1 = nn.Sequential(

            nn.Linear(self.flattened_size, 1024),
            nn.BatchNorm1d(1024),
        )
        self.fc_2 = nn.Sequential(
            nn.Linear(1024, 128),
            nn.BatchNorm1d(128)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def flattened(self):
        with torch.no_grad():
            x = torch.zeros((1, self.input_channels, self.patch_size, self.patch_size,))
            x = self.conv1(x)
            x = self.maxpool(x)
            x = self.conv2(x)
            x = self.maxpool(x)
            x = self.conv3(x)
            x = self.maxpool(x)
            t, w, l, b = x.size()
            return t*w*l*b

    def forward(self, x):
        # x = self.abw(x)
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.maxpool(x)
        x = self.conv3(x)
        x = self.maxpool(x)
        x = x.reshape(-1, self.flattened_size)
        x = self.fc_1(x)
        out = self.fc_2(x)
        return out


def save_model(dataset_name, model, model_name, **kwargs):  # model5 houston patchsize:27
    # other_model1：498和499是UP数据集，500是SA数据集，other_model2:仅有SA数据集。other_model3：Houston数据集
    model_dir = './checkpoints/' + model_name + "/" + dataset_name + "/"
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir, exist_ok=True)
    if isinstance(model, torch.nn.Module):  # 判断是否同类型
        filename = "_epoch{epoch}_{metric:.2f}".format(**kwargs)
        tqdm.write("Saving neural network weights in {}".format(filename))
        torch.save(model.state_dict(), model_dir + filename + '.pth')   # 这里是仅仅保存学到的参数 但是在直接使用参数时，Acc远没有训练时的高
        # torch.save(model, model_dir + filename + '.pth')  # 这里是保存整个网络的状态
    else:
        filename = str('wk_MUFFL')
        tqdm.write("Saving model params in {}".format(filename))
        joblib.dump(model, model_dir + filename + '.pkl')
