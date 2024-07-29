# -*- coding: utf-8 -*-
"""
This file contains the PyTorch dataset for hyperspectral images and
related helpers.
"""
import spectral
import numpy as np
import torch
import torch.utils
import torch.utils.data
import os
import cv2
from tqdm import tqdm

import rs_aug.transforms as T


try:
    # Python 3
    from urllib.request import urlretrieve
except ImportError:
    # Python 2
    from urllib import urlretrieve
from utils import open_file

DATASETS_CONFIG = {
    'PaviaC': {
        'urls': ['http://www.ehu.eus/ccwintco/uploads/e/e3/Pavia.mat',
                 'http://www.ehu.eus/ccwintco/uploads/5/53/Pavia_gt.mat'],
        'img': 'Pavia.mat',
        'gt': 'Pavia_gt.mat'
    },
    'PaviaU': {
        'urls': ['http://www.ehu.eus/ccwintco/uploads/e/ee/PaviaU.mat',
                 'http://www.ehu.eus/ccwintco/uploads/5/50/PaviaU_gt.mat'],
        'img': 'PaviaU.mat',
        'gt': 'PaviaU_gt.mat'
    },
    'KSC': {
        'urls': ['http://www.ehu.es/ccwintco/uploads/2/26/KSC.mat',
                 'http://www.ehu.es/ccwintco/uploads/a/a6/KSC_gt.mat'],
        'img': 'KSC.mat',
        'gt': 'KSC_gt.mat'
    },
    'IndianPines': {
        'urls': ['http://www.ehu.eus/ccwintco/uploads/6/67/Indian_pines_corrected.mat',
                 'http://www.ehu.eus/ccwintco/uploads/c/c4/Indian_pines_gt.mat'],
        'img': 'Indian_pines_corrected.mat',
        'gt': 'Indian_pines_gt.mat'
    },
    'Botswana': {
        'urls': ['http://www.ehu.es/ccwintco/uploads/7/72/Botswana.mat',
                 'http://www.ehu.es/ccwintco/uploads/5/58/Botswana_gt.mat'],
        'img': 'Botswana.mat',
        'gt': 'Botswana_gt.mat',
    },
    'Salinas': {
        'img': 'Salinas.mat',
        'gt': 'Salinas_gt.mat',
    },
    'Houston': {
        'img': 'houston.mat',
        'gt': 'houston_gt.mat',
    },
    'MUFFL': {
        'img': 'HSI.mat',
        'gt': 'gt.mat',
    },
    'HongHu': {
        'img': 'HongHu.tif',
        'gt': 'HongHu_gt.tif',
    }
}

try:
    from custom_datasets import CUSTOM_DATASETS_CONFIG

    DATASETS_CONFIG.update(CUSTOM_DATASETS_CONFIG)
except ImportError:
    pass


class TqdmUpTo(tqdm):
    """Provides `update_to(n)` which uses `tqdm.update(delta_n)`."""

    def update_to(self, b=1, bsize=1, tsize=None):
        """
        b  : int, optional
            Number of blocks transferred so far [default: 1].
        bsize  : int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize  : int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)  # will also set self.n = b * bsize


def get_dataset(dataset_name, target_folder="./", datasets=DATASETS_CONFIG, patch_size=5):
    """ Gets the dataset specified by name and return the related components.
    Args:
        dataset_name: string with the name of the dataset
        target_folder (optional): folder to store the datasets, defaults to ./
        datasets (optional): dataset configuration dictionary, defaults to prebuilt one
    Returns:
        img: 3D hyperspectral image (WxHxB)
        gt: 2D int array of labels
        label_values: list of class names
        ignored_labels: list of int classes to ignore
        rgb_bands: int tuple that correspond to red, green and blue bands
    """
    palette = None

    if dataset_name not in datasets.keys():
        raise ValueError("{} dataset is unknown.".format(dataset_name))

    dataset = datasets[dataset_name]  # 取这个数据的全部信息

    folder = target_folder + datasets[dataset_name].get('folder',
                                                        dataset_name + '/')  # get()函数返回指定键的值，如果指定键的值不存在时，返回该默认值。
    if dataset.get('download', False):  # 我将True改成了False ,默认不下载
        # Download the dataset if is not present
        if not os.path.isdir(folder):  # 用于判断对象是否为一个目录
            os.mkdir(folder)
        for url in datasets[dataset_name]['urls']:
            # download the files
            filename = url.split('/')[-1]
            if not os.path.exists(folder + filename):
                with TqdmUpTo(unit='B', unit_scale=True, miniters=1,
                              desc="Downloading {}".format(filename)) as t:
                    urlretrieve(url, filename=folder + filename,  # 利用urlretrieve()将数据下载到本地。
                                reporthook=t.update_to)
    elif not os.path.isdir(folder):
        print("WARNING: {} is not downloadable.".format(dataset_name))

    if dataset_name == 'PaviaC':
        # Load the image
        img = open_file(folder + 'Pavia.mat')['pavia']

        rgb_bands = (55, 41, 12)

        gt = open_file(folder + 'Pavia_gt.mat')['pavia_gt']

        label_values = ["Undefined", "Water", "Trees", "Asphalt",
                        "Self-Blocking Bricks", "Bitumen", "Tiles", "Shadows",
                        "Meadows", "Bare Soil"]

        ignored_labels = [0]

    elif dataset_name == 'PaviaU':
        # Load the image
        img = open_file(folder + 'PaviaU.mat')['paviaU']
        # img = open_file(folder + 'paviaU_PCA_30.mat')['q']  # PCA
        # feature = h5py.File(folder + 'paviaU_PCA_30.mat')
        # img = feature['q']

        rgb_bands = (55, 41, 12)

        gt = open_file(folder + 'PaviaU_gt.mat')['paviaU_gt']

        label_values = ["Undefined", 'Asphalt', 'Meadows', 'Gravel', 'Trees',
                        'Painted metal sheets', 'Bare Soil', 'Bitumen',
                        'Self-Blocking Bricks', 'Shadows']

        ignored_labels = [0]

    elif dataset_name == 'IndianPines':
        # Load the image
        img = open_file(folder + 'Indian_pines_corrected.mat')
        img = img['indian_pines_corrected']

        # img = open_file(folder + 'indian_pines_80.mat')       # 80层引导滤波
        # img = img['q']

        # img = open_file(folder + 'indian_pines_PCA_30.mat')  # PCA
        # img = img['q']  # PCA

        rgb_bands = (43, 21, 11)  # AVIRIS sensor

        gt = open_file(folder + 'Indian_pines_gt.mat')['indian_pines_gt']
        label_values = ["Undefined", "Alfalfa", "Corn-notill", "Corn-mintill",
                        "Corn", "Grass-pasture", "Grass-trees",
                        "Grass-pasture-mowed", "Hay-windrowed", "Oats",
                        "Soybean-notill", "Soybean-mintill", "Soybean-clean",
                        "Wheat", "Woods", "Buildings-Grass-Trees-Drives",
                        "Stone-Steel-Towers"]

        ignored_labels = [0]

    elif dataset_name == 'Botswana':
        # Load the image
        img = open_file(folder + 'Botswana.mat')['Botswana']

        rgb_bands = (75, 33, 15)

        gt = open_file(folder + 'Botswana_gt.mat')['Botswana_gt']
        label_values = ["Undefined", "Water", "Hippo grass",
                        "Floodplain grasses 1", "Floodplain grasses 2",
                        "Reeds", "Riparian", "Firescar", "Island interior",
                        "Acacia woodlands", "Acacia shrublands",
                        "Acacia grasslands", "Short mopane", "Mixed mopane",
                        "Exposed soils"]

        ignored_labels = [0]

    elif dataset_name == 'KSC':
        # Load the image
        img = open_file(folder + 'KSC.mat')['KSC']

        rgb_bands = (43, 21, 11)  # AVIRIS sensor

        gt = open_file(folder + 'KSC_gt.mat')['KSC_gt']
        label_values = ["Undefined", "Scrub", "Willow swamp",
                        "Cabbage palm hammock", "Cabbage palm/oak hammock",
                        "Slash pine", "Oak/broadleaf hammock",
                        "Hardwood swamp", "Graminoid marsh", "Spartina marsh",
                        "Cattail marsh", "Salt marsh", "Mud flats", "Wate"]

        ignored_labels = [0]

    elif dataset_name == 'Salinas':
        # Load the image
        img = open_file(folder + 'Salinas.mat')
        img = img['salinas_corrected']
        # img = open_file(folder + 'Salinas_PCA_30.mat')
        # img = img['q']

        rgb_bands = (55, 41, 12)

        gt = open_file(folder + 'Salinas_gt.mat')['salinas_gt']

        label_values = ["Undefined", "Brocoli_green_weeds_1", "Brocoli_green_weeds_2", "Fallow",
                        "Fallow_rough_plow", "Fallow_smooth", "Stubble", "Celery",
                        "Grapes_untrained", "Soil_vinyard_develop", "Corn_senesced_green_weeds", "Lettuce_romaine_4wk",
                        "Lettuce_romaine_5wk", "Lettuce_romaine_6wk", "Lettuce_romaine_7wk",
                        "Vinyard_untrained", "Vinyard_vertical_trellis"]

        ignored_labels = [0]

    elif dataset_name == 'Houston':
        img = open_file(folder + 'Houston.mat')
        img = img['Houston']
        rgb_bands = (43, 60, 21)
        gt = open_file(folder + 'Houston_gt.mat')['Houston_gt']
        ignored_labels = [0]
        label_values = ["Undefined", "Healthy grass", "Stressed grass", "Synthetic grass", "Trees", "Soil", "Water",
                        "Residential", "Commercial", "Road", "Highway", "Railway", "Parking Lot 1", "Parking Lot 2"
                        , "Tennis Court", "Running Track"]
        # Normalize
        # muX = np.mean(X, axis=(0, 1), keepdims=True)
        # stdX = np.std(X, axis=(0, 1), keepdims=True) + 1e-6
        # X = (X - muX)/stdX
    elif dataset_name == 'MUFFL':
        img = open_file(folder + 'HSI.mat')
        img = img['HSI']

        rgb_bands = (43, 60, 21)

        gt = open_file(folder + 'gt1.mat')['A']
        # gt = gt.T # 因为原数据GT维度方向与hsi不同
        ignored_labels = [0]
        label_values = ["Undefined", "Trees", "Mostly grass", "Mixed ground surface", "Dirt and sand", "Road", "Water"
            , "Building shadow", "Building", "Sidewalk", "Yellow curb", "Cloth panels"]

    elif dataset_name == 'HongHu':
        # Load the image
        img = open_file(folder + 'HongHu.tif').transpose(1, 2, 0)
        rgb_bands = (55, 41, 12)

        gt = open_file(folder + 'HongHu_gt.tif')

        label_values = ["Undefined", "Red roof", "Road", "Bare soil", "Cotton", "Cotton firewood", "Rape"
                        "Chinese cabbage", "Pakchoi", "Cabbage", "Tuber mustard", "Brassica parachinensis",
                        "Brassica chinensis", "Small Brassica chinensis", "Lactuca sativa", "Celtuce",
                        "Film covered lettuce", "Romaine lettuce", "Carrot", "White radish", "Garlic sprout",
                        "Broad bean", "Tree"]

        ignored_labels = [0]

    else:
        # Custom dataset    # 自定义数据
        img, gt, rgb_bands, ignored_labels, label_values, palette = CUSTOM_DATASETS_CONFIG[dataset_name]['loader'](
            folder)

    # Filter NaN out    检查数据中是否还有NaN数据
    nan_mask = np.isnan(img.sum(axis=-1))  # np.isnan() 返回的是一个数组，其中的值是对应输入数组对元素NaN判断的bool型数据
    if np.count_nonzero(nan_mask) > 0:  # 意义为：数据中有NaN存在
        print( "Warning: NaN have been found in the data. It is preferable to remove them beforehand. Learning on NaN data is disabled.")
    img[nan_mask] = 0  # 将NaN型数据转成0
    gt[nan_mask] = 0
    ignored_labels.append(0)

    img = cv2.copyMakeBorder(img, patch_size // 2, patch_size // 2, patch_size // 2, patch_size // 2,
                             cv2.BORDER_REFLECT)
    gt = cv2.copyMakeBorder(gt, patch_size // 2, patch_size // 2, patch_size // 2, patch_size // 2, cv2.BORDER_REFLECT)

    ignored_labels = list(set(ignored_labels))

    img = np.asarray(img, dtype='float32')
    return img, gt, label_values, ignored_labels, rgb_bands, palette  # , ori_img, ori_gt


class HyperX(torch.utils.data.Dataset):
    """ Generic class for a hyperspectral scene """

    def __init__(self, data, gt, **hyperparams):
        """
        Args:
            data: 3D hyperspectral image
            gt: 2D array of labels
            patch_size: int, size of the spatial neighbourhood
            center_pixel: bool, set to True to consider only the label of the
                          center pixel
            data_augmentation: bool, set to True to perform random flips
            supervision: 'full' or 'semi' supervised algorithms
        """
        super(HyperX, self).__init__()
        self.data = data
        self.label = gt
        self.name = hyperparams['dataset']
        self.patch_size = hyperparams['patch_size']
        self.ignored_labels = set(hyperparams['ignored_labels'])
        self.flip_augmentation = hyperparams['flip_augmentation']
        self.radiation_augmentation = hyperparams['radiation_augmentation']
        self.mixture_augmentation = hyperparams['mixture_augmentation']
        self.center_pixel = hyperparams['center_pixel']
        self.train_sample_extend = hyperparams['train_sample_extend']  # 样本扩展参数
        supervision = hyperparams['supervision']
        # Fully supervised : use all pixels with label not ignored
        if supervision == 'full':
            mask = np.ones_like(gt)
            for l in self.ignored_labels:
                mask[gt == l] = 0
        # Semi-supervised : use all pixels, except padding
        elif supervision == 'semi':
            mask = np.ones_like(gt)  # 返回一个和gt一样大小的全1矩阵
        elif supervision == 'simclr':
            mask = np.ones_like(gt)
            for l in self.ignored_labels:
                mask[gt == l] = 0

        x_pos, y_pos = np.nonzero(mask)  # 返回mask中非零值得索引值 即pix在图中的坐标
        p = self.patch_size // 2  # // 是向下取整的除法,11
        temp_indices = np.array([(x, y) for x, y in zip(x_pos, y_pos) if x > p and x < data.shape[0] - p and y > p and y < data.shape[1] - p])
        # pix的坐标位置
        self.indices = []
        for item_x, item_y in temp_indices:
            newitem = [item_x, item_y, 0]
            self.indices.append(newitem)
            self.labels = [self.label[x, y] for x, y, z in self.indices]
            np.random.shuffle(self.indices)

    @staticmethod
    def flip(*arrays, a):  # 数据增强 之 翻转

        if a is 1:
            arrays = [np.fliplr(arr) for arr in arrays]  # 将数组在左右翻转
        if a is 2:
            arrays = [np.flipud(arr) for arr in arrays]  # 上下翻转
        return arrays

    @staticmethod
    def radiation_noise(data, alpha_range=(0.9, 1.1), beta=1 / 25):
        alpha = np.random.uniform(*alpha_range)
        noise = np.random.normal(loc=0., scale=1.0, size=data.shape)
        return alpha * data + beta * noise

    def mixture_noise(self, data, label, beta=1 / 25):
        alpha1, alpha2 = np.random.uniform(0.01, 1., size=2)
        noise = np.random.normal(loc=0., scale=1.0, size=data.shape)
        data2 = np.zeros_like(data)
        for idx, value in np.ndenumerate(label):
            if value not in self.ignored_labels:
                l_indices = np.nonzero(self.labels == value)[0]
                l_indice = np.random.choice(l_indices)
                assert (self.labels[l_indice] == value)
                x, y = self.indices[l_indice]
                data2[idx] = self.data[x, y]
        return (alpha1 * data + alpha2 * data2) / (alpha1 + alpha2) + beta * noise

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        x, y, z = self.indices[i]  # 先获取坐标
        x1, y1 = x - self.patch_size // 2, y - self.patch_size // 2  # 计算出目标window
        # x2, y2 = x1 + self.patch_size, y1 + self.patch_size
        x2, y2 = x1 + self.patch_size, y1 + self.patch_size

        data = self.data[x1:x2, y1:y2]  # 读取window中的数据和类别
        label = self.label[x1:x2, y1:y2]  # type =array 21*21*30

        if self.radiation_augmentation and np.random.random() < 0.1:  # 10%的概率加噪声
            data = self.radiation_noise(data)
        if self.mixture_augmentation and np.random.random() < 0.2:  # 20%的概率加mixture噪声
            data = self.mixture_noise(data, label)
        # ***augmention***
        train_transform = T.Compose([T.RandomEnlarge(prob=1, min_clip_rate=[0.95, 0.95]),
                                     # T.RandomFlip(prob=0.2, direction='Horizontal'),
                                    T.RandomColor(prob=0.8, alpha_range=[0.8, 1.2], beta_range=[0, 500],
                                                  band_num=data.shape[-1]),
                                     # T.RandomBlur(prob=0.5, ksize=3, band_num=data.shape[2]),  # ksize=1/k
                                     # T.RandomRotate(prob=0.5, ig_pix=0),
                                     T.Normalize(mean=([0] * data.shape[-1]), std=([1] * data.shape[-1]), bit_num=16,
                                                 band_num=data.shape[-1])
                                     ])
        data1 = train_transform(data)
        data2 = train_transform(data)
        data1 = np.asarray(np.copy(data1[0]).transpose((2, 0, 1)))   # type= float32
        data2 = np.asarray(np.copy(data2[0]).transpose((2, 0, 1)))
        label = np.asarray(np.copy(label), dtype='int64')
        # ***Load the data into PyTorch tensors***
        data1 = torch.from_numpy(data1)
        data2 = torch.from_numpy(data2)
        label = torch.from_numpy(label)
        # ***augmention***
        # if self.flip_augmentation and self.patch_size > 1:      # 数据增强 之 翻转
        #     # Perform data augmentation (only on 2D patches)
        #     data, label = self.flip(data, label)
        # Extract the center label if needed
        if self.center_pixel and self.patch_size > 1:
            label = label[self.patch_size // 2, self.patch_size // 2]  # 确定window中心点pix的label
        # Remove unused dimensions when we work with invidual spectrums
        elif self.patch_size == 1:
            data1 = data1[:, 0, 0]
            data2 = data2[:, 0, 0]
            label = label[0, 0]
        return data1, data2, (x, y, label)


class R_HyperX(torch.utils.data.Dataset):
    """ Generic class for a hyperspectral scene """

    def __init__(self, data, gt, **hyperparams):
        """
        Args:
            data: 3D hyperspectral image
            gt: 2D array of labels
            patch_size: int, size of the spatial neighbourhood
            center_pixel: bool, set to True to consider only the label of the
                          center pixel
            data_augmentation: bool, set to True to perform random flips
            supervision: 'full' or 'semi' supervised algorithms
        """
        super(R_HyperX, self).__init__()
        self.data = data
        self.label = gt
        self.name = hyperparams['dataset']
        self.patch_size = hyperparams['patch_size']
        self.ignored_labels = set(hyperparams['ignored_labels'])
        self.flip_augmentation = hyperparams['flip_augmentation']
        self.radiation_augmentation = hyperparams['radiation_augmentation']
        self.mixture_augmentation = hyperparams['mixture_augmentation']
        self.center_pixel = hyperparams['center_pixel']
        self.train_sample_extend = hyperparams['train_sample_extend']  # 样本扩展参数
        supervision = hyperparams['supervision']
        # Fully supervised : use all pixels with label not ignored
        if supervision == 'full':
            mask = np.ones_like(gt)
            for l in self.ignored_labels:
                mask[gt == l] = 0
        # Semi-supervised : use all pixels, except padding
        elif supervision == 'semi':
            mask = np.ones_like(gt)  # 返回一个和gt一样大小的全1矩阵

        elif supervision == 'simclr':
            mask = np.ones_like(gt)
            for l in self.ignored_labels:
                mask[gt == l] = 0

        x_pos, y_pos = np.nonzero(mask)  # 返回mask中非零值得索引值 即pix在图中的坐标
        p = self.patch_size // 2  # // 是向下取整的除法
        # z = []
        # sizes = x_pos.size
        # for i in range(1, sizes):
        #     z.append(i)

        temp_indices = np.array([(x, y) for x, y in zip(x_pos, y_pos) if  # pix的坐标位置
                                 x > p and x < data.shape[0] - p and y > p and y < data.shape[1] - p])

        self.indices = []
        for item_x, item_y in temp_indices:
            newitem = [item_x, item_y, 0]
            # self.indices.append(newitem)
            # newitem = [item_x, item_y, 1]
            # self.indices.append(newitem)
            # newitem = [item_x, item_y, 2]
            self.indices.append(newitem)

            # if self.train_sample_extend is True:
            #     #self.indices = []
            #     self.labels = []
            #     for j in range(1, 4):
            #         z = np.ones_like(x_pos)*j
            #         indices_ = np.array([(x, y, z) for x, y, z in zip(x_pos, y_pos, z) if    # pix的坐标位置 以及 是否进行
            #                                 x > p and x < data.shape[0] - p and y > p and y < data.shape[1] - p])  # 对x,y坐标的限制
            #         self.indices.append(indices_)

            self.labels = [self.label[x, y] for x, y, z in self.indices]
            # self.labels.append(labels)
            # self.labels = [self.label[x, y] for x, y in self.indices]
            np.random.shuffle(self.indices)  # 此处已经打乱了坐标

    @staticmethod
    def flip(*arrays, z):  # 数据增强 之 翻转
        # horizontal = np.random.random() > 0.5   # 50% 几率
        # vertical = np.random.random() > 0.5

        if z is 1:
            arrays = [np.fliplr(arr) for arr in arrays]  # 将数组在左右翻转
        if z is 2:
            arrays = [np.flipud(arr) for arr in arrays]  # 上下翻转
        return arrays

    @staticmethod
    def radiation_noise(data, alpha_range=(0.9, 1.1), beta=1 / 25):
        alpha = np.random.uniform(*alpha_range)
        noise = np.random.normal(loc=0., scale=1.0, size=data.shape)
        return alpha * data + beta * noise

    def mixture_noise(self, data, label, beta=1 / 25):
        alpha1, alpha2 = np.random.uniform(0.01, 1., size=2)
        noise = np.random.normal(loc=0., scale=1.0, size=data.shape)
        data2 = np.zeros_like(data)
        for idx, value in np.ndenumerate(label):
            if value not in self.ignored_labels:
                l_indices = np.nonzero(self.labels == value)[0]
                l_indice = np.random.choice(l_indices)
                assert (self.labels[l_indice] == value)
                x, y = self.indices[l_indice]
                data2[idx] = self.data[x, y]
        return (alpha1 * data + alpha2 * data2) / (alpha1 + alpha2) + beta * noise

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        x, y, z = self.indices[i]  # 先获取坐标
        x1, y1 = x - self.patch_size // 2, y - self.patch_size // 2  # 计算出目标window
        x2, y2 = x1 + self.patch_size, y1 + self.patch_size
        data = self.data[x1:x2, y1:y2]  # 读取window中的数据和类别23*23
        # print(data.shape)
        label = self.label[x1:x2, y1:y2]  # 21*21
        # print(data.shape)  # data.shape(21,21,30),在此处剥离通道，写个函数
        # print(type(data))  # array

        if self.radiation_augmentation and np.random.random() < 0.1:  # 10%的概率加噪声
            data = self.radiation_noise(data)
        if self.mixture_augmentation and np.random.random() < 0.2:  # 20%的概率加mixture噪声
            data = self.mixture_noise(data, label)
        # if z // 3 is 1:
        #     data, label = self.flip(data, label, z=1)  # 左右翻转
        # if z // 3 is 2:
        #     data, label = self.flip(data, label, z=2)  # 上下翻转
        # if self.flip_augmentation and self.patch_size > 1:      # 数据增强 之 翻转
        #     # Perform data augmentation (only on 2D patches)
        #     data, label = self.flip(data, label)
        # ***Copy the data into numpy arrays (PyTorch doesn't like numpy views)***
        retrain_transform = T.Compose([
                                       T.RandomEnlarge(prob=1, min_clip_rate=[0.95, 0.95]),
                                       T.RandomFlip(prob=0.5, direction='Horizontal'),
                                       T.RandomColor(prob=0.8, alpha_range=[0.8, 1.2], beta_range=[0, 500],
                                              band_num=data.shape[-1]),  # 注释了
                                       T.RandomBlur(prob=0.9, ksize=3, band_num=data.shape[2]),  # ksize=1/k 注释了
                                       # T.Normalize(mean=([0] * data.shape[-1]), std=([1] * data.shape[-1]), bit_num=16,
                                       #             band_num=data.shape[-1])
                                     ])
        # data = retrain_transform(data)
        data = np.asarray(np.copy(data).transpose((2, 0, 1)))
        label = np.asarray(np.copy(label), dtype='int64')
        # ***Load the data into PyTorch tensors***
        data = torch.from_numpy(data)  # type = tensor
        label = torch.from_numpy(label)
        # ***augmention***
        # Extract the center label if needed
        if self.center_pixel and self.patch_size > 1:
            label = label[self.patch_size // 2, self.patch_size // 2]  # 确定window中心点pix的label
        # Remove unused dimensions when we work with invidual spectrums
        elif self.patch_size == 1:
            data = data[:, 0, 0]
            # data2 = data2[:, 0, 0]
            label = label[0, 0]

        # Add a fourth dimension for 3D CNN
        # if self.patch_size > 1:
        # Make 4D data ((Batch x) Planes x Channels x Width x Height)
        # data = data.unsqueeze(0)        # 在第(0)个维度的位置加一个维度
        return data, label



