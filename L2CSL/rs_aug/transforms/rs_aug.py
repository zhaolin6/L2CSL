import os
import numpy as np
import cv2
import random
import math
from PIL import Image
from collections import OrderedDict
from functools import reduce
from rs_aug.transforms import utils as uts


# ----- compose -----
class Compose:
    """ 
    根据数据增强算子对输入数据进行操作
    所有操作的输入图像流形状均是 [H, W, C]，其中H为图像高，W为图像宽，C为图像通道数
    Args:
        transforms (list): 数据增强算子
    """
    def __init__(self, transforms):
        if not isinstance(transforms, list):
            raise TypeError('The transforms must be a list!')
        if len(transforms) < 1:
            raise ValueError('The length of transforms ' + \
                             'must be equal or larger than 1!')
        self.transforms = transforms
    def __call__(self, img, img_info=None, label=None):
        """
        Args:
            img (str): 图像路径 (.tif/.img/.npy)
            img_info (dict): 存储与图像相关的信息，dict中的字段如下:
                - shape_before_resize (tuple): 图像resize之前的大小 (h, w)
                - shape_before_padding (tuple): 图像padding之前的大小 (h, w)
            label (str): 标注图像路径 (.png)
        """
        if img_info is None:
            img_info = dict()
        # img = uts.read_img(img)
        img = img
        if img is None:
            raise ValueError('Can\'t read The image file {}!'.format(img))
        if label is not None:
            label = uts.read_img(label, True)
        # 数据增强
        for op in self.transforms:
            outputs = op(img, img_info, label)
            img = outputs[0]
            if len(outputs) >= 2:
                img_info = outputs[1]
            if len(outputs) == 3:
                label = outputs[2]
        return outputs


# ----- transforms -----
class Resize:
    """
    调整图像和标注图大小
    Args:
        target_size (int/list/tuple): 目标大小
        interp (str): 插值方式，可选参数为 ['NEAREST', 'LINEAR', 'CUBIC', 'AREA', 'LANCZOS4']，默认为'NEAREST'
    """
    interp_dict = {
        'NEAREST': cv2.INTER_NEAREST,
        'LINEAR': cv2.INTER_LINEAR,
        'CUBIC': cv2.INTER_CUBIC,
        'AREA': cv2.INTER_AREA,
        'LANCZOS4': cv2.INTER_LANCZOS4
    }
    def __init__(self, target_size, interp='NEAREST'):
        self.target_size = target_size
        self.interp = interp
        assert interp in self.interp_dict, 'interp should be one of {}.'.format(self.interp_dict.keys())
        if isinstance(target_size, list) or isinstance(target_size, tuple):
            if len(target_size) != 2:
                raise ValueError(
                    'when target is list or tuple, it should include 2 elements, but it is {}.'
                    .format(target_size))
        elif not isinstance(target_size, int):
            raise TypeError(
                'Type of target_size is invalid. Must be Integer or List or tuple, now is {}.'
                .format(type(target_size)))
    def __call__(self, img, img_info=None, label=None):
        if img_info is None:
            img_info = OrderedDict()
        img_info['shape_before_resize'] = img.shape[:2]
        if not isinstance(img, np.ndarray):
            raise TypeError("ResizeImage: image type is not np.ndarray.")
        if len(img.shape) != 3:
            raise ValueError('ResizeImage: image is not 3-dimensional.')
        img_shape = img.shape
        img_size_min = np.min(img_shape[0:2])
        # img_size_max = np.max(img_shape[0:2])
        if float(img_size_min) == 0:
            raise ZeroDivisionError('ResizeImage: min size of image is 0.')
        if isinstance(self.target_size, int):
            resize_w = self.target_size
            resize_h = self.target_size
        else:
            resize_w = self.target_size[0]
            resize_h = self.target_size[1]
        img_scale_x = float(resize_w) / float(img_shape[1])
        img_scale_y = float(resize_h) / float(img_shape[0])
        img = cv2.resize(
            img,
            None,
            None,
            fx=img_scale_x,
            fy=img_scale_y,
            interpolation=self.interp_dict[self.interp])
        if label is not None:
            label = cv2.resize(
                label,
                None,
                None,
                fx=img_scale_x,
                fy=img_scale_y,
                interpolation=self.interp_dict['NEAREST'])
        if label is None:
            return (img, img_info)
        else:
            return (img, img_info, label)


class Normalize:
    """
    对图像进行标准化
        1.图像像素归一化到区间 [0.0, 1.0]
        2.对图像进行减均值除以标准差操作
    Args:
        mean (list): 图像数据集的均值列表，有多少波段需要多少个元素
        std (list): 图像数据集的标准差列表，有多少波段需要多少个元素
        bit_num (int): 图像的位数，默认为8
        band_num (int): 操作的波段数，默认为7
    """
    def __init__(self, mean, std, bit_num=8, band_num=7):
        self.mean = mean
        self.std = std
        self.band_num = band_num
        self.min_val = [0] * band_num
        self.max_val = [(2**bit_num)-1] * band_num
        if bit_num not in [8, 16, 24]:
            raise ValueError('{} is not effective bit_num, bit_num should be one of 8, 16, 24.'
                             .format(bit_num))
        if band_num != len(self.mean):
            raise ValueError('band_num should be equal to len of mean/std.')
        if not (isinstance(self.mean, list) and isinstance(self.std, list)):
            raise ValueError('{}: input type is invalid.'.format(self))
        if reduce(lambda x, y: x * y, self.std) == 0:
            raise ValueError('{}: std is invalid!'.format(self))

    def __call__(self, img, img_info=None, label=None):
        mean = np.array(self.mean)[np.newaxis, np.newaxis, :]
        std = np.array(self.std)[np.newaxis, np.newaxis, :]
        img = uts.normalize(img, self.min_val, self.max_val, mean, std, self.band_num)
        if label is None:
            return (img, img_info)
        else:
            return (img, img_info, label)


class RandomFlip:
    """
    对图像和标注图进行翻转
    Args:
        prob (float): 随机翻转的概率。默认值为0.5
        direction (str): 翻转方向，可选参数为 ['Horizontal', 'Vertical', 'Both']，默认为'Both'
    """
    flips_list = ['Horizontal', 'Vertical', 'Both']

    def __init__(self, prob=0.5, direction='Both'):
        self.prob = prob
        self.direction = direction  # 挑选模式
        assert direction in self.flips_list, 'direction should be one of {}.'.format(self.flips_list)
        if prob < 0 or prob > 1:
            raise ValueError('prob should be between 0 and 1.')

    def __call__(self, img, img_info=None, label=None):
        if random.random() < self.prob:
            img = uts.mode_flip(img, self.direction)
            if label is not None:
                label = uts.mode_flip(label, self.direction)
        if label is None:
            return (img, img_info)
        else:
            return (img, img_info, label)


class RandomRotate:
    """
    对图像和标注图进行随机1-89度旋转，保持图像大小
    Args:
        prob (float): 选择的概率。默认值为0.5
        ig_pix (int): 标签旋转后周围填充的忽略值，默认为255
    """
    def __init__(self, prob=0.5, ig_pix=255):
        self.prob = prob
        self.ig_pix = ig_pix
        if prob < 0 or prob > 1:
            raise ValueError('prob should be between 0 and 1.')

    def __call__(self, img, img_info=None, label=None):
        ang = random.randint(1, 89)
        if random.random() < self.prob:
            img = uts.rotate_img(img, ang)
            if label is not None:
                label = uts.rotate_img(label, ang, ig_pix=self.ig_pix)
        if label is None:
            return (img, img_info)
        else:
            return (img, img_info, label)


class RandomEnlarge:
    """
    对图像和标注图进行随机裁剪，然后拉伸到到原来的大小 (局部放大)
    Args:
        prob (float): 裁剪的概率。默认值为0.5
        min_clip_rate (list/tuple): 裁剪图像行列占原图大小的最小倍率。默认为 [0.5, 0.5]
    """
    def __init__(self, prob=0.5, min_clip_rate=[0.5, 0.5]):
        self.prob = prob
        self.min_clip_rate = list(min_clip_rate)
        if prob < 0 or prob > 1:
            raise ValueError('prob should be between 0 and 1.')
        if isinstance(min_clip_rate, list) or isinstance(min_clip_rate, tuple):
            if len(min_clip_rate) != 2:
                raise ValueError(
                    'when min_clip_rate is list or tuple, it should include 2 elements, but it is {}.'
                    .format(min_clip_rate))

    def __call__(self, img, img_info=None, label=None):
        h, w = img.shape[:2]
        h_clip = math.floor(self.min_clip_rate[0] * h)  # 向下取整
        w_clip = math.floor(self.min_clip_rate[1] * w)
        x = random.randint(0, (w - w_clip))  # 生成一个指定范围内的整数
        y = random.randint(0, (h - h_clip))
        if random.random() < self.prob:
            img = uts.enlarge_img(img, x, y, h_clip, w_clip)
            if label is not None:
                label = uts.enlarge_img(label, x, y, h_clip, w_clip)
        if label is None:
            return (img, img_info)
        else:
            return (img, img_info, label)
        # return (img, img_info, label)


class RandomNarrow:
    """
    对图像和标注图进行随机缩小，然后填充到到原来的大小
    Args:
        prob (float): 缩小的概率。默认值为0.5
        min_size_rate (list/tuple): 缩小图像行列为原图大小的倍率。默认为 [0.5, 0.5]
        ig_pix (int): 标签缩小后周围填充的忽略值，默认为255
    """
    def __init__(self, prob=0.5, min_size_rate=[0.5, 0.5], ig_pix=255):
        self.prob = prob
        self.min_size_rate = list(min_size_rate)
        self.ig_pix = ig_pix
        if prob < 0 or prob > 1:
            raise ValueError('prob should be between 0 and 1.')
        if isinstance(min_size_rate, list) or isinstance(min_size_rate, tuple):
            if len(min_size_rate) != 2:
                raise ValueError(
                    'when min_size_rate is list or tuple, it should include 2 elements, but it is {}.'
                    .format(min_size_rate))
    def __call__(self, img, img_info=None, label=None):
        x_rate = random.uniform(self.min_size_rate[0], 1)
        y_rate = random.uniform(self.min_size_rate[1], 1)
        if random.random() < self.prob:
            img = uts.narrow_img(img, x_rate, y_rate)
            if label is not None:
                label = uts.narrow_img(label, x_rate, y_rate, ig_pix=self.ig_pix)
        if label is None:
            return (img, img_info)
        else:
            return (img, img_info, label)
        return (img, img_info, label)


class RandomBlur:
    """
    对图像进行高斯模糊
    Args：
        prob (float): 图像模糊概率。默认为0.1
        ksize (int): 高斯核大小，默认为3
        band_num (int): 操作的波段数，默认为7
    """
    def __init__(self, prob=0.1, ksize=3, band_num=7):
        self.prob = prob
        self.ksize = ksize
        self.band_num = band_num
        if prob < 0 or prob > 1:
            raise ValueError('prob should be between 0 and 1.')

    def __call__(self, img, img_info=None, label=None):
        if random.random() < self.prob:
            img[:, :, :self.band_num] = cv2.GaussianBlur(img[:, :, :self.band_num], (self.ksize, self.ksize), 0)
        if label is None:
            return (img, img_info)
        else:
            return (img, img_info, label)
        # return (img, img_info, label)


class RandomSharpening:
    """
    对图像进行锐化
    Args：
        prob (float): 图像锐化概率。默认为0.1
        laplacian_mode (str): 拉普拉斯算子类型，可选参数为 ['4-1', '8-1', '4-2']，默认为'8-1'
        band_num (int): 操作的波段数，默认为7
    """
    laplacian_dict = {
        '4-1': np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], np.float32),
        '8-1': np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], np.float32),
        '4-2': np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
    }

    def __init__(self, prob=0.1, laplacian_mode='8-1', band_num=7):
        self.prob = prob
        self.band_num = band_num
        self.kernel = self.laplacian_dict[laplacian_mode]
        assert laplacian_mode in self.laplacian_dict,  \
               'laplacian_mode should be one of {}.'.format(self.laplacian_dict.keys())
        if prob < 0 or prob > 1:
            raise ValueError('prob should be between 0 and 1.')

    def __call__(self, img, img_info=None, label=None):
        if random.random() < self.prob:
            img[:, :, :self.band_num] += cv2.filter2D(img[:, :, :self.band_num], -1, kernel=self.kernel)
        if label is None:
            return (img, img_info)
        else:
            return (img, img_info, label)
        return (img, img_info, label)


class RandomColor:
    """
    对图像随机进行对比度及亮度的小范围增减
    Args：
        prob (float): 改变概率。默认为0.5
        alpha_range (list/tuple): 图像对比度调节范围，默认为 [0.8, 1.2]
        beta_range (list/tuple): 图像亮度调节范围，默认为 [-10, 10]
        band_num (int): 操作的波段数，默认为7
    """
    def __init__(self, prob=0.5, alpha_range=[0.8, 1.2], beta_range=[-10, 10], band_num=7):  # 初值定义
        self.prob = prob
        self.alpha_range = list(alpha_range)
        self.beta_range = list(beta_range)
        self.band_num = band_num
        if prob < 0 or prob > 1:
            raise ValueError('prob should be between 0 and 1.')
        if isinstance(alpha_range, list) or isinstance(alpha_range, tuple):
            if len(alpha_range) != 2:
                raise ValueError(
                    'when alpha_range is list or tuple, it should include 2 elements, but it is {}.'
                    .format(alpha_range))
        if isinstance(beta_range, list) or isinstance(beta_range, tuple):
            if len(beta_range) != 2:
                raise ValueError(
                    'when beta_range is list or tuple, it should include 2 elements, but it is {}.'
                    .format(beta_range))

    def __call__(self, img, img_info=None, label=None):
        if random.random() < self.prob:
            alpha = random.uniform(self.alpha_range[0], self.alpha_range[1])  # 生成随机的小数，跟我之前写的函数差不多
            beta = random.uniform(self.beta_range[0], self.beta_range[1])
            img[:, :, :self.band_num] = alpha * img[:, :, :self.band_num] + beta
        if label is None:
            return (img, img_info)
        else:
            return (img, img_info, label)
        # return (img, img_info, label)


class RandomStrip:
    """
    对图像随机加上条带噪声
    Args：
        prob (float): 加上条带噪声的概率。默认为0.5
        strip_rate (float): 条带占比，默认0.05
        direction (str): 条带方向，可选参数 ['Horizontal', 'Vertical'],，默认'Horizontal'
        band_num (int): 操作的波段数，默认为7
    """
    strip_list = ['Horizontal', 'Vertical']
    def __init__(self, prob=0.5, strip_rate=0.05, direction='Horizontal', band_num=7):
        self.prob = prob
        self.strip_rate = strip_rate
        self.direction = direction
        self.band_num = band_num
        assert direction in self.strip_list, 'direction should be one of {}.'.format(self.strip_list)
        if prob < 0 or prob > 1:
            raise ValueError('prob should be between 0 and 1.')
        if strip_rate < 0 or strip_rate > 1:
            raise ValueError('strip_rate should be between 0 and 1.')
    def __call__(self, img, img_info=None, label=None):
        h, w = img.shape[:2]
        if random.random() < self.prob:
            strip_num = self.strip_rate * (h if self.direction == 'Horizontal' else w)
            img = uts.random_strip(img, strip_num, self.direction, self.band_num)
        if label is None:
            return (img, img_info)
        else:
            return (img, img_info, label)
        return (img, img_info, label)


class RandomFog:
    """
    对图像随机加上雾效果
    Args：
        prob (float): 加上雾效果的概率。默认为0.5
        fog_range (list/tuple): 雾的大小范围，范围在0-1之间，默认为 [0.03, 0.28]
        band_num (int): 操作的波段数，默认为7
    """
    def __init__(self, prob=0.5, fog_range=[0.03, 0.28], band_num=7):
        self.prob = prob
        self.fog_range = fog_range
        self.band_num = band_num
        if prob < 0 or prob > 1:
            raise ValueError('prob should be between 0 and 1.')
        if isinstance(fog_range, list) or isinstance(fog_range, tuple):
            if len(fog_range) != 2:
                raise ValueError(
                    'when fog_range is list or tuple, it should include 2 elements, but it is {}.'
                    .format(fog_range))
    def __call__(self, img, img_info=None, label=None):
        if random.random() < self.prob:
            img = uts.add_fog(img, self.fog_range, self.band_num)
        if label is None:
            return (img, img_info)
        else:
            return (img, img_info, label)
        return (img, img_info, label)


class RandomSplicing:
    """
    对图像进行随机划分成两块，并对其中一块改变色彩，营造拼接未匀色的效果
    Args：
        prob (float): 执行此操作的概率。默认为0.1
        direction (str): 分割方向，可选参数 ['Horizontal', 'Vertical'],，默认'Horizontal'
        band_num (int): 操作的波段数，默认为7
    """
    splic_list = ['Horizontal', 'Vertical']
    def __init__(self, prob=0.1, direction='Horizontal', band_num=7):
        self.prob = prob
        self.direction = direction
        self.band_num = band_num
        assert direction in self.splic_list, 'direction should be one of {}.'.format(self.splic_list)
        if prob < 0 or prob > 1:
            raise ValueError('prob should be between 0 and 1.')
    def __call__(self, img, img_info=None, label=None):
        if random.random() < self.prob:
            img = uts.random_splicing(img, self.direction, self.band_num)
        if label is None:
            return (img, img_info)
        else:
            return (img, img_info, label)
        return (img, img_info, label)


class RandomRemoveBand:
    """
    对图像随机置零某个波段
    Args：
        prob (float): 执行此操作的概率。默认为0.1
        kill_bands (list): 必须置零的波段列表，默认为None
        keep_bands (list): 不能置零的波段列表，默认为None
    """
    def __init__(self, prob=0.1, kill_bands=None, keep_bands=None):
        self.prob = prob
        self.kill_bands = [] if kill_bands == None else list(kill_bands)
        self.keep_bands = [] if keep_bands == None else list(keep_bands)
        if prob < 0 or prob > 1:
            raise ValueError('prob should be between 0 and 1.')
        if not(isinstance(kill_bands, list)) and kill_bands != None:
            raise ValueError('kill_bands is list or None.')
        if not(isinstance(keep_bands, list)) and keep_bands != None:
            raise ValueError('keep_bands is list or None.')

    def __call__(self, img, img_info=None, label=None):
        if random.random() < self.prob:
            rand_list = []
            rm_list = []
            c = img.shape[-1]
            for i in range(c):
                if i in self.kill_bands:
                    rm_list.append(i)
                elif i in self.keep_bands:
                    continue
                else:
                    rand_list.append(i)
            rnd = random.choice(rand_list)
            rm_list.append(rnd)
            for j in rm_list:
                img[:, :, j] = 0
        if label is None:
            return (img, img_info)
        else:
            return (img, img_info, label)
        return (img, img_info, label)


class PCA:
    """
    对图像进行PCA变化降维
    Args：
        out_dim (int): 降维后的波段数。默认为3
        keep_bands (list): 不参与降维计算的波段列表，默认为None
    """
    def __init__(self, out_dim=3, keep_bands=None):
        self.out_dim = out_dim
        self.keep_bands = [] if keep_bands == None else list(keep_bands)
        if not(isinstance(keep_bands, list)) and keep_bands != None:
            raise ValueError('keep_bands is list or None.')
    def __call__(self, img, img_info=None, label=None):
        h, w, c = img.shape
        kp_bands = []
        pca_bands = []
        for i in range(c):
            if i in self.keep_bands:
                kp_bands.append(img[:, :, i])
            else:
                pca_bands.append(img[:, :, i])
        kp_bands = np.array(kp_bands).transpose((1, 2, 0))
        pca_bands = np.array(pca_bands).transpose((1, 2, 0))
        out = uts.pca(pca_bands, self.out_dim).reshape((h, w, self.out_dim))
        img = np.concatenate((out, kp_bands), axis=-1)
        if label is None:
            return (img, img_info)
        else:
            return (img, img_info, label)
        return (img, img_info, label)


class NDVI:
    """
    对图像计算NDVI (归一化植被指数)并添加在新的通道中
    Args：
        r_band (int): 红波段序号，默认为landsat TM的第三波段
        nir_band (int): 近红外波段序号，默认为landsat TM的第四波段
    """
    def __init__(self, r_band=2, nir_band=3):
        self.r_band = r_band
        self.nir_band = nir_band

    def __call__(self, img, img_info=None, label=None):
        img = uts.band_comput(img, self.nir_band, self.r_band)
        if label is None:
            return (img, img_info)
        else:
            return (img, img_info, label)
        return (img, img_info, label)


class NDWI:
    """
    对图像计算NDWI (归一化水体指数)并添加在新的通道中
    Args：
        g_band (int): 绿波段序号，默认为landsat TM的第二波段
        nir_band (int): 近红外波段序号，默认为landsat TM的第四波段
    """
    def __init__(self, g_band=1, nir_band=3):
        self.g_band = g_band
        self.nir_band = nir_band

    def __call__(self, img, img_info=None, label=None):
        img = uts.band_comput(img, self.g_band, self.nir_band)
        if label is None:
            return (img, img_info)
        else:
            return (img, img_info, label)
        return (img, img_info, label)


class NDBI:
    """
    对图像计算NDBI (归一化建筑指数)并添加在新的通道中
    Args：
        nir_band (int): 近红外波段序号，默认为landsat TM的第四波段
        mir_band (int): 中红外波段序号，默认为landsat TM的第五波段
    """
    def __init__(self, nir_band=3, mir_band=4):
        self.nir_band = nir_band
        self.mir_band = mir_band

    def __call__(self, img, img_info=None, label=None):
        img = uts.band_comput(img, self.mir_band, self.nir_band)
        if label is None:
            return (img, img_info)
        else:
            return (img, img_info, label)

        return (img, img_info, label)