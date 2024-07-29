# Python 2/3 compatiblity
from __future__ import print_function
from __future__ import division

# Torch
import torch
import torch.utils.data as data
from torchsummary import summary

# Numpy, scipy, scikit-image, spectral
import numpy as np
# Visualization
import seaborn as sns
import visdom       # 可视化工具
from utils import metrics, convert_to_color_, convert_from_color_,\
    display_dataset, display_predictions, explore_spectrums, plot_spectrums, plot_spectrums_, \
    sample_gt, build_dataset, show_results, compute_imf_weights, get_device, camel_to_snake, open_file, get_train
from all_datasets import get_dataset, DATASETS_CONFIG, R_HyperX
from finetune import get_model, test, new_train
import argparse

dataset_names = [v['name'] if 'name' in v.keys() else k for k, v in DATASETS_CONFIG.items()]
# Argument parser for CLI interaction
parser = argparse.ArgumentParser(description="Run deep learning experiments on"
                                             " various hyperspectral datasets")
parser.add_argument('--dataset', type=str, default='PaviaU', choices=dataset_names,
                    help="Dataset to use: IndianPines; PaviaU; Salinas;Houston.HongHu")  # paviau
parser.add_argument('--newmodel', type=str, default="NEWMODEL",
                    help="Model to train. Available:\n" "NEWMODEL")
parser.add_argument('--folder', type=str, help="Folder where to store the "
                    "datasets (defaults to the current working directory).",
                    default="./Datasets/")
parser.add_argument('--cuda', type=int, default=0,
                    help="Specify CUDA device (defaults to -1, which learns on CPU)")
parser.add_argument('--runs', type=int, default=1, help="Number of runs (default: 1)")
parser.add_argument('--restore', type=str,  # default="./checkpoints/model/Salinas/Salinas_CBW.pth",
                    help="Weights to use for initialization, e.g. a checkpoint")

# Dataset options
group_dataset = parser.add_argument_group('Dataset')
group_dataset.add_argument('--training_sample', type=float, default=5,           # rate of training samples
                    help="percentage of samples to use for training (0.-1.)(default: 0.5)")
group_dataset.add_argument('--sampling_mode', type=str, help="Sampling mode (random sampling or disjoint, default:"
                                                             " random)", default='fixed')  # random fixed

group_dataset.add_argument('--train_set', type=str, default=None, help="Path to the train ground truth (optional, this "
                                                                       "supersedes the --sampling_mode option)")
group_dataset.add_argument('--test_set', type=str, default=None,
                           help="Path to the test set (optional, by default the test_set is the entire ground truth"
                                " minus the training)")

# Training options
group_train = parser.add_argument_group('Training')
group_train.add_argument('--epoch', type=int, default=500, help="Training epochs optional, if"
                         " absent will be set by the model)")  # ori 500
group_train.add_argument('--patch_size', type=int, default=21, help="Size of the spatial neighbourhood (optional, if "
                                                                    "absent will be set by the model)")
group_train.add_argument('--lr', type=float, default=0.0001,
                         help="Learning rate, set by the model if not specified.")  # before 0.05
group_train.add_argument('--class_balancing', action='store_true', default=True,
                         help="Inverse median frequency class balancing (default = False)")
group_train.add_argument('--batch_size', type=int, default=10,  # 64
                         help="Batch size (optional, if absent will be set by the model")  # (5:16 10:32,30:128)256
group_train.add_argument('--test_stride', type=int, default=1,
                         help="Sliding window step stride during inference (default = 1)")
# Data augmentation parameters
group_da = parser.add_argument_group('Data augmentation')
group_da.add_argument('--flip_augmentation', action='store_true', default=False,     # 翻转
                        help="Random flips (if patch_size > 1)")
group_da.add_argument('--radiation_augmentation', action='store_true', default=False,    # 加噪声
                        help="Random radiation noise (illumination)")
group_da.add_argument('--mixture_augmentation', action='store_true', default=False,
                        help="Random mixes between spectra")

parser.add_argument('--with_exploration', action='store_true', default=False,
                    help="See data exploration visualization")
parser.add_argument('--download', type=str, default=None, nargs='+',
                    choices=dataset_names,
                    help="Download the specified datasets and quits.")
parser.add_argument('--train_sample_extend', type=bool, default=False,
                    help="train sample extended by flip.")

args = parser.parse_args()

CUDA_DEVICE = get_device(args.cuda)

# % of training samples
SAMPLE_PERCENTAGE = args.training_sample
# Data augmentation ?
FLIP_AUGMENTATION = args.flip_augmentation
RADIATION_AUGMENTATION = args.radiation_augmentation
MIXTURE_AUGMENTATION = args.mixture_augmentation
# Dataset name
DATASET = args.dataset
# Model name
NEWMODEL = args.newmodel
# Number of runs (for cross-validation)
N_RUNS = args.runs
# Spatial context size (number of neighbours in each spatial direction)
PATCH_SIZE = args.patch_size
# Add some visualization of the spectra ?
DATAVIZ = args.with_exploration
# Target folder to store/download/load the datasets
FOLDER = args.folder
# Number of epochs to run
EPOCH = args.epoch
# Sampling mode, e.g random sampling
SAMPLING_MODE = args.sampling_mode
# Pre-computed weights to restore
CHECKPOINT = args.restore
# Learning rate for the SGD
LEARNING_RATE = args.lr
# Automated class balancing
CLASS_BALANCING = args.class_balancing
# Training ground truth file
TRAIN_GT = args.train_set
# Testing ground truth file
TEST_GT = args.test_set
TEST_STRIDE = args.test_stride
# Training sample extended by flip
TRAIN_SAMPLE_EXTEND = args.train_sample_extend

if args.download is not None and len(args.download) > 0:    # 下载数据集
    for dataset in args.download:
        get_dataset(dataset, target_folder=FOLDER, patch_size=PATCH_SIZE)
    quit()
viz = visdom.Visdom(env=DATASET + ' ' + NEWMODEL)
if not viz.check_connection:
    print("Visdom is not connected. Did you run 'python -m visdom.server' ?")

hyperparams = vars(args)    # 返回对象object的属性和属性值的字典对象（超参数）
# Load the dataset
# img, gt, LABEL_VALUES, IGNORED_LABELS, RGB_BANDS, palette = get_dataset(DATASET, FOLDER, patch_size=PATCH_SIZE)
img, gt, LABEL_VALUES, IGNORED_LABELS, RGB_BANDS, palette = get_dataset(DATASET, FOLDER,
                                                                            patch_size=PATCH_SIZE)
# IGNORED_LABELS就是 undefined类别
# Number of classes
N_CLASSES = len(LABEL_VALUES)
# Number of bands (last dimension of the image tensor)
N_BANDS = img.shape[-1]

# 生成数据集的伪三色图
if palette is None:
    # Generate color palette
    palette = {0: (0, 0, 0)}
    for k, color in enumerate(sns.color_palette("hls", len(LABEL_VALUES) - 1)):
        palette[k + 1] = tuple(np.asarray(255 * np.array(color), dtype='uint8'))
invert_palette = {v: k for k, v in palette.items()}


def convert_to_color(x):
    return convert_to_color_(x, palette=palette)


def convert_from_color(x):
    return convert_from_color_(x, palette=invert_palette)


# Instantiate the experiment based on predefined networks
hyperparams.update({'n_classes': N_CLASSES, 'n_bands': N_BANDS, 'ignored_labels': IGNORED_LABELS, 'device': CUDA_DEVICE})  # 更新了部分参数
hyperparams = dict((k, v) for k, v in hyperparams.items() if v is not None)

# Show the image and the ground truth
display_dataset(img, gt, RGB_BANDS, LABEL_VALUES, palette, viz)   # 显示hsi图,填充后的图片
# display_dataset(ori_img, ori_gt, RGB_BANDS, LABEL_VALUES, palette, viz)     # 显示原始hsi图与ground truth
color_gt = convert_to_color(gt)     # 显示ground truth

if DATAVIZ:     # 数据评价波段可视化
    # Data exploration : compute and show the mean spectrums
    mean_spectrums, std_spectrums = explore_spectrums(img, gt, LABEL_VALUES, viz, ignored_labels=IGNORED_LABELS)
    # for i in range(15):
    # plt.plot(range(204), mean_spectrums.values())
    # plt.shown
    plot_spectrums(mean_spectrums, viz, title='Mean spectrum/class')
    plot_spectrums_(std_spectrums, viz, title='Std spectrum/class')


results = []
gt_ = gt[(PATCH_SIZE // 2):-(PATCH_SIZE // 2),
      (PATCH_SIZE // 2):-(PATCH_SIZE // 2)]  # 这里是为了还原原来的gt尺寸，以作后续的计算做准备***
# print(gt_.shape)

for run in range(N_RUNS):
    train_gt, test_gt = sample_gt(gt_, SAMPLE_PERCENTAGE, mode=SAMPLING_MODE)   # 获取样本训练集和测试集
    # ---------------------------------------只在原gt上进行training样本采样-------------------------------------------
    mask = np.zeros_like(gt)
    for l in set(hyperparams['ignored_labels']):  # 未标记的像素样本
        mask[gt == l] = 0
    x_pos, y_pos = np.nonzero(train_gt)  # 读取train_gt中的非0值元素坐标
    indices = np.array([(x, y) for x, y in zip(x_pos, y_pos)])      # 提取train样本的坐标
    for x, y in indices:
        if mask[x+PATCH_SIZE//2, y+PATCH_SIZE//2] is not 0:  # gt位置矫正
            mask[x+PATCH_SIZE//2, y+PATCH_SIZE//2] = gt[x+PATCH_SIZE//2, y+PATCH_SIZE//2]
    train_gt = mask

    print("{} training samples and {} testing samples selected (over {})".format(np.count_nonzero(train_gt), np.count_nonzero(test_gt), np.count_nonzero(gt_)))
    print("Running an experiment with the {} model".format(NEWMODEL),
          "run {}/{}".format(run + 1, N_RUNS))
    '''加载权重'''
    #！！！！此部分为新增加部分，训练完毕之后将模型参数导入
    model, optimizer, loss, hyperparams = get_model(NEWMODEL, **hyperparams)  # 始终未完全收敛
    new_model = model  # 新模型

    pretrained_dict = torch.load('checkpoints/model/PaviaU/_epoch500_1.00.pth')  # model/other_model1/
    new_model_dict = new_model.state_dict()

    num = 0
    for key in pretrained_dict.keys():
        # for k in pre_trained_dict[key].keys():
        if key in new_model_dict.keys():  # todo
            num = num + 1
    print("The number of total keys is {},and {} common keys have been loaded!".format(len(new_model_dict.keys()), num))

    state_dict = {k: v for k, v in pretrained_dict.items() if k in new_model_dict}  # 剔除不同的参数

    new_model_dict.update(state_dict)  # 更新有的参数
    new_model.load_state_dict(new_model_dict)  # 此时参数已经加载到新模型中，成功导入！linear参数重新训练
    model = new_model
    retrain_dataset = R_HyperX(img, train_gt, **hyperparams)  # 数据生成器
    retrain_loader = data.DataLoader(retrain_dataset, batch_size=hyperparams['batch_size'], pin_memory=hyperparams['device'], shuffle=True, drop_last=False)

    try:
        # 进行模型训练
        new_train(model, optimizer, loss, retrain_loader, hyperparams['epoch'],
                  scheduler=hyperparams['scheduler'], device=hyperparams['device'],
                  supervision=hyperparams['supervision'],  # val_loader=val_loader,  # 注释了val集
                  display=viz)
    except KeyboardInterrupt:
        pass

    probabilities = test(model, img, hyperparams)   # 进行模型测试,传入新的model img扩展后的img
    prediction = np.argmax(probabilities, axis=-1)  # 得到预测结果

    prediction = prediction[(PATCH_SIZE // 2):-(PATCH_SIZE // 2), (PATCH_SIZE // 2):-(PATCH_SIZE // 2)]
    # 这里是为了还原原来的img尺寸，以作后续的计算
    # test_gt = test_gt[(PATCH_SIZE // 2):-(PATCH_SIZE // 2), (PATCH_SIZE // 2):-(PATCH_SIZE // 2)]
    # 这里是为了还原原来的test_gt尺寸，以作后续的计算***
    gt_f = gt[(PATCH_SIZE // 2):-(PATCH_SIZE // 2), (PATCH_SIZE // 2):-(PATCH_SIZE // 2)]
    # 这里是为了还原原来的gt尺寸，以作后续的计算***

    # 结构统计与分析
    run_results = metrics(prediction, test_gt, ignored_labels=hyperparams['ignored_labels'], n_classes=N_CLASSES)
    mask = np.zeros(gt_f.shape, dtype='bool')
    for l in IGNORED_LABELS:
        mask[gt_f == l] = True
    prediction[mask] = 0    # 阴影（未标记）像素的类别标记
    # 展示分类效果图
    color_prediction = convert_to_color(prediction)
    display_predictions(color_prediction, viz, gt=convert_to_color(gt_f), caption="Prediction vs. ground truth")

    results.append(run_results)
    show_results(run_results, viz, label_values=LABEL_VALUES)   # 显示

if N_RUNS > 1:
    show_results(results, viz, label_values=LABEL_VALUES, agregated=True)

