from __future__ import print_function
from __future__ import division
# Torch
import torch
import torch.utils.data as data
from torchsummary import summary
from tqdm import tqdm
import torch.optim as optim
import numpy as np
import seaborn as sns
import visdom       # 可视化工具
from utils import metrics, convert_to_color_, convert_from_color_,\
    display_dataset, display_predictions, explore_spectrums, plot_spectrums, plot_spectrums_, \
    sample_gt, build_dataset, show_results, compute_imf_weights, get_device, camel_to_snake, open_file
from all_datasets import HyperX, get_dataset, DATASETS_CONFIG
from pretrain import get_model, save_model
import argparse
# import pandas as pd
# from networkx.linalg import adj_matrix
# from skimage.future import graph


def train(dataset_name, net, optimizer, criterion, data_loader, batch_size, epoch, scheduler=None,
          display_iter=100, device=torch.device('cuda'), display=None,
          val_loader=None, supervision='simclr'):  # loss更改为simclr,在后面加一个batch_size
    """
    Training loop to optimize a network for several epochs and a specified loss

    Args:
        net: a PyTorch model
        optimizer: a PyTorch optimizer
        data_loader: a PyTorch dataset loader
        epoch: int specifying the number of training epochs
        criterion: a PyTorch-compatible loss function, e.g. nn.CrossEntropyLoss
        device (optional): torch device to use (defaults to CPU)
        display_iter (optional): number of iterations before refreshing the
        display (False/None to switch off).
        scheduler (optional): PyTorch scheduler
        val_loader (optional): validation dataset
        supervision (optional): 'full' or 'semi','simclr'
    """

    if criterion is None:
        raise Exception("Missing criterion. You must specify a loss function.")

    net.to(device)
    iter_ = 1

    for e in tqdm(range(1, epoch + 1), desc="Training the network"):  # tqdm(list)方法可以传入任意一种list，tqdm进展显示
        # Set the network to training mode
        net.train()
        avg_loss = 0.
        total_num = 0
        # batch_size = 256  # 参数暂时没传入，手动传入！！！！
        if e == 300:      # lr 衰减
            for p in optimizer.param_groups:
                p['lr'] *= 0.5

        for batch_idx, (data1, data2, (x, y, z)) in enumerate(data_loader):  # 寻得批次数据
            # Load the data into the GPU if required
            indices = np.array([(x_pos, y_pos, z_pos) for x_pos, y_pos, z_pos in zip(x, y, z)])
            data1, data2 = data1.to(device), data2.to(device)
            optimizer.zero_grad()

            if supervision == 'simclr':  # 从元组中索
                out1 = net(data1)  # 前向传播
                out2 = net(data2)
                loss = criterion.compute_loss(out1, out2, indices)  # 在loss处加to（device）

            else:
                raise ValueError("supervision mode \"{}\" is unknown.".format(supervision))
            loss.backward()

            optimizer.step()
            total_num += batch_size
            avg_loss += loss.item()*batch_size
            mean_losses = avg_loss/total_num  # 此部分loss为改动版本

            if display_iter and iter_ % display_iter == 0:  # 可视化模块
                string = 'Train (epoch {}/{}) [{}/{} ({:.0f}%)]\tLoss: {:.6f}'
                string = string.format(
                    e, epoch, batch_idx *
                    len(data1), len(data1) * len(data_loader),
                    100. * batch_idx / len(data_loader), mean_losses)

                tqdm.write(string)

            iter_ += 1
            del(data1, data2, (x, y, z), loss)

        avg_loss /= len(data_loader)

        metric = avg_loss
        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(metric)
        elif scheduler is not None:
            scheduler.step()
        if e >= 50 and e % 10 == 0:  # 保存参数模型,改为epoch
            save_model(dataset_name, net, camel_to_snake(str(net.__class__.__name__)), epoch=e, metric=1)



if __name__ == '__main__':

    dataset_names = [v['name'] if 'name' in v.keys() else k for k, v in DATASETS_CONFIG.items()]
    # Argument parser for CLI interaction
    parser = argparse.ArgumentParser(description="Run deep learning experiments on"
                                                 " various hyperspectral datasets")
    parser.add_argument('--dataset', type=str, default='PaviaU', choices=dataset_names,
                        help="Dataset to use: IndianPines; PaviaU; Salinas;Houston.HongHu")  # paviau
    parser.add_argument('--model', type=str, default="CNN",
                        help="Model to train. Available:\n""CNN""CNN_2dMODEL""CNN_3dMODEL")
    parser.add_argument('--folder', type=str, help="Folder where to store the "
                                                   "datasets (defaults to the current working directory).", default="./Datasets/")
    parser.add_argument('--cuda', type=int, default=0,
                        help="Specify CUDA device (defaults to -1, which learns on CPU)")
    parser.add_argument('--restore', type=str,
                        help="Weights to use for initialization, e.g. a checkpoint")  # 参数存储
    group_dataset = parser.add_argument_group('Dataset')
    group_dataset.add_argument('--training_sample', type=float, default=0.8,  # rate of training samples
                               help="percentage of samples to use for training (0.-1.)(default: 0.5)")
    group_dataset.add_argument('--sampling_mode', type=str, help="Sampling mode"
                                                                 " (random sampling or disjoint, default: random)",
                               default='random')
    group_dataset.add_argument('--train_set', type=str, default=None,
                               help="Path to the train ground truth (optional, this "
                                    "supersedes the --sampling_mode option)")
    group_dataset.add_argument('--test_set', type=str, default=None,
                               help="Path to the test set (optional, by default "
                                    "the test_set is the entire ground truth minus the training)")
    # Training options
    group_train = parser.add_argument_group('Training')
    group_train.add_argument('--epoch', type=int, default=500, help="Training epochs optional, if"
                                                                   " absent will be set by the model)")  # 预训练次数选400
    group_train.add_argument('--patch_size', type=int, default=21,
                             help="Size of the spatial neighbourhood (optional, if "
                                  "absent will be set by the model)")
    group_train.add_argument('--lr', type=float, default=0.005,
                             help="Learning rate, set by the model if not specified.")

    group_train.add_argument('--class_balancing', action='store_true', default=True,
                             help="Inverse median frequency class balancing (default = False)")
    group_train.add_argument('--batch_size', type=int, default=512,
                             help="Batch size (optional, if absent will be set by the model")
    group_train.add_argument('--test_stride', type=int, default=1,
                             help="Sliding window step stride during inference (default = 1)")  #
    # Data augmentation parameters
    group_da = parser.add_argument_group('Data augmentation')
    group_da.add_argument('--flip_augmentation', action='store_true', default=False,  # 翻转
                          help="Random flips (if patch_size > 1)")
    group_da.add_argument('--radiation_augmentation', action='store_true', default=False,  # 加噪声
                          help="Random radiation noise (illumination)")
    group_da.add_argument('--mixture_augmentation', action='store_true', default=False,
                          help="Random mixes between spectra")

    parser.add_argument('--with_exploration', action='store_true', default=False,
                        help="See data exploration visualization")  # 显示结果
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
    MODEL = args.model
    # Number of runs (for cross-validation)
    # N_RUNS = args.runs
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

    if args.download is not None and len(args.download) > 0:  # 下载数据集
        for dataset in args.download:
            get_dataset(dataset, target_folder=FOLDER, patch_size=PATCH_SIZE)  # 输出扩展后的img,gt，以及其他一些的参数
        quit()
    viz = visdom.Visdom(env=DATASET + ' ' + MODEL)  # visdom可视化窗口

    if not viz.check_connection:
        print("visdom无连接！Visdom is not connected. Did you run 'python -m visdom.server' ?")

    hyperparams = vars(args)  # 返回对象object的属性和属性值的字典对象（超参数）
    # Load the dataset
    img, gt, LABEL_VALUES, IGNORED_LABELS, RGB_BANDS, palette = get_dataset(DATASET, FOLDER,
                                                                            patch_size=PATCH_SIZE)

    N_CLASSES = len(LABEL_VALUES)
    # Number of bands (last dimension of the image tensor)
    N_BANDS = img.shape[-1]  # pca,c=30
    # 生成数据集的伪三色图
    if palette is None:
        # Generate color palette
        palette = {0: (0, 0, 0)}
        for k, color in enumerate(sns.color_palette("hls", len(LABEL_VALUES) - 1)):
            palette[k + 1] = tuple(np.asarray(255 * np.array(color), dtype='uint8'))
    invert_palette = {v: k for k, v in palette.items()}  # 结果显示的一个字典


    def convert_to_color(x):
        return convert_to_color_(x, palette=palette)


    def convert_from_color(x):
        return convert_from_color_(x, palette=invert_palette)


    # Instantiate the experiment based on predefined networks
    hyperparams.update({'n_classes': N_CLASSES, 'n_bands': N_BANDS, 'ignored_labels': IGNORED_LABELS,
                        'device': CUDA_DEVICE})  # 更新部分参数
    hyperparams = dict((k, v) for k, v in hyperparams.items() if v is not None)

    color_gt = convert_to_color(gt)  # 显示ground truth

    results = []
    gt_ = gt[(PATCH_SIZE // 2):-(PATCH_SIZE // 2), (PATCH_SIZE // 2):-(PATCH_SIZE // 2)]
    train_gt, test_gt = sample_gt(gt_, SAMPLE_PERCENTAGE, mode=SAMPLING_MODE)   # 获取样本，全部作为训练样本，取gt_标签值不为0的所有的坐标
    # ---------------------------------------只在原gt上进行training样本采样-------------------------------------------
    mask = np.zeros_like(gt)
    for l in set(hyperparams['ignored_labels']):  # 未标记的像素样本
        mask[gt == l] = 0
    x_pos, y_pos = np.nonzero(train_gt)  # 读取train_gt中的非0值元素坐标，取得坐标
    indices = np.array([(x, y) for x, y in zip(x_pos, y_pos)])      # 提取train样本的坐标，只能在原图上面取
    for x, y in indices:
        if mask[x + PATCH_SIZE // 2, y + PATCH_SIZE // 2] is not 0:  # gt位置矫正
            mask[x + PATCH_SIZE // 2, y + PATCH_SIZE // 2] = gt[x + PATCH_SIZE // 2, y + PATCH_SIZE // 2]
    train_gt = mask  # 为矫正后
    # ---------------------------------------只在原gt上进行test样本采样-------------------------------------------
    print("{} samples selected (over {})".format(np.count_nonzero(train_gt), np.count_nonzero(gt_)))
    print("Running an experiment with the {} model".format(MODEL))  # cnn模型
    model, optimizer, loss, hyperparams = get_model(MODEL, **hyperparams)   # 模型加载
    if CLASS_BALANCING:
            weights = compute_imf_weights(train_gt, N_CLASSES, IGNORED_LABELS)  # 参数加载
            hyperparams['weights'] = torch.from_numpy(weights)
            train_dataset = HyperX(img, train_gt, **hyperparams)   # 数据生成器
            train_loader = data.DataLoader(train_dataset, batch_size=hyperparams['batch_size'], pin_memory=hyperparams['device'], shuffle=True, drop_last=True)
            print(hyperparams)
            print("Network :")
            with torch.no_grad():
                for input, _, _ in train_loader:
                    break
                    # summary(model.to(hyperparams['device']), input.size()[1:], device=hyperparams['device'])
                summary(model.to(hyperparams['device']), input.size()[1:])

            try:  # 进行模型训练
                train(DATASET, model, optimizer, loss, train_loader, hyperparams['batch_size'], hyperparams['epoch'],
                      scheduler=hyperparams['scheduler'], device=hyperparams['device'],
                      supervision=hyperparams['supervision'],
                      display=viz)  # 注释了val集
            except KeyboardInterrupt:  # Allow the user to stop the training
                pass

