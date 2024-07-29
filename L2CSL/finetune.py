#!!!此为新的fine_tune模型
import torch.nn as nn
import torch
import numpy as np
import torch.optim as optim
from torchvision.models.resnet import resnet18
from SSFT import SSFTransformer, MODEL
# utils
import math
from tqdm import tqdm
from einops import rearrange
from utils import grouper, sliding_window, count_sliding_window, camel_to_snake

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

    if name =='NEWMODEL':
        patch_size = kwargs.setdefault('patch_size', 15)  # 如果'patch_size'键不存在于字典中，将会添加键并将值设为默认值,5
        center_pixel = True
        model = NEWMODEL(n_bands, n_classes, patch_size=patch_size)  # 模型
        lr = kwargs.setdefault('lr', 0.001)
        optimizer = get_optimizer(model, lr)
        criterion = nn.CrossEntropyLoss()   # loss function     # weight参数分别代表n类的权重
        model = model.to(device)
        epoch = kwargs.setdefault('epoch', 500)  # 300
        kwargs.setdefault('scheduler', optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=epoch // 4,
                                                                            verbose=True))
        # 学习率调整，该方法提供了一些基于训练过程中的某些测量值对学习率进行动态的下降
        # kwargs.setdefault('batch_size', 256)  # （原100）测试一下
        kwargs.setdefault('supervision', 'full')
        kwargs.setdefault('flip_augmentation', False)
        kwargs.setdefault('radiation_augmentation', False)
        kwargs.setdefault('mixture_augmentation', False)
        kwargs['center_pixel'] = center_pixel
        # return model, optimizer, criterion, kwargs

    else:
        raise KeyError("{} model is unknown.".format(name))

    return model, optimizer, criterion, kwargs


class NEWMODEL(nn.Module):
    def __init__(self, input_channels, n_classes, patch_size=21):
        super(NEWMODEL, self).__init__()
        self.patch_size = patch_size
        self.input_channels = input_channels
        self.f = MODEL(self.input_channels, n_classes, patch_size=self.patch_size).f

        self.flattened_size = self.flattened()
        self.fc_new1 = nn.Sequential(
            nn.Linear(self.flattened_size, n_classes))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def flattened(self):  # 此函数没有用到
        with torch.no_grad():
            x = torch.zeros((1, self.input_channels, self.patch_size, self.patch_size,))
            x = self.f(x)
            t, w, l, b = x.size()
            return t*w*l*b  # 卷积完毕

    def forward(self, img):
        img = self.f(img)
        i_features = img.reshape(-1, self.flattened_size)
        out = self.fc_new1(i_features)

        return out

def get_optimizer(model, LR):
    learning_rate = LR
    param_group = []
    param_group += [{'params': model.f.parameters(), 'lr': learning_rate * 0.1}]
    param_group += [{'params': model.fc_new1.parameters(), 'lr': learning_rate}]  #
    optimizer = optim.Adam(param_group, eps=1e-6, weight_decay=1e-6)  # weight_decay=1e-6,在这里的时候只对部分的结构做了优化
    return optimizer  # 优化的时候只对后面一层进行优化

def new_train(net, optimizer, criterion, data_loader, epoch, scheduler=None,
          display_iter=100, device=torch.device('cuda'), display=None,
          val_loader=None, supervision='full'):
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

    #********************** text model  build **************
    losses = np.zeros(1000000)
    mean_losses = np.zeros(100000000)
    iter_ = 1
    loss_win, val_win = None, None
    val_accuracies = []

    for e in tqdm(range(1, epoch + 1), desc="Training the network"):
        # Set the network to training mode
        net.train()
        avg_loss = 0.
        # if e == 400:      # lr 衰减
        #     for p in optimizer.param_groups:
        #         p['lr'] *= 0.1

        for batch_idx, (data, target) in enumerate(data_loader):
            # Load the data into the GPU if required
            data, target = data.to(device), target.to(device)   # 在DataLoader中已经将Tensor封装成了Variable

            optimizer.zero_grad()
            if supervision == 'full':
                output = net(data)
                loss = criterion(output, target)
            else:
                raise ValueError("supervision mode \"{}\" is unknown.".format(supervision))
            loss.backward()
            # loss.backward(retain_graph=True)    # L1 regularization
            optimizer.step()
            avg_loss += loss.item()
            losses[iter_] = loss.item()
            mean_losses[iter_] = np.mean(losses[max(0, iter_ - 100):iter_ + 1])

            if display_iter and iter_ % display_iter == 0:  # 可视化模块
                string = 'Train (epoch {}/{}) [{}/{} ({:.0f}%)]\tLoss: {:.6f}'
                string = string.format(
                    e, epoch, batch_idx *
                    len(data), len(data) * len(data_loader),
                    100. * batch_idx / len(data_loader), mean_losses[iter_])
                update = None if loss_win is None else 'append'
                loss_win = display.line(
                    X=np.arange(iter_ - display_iter, iter_),
                    Y=mean_losses[iter_ - display_iter:iter_],
                    win=loss_win,
                    update=update,
                    opts={'title': "Training loss",
                          'xlabel': "Iterations",
                          'ylabel': "Loss"
                         }
                )
                tqdm.write(string)

                if len(val_accuracies) > 0:
                    val_win = display.line(Y=np.array(val_accuracies),
                                           X=np.arange(len(val_accuracies)),
                                           win=val_win,
                                           opts={'title': "Validation accuracy",
                                                 'xlabel': "Epochs",
                                                 'ylabel': "Accuracy"
                                                 })
            iter_ += 1
            del(data, target, loss)

        # Update the scheduler
        avg_loss /= len(data_loader)
        if val_loader is not None:  # 做验证
            val_acc = val(net, val_loader, device=device, supervision=supervision)
            val_accuracies.append(val_acc)
            metric = -val_acc
        else:
            metric = avg_loss

        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(metric)
        elif scheduler is not None:
            scheduler.step()
    torch.cuda.empty_cache()



def val(net, data_loader, device='cpu', supervision='full'):  # 在train 里面设置成了none
    # TODO : fix me using metrics()
    accuracy, total = 0., 0.
    ignored_labels = data_loader.dataset.ignored_labels
    for batch_idx, (data, target) in enumerate(data_loader):
        with torch.no_grad():
            # Load the data into the GPU if required
            data, target = data.to(device), target.to(device)
            if supervision == 'full':
                output = net(data)
            elif supervision == 'semi':
                outs = net(data)
                output, rec = outs
            _, output = torch.max(output, dim=1)
            for out, pred in zip(output.view(-1), target.view(-1)):
                if out.item() in ignored_labels:
                    continue
                else:
                    accuracy += out.item() == pred.item()
                    total += 1
    return accuracy / total


def test(net, img, hyperparams):
    """
   # Test a model on a specific image
    """
    net.eval()
    patch_size = hyperparams['patch_size']
    center_pixel = hyperparams['center_pixel']
    batch_size, device = hyperparams['batch_size'], hyperparams['device']
    n_classes = hyperparams['n_classes']

    kwargs = {'step': hyperparams['test_stride'], 'window_size': (patch_size, patch_size)}
    probs = np.zeros(img.shape[:2] + (n_classes,))  # 增加一个维度2（H,W,类别）


    iterations = count_sliding_window(img, **kwargs) // batch_size
    for batch in tqdm(grouper(batch_size, sliding_window(img, **kwargs)),
                      total=(iterations),
                      desc="Inference on the image"
                      ):
        with torch.no_grad():
            if patch_size == 1:
                data = [b[0][0, 0] for b in batch]
                data = np.copy(data)
                data = torch.from_numpy(data)
            else:
                data = [b[0] for b in batch]
                data = np.copy(data)
                data = data.transpose(0, 3, 1, 2)
                data = torch.from_numpy(data)
                # data = data.unsqueeze(1)              # 3DConv时执行

            indices = [b[1:] for b in batch]
            data = data.to(device)
            output = net(data)

            if isinstance(output, tuple):
                output = output[0]
            output = output.to('cpu')  # 将cpu 改为 cuda

            if patch_size == 1 or center_pixel:
                output = output.numpy()
            else:
                output = np.transpose(output.numpy(), (0, 2, 3, 1))
            for (x, y, w, h), out in zip(indices, output):
                if center_pixel:
                    # probs[x, y] += out
                    probs[x + w // 2, y + h // 2] += out
                    # probs[x:x + w, y:y + h] += out
                else:
                    probs[x:x + w, y:y + h] += out
    return probs



