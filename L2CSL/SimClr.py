import torch
import torch.nn as nn
import numpy as np
from torch import nn
import torch.nn.functional as F


class SimClr(nn.Module):
    def __init__(self, batch_size, temperature=0.5):  # 分别测试0.25，0.75
        super(SimClr, self).__init__()
        self.batch_size = batch_size
        # self.temperature = torch.tensor(temperature)
        # dataframe = pd.DataFrame(super_pixel)
        # dataframe.to_excel(r"C:\Users\h'p\Desktop\5.xlsx")
        self.temperature = temperature

    def Q(self, indices):
        H = indices.shape[0]  # 256 个包含坐标的列表
        superpixel_count = self.super_pixel.max() + 1  # 用来计算总共有多少个超像素块 2418
        Q = np.zeros([H, superpixel_count], dtype=np.float32)  # 256 2418
        k = 0
        while k < H:
            for i, j, p in indices:  # 得到一个256*2418的矩阵,批次样本属于哪一个超像素块 todo
                Q[k, self.super_pixel[i][j]] = 1
                k = k+1
        # print(Q)
        # dataframe = pd.DataFrame(Q)
        # dataframe.to_excel(r"C:\Users\h'p\Desktop\3.xlsx")
        return torch.from_numpy(Q)  # 256 2418

    def adj_matrix(self, Q, sigma=5):  # A 是每个超像素的特征矩阵
        min = -1
        max = 1
        keep_pixel = Q @ self.A  # 256*8
        keep_pixel_t = torch.t(keep_pixel)

        # todo tips 1: 欧氏距离衡量两个超像素样本之间的相似度
        # h, _ = keep_pixel.shape
        # matrix = torch.zeros(size=[h, h])
        # for i in range(h):
        #     for j in range(h):
        #         matrix[i][j] = torch.exp(-torch.sum(torch.square(keep_pixel[i, :] - keep_pixel_t[:, j])) / sigma ** 2)

        # todo tips 2: 余弦相似度衡量两个超像素样本之间的相似度
        keep_pixel_ave = torch.mean(keep_pixel, dim=-1, keepdim=True).expand_as(keep_pixel)
        keep_pixel_t_ave = torch.mean(keep_pixel_t, dim=-2, keepdim=True).expand_as(keep_pixel_t)
        keep_pixel = keep_pixel - keep_pixel_ave
        keep_pixel_t = keep_pixel_t - keep_pixel_t_ave
        # 上面的数据是准备用来做修正后的余弦相似度的,若使用普通余弦相似度，注释即可

        dot_matrix = keep_pixel @ keep_pixel_t
        row_norm = torch.norm(keep_pixel, p=2, dim=-1, keepdim=True)
        col_norm = torch.norm(keep_pixel_t, p=2, dim=-2, keepdim=True)
        norm_matrix = row_norm @ col_norm
        matrix = torch.div(dot_matrix, norm_matrix)
        matrix[torch.isneginf(matrix)] = 0
        matrix = (matrix-min)/(max-min)
        matrix_mean = torch.mean(matrix, dim=-2, keepdim=True).expand_as(matrix)

        return torch.pow(0.75, (matrix-matrix_mean)/0.25).repeat((2, 2)).to(torch.device('cuda'))
        # matrix, torch.exp(matrix/0.5).repeat((2, 2)).to(torch.device('cuda'))
        # a 越小左边越陡，扩大的值会越大，增大负数的值0.5 0.5  0.75, matrix/0.5 -matrix_mean

    def compute_loss(self, out1, out2, indices):  # 传入的是tensor形式，emb_1是增强1的所有out，emb_2是增强2的所有out
        # Q = self.Q(indices)  # 256 2418
        #weight = self.adj_matrix(Q)  # .repeat((2, 2)).to(torch.device('cuda'))  # TODO nagetive sample weight

        # WEIGHT = (Q @ self.A) @ torch.t(Q)
        # weight = torch.exp(-(Q @ self.A) @ (torch.t(Q))/self.temperature).repeat((2, 2)).to(torch.device('cuda'))

        # torch.set_printoptions(threshold=np.inf)
        # print(weight)
        # work = pd.ExcelWriter(r"C:\Users\h'p\Desktop\4.xlsx")
        # dataframe = pd.DataFrame(weight.detach().numpy())  #
        # dataframe.to_excel(work, sheet_name='2')
        # dataframe = pd.DataFrame(weight_e.detach().numpy())  # indices .detach().numpy()
        # dataframe.to_excel(work)
        # dataframe = pd.DataFrame(indices)  # indices
        # dataframe.to_excel(work, sheet_name='3')
        # work.save()

        z_1 = F.normalize(out1, dim=1)
        z_2 = F.normalize(out2, dim=1)
        out = torch.cat([z_1, z_2], dim=0)
        # [2*B, 2*B]
        sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / self.temperature)

        # torch.set_printoptions(threshold=np.inf)
        # print(sim_matrix[0:2, :])
        # TODO 给这个sim_matrix加上权重
        # sim_matrix = torch.mul(sim_matrix, weight)  # TODO THE SMAE AS TOP

        # torch.set_printoptions(threshold=np.inf)
        # print(sim_matrix[0:2, :])

        mask = (torch.ones_like(sim_matrix) - torch.eye(2 * self.batch_size, device=sim_matrix.device)).bool()
        # [2*B, 2*B-1]
        sim_matrix = sim_matrix.masked_select(mask).view(2 * self.batch_size, -1)

        # compute loss
        pos_sim = torch.exp(torch.sum(z_1 * z_2, dim=-1) / self.temperature)
        # [2*B]
        pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
        loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
        # representations = torch.cat([z_1, z_2], dim=0)
        # negatives_mask = torch.eye(self.batch_size * 2, self.batch_size * 2).to(torch.device('cuda'))
        # similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
        #
        # sim_ij = torch.diag(similarity_matrix, self.batch_size)
        # sim_ji = torch.diag(similarity_matrix, -self.batch_size)
        # positives = torch.cat([sim_ij, sim_ji], dim=0)
        #
        # nominator = torch.exp(positives / self.temperature)
        # denominator = negatives_mask * torch.exp(similarity_matrix / self.temperature)
        #
        # loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))
        # loss = torch.sum(loss_partial) / (2 * self.batch_size)
        return loss




