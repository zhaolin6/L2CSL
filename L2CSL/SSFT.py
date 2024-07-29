import torch
from torch import nn, einsum
from einops import rearrange
from VIT import weight_init
from VIT.drop import DropPath
from timm.models.layers import to_2tuple
from torchvision.models.resnet import resnet18
import torch.nn.functional as F


class MODEL(nn.Module):
    def __init__(self, input_channels, n_classes, patch_size):  # 后面使用3d卷积的时候，输入的channels有变化，ori=30，after=64
        super(MODEL, self).__init__()
        self.f = []
        self.patch_size = patch_size
        self.input_channels = input_channels
        self.n_classes = n_classes

        for name, module in resnet18(pretrained=False).named_children():  # resnet50 may try?
            if name == 'conv1':
                module = nn.Conv2d(self.input_channels, 64, kernel_size=(3, 3), padding=(1, 1), bias=False)
                # HSI通道数与RGB通道数不一致
            if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):  # except adaptavg
                self.f.append(module)

        self.f = nn.Sequential(*self.f)

        backbone_in_channels = resnet18().fc.in_features
        self.project = nn.Sequential(
            nn.Linear(backbone_in_channels, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128, bias=True))  # 128

    def forward(self, x):
        x = self.f(x)
        out = torch.flatten(x, start_dim=1)  # b 512
        out = self.project(out)
        # x = x.reshape(-1, self.flattened_size)
        # print(x.shape)

        return out


class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        std = torch.var(x, dim=1, unbiased=False, keepdim=True).sqrt()
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (std + self.eps) * self.g + self.b


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = LayerNorm(dim)
        # self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)


class PatchEmbed(nn.Module):
    def __init__(self, img_size, in_channels, patch_size, dim, norm=True):
        # TODO THIS IS MORE IMPORTANT
        super(PatchEmbed, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        # ***----------------------spectral------------------------------***
        self.base_con3d_1 = nn.Sequential(nn.Conv3d(1, 8, kernel_size=(7, 1, 1), stride=(2, 1, 1)),
                                          nn.BatchNorm3d(8),
                                          nn.LeakyReLU(inplace=True)
                                          )  # b,5,49,15,15
        self.base_con3d_2 = nn.Sequential(nn.Conv3d(8, 8, kernel_size=(5, 1, 1), stride=(1, 1, 1)), #, padding=(2, 0, 0)
                                          nn.BatchNorm3d(8),
                                          nn.LeakyReLU(inplace=True))

        self.base_con3d_3 = nn.Sequential(nn.Conv3d(8, 8, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0)),
                                          nn.BatchNorm3d(8),
                                          nn.LeakyReLU(inplace=True),
                                          )  # 1 8 49 15 15

        self.pooling = nn.Sequential(nn.MaxPool3d(kernel_size=(3, 1, 1), stride=(1, 1, 1)))
        # ***----------------------spatial------------------------------***
        self.flattened_size = self.flattened()
        self.con_2d_1 = nn.Sequential(
            nn.Conv2d(self.flattened_size, dim // 4, kernel_size=1, stride=1),  # in_channels  360 392 , groups=1
            nn.BatchNorm2d(dim // 4),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(dim // 4, dim // 2, kernel_size=3, padding=1),  # 3 1  , groups=dim//4
            nn.BatchNorm2d(dim // 2),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2, 2, 1),

            nn.Conv2d(dim // 2, dim, kernel_size=3, padding=1),  # 把最后的输出通道改成spa//4就可以 , groups=dim//2
            nn.BatchNorm2d(dim),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2, 2, 1)
        )

        self.con_2d_2 = nn.Sequential(
                                    nn.Conv2d(self.flattened_size, dim // 4, kernel_size=1, stride=1),  # groups=1 360
                                    nn.BatchNorm2d(dim // 4),
                                    nn.LeakyReLU(inplace=True),

                                    nn.Conv2d(dim // 4, dim // 2, kernel_size=5, padding=2),  # 3 1 , groups=dim//4
                                    nn.BatchNorm2d(dim // 2),
                                    nn.LeakyReLU(inplace=True),
                                    nn.MaxPool2d(2, 2, 1),

                                    nn.Conv2d(dim // 2, dim, kernel_size=5, padding=2),
                                    nn.BatchNorm2d(dim),
                                    nn.LeakyReLU(inplace=True),
                                    nn.MaxPool2d(2, 2, 1)
        )

        self.pro_j = nn.Sequential(
                                   nn.Conv2d(dim*2, dim, kernel_size=1, padding=0),  # k = 3, p = 1 , groups=1
                                   nn.BatchNorm2d(dim),
                                   nn.LeakyReLU(inplace=True),
                                   # nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1,  padding=0, groups=dim),
                                   # nn.BatchNorm2d(dim),
                                   # nn.LeakyReLU(inplace=True),  # 这一部分我可能要加到self-attention里面
                                   )
        self.norm = nn.LayerNorm(dim) if norm else nn.Identity()

    def flattened(self):
        with torch.no_grad():
            x = torch.zeros((1, self.in_channels, self.img_size, self.img_size))
            x1 = torch.unsqueeze(x, dim=1)
            x_1 = self.base_con3d_1(x1)
            x_2 = self.base_con3d_2(x_1)
            x_3 = self.base_con3d_3(x_2)
            b, c, h, w = rearrange(x_3, 'B m n H W -> B (m n) H W ').size()
            return c

    def forward(self, x):
        # //-----------***the following are for spectral feature***----------//
        x1 = torch.unsqueeze(x, dim=1)  # [:, :, 6:9, 6:9]
        assert len(x1.size()) == 5, "dim of x must be 5"
        x_1 = self.base_con3d_1(x1)
        # x_1 = self.pooling(x_1)  # 快速降维的操作

        x_2 = self.base_con3d_2(x_1)
        # x_2 = self.pooling(x_2)

        x_3 = self.base_con3d_3(x_2)

        x_spectral = rearrange(x_3, 'B m n H W -> B (m n) H W ')

        x_spatial_1 = self.con_2d_1(x_spectral)
        x_spatial_2 = self.con_2d_2(x_spectral)
        x_fusion = torch.cat((x_spatial_1, x_spatial_2), dim=1)  # todo
        x_out = self.pro_j(x_fusion)  # todo

        x_out = rearrange(x_out, 'B m H W -> B (H W) m ')   # 为了契合attention

        x_out = self.norm(x_out)
        return x_out


class SPAAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, drop=0., attn_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.patch_size = 5

        self.to_q = nn.Conv2d(dim, dim, kernel_size=(3, 3), padding=(1, 1), groups=dim)
        self.to_k = nn.Conv2d(dim, dim, kernel_size=(3, 3), padding=(1, 1), groups=dim)
        self.to_v = nn.Conv2d(dim, dim, kernel_size=(3, 3), padding=(1, 1), groups=dim)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(drop)
        self.proj = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=(1, 1), padding=(0, 0), bias=qkv_bias),
                                  nn.BatchNorm2d(dim)
                                  )  # , groups=1
        # self.LPU = LocalPerceptionUint(dim, act=True)

    def forward(self, x):

        q = self.to_q(x)  # 1 128*4 5 5
        q = rearrange(q, 'b (h d) l w -> b h (l w) d', h=self.num_heads)  # b 4 25 128

        v = self.to_v(x)
        v = rearrange(v, 'b (h d) l w -> b h (l w) d', h=self.num_heads)  # b 4 25 128

        k = self.to_k(x)
        k = rearrange(k, 'b (h d) l w -> b h (l w) d', h=self.num_heads)  # b 4 25 128

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = dots.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h (H W) d -> b (h d) H W', H=self.patch_size)

        x = self.proj(out)  # 可以使用卷积 b 25 4*128
        # x = self.LPU(out)
        x = self.proj_drop(x)
        return x


class SPEAttention(nn.Module):
    # TODO 关于光谱的Attention
    def __init__(self, dim, num_heads=1, qkv_bias=False, drop=0., attn_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.gamma = nn.Parameter(torch.zeros(1))

        self.to_q = nn.Conv2d(dim, dim, kernel_size=(3, 3), padding=(1, 1), groups=dim)
        self.to_k = nn.Conv2d(dim, dim, kernel_size=(3, 3), padding=(1, 1), groups=dim)
        self.to_v = nn.Conv2d(dim, dim, kernel_size=(3, 3), padding=(1, 1), groups=dim)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(drop)
        self.proj = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=(1, 1), padding=(0, 0), bias=qkv_bias),
                                  nn.BatchNorm2d(dim)
                                  )  # 这里看能不能调一下, groups=1
        # self.LPU = LocalPerceptionUint(dim, act=True)

    def forward(self, x):
        B, C, H, W = x.size()
        q = self.to_q(x)  # b 128 5 5
        q = rearrange(q, 'b d l w -> b d (l w)')  # b 128 3 3 -> b 128 9

        k = self.to_k(x)
        k = rearrange(k, 'b d l w -> b (l w) d')  # b 128 3 3 -> b 9 128

        v = self.to_v(x)
        v = rearrange(v, 'b d l w -> b d (l w)')  # b 128 3 3 -> b 128 9

        attention = (q @ k) * self.scale  # -->2 1 128 128

        # attention = torch.max(attention, -1, keepdim=True)[0].expand_as(attention) - attention

        attention = attention.softmax(dim=-1)
        attention = self.attn_drop(attention)  # 不加有啥效果

        out = (attention @ v).view(B, C, H, W)

        # out = self.gamma * out + x  # gamma是一个可更新的参数

        x = self.proj(out)
        x = self.proj_drop(x)
        return x


class FusionAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, drop=0., attn_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.patch_size = 5

        self.to_q = nn.Conv2d(dim, dim, kernel_size=(3, 3), padding=(1, 1), groups=dim)
        self.to_k = nn.Conv2d(dim, dim, kernel_size=(3, 3), padding=(1, 1), groups=dim)
        self.to_v = nn.Conv2d(dim, dim, kernel_size=(3, 3), padding=(1, 1), groups=dim)

        self.spe_to_q = nn.Conv2d(dim, dim, kernel_size=(3, 3), padding=(1, 1), groups=dim)
        self.spe_to_k = nn.Conv2d(dim, dim, kernel_size=(3, 3), padding=(1, 1), groups=dim)
        self.spe_to_v = nn.Conv2d(dim, dim, kernel_size=(3, 3), padding=(1, 1), groups=dim)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(drop)
        self.proj = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=(1, 1), padding=(0, 0), bias=qkv_bias),
                                  nn.BatchNorm2d(dim)
                                  )  # , groups=1
        # self.LPU = LocalPerceptionUint(dim, act=True)

    def forward(self, x):
        # x = self.LPU(x)
        B, C, H, W = x.size()
        # ------------分割线----------------------
        q = self.to_q(x)  # 1 128*4 5 5
        q = rearrange(q, 'b (h d) l w -> b h (l w) d', h=self.num_heads)

        k = self.to_k(x)
        k = rearrange(k, 'b (h d) l w -> b h (l w) d', h=self.num_heads)

        v = self.to_v(x)
        v = rearrange(v, 'b (h d) l w -> b h (l w) d', h=self.num_heads)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = dots.softmax(dim=-1)
        attn = self.attn_drop(attn)  # b 4 25 25  空间attention
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h (H W) d -> b (h d) H W', H=self.patch_size)
        # ------------分割线----------------------
        spe_q = self.spe_to_q(out)  # b 128 5 5
        spe_q = rearrange(spe_q, 'b d l w -> b d (l w)')  # b 128 1

        spe_k = self.spe_to_k(out)
        spe_k = rearrange(spe_k, 'b d l w -> b (l w) d')  # b 128 3 3 -> b 9 128(b 1 128)

        spe_v = self.spe_to_v(out)
        spe_v = rearrange(spe_v, 'b d l w -> b d (l w)')  # b 128 3 3 -> b 128 9

        attention = (spe_q @ spe_k) * self.scale   # -->2 128 128
        attention = attention.softmax(dim=-1)
        attention = self.attn_drop(attention)  # 不加有啥效果
        x_out = (attention @ spe_v).reshape(B, C, H, W)  # 2 128 128 @ 2 128 25->2 128 25
        # x_out = x + x_out
        # ----------------------------------------
        output = self.proj(x_out)  # 可以使用卷积 b 25 4*128
        # x = self.LPU(out)
        output = self.proj_drop(output)
        return output


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        # TODO 原始的
        # self.net = nn.Sequential(
        #     nn.Linear(dim, hidden_dim),
        #     nn.GELU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(hidden_dim, dim),
        #     nn.Dropout(dropout)
        # )
        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, kernel_size=(1, 1)),  # , groups=1
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(hidden_dim, dim, kernel_size=(1, 1)),  # , groups=1
            nn.Dropout(dropout)
        )
        # TODO 第二种
        # self.net = nn.Sequential(
        #     nn.Conv2d(dim, 256, kernel_size=3, padding=1, stride=1, groups=1),
        #     nn.BatchNorm2d(256),
        #     nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1),
        #     nn.GELU(),
        #     nn.Conv2d(in_channels=512, out_channels=dim, kernel_size=1),
        #     nn.GELU(),
        # )
        # TODO 第三种
        # self.conv1 = nn.Sequential(
        #     nn.Conv2d(dim, hidden_dim, 1, 1, 0, bias=True),
        #     nn.GELU(),
        #     nn.BatchNorm2d(hidden_dim, eps=1e-5),
        #     nn.Dropout(dropout)
        # )
        # self.proj = nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, groups=hidden_dim)
        # self.proj_act = nn.GELU()
        # self.proj_bn = nn.BatchNorm2d(hidden_dim, eps=1e-5)
        # self.conv2 = nn.Sequential(
        #     nn.Conv2d(hidden_dim, dim, 1, 1, 0, bias=True),
        #     nn.BatchNorm2d(dim, eps=1e-5),
        # )
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x_out = self.net(x)
        # TODO 第三种配套
        # x = self.conv1(x)
        # x = self.proj(x) + x
        # x = self.proj_act(x)
        # x = self.proj_bn(x)
        # x = self.conv2(x)
        # x = x.flatten(2).permute(0, 2, 1)
        # x_out = self.drop(x)
        return x_out


class LocalPerceptionUint(nn.Module):
    # TODO 暂时没用到，一些论文说加在attention之前有效果
    def __init__(self, dim, kernel_size=3, act=False):
        super(LocalPerceptionUint, self).__init__()
        self.act = act
        self.conv_3x3_dw = nn.Conv2d(in_channels=dim,   out_channels=dim, kernel_size=to_2tuple(kernel_size),
                                     stride=to_2tuple(1), padding=to_2tuple(1), groups=dim)

        if self.act:
            self.actation = nn.Sequential(
                nn.GELU(),
                nn.BatchNorm2d(dim)
            )

    def forward(self, x):
        if self.act:
            out = self.actation(self.conv_3x3_dw(x))
            return out
        else:
            out = self.conv_3x3_dw(x)
            return out


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_dim, qkv_bias=False, drop=0., attn_drop=0., drop_path=0.):
        super().__init__()
        # self.LPU = LocalPerceptionUint(dim)  # LPU结构
        # self.spa_attn = PreNorm(dim, SPAAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, drop=drop, attn_drop=attn_drop))
        # self.spe_attn = PreNorm(dim, SPEAttention(dim, num_heads=1, qkv_bias=qkv_bias, drop=drop, attn_drop=attn_drop))
        self.attn = PreNorm(dim, FusionAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, drop=drop,
                                                 attn_drop=attn_drop))
        self.mlp = PreNorm(dim, FeedForward(dim=dim, hidden_dim=mlp_dim, dropout=drop))
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        # x = self.LPU(x)
        # x_1 = x + self.drop_path(self.spe_attn(x))  # 输出是一个三维的: 1 25 128
        # y_1 = x_1 + self.drop_path(self.spa_attn(x_1))
        y_1 = x + self.drop_path(self.attn(x))
        y_out = y_1 + self.drop_path(self.mlp(y_1))
        return y_out


class SSFTransformer(nn.Module):
    def __init__(self, img_size, in_channels, num_classes, embed_dim, depth, num_heads,  mlp_dim,
                 qkv_bias=True, drop_rate=0.1, attn_drop_rate=0.1, drop_path_rate=0.1, pool='mean'):
        assert pool == 'cls' or pool == 'mean', "pool must be cls or mean"
        super(SSFTransformer, self).__init__()
        self.num_tokens = 1
        self.num_classes = num_classes
        self.pool = pool

        patch_size = 5  # 此处可调3
        num_patches = 25  # 25 9
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_channels=in_channels,
                                      dim=embed_dim)

        # self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.blocks = nn.ModuleList([])
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        for i in range(depth):
            self.blocks.append(
            Block(dim=embed_dim, num_heads=num_heads, mlp_dim=mlp_dim, qkv_bias=qkv_bias, attn_drop=attn_drop_rate,
                  drop=drop_rate, drop_path=dpr[i]))

        # for para in self.blocks.parameters():  # TODO 参数冻结，预训练时注释掉
        #     para.requires_grad = False

        self.norm = nn.LayerNorm(embed_dim)

        # TODO classification
        # self.cls = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()  # TODO 预训练注释掉

        # self.project = nn.Sequential(
        #     nn.Linear(embed_dim, 512),
        #     nn.BatchNorm1d(512),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(512, num_classes),
        # )  # TODO 微调注释掉
        self.apply(self._init_vit_weights)

        # TODO //***************卷积部分的代码***************//
        self.g = MODEL(in_channels, img_size)
        backbone_in_channels = resnet18().fc.in_features
        self.project = nn.Sequential(
                                    nn.Linear(backbone_in_channels, 256, bias=False),
                                    nn.BatchNorm1d(256),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(256, num_classes, bias=True))  # 128

    def _init_vit_weights(self, module):
        """ ViT weight initialization
        """
        if isinstance(module, nn.Linear):
            if module.out_features == self.num_classes:
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            else:
                weight_init.trunc_normal_(module.weight, std=.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Conv2d):
            # NOTE conv was left to pytorch default in my original init
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def add_token(self, x):
        x = x + self.pos_embed[:, 1:, :]
        x = self.pos_drop(x)
        # cls_token = self.cls_token + self.pos_embed[:, :1, :]
        # cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        # x = torch.cat((cls_tokens, x), dim=1)
        return x

    def feature_forward(self, x):
        x_1 = self.patch_embed(x)  # 这里集合了空间和光谱操作
        x_1 = self.add_token(x_1)  # 位置编码
        x_1 = rearrange(x_1, 'b (h w) c -> b c h w', h=5)  # 契合卷积的  # 2 128 5 5

        for block in self.blocks:  # transformer
            x_1 = block(x_1)  # b 25 128
        # 预训练的下面这部分要保留
        x_1 = rearrange(x_1, 'b c h w -> b (h w) c')
        x_2 = x_1.mean(dim=1) if self.pool == 'mean' else x_1[:, 0]  # b 128
        # feature = self.norm(x_2)  # b 128

        # x_out = self.project(x_2)  # TODO 预训练用这个，记得分别注释掉
        # x_out = self.cls(x_2)  # TODO 微调用这个

        return x_2  # x_out

    def forward(self, x):
        # 下面的是transformer编码器
        out1 = self.feature_forward(x)  # 2 128
        # 下面的是卷积的编码器
        out2 = self.g(x)
        out2 = self.project(out2)  # 2 128
        return F.normalize(out1, dim=-1), F.normalize(out2, dim=-1)