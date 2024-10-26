from __future__ import division, print_function
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from torch.nn import functional as F
import numpy as np
import cv2
import math
from einops import rearrange
from torch.distributions.uniform import Uniform

torch.cuda.set_device(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def kaiming_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            torch.nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model


def sparse_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            torch.nn.init.sparse_(m.weight, sparsity=0.1)
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model


class ConvBlock(nn.Module):
    """two convolution layers with batch norm and leaky relu"""

    def __init__(self, in_channels, out_channels, dropout_p):
        super(ConvBlock, self).__init__()
        self.conv_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Dropout(dropout_p),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.conv_conv(x)


class DownBlock(nn.Module):
    """Downsampling followed by ConvBlock"""

    def __init__(self, in_channels, out_channels, dropout_p):
        super(DownBlock, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_channels, out_channels, dropout_p)

        )

    def forward(self, x):
        return self.maxpool_conv(x)


class UpBlock(nn.Module):
    """Upssampling followed by ConvBlock"""

    def __init__(self, in_channels1, in_channels2, out_channels, dropout_p,
                 bilinear=True):
        super(UpBlock, self).__init__()
        self.bilinear = bilinear
        if bilinear:
            self.conv1x1 = nn.Conv2d(in_channels1, in_channels2, kernel_size=1)
            self.up = nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels1, in_channels2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels2 * 2, out_channels, dropout_p)

    def forward(self, x1, x2):
        if self.bilinear:
            x1 = self.conv1x1(x1)
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

# 不同的上采样方式
class UpBlock1(nn.Module):
    """Upssampling followed by ConvBlock"""

    def __init__(self, in_channels1, in_channels2, out_channels, dropout_p,
                 bilinear=True):
        super(UpBlock1, self).__init__()
        self.bilinear = bilinear
        if bilinear:
            self.conv1x1 = nn.Conv2d(in_channels1, in_channels2, kernel_size=1)
            self.up = nn.Upsample(
                scale_factor=2, mode='bicubic', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels1, in_channels2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels2 * 2, out_channels, dropout_p)

    def forward(self, x1, x2):
        if self.bilinear:
            x1 = self.conv1x1(x1)
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class Encoder_cnn(nn.Module):
    def __init__(self, params):
        super(Encoder_cnn, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        self.dropout = self.params['dropout']
        assert (len(self.ft_chns) == 5)
        self.in_conv = ConvBlock(
            self.in_chns, self.ft_chns[0], self.dropout[0])
        self.down1 = DownBlock(
            self.ft_chns[0], self.ft_chns[1], self.dropout[1])
        self.down2 = DownBlock(
            self.ft_chns[1], self.ft_chns[2], self.dropout[2])
        self.down3 = DownBlock(
            self.ft_chns[2], self.ft_chns[3], self.dropout[3])
        self.down4 = DownBlock(
            self.ft_chns[3], self.ft_chns[4], self.dropout[4])

    def forward(self, x):
        x0 = self.in_conv(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        return [x0, x1, x2, x3, x4]



# 不同的上采样方式
def _upsample_like1(src, tar):
    src = F.upsample(src, size=tar.shape[2:], mode='bilinear') # old version torch
    # src = F.interpolate(src, size=tar.shape[2:], mode='bicubic', align_corners=True)
    return src


class Decoder(nn.Module):
    def __init__(self, params):
        super(Decoder, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        assert (len(self.ft_chns) == 5)

        self.up1 = UpBlock(
            self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], dropout_p=0.0)
        self.up2 = UpBlock(
            self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p=0.0)
        self.up3 = UpBlock(
            self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=0.0)
        self.up4 = UpBlock(
            self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=0.0)

        self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class,
                                  kernel_size=3, padding=1)

    def forward(self, feature):
        x0 = feature[0]
        x1 = feature[1]
        x2 = feature[2]
        x3 = feature[3]
        x4 = feature[4]

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.up4(x, x0)
        output = self.out_conv(x)
        return output


class Decoder2(nn.Module):
    def __init__(self, params):
        super(Decoder2, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.bilinear = self.params['bilinear']
        assert (len(self.ft_chns) == 5)

        self.up1 = UpBlock(
            self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], dropout_p=0.0)
        self.up2 = UpBlock(
            self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p=0.0)
        self.up3 = UpBlock(
            self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=0.0)
        self.up4 = UpBlock(
            self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=0.0)



        self.out_conv4 = nn.Conv2d(self.ft_chns[4], self.n_class,
                                  kernel_size=3, padding=1)
        self.out_conv3 = nn.Conv2d(self.ft_chns[3], self.n_class,
                                   kernel_size=3, padding=1)
        self.out_conv2 = nn.Conv2d(self.ft_chns[2], self.n_class,
                                   kernel_size=3, padding=1)
        self.out_conv1 = nn.Conv2d(self.ft_chns[1], self.n_class,
                                   kernel_size=3, padding=1)
        self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class,
                                  kernel_size=3, padding=1)

        self.out_conv0 = nn.Conv2d(self.n_class * 5, self.n_class, kernel_size=3, padding=1)

    def forward(self, feature):
        x0 = feature[0]
        x1 = feature[1]
        x2 = feature[2]
        x3 = feature[3]
        x4 = feature[4]
        x4_1 = self.out_conv4(x4)
        x = self.up1(x4, x3)
        x3_1 = self.out_conv3(x)
        x = self.up2(x, x2)
        x2_1 = self.out_conv2(x)
        x = self.up3(x, x1)
        x1_1 = self.out_conv1(x)
        x = self.up4(x, x0)
        output = self.out_conv(x)

        x1_1 = _upsample_like1(x1_1, output)
        x2_1 = _upsample_like1(x2_1, output)
        x3_1 = _upsample_like1(x3_1, output)
        x4_1 = _upsample_like1(x4_1, output)
        output0 = self.out_conv0(torch.cat((output, x1_1, x2_1, x3_1, x4_1), 1))
        return output0


class DWConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

    def forward(self, x: torch.Tensor, H, W) -> torch.Tensor:
        B, N, C = x.shape
        tx = x.transpose(1, 2).view(B, C, H, W)
        conv_x = self.dwconv(tx)
        return conv_x.flatten(2).transpose(1, 2)


class MixFFN(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.fc1 = nn.Linear(c1, c2)
        self.dwconv = DWConv(c2)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(c2, c1)

    def forward(self, x: torch.Tensor, H, W) -> torch.Tensor:
        ax = self.act(self.dwconv(self.fc1(x), H, W))
        out = self.fc2(ax)
        return out


class MixFFN_skip(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.fc1 = nn.Linear(c1, c2)
        self.dwconv = DWConv(c2)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(c2, c1)
        self.norm1 = nn.LayerNorm(c2)
        self.norm2 = nn.LayerNorm(c2)
        self.norm3 = nn.LayerNorm(c2)

    def forward(self, x: torch.Tensor, H, W) -> torch.Tensor:
        ax = self.act(self.norm1(self.dwconv(self.fc1(x), H, W) + self.fc1(x)))
        out = self.fc2(ax)
        return out


class MLP_FFN(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.fc1 = nn.Linear(c1, c2)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(c2, c1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

class OverlapPatchEmbeddings(nn.Module):
    def __init__(self, img_size=256, patch_size=8, stride=4, padding=1, in_ch=3, dim=768):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_ch, dim, patch_size, stride, padding)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        px = self.proj(x)
        _, _, H, W = px.shape
        fx = px.flatten(2).transpose(1, 2)
        nfx = self.norm(fx)
        return nfx, H, W

class OverlapPatchEmbeddings1(nn.Module):
    def __init__(self, img_size=256, patch_size=8, padding=1, in_ch=3, dim=768):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_ch, dim, kernel_size=patch_size, padding=padding)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        px = self.proj(x)
        _, _, H, W = px.shape
        fx = px.flatten(2).transpose(1, 2)
        nfx = self.norm(fx)
        return nfx, H, W


class MLP(nn.Module):
    def __init__(self, dim, embed_dim):
        super().__init__()
        self.proj = nn.Linear(dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.flatten(2).transpose(1, 2)
        return self.proj(x)


class ConvModule(nn.Module):
    def __init__(self, c1, c2, k):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.activate = nn.ReLU(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activate(self.bn(self.conv(x)))


class LaplacianPyramid(nn.Module):
    def __init__(self, in_channels=64, pyramid_levels=3):
        """
        Constructs a Laplacian pyramid from an input tensor.

        Args:
            in_channels    (int): Number of input channels.
            pyramid_levels (int): Number of pyramid levels.

        Input:
            x : (B, in_channels, H, W)
        Output:
            Fused frequency attention map : (B, in_channels, in_channels)
        """
        super().__init__()
        self.in_channels = in_channels
        self.pyramid_levels = pyramid_levels
        sigma = 1.6
        s_value = 2 ** (1 / 3)

        self.sigma_kernels = [
            self.get_gaussian_kernel(2 * i + 3, sigma * s_value ** i)
            for i in range(pyramid_levels)
        ]

    def get_gaussian_kernel(self, kernel_size, sigma):
        kernel_weights = cv2.getGaussianKernel(ksize=kernel_size, sigma=sigma)
        kernel_weights = kernel_weights * kernel_weights.T
        kernel_weights = np.repeat(kernel_weights[None, ...], self.in_channels, axis=0)[:, None, ...]

        return torch.from_numpy(kernel_weights).float().to(device)

    def forward(self, x):
        G = x

        # Level 1
        L0 = Rearrange('b d h w -> b d (h w)')(G)
        L0_att = F.softmax(L0, dim=2) @ L0.transpose(1, 2)  # L_k * L_v
        L0_att = F.softmax(L0_att, dim=-1)

        # Next Levels
        attention_maps = [L0_att]
        pyramid = [G]

        for kernel in self.sigma_kernels:
            G = F.conv2d(input=G, weight=kernel, bias=None, padding='same', groups=self.in_channels)
            pyramid.append(G)

        for i in range(1, self.pyramid_levels):
            L = torch.sub(pyramid[i - 1], pyramid[i])
            L = Rearrange('b d h w -> b d (h w)')(L)
            L_att = F.softmax(L, dim=2) @ L.transpose(1, 2)
            attention_maps.append(L_att)

        return sum(attention_maps)


class DES(nn.Module):
    """
    Diversity-Enhanced Shortcut (DES) based on: "Gu et al.,
    Multi-Scale High-Resolution Vision Transformer for Semantic Segmentation.
    https://github.com/facebookresearch/HRViT
    """

    def __init__(self, in_features, out_features, bias=True, act_func: nn.Module = nn.GELU):
        super().__init__()
        _, self.p = self._decompose(min(in_features, out_features))
        self.k_out = out_features // self.p
        self.k_in = in_features // self.p
        self.proj_right = nn.Linear(self.p, self.p, bias=bias)
        self.act = act_func()
        self.proj_left = nn.Linear(self.k_in, self.k_out, bias=bias)

    def _decompose(self, n):
        assert n % 2 == 0, f"Feature dimension has to be a multiple of 2, but got {n}"
        e = int(math.log2(n))
        e1 = e // 2
        e2 = e - e // 2
        return 2 ** e1, 2 ** e2

    def forward(self, x):
        B = x.shape[:-1]
        x = x.view(*B, self.k_in, self.p)
        x = self.proj_right(x).transpose(-1, -2)

        if self.act is not None:
            x = self.act(x)

        x = self.proj_left(x).transpose(-1, -2).flatten(-2, -1)

        return x


class EfficientFrequencyAttention(nn.Module):
    """
    args:
        in_channels:    (int) : Embedding Dimension.
        key_channels:   (int) : Key Embedding Dimension,   Best: (in_channels).
        value_channels: (int) : Value Embedding Dimension, Best: (in_channels or in_channels//2).
        pyramid_levels  (int) : Number of pyramid levels.
    input:
        x : [B, D, H, W]
    output:
        Efficient Attention : [B, D, H, W]

    """

    def __init__(self, in_channels, key_channels, value_channels, pyramid_levels=3):
        super().__init__()
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.value_channels = value_channels

        self.keys = nn.Conv2d(in_channels, key_channels, 1)
        self.queries = nn.Conv2d(in_channels, key_channels, 1)
        self.values = nn.Conv2d(in_channels, value_channels, 1)
        self.reprojection = nn.Conv2d(value_channels, in_channels, 1)

        # Build a laplacian pyramid
        self.freq_attention = LaplacianPyramid(in_channels=in_channels, pyramid_levels=pyramid_levels)

        self.conv_dw = nn.Conv3d(in_channels, in_channels, kernel_size=(2, 1, 1), bias=False, groups=in_channels)

    def forward(self, x):
        n, _, h, w = x.size()

        # Efficient Attention
        keys = F.softmax(self.keys(x).reshape((n, self.key_channels, h * w)), dim=2)
        queries = F.softmax(self.queries(x).reshape(n, self.key_channels, h * w), dim=1)
        values = self.values(x).reshape((n, self.value_channels, h * w))
        context = keys @ values.transpose(1, 2)  # dk*dv
        attended_value = (context.transpose(1, 2) @ queries).reshape(n, self.value_channels, h, w)  # n*dv
        eff_attention = self.reprojection(attended_value)

        # Freqency Attention
        freq_context = self.freq_attention(x)
        freq_attention = (freq_context.transpose(1, 2) @ queries).reshape(n, self.value_channels, h, w)

        # Attention Aggregation: Efficient Frequency Attention (EF-Att) Block
        attention = torch.cat([eff_attention[:, :, None, ...], freq_attention[:, :, None, ...]], dim=2)
        attention = self.conv_dw(attention)[:, :, 0, ...]

        return attention


class FrequencyTransformerBlock(nn.Module):
    """
        Input:
            x : [b, (H*W), d], H, W

        Output:
            mx : [b, (H*W), d]
    """

    def __init__(self, in_dim, key_dim, value_dim, pyramid_levels=3, token_mlp='mix'):
        super().__init__()

        self.in_dim = in_dim
        self.norm1 = nn.LayerNorm(in_dim)
        self.attn = EfficientFrequencyAttention(in_channels=in_dim, key_channels=key_dim, value_channels=value_dim,
                                                pyramid_levels=pyramid_levels)

        self.norm2 = nn.LayerNorm(in_dim)
        if token_mlp == 'mix':
            self.mlp = MixFFN(in_dim, int(in_dim * 4))
        elif token_mlp == 'mix_skip':
            self.mlp = MixFFN_skip(in_dim, int(in_dim * 4))
        else:
            self.mlp = MLP_FFN(in_dim, int(in_dim * 4))

        self.des = DES(in_features=in_dim, out_features=in_dim, bias=True, act_func=nn.GELU)

    def forward(self, x: torch.Tensor, H, W) -> torch.Tensor:
        norm_1 = self.norm1(x)
        norm_1 = Rearrange('b (h w) d -> b d h w', h=H, w=W)(norm_1)

        attn = self.attn(norm_1)
        attn = Rearrange('b d h w -> b (h w) d')(attn)

        # DES Shortcut
        shortcut = self.des(x.reshape(x.shape[0], self.in_dim, -1).permute(0, 2, 1))

        tx = x + attn + shortcut
        mx = tx + self.mlp(self.norm2(tx), H, W)

        return mx


class Encoder(nn.Module):
    def __init__(self, image_size, in_dim, key_dim, value_dim, layers, pyramid_levels=3, token_mlp='mix_skip'):
        super().__init__()

        patch_specs = [
            (3, 2, 1),
            (3, 2, 1),
            (3, 2, 1),
            (3, 2, 1),
            (3, 2, 1)
        ]

        self.patch_embeds = nn.ModuleList()
        self.blocks = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(len(patch_specs)):
            patch_size, stride, padding = patch_specs[i]
            in_channels = in_dim[i - 1] if i > 0 else 3  # Input channels for the first patch_embed
            out_channels = in_dim[i]

            # Patch Embedding
            if i == 0:
                patch_embed = OverlapPatchEmbeddings1(image_size // (2 ** i), patch_size, padding,
                                                 in_channels, out_channels)
            if i > 0:
                patch_embed = OverlapPatchEmbeddings(image_size // (2 ** i), patch_size, stride, padding,
                                                 in_channels, out_channels)
            self.patch_embeds.append(patch_embed)

            # Transformer Blocks
            transformer_block = nn.ModuleList([
                FrequencyTransformerBlock(out_channels, key_dim[i], value_dim[i], pyramid_levels, token_mlp)
                for _ in range(layers[i])
            ])
            self.blocks.append(transformer_block)

            # Layer Normalization
            norm = nn.LayerNorm(out_channels)
            self.norms.append(norm)

    def forward(self, x):
        B = x.shape[0]
        outs = []

        for i in range(len(self.patch_embeds)):
            x, H, W = self.patch_embeds[i](x)
            for blk in self.blocks[i]:
                x = blk(x, H, W)
            x = self.norms[i](x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            outs.append(x)

        return outs


def Dropout(x, p=0.3):
    x = torch.nn.functional.dropout(x, p)
    return x


def FeatureDropout(x):
    attention = torch.mean(x, dim=1, keepdim=True)
    max_val, _ = torch.max(attention.view(
        x.size(0), -1), dim=1, keepdim=True)
    threshold = max_val * np.random.uniform(0.7, 0.9)
    threshold = threshold.view(x.size(0), 1, 1, 1).expand_as(attention)
    drop_mask = (attention < threshold).float()
    x = x.mul(drop_mask)
    return x


class FeatureNoise(nn.Module):
    def __init__(self, uniform_range=0.3):
        super(FeatureNoise, self).__init__()
        self.uni_dist = Uniform(-uniform_range, uniform_range)

    def feature_based_noise(self, x):
        noise_vector = self.uni_dist.sample(
            x.shape[1:]).to(x.device).unsqueeze(0)
        x_noise = x.mul(noise_vector) + x
        return x_noise

    def forward(self, x):
        x = self.feature_based_noise(x)
        return x



class CUT-Net(nn.Module):
    def __init__(self, in_chns, num_classes=1, n_skip_bridge=1, pyramid_levels=3, token_mlp_mode="mix_skip"):
        super(CUT-Net, self).__init__()
        self.n_skip_bridge = n_skip_bridge

        # Encoder configurations
        params1 = [[16, 32, 64, 128, 256],  # dims
                  [16, 32, 64, 128, 256],  # key_dim
                  [16, 32, 64, 128, 256],  # value_dim
                  [2, 2, 2, 2, 2]]  # layers

        params = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': num_classes,
                  'bilinear': False,
                  'acti_func': 'relu'}
        self.encoder1 = Encoder(image_size=256, in_dim=params1[0], key_dim=params1[1], value_dim=params1[2],
                                layers=params1[3], pyramid_levels=pyramid_levels, token_mlp=token_mlp_mode)
        self.encoder2 = Encoder_cnn(params)

        self.decoder = Decoder(params)
        self.decoder2 = Decoder2(params)



    def forward(self, x):
        x1 = x
        if x1.size()[1] == 1:
            x1 = x1.repeat(1, 3, 1, 1)
        output_enc = self.encoder1(x1)

        feature = self.encoder2(x)

        cnn_tras = []
        cnn_tras1 = []
        cnn_tras2 = []
        for i in range(len(feature)):
            cnn_tras.append(output_enc[i] + feature[i])
        for i in range(len(feature)):
            diff = torch.abs(output_enc[i] - feature[i])
            cnn_tras1.append(diff)
        for i in range(len(feature)):
            cnn_tras2.append(cnn_tras[i] + cnn_tras1[i])
        output = self.decoder(cnn_tras2)
        output1 = self.decoder2(cnn_tras2)
        return output, output1




if __name__ == "__main__":
    model = CUT-Net(in_chns=3, num_classes=1).to(device)
    x = torch.rand(4, 3, 256, 256).to(device)
    y, y1 = model(x)
    print(y.shape)
