import logging
import math
from timm.models.layers import trunc_normal_, DropPath
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import BatchNorm2d
logger = logging.getLogger(__name__)

class FourierUnit(nn.Module):

    def __init__(self, in_channels, out_channels, groups=1, spectral_pos_encoding=False, fft_norm='ortho'):
        # bn_layer not used
        super(FourierUnit, self).__init__()
        self.groups = groups
        self.fft_norm = fft_norm

        self.conv_layer = torch.nn.Conv2d(in_channels=in_channels * 2 + (2 if spectral_pos_encoding else 0),
                                          out_channels=out_channels * 2,
                                          kernel_size=1, stride=1, padding=0, groups=self.groups, bias=False)
        self.bn = torch.nn.BatchNorm2d(out_channels * 2)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        if torch.__version__ > '1.7.1' and '1.7.1' not in torch.__version__:
            x = x.to(torch.float32)
            batch = x.shape[0]

            # (batch, c, h, w/2+1, 2)
            fft_dim = (-2, -1)
            ffted = torch.fft.rfftn(x, dim=fft_dim, norm=self.fft_norm)
            ffted = torch.stack((ffted.real, ffted.imag), dim=-1)
            ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()  # (batch, c, 2, h, w/2+1)
            ffted = ffted.view((batch, -1,) + ffted.size()[3:])

            ffted = self.conv_layer(ffted)  # (batch, c*2, h, w/2+1)
            ffted = self.relu(self.bn(ffted.to(torch.float32)))
            ffted = ffted.to(torch.float32)

            ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(
                0, 1, 3, 4, 2).contiguous()  # (batch,c, t, h, w/2+1, 2)
            ffted = torch.complex(ffted[..., 0], ffted[..., 1])

            ifft_shape_slice = x.shape[-2:]
            output = torch.fft.irfftn(ffted, s=ifft_shape_slice, dim=fft_dim, norm=self.fft_norm)
        else:
            batch, c, h, w = x.size()
            r_size = x.size()

            # (batch, c, h, w/2+1, 2)
            ffted = torch.rfft(x, signal_ndim=2, normalized=True)
            # (batch, c, 2, h, w/2+1)
            ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()
            ffted = ffted.view((batch, -1,) + ffted.size()[3:])

            ffted = self.conv_layer(ffted)  # (batch, c*2, h, w/2+1)
            ffted = self.relu(self.bn(ffted))

            ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(
                0, 1, 3, 4, 2).contiguous()  # (batch,c, t, h, w/2+1, 2)

            output = torch.irfft(ffted, signal_ndim=2,
                                 signal_sizes=r_size[2:], normalized=True)

        return output
    
class SpectralTransform(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, groups=1, enable_lfu=True, separable_fu=False, **fu_kwargs):
        # bn_layer not used
        super(SpectralTransform, self).__init__()
        self.enable_lfu = enable_lfu
        if stride == 2:
            self.downsample = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        else:
            self.downsample = nn.Identity()

        self.stride = stride
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels //
                      2, kernel_size=1, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True)
        )
        fu_class = FourierUnit
        self.fu = fu_class(
            out_channels // 2, out_channels // 2, groups, **fu_kwargs)
        if self.enable_lfu:
            self.lfu = fu_class(
                out_channels // 2, out_channels // 2, groups)
        self.conv2 = torch.nn.Conv2d(
            out_channels // 2, out_channels, kernel_size=1, groups=groups, bias=False)

    def forward(self, x):

        x = self.downsample(x)
        x = self.conv1(x)
        output = self.fu(x)

        if self.enable_lfu:
            n, c, h, w = x.shape
            split_no = 2
            split_s = h // split_no
            xs = torch.cat(torch.split(
                x[:, :c // 4], split_s, dim=-2), dim=1).contiguous()
            xs = torch.cat(torch.split(xs, split_s, dim=-1),
                           dim=1).contiguous()
            xs = self.lfu(xs)
            xs = xs.repeat(1, 1, split_no, split_no).contiguous()
        else:
            xs = 0

        output = self.conv2(x + output + xs)

        return output


class FFC(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 ratio_gin, ratio_gout, stride=1, padding=0,
                 dilation=1, groups=1, bias=False, enable_lfu=True,
                 padding_type='reflect', gated=False, **spectral_kwargs):
        super(FFC, self).__init__()

        assert stride == 1 or stride == 2, "Stride should be 1 or 2."
        self.stride = stride

        in_cg = int(in_channels * ratio_gin)
        in_cl = in_channels - in_cg
        out_cg = int(out_channels * ratio_gout)
        out_cl = out_channels - out_cg
        # self.gnconv = gnconv(in_channels)
        self.ratio_gin = ratio_gin
        self.ratio_gout = ratio_gout
        self.global_in_num = in_cg

        module = nn.Identity if in_cl == 0 or out_cl == 0 else nn.Conv2d
        self.convl2l = module(in_cl, out_cl, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode=padding_type)
        module = nn.Identity if in_cl == 0 or out_cg == 0 else nn.Conv2d
        self.convl2g = module(in_cl, out_cg, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode=padding_type)
        module = nn.Identity if in_cg == 0 or out_cl == 0 else nn.Conv2d
        self.convg2l = module(in_cg, out_cl, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode=padding_type)
        module = nn.Identity if in_cg == 0 or out_cg == 0 else SpectralTransform
        self.convg2g = module(
            in_cg, out_cg, stride, 1 if groups == 1 else groups // 2, enable_lfu, **spectral_kwargs)

        self.gated = gated
        module = nn.Identity if in_cg == 0 or out_cl == 0 or not self.gated else nn.Conv2d
        self.gate = module(in_channels, 2, 1)

    def forward(self, x):
        x_l, x_g = x if type(x) is tuple else (x, 0)
        out_xl, out_xg = 0, 0

        if self.ratio_gout != 1:
            out_xl = self.convl2l(x_l) + self.convg2l(x_g)
        if self.ratio_gout != 0:
            out_xg = self.convl2g(x_l) + self.convg2g(x_g)

        return out_xl, out_xg


class FFC_BN_ACT(nn.Module):

    def __init__(self, in_channels, out_channels,
                 kernel_size, ratio_gin, ratio_gout,
                 stride=1, padding=0, dilation=1, groups=1, bias=False,
                 norm_layer=nn.BatchNorm2d, activation_layer=nn.Identity,
                 padding_type='reflect',
                 enable_lfu=True, **kwargs):
        super(FFC_BN_ACT, self).__init__()
        self.ffc = FFC(in_channels, out_channels, kernel_size,
                       ratio_gin, ratio_gout, stride, padding, dilation,
                       groups, bias, enable_lfu, padding_type=padding_type, **kwargs)
        lnorm = nn.Identity if ratio_gout == 1 else norm_layer
        gnorm = nn.Identity if ratio_gout == 0 else norm_layer
        global_channels = int(out_channels * ratio_gout)
        self.bn_l = lnorm(out_channels - global_channels)
        self.bn_g = gnorm(global_channels)

        lact = nn.Identity if ratio_gout == 1 else activation_layer
        gact = nn.Identity if ratio_gout == 0 else activation_layer
        self.act_l = lact(inplace=True)
        self.act_g = gact(inplace=True)

    def forward(self, x):
        x_l, x_g = self.ffc(x)
        x_l = self.act_l(self.bn_l(x_l.to(torch.float32)))
        x_g = self.act_g(self.bn_g(x_g.to(torch.float32)))
        return x_l, x_g
def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return gelu(x)


class GELU2(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)


def get_dwconv(dim, kernel, bias):
    return nn.Conv2d(dim, dim, kernel_size=kernel, padding=(kernel-1)//2 ,bias=bias, groups=dim)


class gnconv(nn.Module):
    def __init__(self, dim, order=5, gflayer=None, h=14, w=8, s=1.0):
        super().__init__()
        self.order = order
        self.dims = [dim // 2 ** i for i in range(order)]
        self.dims.reverse()
        self.proj_in = nn.Conv2d(dim, 2*dim, 1)

        if gflayer is None:
            self.dwconv = get_dwconv(sum(self.dims), 7, True)
        else:
            self.dwconv = gflayer(sum(self.dims), h=h, w=w)

        self.proj_out = nn.Conv2d(dim, dim, 1)

        self.pws = nn.ModuleList(
            [nn.Conv2d(self.dims[i], self.dims[i + 1], 1) for i in range(order - 1)]
        )

        self.scale = s
        print('[gnconv]', order, 'order with dims=', self.dims, 'scale=%.4f' % self.scale)

    def forward(self, x, mask=None, dummy=False):
        B, C, H, W = x.shape

        fused_x = self.proj_in(x)
        pwa, abc = torch.split(fused_x, (self.dims[0], sum(self.dims)), dim=1)

        dw_abc = self.dwconv(abc) * self.scale

        dw_list = torch.split(dw_abc, self.dims, dim=1)
        x = pwa * dw_list[0]

        for i in range(self.order - 1):
            x = self.pws[i](x) * dw_list[i + 1]

        x = self.proj_out(x)

        return x

class Block(nn.Module):
    r""" HorNet block
    """
    def __init__(self, dim, order = 5 , drop_path=0.1, layer_scale_init_value=1e-6, gnconv=gnconv):
        super().__init__()

        self.norm1 = LayerNorm(dim, eps=1e-6, data_format='channels_first')
        self.gnconv = gnconv(dim,order) # depthwise conv
        self.norm2 = LayerNorm(dim, eps=1e-6)
        self.conv0 = nn.Conv2d(in_channels=dim//2, out_channels=dim, kernel_size=3, stride=1, padding=1)
        self.conv1 = nn.Conv2d(in_channels=dim//2, out_channels=dim, kernel_size=1, stride=1, padding=0)
        # self.conv2 = nn.Conv2d(in_channels=dim//2, out_channels=dim//2, kernel_size=3, stride=1, padding=1)
        self.HydraAttention = HydraAttention(dim//2)
        self.bn3 = nn.BatchNorm2d(256)
        # self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.bn = torch.nn.BatchNorm2d(512)
        self.relu = torch.nn.ReLU(inplace=True)
        self.act = nn.GELU()
        self.ln2 = nn.LayerNorm(dim)
        # self.pwconv2 = nn.Linear(4 * dim, dim)

        self.gamma1 = nn.Parameter(layer_scale_init_value * torch.ones(dim),
                                    requires_grad=True) if layer_scale_init_value > 0 else None

        self.gamma2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            GELU2(),
            nn.Linear(4 * dim, dim),
            nn.Dropout(drop_path),
        )

    def forward(self, x,mask):
        B, C, H, W  = x.shape
        
        if self.gamma1 is not None:
            gamma1 = self.gamma1.view(C, 1, 1)
        else:
            gamma1 = 1
        x = x + self.drop_path(gamma1 * self.gnconv(self.norm1(x)))

        input = x
        [b, c, h, w] = x.shape
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm2(x) #(N, H, W, C)
        # print(x.size())
        x =x.permute(0, 3, 1, 2) # (N, C, H, W) 
        C = x.size(1)
        x_l, x_g = torch.split(x, C//2, dim=1)
        x_l = self.conv0(x_l)
        x_l_G,x_l_F = torch.split(x_l, C//2, dim=1)
        x_l = x_l_F*torch.sigmoid(x_l_G)
        x_g = self.conv1(x_g)
        x_g_G,x_g_F = torch.split(x_g, C//2, dim=1)
        x_g_F = self.HydraAttention(x_g_F,mask)
        x_g_F = x_g_F.reshape(b, h, w, C//2).permute(0, 3, 1, 2)
        x_g = x_g_F*torch.sigmoid(x_g_G)
        # x_l = torch.split(x_l1, C//4, dim=1)
        # x_g = torch.split(x_g1, C//4, dim=1)
        # x_l_l = self.conv0(x_l)
        # x_l_g = self.conv1(x_l)
        # x_l = x_l_g * torch.sigmoid(x_l_l)
        # x_g_l = self.conv2(x_g)
        # x_g_zp = x_g.permute(0,2,3,1).reshape(b,h * w,c//2)
        # x_g_g = self.HydraAttention(x_g,mask)
        # print("======x_g_g======")
        # print(x_g_g.size())
        # x_g_g = x_g_g.reshape(b, h, w, c//2).permute(0, 3, 1, 2)
        # x_l  = x_l_l + x_g_l
        # x_g = x_g_g+x_l_g
        # x_l = self.bn(x_l)
        # x_g = self.bn(x_g)
        # x_l = self.relu(x_l)
        # x_g = self.relu(x_g)
        x =torch.cat([x_l,x_g],dim=1)
        x = self.bn(x)
        x = self.act(x)
        # x = torch.cat([x_l,x_g],dim=1)
        # x = self.pwconv1(x)
        # x = self.act(x)
        # x = self.pwconv2(x)
        # if self.gamma2 is not None:
        #     x = self.gamma2 * x
        # x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        x = x.permute(0,2,3,1).reshape(b,h * w,c)
        x = x + self.mlp(self.ln2(x))
        x = x.reshape(b,h,w,c).permute(0,3,1,2)
        x = x.contiguous()
        return x
    
class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class HydraAttention(nn.Module):
    def __init__(self, d_model, output_layer='linear', dropout=0.1):
        super(HydraAttention, self).__init__()
        self.d_model = d_model
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out = nn.Linear(d_model, d_model) if output_layer == 'linear' else nn.Identity()
        self.dropout = nn.Dropout(dropout) 

    def forward(self, x, mask):
        '''x: (B, T, D)'''
        b, c, h, w = x.size()
        # print("============")
        # print(x.size())
        x = x.permute(0, 2, 3, 1).reshape(b, h * w, c)
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q / q.norm(dim=-1, keepdim=True)
        k = k / k.norm(dim=-1, keepdim=True)
        k = k.reshape(b, h, w, c).permute(0, 3, 1, 2)
        q = q.reshape(b, h, w, c).permute(0, 3, 1, 2)
        if mask is not None:
            mask = F.interpolate(mask, size=[h, w], mode='nearest')
            # print(mask.size())
            # print(k.size())
            # mask = mask.unsqueeze(1).expand(1, c, h, w) 
            k = k * (1-mask)
            q = q * (1-mask)
            k_mask = k * mask
            q_mask = q * mask
            k = k.permute(0, 2, 3, 1).reshape(b, h * w, c)
            q = q.permute(0, 2, 3, 1).reshape(b, h * w, c)
            k_mask = k_mask.permute(0, 2, 3, 1).reshape(b, h * w, c)
            q_mask = q_mask.permute(0, 2, 3, 1).reshape(b, h * w, c)
        # print("======++++++======")
        # print(k.size())
        # print(v.size())
        kvw = k * v
        kvw_mask = k_mask * v
        if self.dropout.p > 0:
            kvw = self.dropout(kvw.transpose(-1, -2)).transpose(-1, -2) # dropout in seq dimension 
        out_know = kvw.sum(dim=-2, keepdim=True) * q
        out_mask = kvw_mask.sum(dim=-2,keepdim=True) * q_mask
        out_know = out_know.reshape(b, h, w, c).permute(0, 3, 1, 2)
        out_mask = out_mask.reshape(b, h, w, c).permute(0, 3, 1, 2)
        out = out_know * (1 - mask) + mask * (0.3 * out_know + 0.7 * out_mask)
        out = out.permute(0, 2, 3, 1).reshape(b, h * w, c)
        return self.out(out)
# class HydraAttention(nn.Module):
#     def __init__(self, d_model, output_layer='linear', dropout=0.0):
#         super(HydraAttention, self).__init__()
#         self.d_model = d_model
#         self.qkv = nn.Linear(d_model, d_model * 3)
#         self.out = nn.Linear(d_model, d_model) if output_layer == 'linear' else nn.Identity()
#         self.dropout = nn.Dropout(dropout) 

#     def forward(self, x, mask=None):
#         '''x: (B, T, D)'''
#         q, k, v = self.qkv(x).chunk(3, dim=-1)
#         q = q / q.norm(dim=-1, keepdim=True)
#         k = k / k.norm(dim=-1, keepdim=True)
#         if mask is not None:
#             k = k.masked_fill(mask.unsqueeze(-1), 0)
#         kvw = k * v
#         if self.dropout.p > 0:
#             kvw = self.dropout(kvw.transpose(-1, -2)).transpose(-1, -2) # dropout in seq dimension 
#         out = kvw.sum(dim=-2, keepdim=True) * q
#         return self.out(out)
# class FullyAttentionalBlock(nn.Module):
#     def __init__(self, dim, norm_layer=BatchNorm2d):
#         super(FullyAttentionalBlock, self).__init__()
#         self.conv1 = nn.Linear(dim, dim)
#         self.conv2 = nn.Linear(dim, dim)
#         self.conv = nn.Sequential(nn.Conv2d(dim, dim, 3, stride=1, padding=1, bias=False),
#                                   norm_layer(dim),
#                                   nn.ReLU())

#         self.softmax = nn.Softmax(dim=-1)
#         self.gamma = nn.Parameter(torch.zeros(1))

#     def forward(self, x):
#         batch_size, _, height, width = x.size()

#         feat_h = x.permute(0, 3, 1, 2).contiguous().view(batch_size * width, -1, height)
#         feat_w = x.permute(0, 2, 1, 3).contiguous().view(batch_size * height, -1, width)
#         encode_h = self.conv1(F.avg_pool2d(x, [1, width]).view(batch_size, -1, height).permute(0, 2, 1).contiguous())
#         encode_w = self.conv2(F.avg_pool2d(x, [height, 1]).view(batch_size, -1, width).permute(0, 2, 1).contiguous())

#         energy_h = torch.matmul(feat_h, encode_h.repeat(width, 1, 1))
#         energy_w = torch.matmul(feat_w, encode_w.repeat(height, 1, 1))
#         full_relation_h = self.softmax(energy_h)  # [b*w, c, c]
#         full_relation_w = self.softmax(energy_w)
#         '''
#         计算两个tensor的矩阵乘法
#         '''
#         full_aug_h = torch.bmm(full_relation_h, feat_h).view(batch_size, width, -1, height).permute(0, 2, 3, 1)
#         full_aug_w = torch.bmm(full_relation_w, feat_w).view(batch_size, height, -1, width).permute(0, 2, 1, 3)
#         out = self.gamma * (full_aug_h + full_aug_w) + x
#         out = self.conv(out)
#         return out
# class SelfAttention(nn.Module):
#     """
#     A vanilla multi-head masked self-attention layer with a projection at the end.
#     It is possible to use torch.nn.MultiheadAttention here but I am including an
#     explicit implementation here to show that there is nothing too scary here.
#     """

#     def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop):
#         super().__init__()
#         assert n_embd % n_head == 0
#         # key, query, value projections for all heads
#         self.key = nn.Linear(n_embd, n_embd)
#         self.query = nn.Linear(n_embd, n_embd)
#         self.value = nn.Linear(n_embd, n_embd)
#         # regularization
#         self.attn_drop = nn.Dropout(attn_pdrop)
#         self.resid_drop = nn.Dropout(resid_pdrop)
#         # output projection
#         self.proj = nn.Linear(n_embd, n_embd)
#         self.n_head = n_head

#     def forward(self, x, mask=None, rel_pos=None, return_att=False):
#         B, T, C = x.size()

#         # calculate query, key, values for all heads in batch and move head forward to be the batch dim
#         k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
#         q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
#         v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

#         # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
#         att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
#         if rel_pos is not None:
#             att += rel_pos
#         if mask is not None:  # maybe we don't need mask in axial-transformer
#             # mask:[B,1,L(1),L]
#             att = att.masked_fill(mask == 1, float('-inf'))

#         att = F.softmax(att, dim=-1)
#         att = self.attn_drop(att)
#         y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
#         y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

#         # output projection
#         y = self.resid_drop(self.proj(y))

#         if return_att:
#             return y, att
#         else:
#             return y


# class AxialAttention(nn.Module):
#     def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop, H, W,
#                  add_rel_pos=False, rel_pos_bins=32):
#         super().__init__()

#         self.rln1 = nn.LayerNorm(n_embd, eps=1e-4)
#         self.cln1 = nn.LayerNorm(n_embd, eps=1e-4)
#         self.ln2 = nn.LayerNorm(n_embd, eps=1e-4)
#         self.attn_row = SelfAttention(n_embd, n_head, attn_pdrop, resid_pdrop)
#         self.attn_col = SelfAttention(n_embd, n_head, attn_pdrop, resid_pdrop)
#         self.ff = nn.Sequential(
#             nn.Linear(n_embd, 4 * n_embd),
#             GELU(),
#             nn.Linear(4 * n_embd, n_embd),
#             nn.Dropout(resid_pdrop),
#         )

#         self.add_rel_pos = add_rel_pos
#         # self.rel_pos_bins = rel_pos_bins
#         self.row_rel_pos_bias = nn.Linear(2 * H - 1, n_head, bias=False)
#         self.col_rel_pos_bias = nn.Linear(2 * W - 1, n_head, bias=False)

#     def _cal_1d_pos_emb(self, hidden_states, rel_pos_onehot_size, row=True):
#         # hidden_states:[B,L,D], [1,L]
#         position_ids = torch.arange(hidden_states.shape[1], dtype=torch.long).unsqueeze(0)
#         # [1,1,L]-[1,L,1]-->[1,L,L]
#         rel_pos_mat = position_ids.unsqueeze(-2) - position_ids.unsqueeze(-1)
#         rel_pos_mat -= torch.min(rel_pos_mat)
#         # [1,L,L]->[1,L,L,D]
#         rel_pos = F.one_hot(rel_pos_mat, num_classes=rel_pos_onehot_size * 2 - 1).type_as(hidden_states)
#         # [1,L,L,D]->[1,L,L,H]->[1,H,L,L]
#         if row:
#             rel_pos = self.row_rel_pos_bias(rel_pos).permute(0, 3, 1, 2)
#         else:
#             rel_pos = self.col_rel_pos_bias(rel_pos).permute(0, 3, 1, 2)

#         rel_pos = rel_pos.contiguous()
#         return rel_pos

#     def forward(self, x, return_att=False, debug=False):  # x:[B,C,H,W], mask:[B,1,H,W]
#         [b, c, h, w] = x.shape
#         x0 = x.clone()
#         x0 = x0.permute(0, 2, 3, 1).reshape(b, h * w, c)
#         mask_row = None
#         mask_col = None

#         # ROW ATTENTION
#         x = x.permute(0, 3, 2, 1).reshape(b * w, h, c)
#         if self.add_rel_pos:
#             row_rel_pos = self._cal_1d_pos_emb(x, rel_pos_onehot_size=h, row=True)
#         else:
#             row_rel_pos = None
#         x_row = self.attn_row(self.rln1(x), mask_row, row_rel_pos, return_att=return_att)
#         if return_att:
#             x_row, att_row = x_row
#         else:
#             att_row = None
#         x_row = x_row.reshape(b, w, h, c).permute(0, 2, 1, 3).reshape(b, h * w, c)

#         # COL ATTENTION
#         x = x.reshape(b, w, h, c).permute(0, 2, 1, 3).reshape(b * h, w, c)
#         if self.add_rel_pos:
#             col_rel_pos = self._cal_1d_pos_emb(x, rel_pos_onehot_size=w, row=False)
#         else:
#             col_rel_pos = None
#         x_col = self.attn_col(self.cln1(x), mask_col, col_rel_pos, return_att=return_att)
#         if return_att:
#             x_col, att_col = x_col
#         else:
#             att_col = None
#         x_col = x_col.reshape(b, h, w, c).reshape(b, h * w, c)

#         # [B,HW,C]
#         x = x0 + x_row + x_col
#         x = x + self.ff(self.ln2(x))
#         x = x.reshape(b, h, w, c).permute(0, 3, 1, 2)

#         x = x.contiguous()

#         if return_att:
#             # att_row:[BW,head,H,H]
#             att_row = torch.mean(att_row, dim=1).reshape(b, w, h, h)
#             att_row = torch.sum(att_row, dim=2).permute(0, 2, 1)  # [b,h,w]
#             # att_col:[BH,head,W,W]
#             att_col = torch.mean(att_col, dim=1).reshape(b, h, w, w)
#             att_col = torch.sum(att_col, dim=2)
#             att_score = att_row * att_col
#             return x, att_score
#         else:
#             return x



