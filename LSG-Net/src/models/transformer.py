import logging
import math

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import BatchNorm2d
logger = logging.getLogger(__name__)


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


class SelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        # output projection
        self.proj = nn.Linear(n_embd, n_embd)
        self.n_head = n_head

    def forward(self, x, mask=None, rel_pos=None, return_att=False):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        if rel_pos is not None:
            att += rel_pos
        if mask is not None:  # maybe we don't need mask in axial-transformer
            # mask:[B,1,L(1),L]
            att = att.masked_fill(mask == 1, float('-inf'))

        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))

        if return_att:
            return y, att
        else:
            return y


class AxialAttention(nn.Module):
    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop, H, W,
                 add_rel_pos=True, rel_pos_bins=32):
        super().__init__()

        self.rln1 = nn.LayerNorm(n_embd, eps=1e-4)
        self.cln1 = nn.LayerNorm(n_embd, eps=1e-4)
        self.ln2 = nn.LayerNorm(n_embd, eps=1e-4)
        self.attn_row = SelfAttention(n_embd, n_head, attn_pdrop, resid_pdrop)
        self.attn_col = SelfAttention(n_embd, n_head, attn_pdrop, resid_pdrop)
        self.ff = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(resid_pdrop),
        )

        self.add_rel_pos = add_rel_pos
        # self.rel_pos_bins = rel_pos_bins
        self.row_rel_pos_bias = nn.Linear(2 * H - 1, n_head, bias=False)
        self.col_rel_pos_bias = nn.Linear(2 * W - 1, n_head, bias=False)

    def _cal_1d_pos_emb(self, hidden_states, rel_pos_onehot_size, row=True):
        # hidden_states:[B,L,D], [1,L]
        position_ids = torch.arange(hidden_states.shape[1], dtype=torch.long).unsqueeze(0)
        # [1,1,L]-[1,L,1]-->[1,L,L]
        rel_pos_mat = position_ids.unsqueeze(-2) - position_ids.unsqueeze(-1)
        rel_pos_mat -= torch.min(rel_pos_mat)
        # [1,L,L]->[1,L,L,D]
        rel_pos = F.one_hot(rel_pos_mat, num_classes=rel_pos_onehot_size * 2 - 1).type_as(hidden_states)
        # [1,L,L,D]->[1,L,L,H]->[1,H,L,L]
        if row:
            rel_pos = self.row_rel_pos_bias(rel_pos).permute(0, 3, 1, 2)
        else:
            rel_pos = self.col_rel_pos_bias(rel_pos).permute(0, 3, 1, 2)

        rel_pos = rel_pos.contiguous()
        return rel_pos

    def forward(self, x, return_att=False, debug=False):  # x:[B,C,H,W], mask:[B,1,H,W]
        [b, c, h, w] = x.shape
        x0 = x.clone()
        x0 = x0.permute(0, 2, 3, 1).reshape(b, h * w, c)
        mask_row = None
        mask_col = None

        # ROW ATTENTION
        x = x.permute(0, 3, 2, 1).reshape(b * w, h, c)
        if self.add_rel_pos:
            row_rel_pos = self._cal_1d_pos_emb(x, rel_pos_onehot_size=h, row=True)
        else:
            row_rel_pos = None
        x_row = self.attn_row(self.rln1(x), mask_row, row_rel_pos, return_att=return_att)
        if return_att:
            x_row, att_row = x_row
        else:
            att_row = None
        x_row = x_row.reshape(b, w, h, c).permute(0, 2, 1, 3).reshape(b, h * w, c)

        # COL ATTENTION
        x = x.reshape(b, w, h, c).permute(0, 2, 1, 3).reshape(b * h, w, c)
        if self.add_rel_pos:
            col_rel_pos = self._cal_1d_pos_emb(x, rel_pos_onehot_size=w, row=False)
        else:
            col_rel_pos = None
        x_col = self.attn_col(self.cln1(x), mask_col, col_rel_pos, return_att=return_att)
        if return_att:
            x_col, att_col = x_col
        else:
            att_col = None
        x_col = x_col.reshape(b, h, w, c).reshape(b, h * w, c)

        # [B,HW,C]
        x = x0 + x_row + x_col
        x = x + self.ff(self.ln2(x))
        x = x.reshape(b, h, w, c).permute(0, 3, 1, 2)

        x = x.contiguous()

        if return_att:
            # att_row:[BW,head,H,H]
            att_row = torch.mean(att_row, dim=1).reshape(b, w, h, h)
            att_row = torch.sum(att_row, dim=2).permute(0, 2, 1)  # [b,h,w]
            # att_col:[BH,head,W,W]
            att_col = torch.mean(att_col, dim=1).reshape(b, h, w, w)
            att_col = torch.sum(att_col, dim=2)
            att_score = att_row * att_col
            return x, att_score
        else:
            return x


class BlockAxial(AxialAttention):

    def __init__(self, config):
        super().__init__(config.n_embd, config.n_head, config.attn_pdrop, config.resid_pdrop, 32, 32)


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)

        self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head

        self.config = config

    def forward(self, x, layer_past=None):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y
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
    def __init__(self, dim, order = 5 , drop_path=0., layer_scale_init_value=1e-6, gnconv=gnconv):
        super().__init__()

        self.norm1 = LayerNorm(dim, eps=1e-6, data_format='channels_first')
        self.gnconv = gnconv(dim,order) # depthwise conv
        self.norm2 = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)

        self.gamma1 = nn.Parameter(layer_scale_init_value * torch.ones(dim),
                                    requires_grad=True) if layer_scale_init_value > 0 else None

        self.gamma2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        B, C, H, W  = x.shape
        
        if self.gamma1 is not None:
            gamma1 = self.gamma1.view(C, 1, 1)
        else:
            gamma1 = 1
        x = x + self.drop_path(gamma1 * self.gnconv(self.norm1(x)))

        input = x
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm2(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma2 is not None:
            x = self.gamma2 * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
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

class FullyAttentionalBlock(nn.Module):
    def __init__(self, dim, norm_layer=BatchNorm2d):
        super(FullyAttentionalBlock, self).__init__()
        self.conv1 = nn.Linear(dim, dim)
        self.conv2 = nn.Linear(dim, dim)
        self.conv = nn.Sequential(nn.Conv2d(dim, dim, 3, stride=1, padding=1, bias=False),
                                  norm_layer(dim),
                                  nn.ReLU())

        self.softmax = nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, _, height, width = x.size()

        feat_h = x.permute(0, 3, 1, 2).contiguous().view(batch_size * width, -1, height)
        feat_w = x.permute(0, 2, 1, 3).contiguous().view(batch_size * height, -1, width)
        encode_h = self.conv1(F.avg_pool2d(x, [1, width]).view(batch_size, -1, height).permute(0, 2, 1).contiguous())
        encode_w = self.conv2(F.avg_pool2d(x, [height, 1]).view(batch_size, -1, width).permute(0, 2, 1).contiguous())

        energy_h = torch.matmul(feat_h, encode_h.repeat(width, 1, 1))
        energy_w = torch.matmul(feat_w, encode_w.repeat(height, 1, 1))
        full_relation_h = self.softmax(energy_h)  # [b*w, c, c]
        full_relation_w = self.softmax(energy_w)
        '''
        计算两个tensor的矩阵乘法
        '''
        full_aug_h = torch.bmm(full_relation_h, feat_h).view(batch_size, width, -1, height).permute(0, 2, 3, 1)
        full_aug_w = torch.bmm(full_relation_w, feat_w).view(batch_size, height, -1, width).permute(0, 2, 1, 3)
        out = self.gamma * (full_aug_h + full_aug_w) + x
        out = self.conv(out)
        return out
    
    
class HydraAttention(nn.Module):
    def __init__(self, d_model, output_layer='linear', dropout=0.0):
        super(HydraAttention, self).__init__()
        self.d_model = d_model
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out = nn.Linear(d_model, d_model) if output_layer == 'linear' else nn.Identity()
        self.dropout = nn.Dropout(dropout) 

    def forward(self, x, mask=None):
        '''x: (B, T, D)'''
        b, c, h, w = x.size()
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
        kvw = k * v
        kvw_mask = k_mask * v
        if self.dropout.p > 0:
            kvw = self.dropout(kvw.transpose(-1, -2)).transpose(-1, -2) # dropout in seq dimension 
        out_know = kvw.sum(dim=-2, keepdim=True) * q
        out_mask = kvw_mask.sum(dim=-2,keepdim=True) * q_mask
        out_know = out_know.reshape(b, h, w, c).permute(0, 3, 1, 2)
        out_mask = out_mask.reshape(b, h, w, c).permute(0, 3, 1, 2)
        out = out_know * (1 - mask) + mask * (0.7 * out_know + 0.3 * out_mask)
        out = out.permute(0, 2, 3, 1).reshape(b, h * w, c)
        return self.out(out)
    
class EfficientAttention(nn.Module):
    def __init__(self, in_channels, dim, head_count, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.head_count = head_count
        self.dim = dim

        self.keys = nn.Conv2d(in_channels, dim, 1)
        self.queries = nn.Conv2d(in_channels, dim, 1)
        self.values = nn.Conv2d(in_channels, dim, 1)
        if dim != out_channels:
            self.reprojection = nn.Conv2d(dim, out_channels, 1)
        else:
            self.reprojection = None

    def forward(self, input_, mask=None, return_scores=False):
        n, _, h, w = input_.size()
        keys = self.keys(input_)
        queries = self.queries(input_)
        values = self.values(input_)
        head_channels = self.dim // self.head_count

        if mask is not None:
            # [b,1,h,w]
            mask = F.interpolate(mask, size=[h, w], mode='nearest')
            keys += (mask * -10000.0)
            queries += (mask * -10000.0)

        keys = keys.reshape((n, self.dim, h * w))  # [b,d,h*w]
        queries = queries.reshape(n, self.dim, h * w)
        values = values.reshape((n, self.dim, h * w))

        attended_values = []
        scores = 0
        '''
        为什么要分成4次 对其进行softmax？？？？
        为什么通道那里不是卷积之后是256/4，而是reshape？？？？
         @ 表示矩阵乘法
        '''
        for i in range(self.head_count):
            key = F.softmax(keys[:, i * head_channels: (i + 1) * head_channels, :], dim=2)
            query = F.softmax(queries[:, i * head_channels: (i + 1) * head_channels, :], dim=1)
            value = values[:, i * head_channels: (i + 1) * head_channels, :]
            context = key @ value.transpose(1, 2)  # [b, d, d]
            attended_value = (context.transpose(1, 2) @ query).reshape(n, head_channels, h, w)
            attended_values.append(attended_value)
            if return_scores:
                score = torch.matmul(query.transpose(1, 2), key)  # [b, hw, hw]
                score = torch.mean(score, dim=1).reshape([n, h, w])
                scores += score

        aggregated_values = torch.cat(attended_values, dim=1)
        if self.reprojection is not None:
            reprojected_value = self.reprojection(aggregated_values)
        else:
            reprojected_value = aggregated_values

        attention = reprojected_value + input_

        if return_scores:
            max_value, _ = torch.max(scores.reshape([n, h * w]), dim=1)
            max_value = max_value[:, None, None]
            scores = scores / (max_value + 1e-5)
            scores = scores.unsqueeze(1)
            scores = scores.detach()
            return attention, scores
        else:
            return attention
class my_Block_2(nn.Module):
    """ Transformer block with original GELU2 """

    def __init__(self, config,ch):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = HydraAttention(ch * 4)
        self.fully = FullyAttentionalBlock(ch * 4)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            GELU2(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x,mask=None):
        [b, c, h, w] = x.shape
        x = x.permute(0, 2, 3, 1).reshape(b, h * w, c)
        x1 = self.ln1(x)
        x = x1.reshape(b, h, w, c).permute(0, 3, 1, 2)
        x2 = self.attn(x,mask)
        x2 = x2.reshape(b, h, w, c).permute(0, 3, 1, 2)
        x3 = self.fully(x)
        x = x + x2 + x3
        x = x.permute(0, 2, 3, 1).reshape(b, h * w, c)
        x = x + self.mlp(self.ln2(x))
        x = x.reshape(b, h, w, c).permute(0, 3, 1, 2)
        x = x.contiguous()
        return x

