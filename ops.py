import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
from timm.layers import trunc_normal_, to_2tuple
from timm.layers.norm import GroupNorm1, LayerNorm2d

GLOBAL_EPS = 5e-4  # fp16: 2^(-14) ~ 65504

# for reparameterization trick
def fuse_conv_bn(kernel, bn):
    weight = kernel * (bn.weight / (bn.running_var + bn.eps).sqrt()).reshape(-1, 1, 1, 1)
    bias = bn.bias - bn.running_mean * bn.weight / (bn.running_var + bn.eps).sqrt()
    return weight, bias


# for reparameterization trick
def get_id_tensor(c):
    oup = c.out_channels
    g_d, k1, k2 = c.weight.shape[1:]
    id_weight = torch.zeros(
        (oup, g_d, k1, k2), 
        device=c.weight.device, 
        dtype=c.weight.dtype
    )
    for i in range(oup):
        id_weight[i, i % g_d, k1 // 2, k2 // 2] = 1
    return id_weight


# for linear attention
def use_linear(q, v):
    Dkq, Dv = q.shape[-2], v.shape[-2]
    Nvk, Nq = v.shape[-1], q.shape[-1]
    return Nvk * Nq * (Dkq + Dv) > (Nvk + Nq) * Dkq * Dv


# remove prefix
def remove_prefix(name, prefix):
    for p in [f"{prefix}{x}" for x in ['_', '.', '/', '-', '+']]:
        name = name.replace(p, '')
    return name



class Scale(nn.Module):
    r""" Learnable scale for specified dimension(s)

    Args:
        dim (int): number of channels
        init_value (float): initial value of scale
        shape (tuple): shape of scale vector when element-wise multiply
            Default: None, which means shape = (1, dim, 1, 1), suitable for (B, C, H, W)
    """
    def __init__(self, dim, init_value=1., shape=None):
        super().__init__()
        self.shape = (1, dim, 1, 1) if shape is None else shape
        self.alpha = nn.Parameter(init_value * torch.ones(dim))

    def forward(self, x):
        return x * self.alpha.view(*self.shape)


class ConvNorm(nn.Sequential):
    r""" Convolution with normalization

    Args:
        inp, oup (int): number of input / output channels
        k, s, p, d, g (int): kernel size, stride, padding, dilation, groups
        bn_w_init (float): weight initialization, Default: 1.
            Suggestion: use 0. when this module directly add residual, kinda like LayerScaleInit=0. ?
    """
    def __init__(self, inp, oup, k=1, s=1, p=0, d=1, g=1, bn_w_init=1.):
        super().__init__()
        self.conv_args = (inp, oup, k, s, p, d, g)
        self.add_module('c', nn.Conv2d(*self.conv_args, bias=False))
        self.add_module('bn', nn.BatchNorm2d(oup))
        nn.init.constant_(self.bn.weight, bn_w_init)
        nn.init.constant_(self.bn.bias, 0.)

    @torch.no_grad()
    def fuse(self):
        w, b = fuse_conv_bn(self.c.weight, self.bn)
        m = nn.Conv2d(*self.conv_args, device=w.device, dtype=w.dtype)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class RepConv(nn.Module):
    r""" Re-parameterized Convolution
    
    Args:
        inp, oup (int): number of input / output channels
        k, s, g (int): kernel size, stride, groups
        use_rep (bool): whether to use reparameterization (mutil-kernel conv), Default: True
        res (bool): whether to use skip connection, Default: False
        bn_w_init (float): weight initialization, Default: 1.
            Suggestion: use 0. when this module directly add residual
    """
    def __init__(self, inp, oup, k=1, s=1, g=1, use_rep=True, res=False, bn_w_init=1.):
        super().__init__()
        self.kernel = to_2tuple(k)
        self.res = res
        if self.res:
            assert inp == oup and s == 1, \
                f"make sure inp({inp}) == oup({oup}) and stride({s}) == 1 when using skip connection"
            self.scale_res = Scale(oup)
            bn_w_init = 0.
        
        # make sure k > 0
        k_lst = [x for x in range(k, -1, -2) if x > 0] if use_rep else [k]
        self.ops = nn.ModuleList([
            ConvNorm(inp, oup, _k, s, (_k // 2), g=g, bn_w_init=bn_w_init) 
            for _k in k_lst
        ])
        self.repr_str = (
            f"# {'RepConv' if g == 1 else 'RepDWConv'}:"
            f" kernels={k_lst}"
            f"{', w. res' if res else ''}"
        )
    
    def extra_repr(self):
        return self.repr_str
    
    def forward(self, x, out=0):
        for op in self.ops:
            out = out + op(x)
        if self.res:
            out = out + self.scale_res(x)
        return out

    @torch.no_grad()
    def fuse(self):
        c = self.ops[0]
        if hasattr(c, 'fuse'):
            c = c.fuse()
        
        lk = self.kernel
        weight, bias = 0, 0
        for op in self.ops:
            if hasattr(op, 'fuse'):
                op = op.fuse()
            w, b, sk = op.weight, op.bias, op.kernel_size
            if sk != lk:
                pad = (lk[0] - sk[0]) // 2, (lk[1] - sk[1]) // 2
                w = nn.functional.pad(w, (pad[1], pad[1], pad[0], pad[0]))
            b = b if b is not None else 0
            weight, bias = weight + w, bias + b
        
        if self.res:
            weight += get_id_tensor(c) * self.scale_res.alpha.view(-1, 1, 1, 1)

        # fuse into one conv
        rep_conv = nn.Conv2d(
            in_channels=c.in_channels, out_channels=c.out_channels, kernel_size=c.kernel_size, 
            stride=c.stride, padding=c.padding, dilation=c.dilation, groups=c.groups, 
            device=weight.device, dtype=weight.dtype
        )
        rep_conv.weight.data.copy_(weight)
        rep_conv.bias.data.copy_(bias)

        # set extra_repr for debug
        if len(self.ops) > 1:
            repr_str = f"{self.repr_str}\n{rep_conv.extra_repr()}"
            rep_conv.extra_repr = partial(lambda m: repr_str, rep_conv)
        return rep_conv


class BNLinear(nn.Sequential):
    r""" Batch Normalization + Linear
    
    Args:
        inp, oup (int): number of input / output channels
        std (float): standard deviation, Default: 0.02
        use_conv2d (bool): whether to use conv2d (kernel=1) instead of linear, Default: False
    """
    def __init__(self, inp, oup, std=0.02, use_conv2d=False):
        super().__init__()
        bn = nn.BatchNorm2d(inp) if use_conv2d else nn.BatchNorm1d(inp)
        l = nn.Conv2d(inp, oup, 1) if use_conv2d else nn.Linear(inp, oup)
        
        trunc_normal_(l.weight, std=std)
        nn.init.constant_(l.bias, 0)
        self.add_module('bn', bn)
        self.add_module('l', l)

    @torch.no_grad()
    def fuse(self):
        bn, l = self._modules.values()
        use_conv2d = isinstance(l, nn.Conv2d)
        weight = l.weight[:,:,0,0] if use_conv2d else l.weight
        
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        b = bn.bias - bn.running_mean * bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = weight * w[None, :]
        b = (weight @ b[:, None]).view(-1) + l.bias
        inp, oup, device = w.size(1), w.size(0), w.device
        if use_conv2d:
            m = nn.Conv2d(inp, oup, 1, 1, 0, device=device)
            w = w[:,:,None,None]
        else:
            m = nn.Linear(inp, oup, device=device)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


def ConvGate(inp, oup, act, conv1d=False):
    r""" Conv(Linear) + Gate
    
    Args:
        inp, oup (int): number of input / output channels
        act (None or nn.Module): activation function
        conv1d (bool): whether to use conv1d instead of conv2d, Default: False
    """
    Conv = nn.Conv1d if conv1d else nn.Conv2d
    c = Conv(inp, oup, kernel_size=1, bias=True)
    # override init of c, set weight near zero, bias=1
    # (1.make this `gate branch` more stable; 2.reduces the impact to another branch) for early epochs
    trunc_normal_(c.weight, std=GLOBAL_EPS)
    nn.init.ones_(c.bias)
    if act is None:
        return c
    
    assert isinstance(act, nn.Module), f"Expected `act({act})` to be an `nn.Module` instance."
    return nn.Sequential(c, act)



def get_act(name):
    name = name.lower()
    if name == 'relu':
        return nn.ReLU()
    if name == 'gelu':
        return nn.GELU()
    if name == 'sigmoid':
        return nn.Sigmoid()
    if name == 'silu':
        return nn.SiLU()
    if name == 'elu1':
        return ELU_1()
    if name in ['id', 'identity']:
        return nn.Identity()
    raise NotImplementedError(f'Unknown act: {name}')


class ELU_1(nn.Module):
    """ for linear attention """
    def forward(self, x):
        return F.elu(x) + 1


def get_norm(name, dim, **kwargs):
    name = name.lower()
    if name in ['id', 'identity']:
        return nn.Identity()
    
    w_init = kwargs.pop('w_init', 1.)  # initial value for weight
    b_init = kwargs.pop('b_init', 0.)  # initial value for bias
    norm = None
    
    # BatchNorm
    if name == 'bn':
        norm = nn.BatchNorm1d(dim, **kwargs)
    if name == 'bn2d':
        norm = nn.BatchNorm2d(dim, **kwargs)
    
    # LayerNorm
    if name == 'ln':
        norm = nn.LayerNorm(dim, **kwargs)
    if name == 'ln2d':  # implemented by timm
        norm = LayerNorm2d(dim, **kwargs)
    
    # GroupNorm
    if name == 'gn':
        norm = nn.GroupNorm(dim, **kwargs)
    if name == 'gn1' or name == 'mln':
        # mln is for `modified layer norm`, from metaformer
        # can be implemented by setting group=1 in GroupNorm
        norm = GroupNorm1(dim, **kwargs)
        
    # RMSNorm
    if name == 'rms':
        norm = RMSNorm(dim, **kwargs)
        
    # Ours
    if name == 'mrms':
        norm = ModifiedRMSNorm(dim, w_init=w_init, **kwargs)
    
    # init the values of weight and bias
    if norm is not None:
        if hasattr(norm, 'weight'):
            nn.init.constant_(norm.weight, w_init)
        if hasattr(norm, 'bias'):
            nn.init.constant_(norm.bias, b_init)
        return norm
    else:
        raise NotImplementedError(f'Unknown norm: {name}')


class ModifiedRMSNorm(nn.Module):
    r""" Modified Root Mean Square Normalization.
    The only difference with RMSNorm is that MRMSNorm is taken over all dimensions 
    except the batch dimension.
    
    Modified RMSNorm:
    y = x / MRMS(x) * gamma, where MRMS(x) = sqrt( sum_{i=1}^{n}(x^2) / n + eps )

    Args:
        dim (int): number of channels
        eps (float): small number to avoid division by zero, default: 1e-5
        w_init (float): initial value for weight, default: 1.
        affine (boolean): whether to use affine transformation, default: True
        
    Shape:
        - Input / Output: (B, C, *)
    """
    def __init__(self, dim, eps=1e-5, w_init=1., affine=True):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.w_init = w_init
        self.affine = affine
        if self.affine:
            self.weight = nn.Parameter(w_init * torch.ones(dim))

    def extra_repr(self):
        return "{dim}, eps={eps}, affine={affine}, weight_init={w_init}".format(**self.__dict__)
    
    def get_dims_shape(self, x):
        dims = tuple(range(1, x.dim()))
        shape = [-1 if i == 1 else 1 for i in range(x.dim())]
        return dims, shape
    
    def _norm(self, x, dims):
        return x * torch.rsqrt(x.pow(2).mean(dims,True) + self.eps)

    def forward(self, x):
        dims, shape = self.get_dims_shape(x)
        normlized_x = self._norm(x.float(), dims=dims).to(x.dtype)
        if self.affine:
            return normlized_x * self.weight.view(*shape)
        else:
            return normlized_x


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        """ modified from LlamaRMSNorm """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
    
    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(1, True) + self.eps)

    def forward(self, x):
        if x.ndim == 4:
            weight = self.weight.unsqueeze(-1).unsqueeze(-1)
        elif x.ndim == 3:
            weight = self.weight.unsqueeze(-1)
        elif x.ndim == 2:
            weight = self.weight
        else:
            raise NotImplementedError
        return weight * self._norm(x.float()).to(x.dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.eps}"