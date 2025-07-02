#%%

import torch
import torch.nn as nn
import math
from timm.models.vision_transformer import Mlp, PatchEmbed
from timm.layers import DropPath
import torch.nn.functional as F
#from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from csms6s import selective_scan_fn, selective_scan_flop_jit
from einops import rearrange, repeat
import torch.distributed as dist
from csm_triton import cross_scan_fn, cross_merge_fn

def window_partition(x,window_size):
    """
    Args:
        x: (B, C, H, W)
        window_size: window size
    Returns:
        local window features (num_windows*B, window_size,window_size, C)
    """
    B,C,H,W=x.shape
    x=x.view(B,C,H//window_size,window_size,W//window_size,window_size).contiguous()
    x=rearrange(x,'b c hw hs ww ws -> (b hw ww)  (hs ws) c' )
    return x
def window_reverse(x, window_size, H, W):
    """
    Args:
        x: local window features (num_windows * B, window_size * window_size, C)
        window_size: size of each window
        H: original height
        W: original width
    Returns:
        x: (B, C, H, W)
    """
    assert H % window_size == 0 and W % window_size == 0, "H and W must be divisible by window_size"

    num_windows_h = H // window_size
    num_windows_w = W // window_size
    B = x.shape[0] // (num_windows_h * num_windows_w)

    x = rearrange(x, '(b nh nw) (wh ww) c -> b c (nh wh) (nw ww)',
                  b=B, nh=num_windows_h, nw=num_windows_w,
                  wh=window_size, ww=window_size)
    return x

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # 关键改动：在 float32 下进行中间计算以防止溢出
        var = x.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
        rms = torch.sqrt(var + self.eps)

        # 将结果转换回原始数据类型再进行后续计算
        return (x / rms.to(x.dtype)) * self.weight

class RMSNorm2D(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))  # 可学习缩放因子 (gamma)

    def forward(self, x):
        # x: (B, C, H, W)
        
        # 中间计算转为 float32 防止精度问题（特别是混合精度训练中）
        var = x.to(torch.float32).pow(2).mean(dim=1, keepdim=True)  # 按通道求均值
        
        rms = torch.sqrt(var + self.eps)  # 均方根
        
        # 除以 rms 并乘以缩放系数
        return (x / rms.to(x.dtype)) * self.weight.view(1, -1, 1, 1)


class PatchEmbed(nn.Module):
    """
    Patch embedding block"
    """
    def __init__(self,in_channels=3,in_dim=64,dim=96):
        super().__init__()
        self.conv_down=nn.Sequential(
            nn.Conv2d(in_channels, in_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(in_dim,eps=1e-4),
            nn.ReLU(),
            nn.Conv2d(in_dim, dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(dim,eps=1e-4),
            nn.ReLU()            
        )
    def forward(self,x):
        """
        Args:
            x: (B, C, H, W)
        Returns:
            x: (B, dim, H/4, W/4)
        """
        x=self.conv_down(x)#x (B, C, H, W)
        return x

class Downsample(nn.Module):
    def __init__(self,dim,keep_dim=False):
        super().__init__()
        if keep_dim:
            self.downsampler=nn.Identity()
            dimout=dim
        else:
            dimout=dim*2
        self.downsampler=nn.Conv2d(dim,dimout,kernel_size=3,stride=2,padding=1)
    def forward(self,x):
        """
        Args:
            x: (B, C, H, W)
        Returns:
            x: (B, dim*2, H/2, W/2)
        """
        x=self.downsampler(x)
        return x

class ConvBlock(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale=0., kernel_size=3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(dim, eps=1e-4),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(dim, eps=1e-4),
        )
        self.layer_scale = layer_scale
        if type(layer_scale) in [int, float] and layer_scale >0. :
            self.gamma = nn.Parameter(layer_scale * torch.ones(dim))
            self.layer_scale_enabled = True
        else:
            self.layer_scale_enabled = False
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
    
    def forward(self, x):
        #(B, C, H, W)
        residual = x
        x = self.conv(x)
        if self.layer_scale_enabled:
            x = x * self.gamma.view(1, -1, 1, 1).contiguous()
        return residual + self.drop_path(x)

class Attention(nn.Module):
    """
    接收独立 Q, K, V 输入的 Attention 模块
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_norm=False, attn_drop=0., proj_drop=0., norm_layer=RMSNorm):
        super().__init__()
        assert dim % num_heads == 0, "dim 必须能被 num_heads 整除"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)

        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q_in, k_in, v_in):
        B, N, C = q_in.shape
        q = self.q_proj(q_in).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(k_in).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(v_in).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        q, k = self.q_norm(q), self.k_norm(k)
        x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class DepthwiseAggregate(nn.Module):
    """
    MUDDFormer 的核心：深度聚合模块 (DA) 
    """
    def __init__(self, dim, layer_idx, num_ways=4, da_hidden_dim_ratio=2.0):
        super().__init__()
        self.dim = dim
        self.layer_idx = layer_idx
        self.num_ways = num_ways
        # num_prev represents the number of previous layers including the current one (0 to layer_idx)
        self.num_prev = self.layer_idx + 1 

        self.weight_gen = nn.Sequential(
            nn.Conv2d(dim, int(dim*da_hidden_dim_ratio), kernel_size=1),  # 大幅减少中间通道
            nn.GELU(),
            nn.Conv2d(int(dim*da_hidden_dim_ratio), self.num_ways * self.num_prev, kernel_size=1)
        )
        
        self.static_bias = nn.Parameter(torch.zeros(self.num_ways, self.num_prev))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_normal_(self.weight_gen[0].weight)
        nn.init.zeros_(self.weight_gen[2].weight)  # 初始化为零变换
        # Set the bias for the current layer to 1.0 for each way, for the last 'num_prev' element
        nn.init.constant_(self.static_bias[:, -1], 1.0) 

    def forward(self, Xs):
        """
        Args:
            Xs (list[torch.Tensor]): 包含从第 0 层到当前层所有输出的列表。
                                     每个 Tensor 的形状为 (B, C, H, W)。
        Returns:
            tuple[torch.Tensor]: 包含 num_ways 个聚合后流的元组。
        """
        B, C, H, W = Xs[-1].shape # Get batch, channel, height, width from the last layer's output

        dynamic_w = self.weight_gen(Xs[-1]) 
        
       
        weights = dynamic_w + self.static_bias.view(1, -1, 1, 1)

        weights = weights.view(B, self.num_ways, self.num_prev, H, W)

        max_vals, _ = weights.max(dim=2, keepdim=True)
        weights = torch.softmax(weights - max_vals, dim=2)
        X_stack = torch.stack(Xs, dim=2) 

        aggregated_output = torch.einsum('bmthw,bcthw->bmchw', weights, X_stack)
        return aggregated_output.unbind(dim=1)

# ==============================================================================
# 2. 计算模块 Block (重构后)
# ==============================================================================

class Block(nn.Module):
    """
    混合计算模块 (Attention + Mamba)
    现在是一个纯粹的计算单元，接收来自外部 DA 模块的输入。
    """
    def __init__(self, dim,mlp_ratio=4., drop_path=0.,
                 layer_scale=1e-6, window_size=7, reduction_ratio=4, channel_scaler=True,
                 # 假设 Mamba 和 Attention 的配置通过这些参数传入
                 mamba_config=None, attn_config=None):
        super().__init__()
        self.dim = dim
        self.half_dim = dim // 2
        self.window_size = window_size
        
        # --- 定义子模块 ---
        self.atten = Attention(dim=self.half_dim, **(attn_config or {}))
        self.mamba = SS2D_tiny(d_model=self.half_dim, **(mamba_config or {}))
        
        # Pre-Norm for each input stream
        self.norm_q = RMSNorm(self.half_dim)
        self.norm_k = RMSNorm(self.half_dim)
        self.norm_v_attn = RMSNorm(self.half_dim)
        self.norm_v_mamba = RMSNorm2D(self.half_dim)
        self.norm_mlp=RMSNorm2D(self.dim)
        self.channel_scaler = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // reduction_ratio, 1),
            nn.GELU(),
            nn.Conv2d(dim // reduction_ratio, dim, 1),
            nn.Sigmoid()
        ) if channel_scaler else None

        self.mlp = nn.Sequential(
            nn.Conv2d(dim, int(dim * mlp_ratio),1),
            nn.GELU(),
            nn.Conv2d(int(dim * mlp_ratio), dim,1)
        )
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        use_layer_scale = layer_scale is not None and type(layer_scale) in [int, float]
        self.gamma_1 = nn.Parameter(layer_scale * torch.ones(dim))  if use_layer_scale else 1
        self.gamma_2 = nn.Parameter(layer_scale * torch.ones(dim))  if use_layer_scale else 1
  
    def forward(self, xq, xk, xv, xr):
        """
        xq, xk, xv, xr 均由外部的 DA 模块提供。
        """
        B, C, H, W = xr.shape
        
        # --- 1. 准备输入流 ---
        # 将 DA 模块提供的4个流，根据我们的混合设计进行拆分
        xq_attn, _ = torch.chunk(xq, 2, dim=1)
        xk_attn, _ = torch.chunk(xk, 2, dim=1)
        xv_attn, xv_mamba = torch.chunk(xv, 2, dim=1)
        
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        if pad_r > 0 or pad_b > 0:
                xq_attn = torch.nn.functional.pad(xq_attn, (0,pad_r,0,pad_b))
                xk_attn = torch.nn.functional.pad(xk_attn, (0,pad_r,0,pad_b))
                xv_attn = torch.nn.functional.pad(xv_attn, (0,pad_r,0,pad_b))
                _, _, Hp, Wp = xq_attn.shape
        else:
            Hp, Wp = H, W
            
        # --- 2. Attention 分支 ---
        # 窗口化处理
        # (B, C/2, H, W) -> (B*num_win, win*win, C/2)
        q_win = window_partition(xq_attn.contiguous(), self.window_size)
        k_win = window_partition(xk_attn.contiguous(), self.window_size)
        v_win = window_partition(xv_attn.contiguous(), self.window_size)
        
        # Pre-Norm
        q = self.norm_q(q_win)
        k = self.norm_k(k_win)
        v = self.norm_v_attn(v_win)
        
        # Attention 计算
        attn_out_win = self.atten(q, k, v)
        
        # 逆窗口化
        # (B*num_win, win*win, C/2) -> (B, C/2, H, W)
        attn_out = window_reverse(attn_out_win, self.window_size, Hp, Wp)
        if pad_r > 0 or pad_b > 0:
                attn_out = attn_out[:, :, :H, :W].contiguous()
        # --- 3. Mamba 分支 ---
        # (B, C/2, H, W) -> (B, H*W, C/2)
        #v_mamba_seq = rearrange(xv_mamba, 'b c h w -> b (h w) c')
        
        # Pre-Norm
        v_mamba_norm = self.norm_v_mamba(xv_mamba)
        
        # Mamba 计算
        mamba_out = self.mamba(v_mamba_norm)
        
        # (B, H*W, C/2) -> (B, C/2, H, W)
        #mamba_out = rearrange(mamba_out_seq, 'b (h w) c -> b c h w', h=H, w=W)
        
        # --- 4. 合并与残差连接 ---
        # 合并 Attn 和 Mamba 的输出
        y_mixed = torch.cat([attn_out, mamba_out], dim=1)
        
        # Channel Scaler (如果启用)
        if self.channel_scaler is not None:
            y_mixed = y_mixed * self.channel_scaler(xr) # 使用 xv 作为内容引导
        

        y = xr + self.drop_path(self.gamma_1.view(1, -1, 1, 1) * y_mixed)
        final_out = y + self.drop_path(self.gamma_2.view(1, -1, 1, 1) * self.mlp(self.norm_mlp(y)))
 
        return final_out


# =====================================================
class mamba_init:
    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # dt_proj.bias._no_reinit = True
        
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        # S4D real initialization
        A = torch.arange(1, d_state + 1, dtype=torch.float32, device=device).view(1, -1).repeat(d_inner, 1).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 0:
            A_log = A_log[None].repeat(copies, 1, 1).contiguous()
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = D[None].repeat(copies, 1).contiguous()
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    @classmethod
    def init_dt_A_D(cls, d_state, dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, k_group=4):
        # dt proj ============================
        dt_projs = [
            cls.dt_init(dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor)
            for _ in range(k_group)
        ]
        dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in dt_projs], dim=0)).contiguous() # (K, inner, rank)
        dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in dt_projs], dim=0)).contiguous() # (K, inner)
        del dt_projs
            
        # A, D =======================================
        A_logs = cls.A_log_init(d_state, d_inner, copies=k_group, merge=True) # (K * D, N)
        Ds = cls.D_init(d_inner, copies=k_group, merge=True) # (K * D)  
        return A_logs, Ds, dt_projs_weight, dt_projs_bias


class SS2D_tiny(nn.Module):
    """

    """
    def __init__(
        self,
        d_model,
        d_state=16,
        expand=2.0,
        dt_rank="auto",
        act_layer=nn.SiLU,
        d_conv=3,
        conv_bias=True,
        dropout=0.0,
        bias=False,
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        disable_z=True,
        disable_z_act=True,
        oact=True,

    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(expand * d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.k_group = 4 # for cross-scan
        self.with_dconv = d_conv > 1
        self.disable_z=disable_z
        self.disable_z_act=disable_z_act
        self.ocat=oact
        # 输入投影 (in_proj): 由于 _noz, 输出维度是 d_inner, 而不是 2 * d_inner
        d_proj = self.d_inner if self.disable_z else (self.d_inner * 2)
        self.in_proj = nn.Conv2d(self.d_model, d_proj, bias=bias,kernel_size=1)
        self.act: nn.Module = act_layer()

        # 深度可分离卷积 (DWConv)
        if self.with_dconv:
            self.conv2d = nn.Conv2d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                groups=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
            )

        # SSM 参数投影 (x_proj for B, C, dt)
        # 使用单个权重矩阵以提高效率
        self.x_proj_weight = nn.Parameter(torch.randn(self.k_group, self.d_inner, self.dt_rank + self.d_state * 2))
        
        
        # 初始化 SSM 参数 (A, D, dt)
        # 只保留 'v0' 初始化逻辑
        self.A_logs, self.Ds, self.dt_projs_weight, self.dt_projs_bias = mamba_init.init_dt_A_D(
            self.d_state, self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, k_group=self.k_group)
        
        self.out_act = nn.GELU() if oact else nn.Identity()
        # 输出投影 (out_proj)
        self.out_proj = nn.Conv2d(self.d_inner, self.d_model, bias=bias,kernel_size=1)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

    def forward(self, x):
        # x: (B, C, H, W)
        #return (B, C, H, W)
        B, C, H, W = x.shape
        x = self.in_proj(x) # (B, H, W, d_inner)
        if not self.disable_z:
            x, z = x.chunk(2, dim=1)
            if not self.disable_z_act:
                z = self.act(z)
        # 2. 卷积和激活
        if self.with_dconv:
            x = self.conv2d(x).contiguous()
        x = self.act(x)
        y = self.ssm_core(x)
        y = self.out_act(y)
        if not self.disable_z:
            y = y * z
        out = self.dropout(self.out_proj(y))
        return out


    def ssm_core(self, x: torch.Tensor):
        B, D, H, W = x.shape
        N = self.d_state
        K = self.k_group
        R = self.dt_rank
        L = H * W

        x_proj_bias = getattr(self, "x_proj_bias", None)
        # a. 4-way scan preparation
        # cross_scan_fn 应该返回 (B, K, D, L)
        xs = cross_scan_fn(x, in_channel_first=True, out_channel_first=True, scans=0)

        # b. 计算 B, C, delta
        # 使用 einsum 保持逻辑清晰，也可以替换为 conv1d
        x_dbl = torch.einsum("b k d l, k d c -> b k c l", xs, self.x_proj_weight)
        if x_proj_bias is not None:
            x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
            
        dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
        if hasattr(self, "dt_projs_weight"):
            dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)

        # c. 准备 selective_scan 的输入
        xs = xs.contiguous().view(B, -1, L)
        dts = dts.contiguous().view(B, -1, L)
        As = -self.A_logs.to(torch.float).exp() # (K * D, N)
        Ds = self.Ds.to(torch.float) # (K * D)
        Bs = Bs.contiguous().view(B, K, N, L)
        Cs = Cs.contiguous().view(B, K, N, L)
        delta_bias = self.dt_projs_bias.view(-1).to(torch.float)
        
        # d. 执行 selective_scan
        # 注意：这里假设 selective_scan_fn 支持 (B*K, D, L) 形式的输入
        # 或者需要根据你的实现调整输入形状
        ys: torch.Tensor = selective_scan_fn(
            xs, dts, As, Bs, Cs, Ds, delta_bias)
        ys=ys.view(B, K, D, H, W)
        
        # e. 合并 4-way scan 的结果
        # cross_merge_fn 应该接收 (B, K, D, H, W) 并返回 (B, D, H, W)
        y = cross_merge_fn(ys, in_channel_first=True, out_channel_first=True, scans=0)
            #(b d l)
        return y.view(B, D, H, W)










class GlobalLocalLayer(nn.Module):
    """
    MUDDFormer 控制器，实现了正确的 MUDD 循环逻辑。
    """
    def __init__(self, dim, depth, 
                 window_size,
                 attn_config,
                 mamba_config,
                 conv=False,
                 mlp_ratio=4,
                 drop_path=0.,
                 layer_scale_conv=None,
                layer_scale=None,
                 downsample=False,
                use_param_reallocation=True,
                da_hidden_dim_ratio=0.2,
                channel_scaler=False,
                 ):
        super().__init__()
        self.depth = depth
        self.conv_blocks=nn.ModuleList()
        self.mix_blocks=nn.ModuleList()
        self.mlp_ratios = []
        for i in range(depth):
            if depth > 1 and use_param_reallocation:
                # 应用论文公式(9)计算缩放因子
                scale_factor = (0.5 * (depth - i - 1) + 1.5 * i) / (depth - 1)
                self.mlp_ratios.append(mlp_ratio * scale_factor)
            else:
                self.mlp_ratios.append(mlp_ratio)


        if conv:
            for i in range(depth):
                self.conv_blocks.append(ConvBlock(dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                layer_scale=layer_scale_conv))
        else:      
            self.da_modules = nn.ModuleList([
            DepthwiseAggregate(dim=dim, layer_idx=i, num_ways=4,da_hidden_dim_ratio=da_hidden_dim_ratio) for i in range(depth)
        ])                        
            for i in range(depth):
                self.mix_blocks.append(
                Block(dim=dim, 
                      attn_config=attn_config,
                      mamba_config=mamba_config,
                        window_size=window_size,
                        layer_scale=layer_scale,
                        mlp_ratio=self.mlp_ratios[i],
                        drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                        channel_scaler=channel_scaler,
                        
                            ))

   
        self.downsample = None if not downsample else Downsample(dim=dim)
        
        # 为每一层创建一个 DA 模块

        

    def forward(self, x):
        """
        MUDDFormer 主循环：聚合历史 -> 计算当前 -> 保存历史
        """
        # 初始化历史状态列表，包含初始输入 x
        for block in self.conv_blocks:
            x=block(x)
        
        Xs = [x]
        for i,block in enumerate(self.mix_blocks):
            # 1. 调用第 i 层的 DA 模块，聚合所有历史输出 (Xs)
            # Xs 包含了从 X_0 到 X_i 的所有输出
            xq, xk, xv, xr = self.da_modules[i](Xs)
            
            # 2. 将聚合后的4个流送入第 i 层的 Block
            x_out = block(xq, xk, xv, xr)
            
            # 3. 将当前 Block 的输出追加到历史列表中，以备后用
            Xs.append(x_out)
            
        # 循环结束后，最后一个 Block 的输出就是最终结果
        final_output = Xs[-1]
        
        if self.downsample is not None:
            final_output = self.downsample(final_output)
            
        return final_output


class DAMNet(nn.Module):
    """
    Args:
        dim: Model dimension, typically the feature dimension (e.g., 224 or 512).
        depths: A list of integers specifying the depth (number of blocks) at each level of the model.
        window_size: The size of the local window used for window-based attention.
        mlp_ratio: The ratio of the MLP hidden dimension to the input dimension in the MLP block.
        num_heads: The number of attention heads in the multi-head attention mechanism.
        drop_path_rate: DropPath rate for regularizing the model, reducing overfitting.
        in_channels: Number of input channels, typically 3 for RGB images.
        num_classes: Number of output classes for the classification task.
        qkv_bias: Boolean, whether to add bias to the query, key, and value in the attention layer.
        qk_norm: Scaling factor for the query, key, and value projections.
        attn_drop: Dropout rate for the attention layer to prevent overfitting.
        layer_scale: Scaling factor applied to each layer's output, used for layer normalization.
        layer_scale_conv: Scaling factor applied specifically to convolutional layers.
        expert_capacity_factor: Factor that controls the load capacity of each expert in the MoE mechanism.
        balance_loss_weight: Weight for the balance loss, used for balancing the expert capacity.
        top_k: The number of top experts selected in the MoE mechanism for each sample.
        proj_drop: Dropout rate applied to the final projection layer.
    """
    def __init__(self, 
                 dim,
                 in_dim,
                 depths,
                 window_size,
                 mlp_ratio,
                 drop_path_rate=0.2,
                 in_channels=3,
                 num_classes=1000,
                 layer_scale=None,
                 layer_scale_conv=None,
                 use_param_reallocation=False,
                 da_hidden_dim_ratio=0.5,
                 channel_scaler=False,
                 #==============
                d_state=16,
                d_conv=3,
                expand=2,
                #=================
                num_heads=[2,4,8,16],
                qkv_bias=True,
                qk_norm=None,
                proj_drop=0.,
                attn_drop=0.,

           
    ):
        super().__init__()
        num_features=int(dim*2**(len(depths)-1))
        self.num_classes = num_classes
        dpr= [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.patch_embed = PatchEmbed(in_channels=in_channels,in_dim=in_dim,dim=dim)
        self.levels = nn.ModuleList()
        mamba_config={
            'd_state':d_state,
            'd_conv':d_conv,
            'expand':expand,
        }
        for i, depth in enumerate(depths):
            conv=True if i<2 else False
            attn_config={
            'num_heads':num_heads[i],
            'qkv_bias':qkv_bias,
            'qk_norm':qk_norm,
            'attn_drop':attn_drop,
            'proj_drop':proj_drop,
        }
            level=GlobalLocalLayer(
                dim=dim*2**i,
                depth=depth,
                window_size=window_size[i],
                conv=conv,
                mlp_ratio=mlp_ratio,
                drop_path=dpr[sum(depths[:i]):sum(depths[:i+1])],
                layer_scale_conv=layer_scale_conv,
                layer_scale=layer_scale,
                downsample=True if i < len(depths)-1 else False,
                use_param_reallocation=use_param_reallocation,
                da_hidden_dim_ratio=da_hidden_dim_ratio,
                channel_scaler=channel_scaler,
                
                #==============
                attn_config=attn_config,
                mamba_config=mamba_config
                )
            self.levels.append(level)
        self.norm = nn.BatchNorm2d(num_features)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(num_features, num_classes) if num_classes > 0 else nn.Identity()
    def forward(self, x):
        #(B, C, H, W)
        x=self.patch_embed(x)#(B, C, H, W)->(B, dim, H/4, W/4)

        for i, level in enumerate(self.levels):
            x= level(x)
        #x=rearrange(x, "b c h w -> b (h w) c")
        x= self.norm(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x= self.head(x)

        return x




#%%
if __name__=="__main__":
    model = DAMNet(
        dim=96,
        in_dim=32,
        depths=[1, 3, 8, 4],
        window_size=[8, 8, 14, 7],
        mlp_ratio=4,
        num_heads=[2, 4, 8, 16],
        drop_path_rate=0.2,
        in_channels=3,
        num_classes=1000,
        qkv_bias=True,
        qk_norm=None,
        attn_drop=0.1,
        layer_scale=1e-5,
        layer_scale_conv=1e-5,
        proj_drop=0.1,
        use_param_reallocation=True,
        da_hidden_dim_ratio=0.25,
        channel_scaler=True,
        #===============
        d_state=16,
        d_conv=3,
        expand=2,
    )
    checkpoint=torch.load('/home/shulab/wyc/mambavision_moe_simple/checkpoint_epoch_130.pth')
    model = model.to('cuda:0')
    model.load_state_dict(checkpoint['model'],strict=False)

        

    # 将模型放到主GPU上，通常是cuda:0
    model = model.to('cuda:0')

    # 使用DataParallel将模型并行化
    #model = nn.DataParallel(model, device_ids=[0, 1]) 

    # 计算总参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # 计算可训练参数量
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params:,}")
    x = torch.randn(10, 3, 224, 224)
    x=x.to('cuda')
    #%%
    y=model(x)
    y.sum().backward()
    print(y.shape)
    #%%
    for name, param in model.named_parameters():
        if param.grad is None:
            print(f"[NO GRAD] {name}")
# %%
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad = param.grad
            if grad.stride() != grad.contiguous().stride():
                print(f"[NON-CONTIGUOUS GRAD] {name}")
                print(f"  size   : {grad.size()}")
                print(f"  stride : {grad.stride()}")
# %%
#--resume='/public/home/liuzhenzhao/wyc/OmniVision_v2/checkpoint_epoch_240.pth'
    model=SS2D_tiny(d_model=96,d_state=16,d_conv=3,expand=2).cuda()
    x=torch.randn(10, 96, 54, 54).cuda()
# %%
    y=model(x)
# %%
