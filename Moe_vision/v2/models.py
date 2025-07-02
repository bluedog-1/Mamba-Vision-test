#%%

import torch
import torch.nn as nn
import math
from timm.models.vision_transformer import Mlp, PatchEmbed
from timm.layers import DropPath
import torch.nn.functional as F
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from einops import rearrange, repeat
import torch.distributed as dist


def windows_partition(x,window_size):
    """
    Args:
        x: (B, C, H, W)
        window_size: window size
    Returns:
        local window features (num_windows*B, window_size*window_size, C)
    """
    B,C,H,W=x.shape
    x=x.view(B,C,H//window_size,window_size,W//window_size,window_size).contiguous()
    x=x.permute(0,2,4,3,5,1).reshape(-1,window_size*window_size,C).contiguous()# (Bn,wz*wz,C)
    return x
def windows_reverse(x,window_size,H,W):
    """
    Args:
        windows: local window features (num_windows*B, window_size, window_size, C)
        window_size: Window size
        H: Height of image
        W: Width of image
    Returns:
        x: (B, C, H, W)
    """
    B=int(x.shape[0]/(H*W/window_size/window_size))
    x=x.reshape(B,H//window_size,W//window_size,window_size,window_size,-1).contiguous()
    x=x.permute(0,-1,1,3,2,4).reshape(B,-1,H,W).contiguous()
    return x


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

class MambaVisionMixer(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True, 
        layer_idx=None,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        self.in_proj = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)    
        self.x_proj = nn.Linear(
            self.d_inner//2, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner//2, bias=True, **factory_kwargs)
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        dt = torch.exp(
            torch.rand(self.d_inner//2, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner//2,
        ).contiguous()
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(self.d_inner//2, device=device))
        self.D._no_weight_decay = True
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.conv1d_x = nn.Conv1d(
            in_channels=self.d_inner//2,
            out_channels=self.d_inner//2,
            bias=conv_bias//2,
            kernel_size=d_conv,
            groups=self.d_inner//2,
            **factory_kwargs,
        )
        self.conv1d_z = nn.Conv1d(
            in_channels=self.d_inner//2,
            out_channels=self.d_inner//2,
            bias=conv_bias//2,
            kernel_size=d_conv,
            groups=self.d_inner//2,
            **factory_kwargs,
        )

    def forward(self, hidden_states):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        _, seqlen, _ = hidden_states.shape
        xz = self.in_proj(hidden_states)
        xz = rearrange(xz, "b l d -> b d l")
        x, z = xz.chunk(2, dim=1)
        A = -torch.exp(self.A_log.float())
        x = F.silu(F.conv1d(input=x, weight=self.conv1d_x.weight, bias=self.conv1d_x.bias, padding='same', groups=self.d_inner//2))
        z = F.silu(F.conv1d(input=z, weight=self.conv1d_z.weight, bias=self.conv1d_z.bias, padding='same', groups=self.d_inner//2))
        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = rearrange(self.dt_proj(dt), "(b l) d -> b d l", l=seqlen)
        B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        y = selective_scan_fn(x, 
                              dt, 
                              A, 
                              B, 
                              C, 
                              self.D.float(), 
                              z=None, 
                              delta_bias=self.dt_proj.bias.float(), 
                              delta_softplus=True, 
                              return_last_state=None)
        
        y = torch.cat([y, z], dim=1)
        y = rearrange(y, "b d l -> b l d")
        out = self.out_proj(y)
        return out


 
class Attention(nn.Module):

    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = True

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
             q, k, v,
                dropout_p=self.attn_drop.p,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C).contiguous()
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LightweightHybridGate(nn.Module):
    def __init__(self, dim: int, num_experts: int = 3, reduction_ratio: int = 16):  # 增大压缩比减少参数
        super().__init__()
        self.num_experts = num_experts
        
        # 确保最小维度（进一步减小）
        reduced_dim = max(4, dim // reduction_ratio)
        edge_dim = max(4, dim // 16)  # 减少边缘特征维度
        
        # 1. 轻量级序列特征提取（使用分组卷积）
        self.sequence_conv = nn.Sequential(
            nn.Conv1d(dim, reduced_dim, kernel_size=3, padding=1, groups=4),  # 分组减少参数
            nn.BatchNorm1d(reduced_dim),
            nn.ReLU()
        )
        self.sequence_pool = nn.AdaptiveAvgPool1d(1)  # 只用一个池化层
        
        # 2. 轻量级边缘特征提取（单层卷积）
        self.edge_detector = nn.Sequential(
            nn.Conv2d(dim, edge_dim, kernel_size=3, padding=1, groups=4),  # 分组减少参数
            nn.BatchNorm2d(edge_dim),
            nn.ReLU()
        )
        self.edge_pool = nn.AdaptiveAvgPool2d(1)
        
        # 3. 移除纹理分支（减少计算量）
        
        # 计算融合维度（大幅减少）
        fusion_dim = reduced_dim + edge_dim
        
        # 更轻量的特征压缩层（单层线性）
        self.gate_output = nn.Linear(fusion_dim, num_experts)
        
        # 温度参数
        self.temperature = nn.Parameter(torch.tensor(0.5))
        
        # 负载均衡参数
        self.balance_loss_weight = 0.01

    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape
        
        # 1. 序列特征提取（简化）
        x_seq = rearrange(x, "b c h w -> b c (h w)")
        conv_seq_feat = self.sequence_conv(x_seq)
        seq_feat = self.sequence_pool(conv_seq_feat).flatten(1)
        
        # 2. 边缘特征提取（简化）
        edge_feat = self.edge_detector(x)
        edge_intensity = self.edge_pool(edge_feat).flatten(1)
        
        # 3. 特征融合（仅两个分支）
        combined = torch.cat([seq_feat, edge_intensity], dim=1)
        
        # 4. 门控输出（直接线性层）
        expert_logits = self.gate_output(combined)
        
        # 5. 负载均衡损失计算
        if self.training:
            probs = F.softmax(expert_logits, dim=-1)
            load = probs.mean(dim=0)
            importance = probs.sum(dim=0)
            balance_loss = self.balance_loss_weight * (
                (load - load.mean()).abs().mean() + 
                (importance - importance.mean()).abs().mean()
            )
        else:
            balance_loss = torch.tensor(0.0, device=x.device)
        
        # 6. 带温度控制的权重输出
        weights = F.softmax(expert_logits / self.temperature.clamp(min=0.1), dim=-1)
        
        return weights, balance_loss


class EfficientMoEBlock(nn.Module):
    def __init__(self, 
                 dim, 
                 num_heads, 
                 top_k=2, 
                 num_experts=3,
                 window_size=7, 
                 qkv_bias=False,
                 qk_norm=False,
                 attn_drop=0.,
                 proj_drop=0.,
                 drop_path=0.,
                 mlp_ratio=4,
                 layer_scale=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 layer_scale_conv=0.):
        super().__init__()
        self.top_k = top_k
        self.window_size = window_size
        self.dim = dim
        self.num_experts = num_experts
        
        # 轻量级门控网络
        self.gate = LightweightHybridGate(dim=dim, num_experts=num_experts)
        
        # 专家模块
        self.norm1 = norm_layer(dim)
        self.mamba_expert = MambaVisionMixer(d_model=dim, d_state=8, d_conv=3, expand=1)
        self.attn_expert = Attention(dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, 
                                     qk_norm=qk_norm, attn_drop=attn_drop, proj_drop=proj_drop, 
                                     norm_layer=norm_layer)
        self.conv_expert = ConvBlock(dim=dim)
        
        # FFN
        self.norm2 = norm_layer(dim)
        self.ffn = Mlp(dim, hidden_features=int(mlp_ratio * dim), 
                       act_layer=act_layer, drop=proj_drop)
        
        # 层缩放
        use_layer_scale = layer_scale > 0 and isinstance(layer_scale, (int, float))
        self.gamma_1 = nn.Parameter(layer_scale * torch.ones(dim)) if use_layer_scale else 1
        self.gamma_2 = nn.Parameter(layer_scale * torch.ones(dim)) if use_layer_scale else 1

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        # 专家统计
        self.register_buffer('expert_counts', torch.zeros(num_experts))
        self.register_buffer('total_samples', torch.tensor(0))
        
        # 专家掩码（用于冻结）
        self.expert_mask = torch.ones(num_experts, dtype=torch.bool)

    def forward(self, x):
        B, C, H, W = x.shape
        device = x.device
        
        # 重置小批量统计
        batch_expert_counts = torch.zeros(self.num_experts, device=device)
        batch_total_samples = torch.tensor(0, device=device)
        
        # 序列化输入
        x_seq = rearrange(x, "b c h w -> b (h w) c").contiguous()
        x_seq_norm = self.norm1(x_seq)
        
        # 门控计算（包含负载均衡损失）
        expert_weights, balance_loss = self.gate(x)
        topk_weights, topk_indices = torch.topk(expert_weights, k=self.top_k, dim=-1)
        normalized_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-6)
        
        # 构建专家权重矩阵
        expert_weights_matrix = torch.zeros(B, self.num_experts, device=device)
        expert_weights_matrix.scatter_(1, topk_indices, normalized_weights)
        
        # === 按需处理专家 ===
        expert_output = torch.zeros_like(x_seq)
        
        expert_handlers = {
            0: self._handle_mamba_expert,
            1: self._handle_attention_expert,
            2: self._handle_conv_expert,
        }
        
        for expert_idx in range(self.num_experts):
            mask = expert_weights_matrix[:, expert_idx] > 0
            if not mask.any():
                continue
                
            samples = x_seq_norm[mask]
            expert_out = expert_handlers[expert_idx](samples, H, W)
            
            # 更新统计
            batch_expert_counts[expert_idx] += mask.sum()
            batch_total_samples += mask.sum()
            
            # 加权融合
            sample_weights = expert_weights_matrix[mask, expert_idx].view(-1, 1, 1)
            expert_output[mask] += sample_weights * expert_out
        
        # 残差连接
        x_out = x_seq + self.drop_path(self.gamma_1 * expert_output)
        x_out = self.norm2(x_out)
        x_out = x_out + self.drop_path(self.gamma_2 * self.ffn(x_out))
        
        # 转换回空间格式
        spatial_out = rearrange(x_out, "b (h w) c -> b c h w", h=H, w=W).contiguous()
        
        # 更新全局统计
        self._update_expert_stats(batch_expert_counts.detach(), batch_total_samples.item())
        
        return spatial_out, balance_loss
    
    # 以下辅助函数保持不变（与你的原始代码一致）
    def _handle_mamba_expert(self, samples, H, W):
        """处理Mamba专家（带掩码支持）"""
        if not self.expert_mask[0]:
            return torch.zeros_like(samples)
        return self._handle_expert(self.mamba_expert, samples, H, W)
    
    def _handle_attention_expert(self, samples, H, W):
        """处理Attention专家（带掩码支持）"""
        if not self.expert_mask[1]:
            return torch.zeros_like(samples)
        return self._handle_expert(self.attn_expert, samples, H, W)
    
    def _handle_conv_expert(self, samples, H, W):
        """处理Conv专家（带掩码支持）"""
        if not self.expert_mask[2]:
            return torch.zeros_like(samples)
        samples_spatial = rearrange(samples, "b (h w) c -> b c h w", h=H, w=W)
        conv_out = self.conv_expert(samples_spatial)
        return rearrange(conv_out, "b c h w -> b (h w) c")
    
    def _handle_expert(self, expert, samples, H, W):
        """处理专家通用逻辑"""
        pad_r, pad_b = self._compute_padding(H, W)
        samples_spatial = rearrange(samples, "b (h w) c -> b c h w", h=H, w=W)
        
        if pad_r > 0 or pad_b > 0:
            samples_spatial = F.pad(samples_spatial, (0, pad_r, 0, pad_b))
            Hp, Wp = H + pad_b, W + pad_r
        else:
            Hp, Wp = H, W
        
        samples_win = windows_partition(samples_spatial, self.window_size)
        x = expert(samples_win)
        x = windows_reverse(x, self.window_size, Hp, Wp)
        
        if pad_r > 0 or pad_b > 0:
            x = x[:, :, :H, :W]
        return rearrange(x, "b c h w -> b (h w) c")
    
    def _compute_padding(self, H, W):
        """计算 padding"""
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        return pad_r, pad_b
    
    def _update_expert_stats(self, batch_counts, total):
        """更新专家使用统计"""
        self.expert_counts += batch_counts.cpu()
        self.total_samples += total
    
    def get_expert_utilization(self):
        """获取专家利用率"""
        if self.total_samples > 0:
            return self.expert_counts / self.total_samples
        return torch.zeros_like(self.expert_counts)
    
    def reset_expert_stats(self):
        """重置专家统计"""
        self.expert_counts.zero_()
        self.total_samples = 0
    
    def adjust_expert_mask(self, min_utilization=0.1):
        """动态调整专家掩码（防止塌陷）"""
        utilization = self.get_expert_utilization()
        for i in range(self.num_experts):
            if utilization[i] < min_utilization:
                self.expert_mask[i] = False
            else:
                self.expert_mask[i] = True
        self.reset_expert_stats()
        return utilization




class HybridGate(nn.Module):
    def __init__(self, dim: int, num_experts: int = 2, reduction_ratio: int = 8):
        super().__init__()
        self.num_experts = num_experts
        
        # 确保最小维度
        reduced_dim = max(1, dim // reduction_ratio)
        edge_dim = max(1, dim // 8)
        texture_dim = max(1, dim // 16)
        
        # 1. 序列特征提取
        self.sequence_conv = nn.Sequential(
            nn.Conv1d(dim, reduced_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(reduced_dim),  # 添加BN
            nn.ReLU()
        )
        self.sequence_max_pool = nn.AdaptiveMaxPool1d(1)
        self.sequence_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # 2. 边缘特征提取
        self.edge_detector = nn.Sequential(
            nn.Conv2d(dim, edge_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(edge_dim),  # 添加BN
            nn.ReLU()
        )
        
        # 3. 纹理特征提取（独立分支）
        self.texture_analyzer = nn.Sequential(
            nn.AdaptiveAvgPool2d(7),
            nn.Conv2d(dim, texture_dim, kernel_size=3, padding=1),  # 直接使用原始输入
            nn.BatchNorm2d(texture_dim),  # 添加BN
            nn.ReLU()
        )
        
        # 计算融合维度
        fusion_dim = (reduced_dim * 2) + edge_dim + (texture_dim * 7 * 7)
        
        # 特征压缩层
        self.feature_compressor = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.ReLU()
        )
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_experts)
        )
        
        # 温度参数
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        
        # 1. 序列特征提取
        x_seq = rearrange(x, "b c h w -> b c (h w)")
        conv_seq_feat = self.sequence_conv(x_seq)
        seq_feat_max = self.sequence_max_pool(conv_seq_feat).flatten(1)
        seq_feat_avg = self.sequence_avg_pool(conv_seq_feat).flatten(1)
        
        # 2. 边缘特征提取
        edge_feat = self.edge_detector(x)
        edge_intensity = F.adaptive_avg_pool2d(edge_feat, 1).flatten(1)                 
        
        # 3. 纹理特征提取（独立分支）
        texture_feat = self.texture_analyzer(x).flatten(1)  # 直接使用原始输入
        
        # 4. 特征融合
        combined = torch.cat([seq_feat_max, seq_feat_avg, edge_intensity, texture_feat], dim=1)
        compressed = self.feature_compressor(combined)
        expert_scores = self.fusion(compressed)
        
        # 5. 专家权重输出（带温度控制）
        return F.softmax(expert_scores / self.temperature, dim=-1)


class SimpleMoEBlock(nn.Module):
    def __init__(self, 
                 dim, 
                 gate,
                 num_heads, 
                 top_k=1, 
                 num_experts=2,
                 window_size=7, 
                 qkv_bias=False,
                 qk_norm=False,
                 attn_drop=0.,
                 proj_drop=0.,
                 drop_path=0.,
                 mlp_ratio=4,
                 layer_scale=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 mamba_partition=False,
                 attention_partition=True,
                 layer_scale_conv=0.,):
        super().__init__()
        self.top_k = top_k
        self.window_size = window_size
        self.dim = dim
        self.num_experts=num_experts
        self.mamba_partition = mamba_partition
        self.attention_partition = attention_partition
        self.norm1 = norm_layer(dim)
        self.gate = gate if gate is not None else HybridGate(dim=dim,num_experts=num_experts)
        # 专家模块
        self.mamba_expert = MambaVisionMixer(d_model=dim, d_state=8, d_conv=3, expand=1)
        self.attn_expert = Attention(dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_norm=qk_norm, 
                                     attn_drop=attn_drop, proj_drop=proj_drop, norm_layer=norm_layer)
        #self.conv_expert=ConvBlock(dim=dim,drop_path=drop_path,layer_scale=layer_scale_conv)
        # FFN
        self.ffn = Mlp(dim, hidden_features=int(mlp_ratio * dim), act_layer=act_layer, drop=proj_drop)
        
        # 层缩放
        use_layer_scale = layer_scale > 0 and isinstance(layer_scale, (int, float))
        self.gamma_1 = nn.Parameter(layer_scale * torch.ones(dim)) if use_layer_scale else 1
        self.gamma_2 = nn.Parameter(layer_scale * torch.ones(dim)) if use_layer_scale else 1

        self.norm2 = norm_layer(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        # 使用普通属性而非缓冲区 - 解决多GPU问题
        self.current_batch_expert_counts = None
        self.current_batch_total_samples = None
        
    def forward(self, x):
        self._reset_current_batch_stats(x.device)
        B, C, H, W = x.shape

        x_seq = rearrange(x, "b c h w -> b (h w) c").contiguous()
        x_seq_norm = self.norm1(x_seq)

        # 门控计算
        global_weights= self.gate(x)
        topk_weights, topk_indices = torch.topk(global_weights, k=self.top_k, dim=-1)
        normalized_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-6)
        
        # 构建专家权重矩阵
        expert_weights_matrix = torch.zeros(B, self.num_experts, device=x.device).contiguous()
        expert_weights_matrix.scatter_(1, topk_indices, normalized_weights)
        
        # === 专家处理 ===
        expert_output = torch.zeros_like(x_seq)
        
        expert_handlers = {
            1:self._handle_attention_expert,
            0:self._handle_mamba_expert,
            #2:self._handle_conv_expert,
        }
        
        for expert_idx in range(self.num_experts):
            mask = expert_weights_matrix[:, expert_idx] > 0
            if not mask.any():
                continue
            samples = x_seq_norm[mask]
            expert_out = expert_handlers[expert_idx](samples, H, W)#b n c
            
            self.current_batch_expert_counts[expert_idx] += mask.sum()
            self.current_batch_total_samples += mask.sum()
            
            sample_weights = expert_weights_matrix[mask, expert_idx].contiguous().reshape(-1, 1, 1)
            expert_output[mask] += sample_weights * expert_out
        
        # 残差连接
        x_out = x_seq + self.drop_path(self.gamma_1 * expert_output)
        x_out = self.norm2(x_out)
        x_out = x_out + self.drop_path(self.gamma_2 * self.ffn(x_out))
        
        # 转换回空间格式
        spatial_out = rearrange(x_out, "b (h w) c -> b c h w", h=H, w=W).contiguous()
        
        return spatial_out
    
    def _handle_mamba_expert(self, samples, H, W):
        """处理Mamba专家"""
        return self._handle_expert(self.mamba_expert, samples, H, W)
    
    def _handle_attention_expert(self, samples, H, W):
        """处理Attention专家"""
        return self._handle_expert(self.attn_expert, samples, H, W)
    
    def _handle_expert(self, expert, samples, H, W):
        """处理专家通用逻辑"""
        pad_r, pad_b = self._compute_padding(H, W)
        samples_spatial = rearrange(samples, "b (h w) c -> b c h w", h=H, w=W).contiguous()
        
        if pad_r > 0 or pad_b > 0:
            samples_spatial = F.pad(samples_spatial, (0, pad_r, 0, pad_b))
            Hp, Wp = H + pad_b, W + pad_r
        else:
            Hp, Wp = H, W
        
        samples_win = windows_partition(samples_spatial, self.window_size).contiguous()
        x = expert(samples_win)
        x = windows_reverse(x, self.window_size, Hp, Wp).contiguous()
        if pad_r > 0 or pad_b > 0:
            x = x[:, :, :H, :W].contiguous()
        return rearrange(x, "b c h w -> b (h w) c").contiguous()
    
    def _compute_padding(self, H, W):
        """计算 padding"""
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        return pad_r, pad_b
    
    def _reset_current_batch_stats(self, device):
        """重置当前批次统计，并同步跨多个GPU"""
        self.current_batch_expert_counts = torch.zeros(self.num_experts, device=device)
        self.current_batch_total_samples = torch.tensor(0, device=device)
    
    def current_batch_utilization(self):
        """获取当前批次的专家利用率"""
        if self.current_batch_total_samples > 0:
            if dist.is_initialized():
                dist.all_reduce(self.current_batch_expert_counts, op=dist.ReduceOp.SUM)
                dist.all_reduce(self.current_batch_total_samples, op=dist.ReduceOp.SUM)
            
            return self.current_batch_expert_counts / self.current_batch_total_samples
        return torch.zeros_like(self.current_batch_expert_counts)

    def _handle_conv_expert(self, samples, H, W):
            """处理Conv专家"""
            samples_spatial = rearrange(samples, "b (h w) c -> b c h w", h=H, w=W).contiguous()
            conv_out = self.conv_expert(samples_spatial)
            return rearrange(conv_out, "b c h w -> b (h w) c").contiguous()
    
class GlobalLocalLayer(nn.Module):
    def __init__(self, dim, depth, 
                 window_size,
                 num_heads,
                 share_gate=True,
                 conv=False,
                 mlp_ratio=4,
                 drop_path=0.,
                 top_k=1,
                 layer_scale_conv=None,
                layer_scale=None,
                qkv_bias=True,
                 qk_norm=None,
                proj_drop=0.1,
                 attn_drop=0.,
                 downsample=False,
                 mamba_partition=False,
                 attention_partition=True,
                 num_experts=2,
                 ):
        super().__init__()
        self.conv_blocks=nn.ModuleList()
        self.moe_blocks=nn.ModuleList()
        
        if conv:
            for i in range(depth):
                self.conv_blocks.append = ConvBlock(dim=dim,
                                                   drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                                   layer_scale=layer_scale_conv)
                                                   
        else:
            if share_gate:
                self.gate=HybridGate(dim=dim,num_experts=num_experts)
            else:
                self.gate=None
            for i in range(depth):
                
                self.moe_blocks.append(
                SimpleMoEBlock(dim=dim, 
                               gate=self.gate,
                              num_heads=num_heads, 
                              top_k=top_k,
                              window_size=window_size,
                            qkv_bias=qkv_bias,
                            qk_norm=qk_norm,
                            attn_drop=attn_drop,
                            proj_drop=proj_drop,
                            layer_scale=layer_scale,
                            mlp_ratio=mlp_ratio,
                            drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                            mamba_partition=mamba_partition,  
                            attention_partition=attention_partition,
                            layer_scale_conv=layer_scale_conv,
                            num_experts=num_experts
                            ))
   
        self.downsample = None if not downsample else Downsample(dim=dim)
    def forward(self, x):
        #(B, C, H, W)

        for blk in self.conv_blocks:
            x=blk(x)
        
        for blk in self.moe_blocks:
                x = blk(x)
        if self.downsample is not None:
            x= self.downsample(x)
        return x
    


class M3Avision(nn.Module):
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
                 num_heads,
                 drop_path_rate=0.2,
                 in_channels=3,
                 num_classes=1000,
                 qkv_bias=True,
                 qk_norm=None,
                 attn_drop=0.,
                 layer_scale=None,
                 layer_scale_conv=None,
                 top_k=1,
                 proj_drop=0.,
                 mamba_partition=[False, True, True, False],
                 attention_partition=[False, True, True, False],
                 share_gate=True,
    ):
        super().__init__()
        num_features=int(dim*2**(len(depths)-1))
        self.num_classes = num_classes
        dpr= [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.patch_embed = PatchEmbed(in_channels=in_channels,in_dim=in_dim,dim=dim)
        self.levels = nn.ModuleList()
        for i, depth in enumerate(depths):
            conv=True if i<2 else False
            level=GlobalLocalLayer(
                dim=dim*2**i,
                depth=depth,
                window_size=window_size[i],
                num_heads=num_heads[i],
                conv=conv,
                mlp_ratio=mlp_ratio,
                top_k=top_k,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                drop_path=dpr[sum(depths[:i]):sum(depths[:i+1])],
                layer_scale_conv=layer_scale_conv,
                layer_scale=layer_scale,
                downsample=True if i < len(depths)-1 else False,
                mamba_partition=mamba_partition[i],
                attention_partition=attention_partition[i],
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
    model = M3Avision(
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
        top_k=1,
        proj_drop=0.1,
        mamba_partition=[False, False, True, False],
        attention_partition=[True, True, True, False],
        share_gate=True
    )
    checkpoint=torch.load('/home/shulab/wyc/mambavision_moe_simple/checkpoint_epoch_130.pth')
    model = model.to('cuda:0')
    #model.load_state_dict(checkpoint['model'])
    for name,params in model.named_parameters():
        if name in checkpoint['model'] and 'gate' not in name:
            params.data.copy_(checkpoint['model'][name].to(params.device))
        

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
    print(model.levels[2].moe_blocks[3].current_batch_expert_counts)
    #%%
    for name,p in model.named_parameters():
        print(name,p.numel())

# %%

