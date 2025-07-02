#%%
import torch
import torch.nn as nn
import math
from timm.layers import trunc_normal_, DropPath, LayerNorm2d,to_2tuple

from timm.models.vision_transformer import Mlp, PatchEmbed
import torch.nn.functional as F
from csms6s import selective_scan_fn, selective_scan_flop_jit


from einops import rearrange, repeat
from torchvision.ops import deform_conv2d
from copy import deepcopy
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

    
class ChannelLayerNorm(nn.Module):
    def __init__(self, num_channels, eps=1e-5):
        super().__init__()
        self.eps = eps
        # 可学习的缩放和偏移参数
        self.weight = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        
    def forward(self, x):
        """
        x: 输入张量，形状为 (b, c, h, w)
        返回: 归一化后的张量，形状仍为 (b, c, h, w)
        """
        # 计算每个样本每个空间位置的通道维度的均值和方差
        mean = torch.mean(x, dim=1, keepdim=True)  # 形状: (b, 1, h, w)
        var = torch.var(x, dim=1, keepdim=True, unbiased=False)  # 形状: (b, 1, h, w)
        
        # 归一化
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        
        # 应用可学习的缩放和偏移
        return x_norm * self.weight + self.bias
    
class RelativePositionEncoding(nn.Module):
    def __init__(self, channel: int, radius_bins: int = 100):
        """
        参数:
            channel: 输出通道数
            radius_bins: 半径离散化区间数（非最大半径）
        """
        super().__init__()
        self.channel = channel
        self.radius_bins = radius_bins
        self.radius_emb = nn.Embedding(radius_bins + 1, channel)  # +1 确保安全索引
        nn.init.trunc_normal_(self.radius_emb.weight, std=0.02)
        
    def forward(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        # 计算中心坐标
        center_y, center_x = H / 2, W / 2
        
        # 生成坐标网格
        y_coords = torch.arange(H, device=device, dtype=torch.float32) - center_y
        x_coords = torch.arange(W, device=device, dtype=torch.float32) - center_x
        y_rel, x_rel = torch.meshgrid(y_coords, x_coords, indexing='ij')
        
        # 计算径向距离（0到max_radius）
        radius = torch.sqrt(y_rel ** 2 + x_rel ** 2)
        max_radius = torch.sqrt(torch.tensor(center_y**2 + center_x**2))+ 1e-6
        
        # 归一化并映射到[0, radius_bins-1]区间
        normalized_radius = (radius / max_radius) * (self.radius_bins - 1)
        
        # 计算插值权重
        idx_floor = normalized_radius.floor().long()
        idx_ceil = normalized_radius.ceil().long()
        weight_ceil = normalized_radius - idx_floor
        weight_floor = 1 - weight_ceil
        
        # 安全索引（确保不越界）
        idx_floor = idx_floor.clamp(0, self.radius_bins - 1)
        idx_ceil = idx_ceil.clamp(0, self.radius_bins - 1)
        
        # 双线性插值获取位置编码
        emb_floor = self.radius_emb(idx_floor)
        emb_ceil = self.radius_emb(idx_ceil)
        pos_enc = weight_floor.unsqueeze(-1) * emb_floor + weight_ceil.unsqueeze(-1) * emb_ceil
        
        return pos_enc.permute(2, 0, 1)  # (channel, H, W)



class MambaRFlayer(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        conv_bias=True,
        bias=False,
        fusion_type="hybrid",  # "mean" 或 "conv"
        fusion=None,
        device=None,
        dtype=None,
        first='l',
        drop_out=0.1,
        share_mamba=False,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.share_mamba = share_mamba
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.num_rotations = 4  # 4个旋转方向
        d=d_model if first=='r' else self.d_inner
        self.fusion = fusion if fusion !=None else RotationAwareFusion(d, fusion_type)
        self.first=first
        self.out_act = nn.GELU() 
        
        # 初始化共享参数
        self.A_base = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner//2,
        )#(d_inner//2,d_state)
        A_log_base = torch.log(self.A_base)
        
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        

        self.in_proj = nn.Conv2d(self.d_model, self.d_inner,kernel_size=1,stride=1, bias=bias, **factory_kwargs)
        self.out_proj = nn.Conv2d(self.d_inner, self.d_model, kernel_size=1,stride=1,bias=bias, **factory_kwargs)
        
        #self.norm=nn.LayerNorm(self.d_inner)
        # 方向特定的参数
        if share_mamba:
            self.D = nn.Parameter(torch.ones(self.d_inner//2, **factory_kwargs))
            self.D._no_weight_decay = True
            # 共享参数 - 所有方向使用同一组参数
            self.A_log = nn.Parameter(A_log_base)
            self.A_log._no_weight_decay = True
            
            # 卷积层
            self.conv1d_x = nn.Conv1d(
                in_channels=self.d_inner//2,
                out_channels=self.d_inner//2,
                bias=conv_bias,
                kernel_size=d_conv,
                padding='same',
                groups=self.d_inner//2,
                **factory_kwargs,
            )
            self.conv1d_z = nn.Conv1d(
                in_channels=self.d_inner//2,
                out_channels=self.d_inner//2,
                bias=conv_bias,
                kernel_size=d_conv,
                padding='same',
                groups=self.d_inner//2,
                **factory_kwargs,
            )
            
            # SSM参数投影
            self.x_proj = nn.Linear(
                self.d_inner//2, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
            )
            self.dt_proj = nn.Linear(self.dt_rank, self.d_inner//2, bias=True, **factory_kwargs)
            
        else:
            self.conv1d_x=nn.Conv1d(
                in_channels=(self.d_inner//2) * self.num_rotations,
                out_channels=(self.d_inner//2) * self.num_rotations,
                bias=conv_bias,
                kernel_size=d_conv,
                padding='same',
                groups=(self.d_inner//2) * self.num_rotations,
                **factory_kwargs
            )
            
            self.conv1d_z = nn.Conv1d(
                in_channels=(self.d_inner//2) * self.num_rotations,
                out_channels=(self.d_inner//2) * self.num_rotations,
                bias=conv_bias,
                kernel_size=d_conv,
                padding='same',
                groups=(self.d_inner//2)*self.num_rotations,
                **factory_kwargs
            )
            
            # x proj ============================
            self.x_proj = [
                nn.Linear(self.d_inner//2, (self.dt_rank + self.d_state * 2), bias=False)
                for _ in range(self.num_rotations)
            ]
            self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K, N, inner)
            del self.x_proj
            
            # out proj =======================================
            
            self.dropout = nn.Dropout(drop_out) if drop_out > 0. else nn.Identity()
            
            # simple init dt_projs, A_logs, Ds
            self.Ds = nn.Parameter(torch.ones((self.num_rotations * self.d_inner//2)))
            self.A_logs = nn.Parameter(torch.zeros((self.num_rotations * self.d_inner//2, self.d_state))) # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
            self.dt_projs_weight = nn.Parameter(0.1 * torch.rand((self.num_rotations, self.d_inner//2, self.dt_rank)))
            self.dt_projs_bias = nn.Parameter(0.1 * torch.rand((self.num_rotations, self.d_inner//2)))

            

    
    def apply_rotations(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        # 预先在同样设备上申请好输出 tensor
        out = x.new_empty((self.num_rotations ,B, C, H, W))
        for r in range(self.num_rotations):
            # 直接写入预分配好的内存，避免 list append + torch.cat
            out[r] = torch.rot90(x, r, dims=(2, 3))
        return out
    def forward(self, x):
        
        def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True,ssoflex=True,selective_scan_backend='mamba'):
            return selective_scan_fn(u, delta, A, B, C, D, delta_bias, delta_softplus, ssoflex, backend=selective_scan_backend)
        
        """
        x: (B, C, H, W)
        返回: (B, C, H, W)
        """
        B, C, H, W = x.shape
        seqlen = H * W
        dtype = x.dtype
        total_B = B * self.num_rotations
        if self.first == 'r':
            # 选项1: 先旋转后投影
            x_rot = self.apply_rotations(x).view(-1,C,H,W) # (r*b c h w)
            xz = self.in_proj(x_rot)
            xz = rearrange(xz, "rb d_inner h w -> rb d_inner (h w)")

        elif self.first == 'l':
            # 选项2: 先投影后旋转
            x = self.in_proj(x)  # (b, d_inner h w)
            x_rot = self.apply_rotations(x).view(-1,C,H,W)  # (r*b c h w)
            xz = rearrange(x_rot, 'rb c h w -> rb c (h w)')

        x_half, z_half = xz.chunk(2, dim=1)  # 各 (r*B, d_inner//2, L)
        
        # 卷积处理 - 使用'same' padding
        if self.share_mamba:
            x_conv = F.silu(self.conv1d_x(x_half))
            z_conv = F.silu(self.conv1d_z(z_half)) 
            A = -torch.exp(self.A_log)  # (d_inner//2, d_state)
            x_dbl = self.x_proj(rearrange(x_conv, "b n l -> (b l) n"))  # (r*B*L, dt_rank+2*d_state)
            dt, B_param, C_param = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
            
            # 转换为统一的 dtype
            dt = self.dt_proj(dt)  # (r*B*L, d_inner//2)
            dt = rearrange(dt, "(b l) d -> b d l", b=total_B, l=seqlen)
            B_param = rearrange(B_param, "(b l) dstate -> b dstate l", b=total_B, l=seqlen)
            C_param = rearrange(C_param, "(b l) dstate -> b dstate l", b=total_B, l=seqlen)

            # 批量SSM处理，转换为统一 dtype
            y = selective_scan_fn(
                x_conv, dt.to(dtype), A.to(dtype), B_param.to(dtype), C_param.to(dtype), self.D.to(dtype), 
                delta_bias=self.dt_proj.bias.to(dtype),
                delta_softplus=True,
            )  # (4*B, d_inner//2, L)
            yz = torch.cat([y, z_conv], dim=1)  # (r*B, d_inner, L)
        else:
            R=self.dt_rank
            N=self.d_state
            K=self.num_rotations
            x_half_combined = rearrange(x_half, '(r b) d l -> b (r d) l', r=K)
            z_half_combined = rearrange(z_half, '(r b) d l -> b (r d) l', r=K)

            x_conv_combined = F.silu(self.conv1d_x(x_half_combined))
            z_conv_combined = F.silu(self.conv1d_z(z_half_combined))

            # --- 混合式优化开始 ---

            # 步骤1: 在循环外，一次性计算出所有旋转的所有参数 (B, C, dt)
            # 这部分和之前的方案一样，是批处理的，非常高效

            x_proj_bias = getattr(self, "x_proj_bias", None)
            xs=x_conv_combined.view(B,self.num_rotations,-1,seqlen)

            x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight)
            if x_proj_bias is not None:
                x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
            dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
            if hasattr(self, "dt_projs_weight"):
                dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)
    
            xs = xs.view(B, -1, seqlen)#(b rd seqlen)
            dts = dts.contiguous().view(B, -1, seqlen)
            As = -self.A_logs.to(torch.float).exp() # (k * c, d_state)
            Ds = self.Ds.to(torch.float) # (K * c)
            Bs = Bs.contiguous().view(B, K, N, seqlen)
            Cs = Cs.contiguous().view(B, K, N, seqlen)
            delta_bias = self.dt_projs_bias.view(-1).to(torch.float)
            ys: torch.Tensor = selective_scan(
                xs, dts, As, Bs, Cs, Ds, delta_bias
            )#(b rd seqlen)
            yz = torch.cat([ys.view(B,K,-1,seqlen), z_conv_combined.view(B,K,-1,seqlen)], dim=2)  # (B,r, d_inner, L)
        if self.first == 'r':
            yz = rearrange(yz, "b r d_inner (h w) -> (b r) d_inner h w", h=H, w=W)
            yz=self.out_act(yz)
            out = self.out_proj(yz).view(B,K,-1,H,W)
            out=self.dropout(out)
            #  b r c h w
            # 选项1: 先输出后逆旋转
            restored = torch.zeros_like(out)
            for i in range(K):
                restored[:, i] = torch.rot90(out[:, i], -i, dims=(2, 3))
            fused = self.fusion(restored)

        elif self.first == 'l':
            # 选项2: 先逆旋转后输出
            # 转换为图像格式 (r*B, d_model, H, W)
            yz = rearrange(yz, "b r d_inner (h w) -> (b r) d_inner h w", h=H, w=W)
            yz=self.out_act(yz).view(B,K,-1,H,W) 
            yz=self.dropout(yz)
            # 拆分并逆旋转
            restored = torch.zeros_like(yz)
            for i in range(K):
                restored[:, i] = torch.rot90(yz[:, i], -i, dims=(2, 3))            
            fused = self.fusion(restored)  # b d_inner h w
            fused = self.out_proj(fused)  # b c h w
        return fused



class RotationAwareFusion(nn.Module):
    def __init__(self, d_model, fusion_type="hybrid"):
        """
        d_model: 特征维度
        fusion_type: 融合策略类型
            - "mean": 平均融合
            - "max": 最大融合
            - "attn": 基本注意力融合
            - "gated": 门控融合
            - "spatial": 空间卷积融合
            - "spatial_attn": 空间注意力融合
            - "channel_attn": 通道注意力融合
            - "hybrid": 混合注意力融合
            - "deformable": 可变形卷积融合
        """
        super().__init__()
        self.fusion_type = fusion_type
        self.d_model = d_model
        self.num_rotations = 4  # 明确指定旋转方向数量

        # 根据融合类型初始化对应模块
        if fusion_type == "spatial_attn":
            # 空间注意力融合
            self.spatial_attn = nn.Sequential(
                nn.Conv2d(d_model * 4, d_model // 4, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(d_model // 4, 4, kernel_size=1),
                nn.Softmax(dim=1)
            )
        
        elif fusion_type == "channel_attn":
            # 通道注意力融合
            self.channel_attn = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(d_model, d_model // 16, kernel_size=1, bias=False),
                nn.ReLU(),
                nn.Conv2d(d_model // 16, 4, kernel_size=1, bias=False),
                nn.Softmax(dim=1)  # 对4个方向进行softmax
            )
        
        elif fusion_type == "hybrid":
            # 混合注意力融合
            self.spatial_attn = nn.Sequential(
                nn.Conv2d(d_model * 4, d_model // 4, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(d_model // 4, 4, kernel_size=1),
                nn.Softmax(dim=1)
            )
            self.channel_attn = nn.Sequential(
                nn.Conv2d(d_model * 4, d_model // 16, kernel_size=1, bias=False),
                nn.ReLU(),
                nn.Conv2d(d_model // 16, 4, kernel_size=1, bias=False),
                nn.Softmax(dim=1)
            )
            self.gate = nn.Sequential(
                nn.Conv2d(d_model * 2, d_model, kernel_size=1),
                nn.Sigmoid()
            )
        
        elif fusion_type == "deformable":
            # 可变形卷积融合
            self.offset_net = nn.Sequential(
                nn.Conv2d(d_model * 4, d_model // 2, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(d_model // 2, 18, kernel_size=3, padding=1)  # 2*3*3个偏移量
            )
            self.deform_conv = nn.Conv2d(d_model, d_model, kernel_size=3, padding=1)
        
        elif fusion_type == "attn":
            # 基本注意力融合
            self.query = nn.Conv2d(d_model, d_model, kernel_size=1)
            self.key = nn.Conv2d(d_model, d_model, kernel_size=1)
            self.value = nn.Conv2d(d_model, d_model, kernel_size=1)
            self.scale = d_model ** -0.5
        
        elif fusion_type == "gated":
            # 门控融合
            self.gate = nn.Sequential(
                nn.Conv2d(4 * d_model, d_model, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(d_model, 4, kernel_size=1),
                nn.Sigmoid()
            )
        
        elif fusion_type == "spatial":
            # 空间卷积融合
            self.conv = nn.Sequential(
                nn.Conv2d(4 * d_model, d_model, kernel_size=3, padding=1),
                nn.ReLU()
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: 输入张量，形状为 [B, R, C, H, W]，其中 R=4 表示旋转方向
        返回: 融合后的特征 [B, C, H, W]
        """
        B, R, C, H, W = x.shape
        assert R == self.num_rotations, f"必须有 {self.num_rotations} 个旋转方向，实际输入 {R} 个"
        
        # 将所有融合方法重构为使用张量操作而不是列表
        if self.fusion_type == "mean":
            # 平均融合 - 在旋转维度上平均
            return x.mean(dim=1)
        
        elif self.fusion_type == "max":
            # 最大值融合 - 在旋转维度上取最大值
            return x.max(dim=1).values
        
        elif self.fusion_type == "spatial_attn":
            # 空间注意力融合
            # 将旋转维度移动到通道维度: [B, R, C, H, W] -> [B, R*C, H, W]
            x_flat = x.reshape(B, R * C, H, W)
            
            # 计算注意力权重: [B, R, H, W]
            attn_weights = self.spatial_attn(x_flat)
            
            # 应用注意力权重: [B, R, C, H, W] * [B, R, 1, H, W]
            fused = (x * attn_weights.unsqueeze(2)).sum(dim=1)
            return fused
        
        elif self.fusion_type == "channel_attn":
            # 通道注意力融合
            # 计算每个旋转方向的全局平均: [B, R, C, 1, 1]
            channel_avg = x.mean(dim=[3, 4], keepdim=True)
            
            # 压缩旋转维度到通道维度: [B, R*C, 1, 1]
            channel_avg_flat = channel_avg.reshape(B, R * C, 1, 1)
            
            # 计算通道注意力权重: [B, R, 1, 1]
            channel_weights = self.channel_attn(channel_avg_flat)
            
            # 应用通道注意力权重: [B, R, C, H, W] * [B, R, 1, 1, 1]
            fused = (x * channel_weights.unsqueeze(2)).sum(dim=1)
            return fused
        
        elif self.fusion_type == "hybrid":
            # 混合注意力融合
            # 空间注意力部分
            x_flat = x.reshape(B, R * C, H, W)
            spatial_weights = self.spatial_attn(x_flat)  # [B, R, H, W]
            spatial_fused = (x * spatial_weights.unsqueeze(2)).sum(dim=1)  # [B, C, H, W]
            
            # 通道注意力部分
            channel_avg = x.mean(dim=[3, 4], keepdim=True)
            channel_avg_flat = channel_avg.reshape(B, R * C, 1, 1)
            channel_weights = self.channel_attn(channel_avg_flat)  # [B, R, 1, 1]
            channel_fused = (x * channel_weights.unsqueeze(2)).sum(dim=1)  # [B, C, H, W]
            
            # 门控融合
            gate_input = torch.cat([spatial_fused, channel_fused], dim=1)
            gate = self.gate(gate_input)
            return gate * spatial_fused + (1 - gate) * channel_fused
        
        elif self.fusion_type == "deformable":
            # 可变形卷积融合
            # 将旋转维度移动到通道维度: [B, R, C, H, W] -> [B, R*C, H, W]
            x_flat = x.reshape(B, R * C, H, W)
            
            # 计算偏移量
            offsets = self.offset_net(x_flat)  # [B, 18, H, W]
            
            # 对每个旋转方向应用可变形卷积
            deformed_features = []
            for i in range(R):
                # 提取当前旋转方向: [B, C, H, W]
                rot_feature = x[:, i]
                
                # 应用可变形卷积
                deformed = deform_conv2d(
                    rot_feature, 
                    offsets, 
                    self.deform_conv.weight, 
                    self.deform_conv.bias,
                    padding=1
                )
                deformed_features.append(deformed)
            
            # 堆叠并平均
            return torch.stack(deformed_features, dim=1).mean(dim=1)
        
        elif self.fusion_type == "attn":
            # 基本注意力融合
            # 计算Q,K,V
            queries = self.query(x)  # [B, R, C, H, W]
            keys = self.key(x)       # [B, R, C, H, W]
            values = self.value(x)   # [B, R, C, H, W]
            
            # 计算注意力权重
            # 点积: [B, R, R, H, W]
            sim = torch.einsum('brchw,bschw->brshw', queries, keys) * self.scale
            
            # Softmax归一化
            attn_weights = F.softmax(sim, dim=2)  # 对s维度归一化
            
            # 应用注意力
            fused = torch.einsum('brshw,bschw->brchw', attn_weights, values)
            
            # 平均所有旋转方向的输出
            return fused.mean(dim=1)
        
        elif self.fusion_type == "gated":
            # 门控融合
            # 将旋转维度移动到通道维度: [B, R, C, H, W] -> [B, R*C, H, W]
            x_flat = x.reshape(B, R * C, H, W)
            
            # 计算门控权重: [B, R, H, W]
            gates = self.gate(x_flat)
            
            # 应用门控权重
            fused = (x * gates.unsqueeze(2)).sum(dim=1)
            return fused
        
        elif self.fusion_type == "spatial":
            # 空间卷积融合
            # 将旋转维度移动到通道维度: [B, R, C, H, W] -> [B, R*C, H, W]
            x_flat = x.reshape(B, R * C, H, W)
            
            # 应用卷积
            return self.conv(x_flat)
        
        else:
            raise ValueError(f"未知的融合类型: {self.fusion_type}")


class Block(nn.Module):
    def __init__(self, 
                 d_model,
                 mamba_config,
                 mlp_ratio=4.0, 
                 drop_rate=0., 
                 drop_path=0.,
                 norm_layer=ChannelLayerNorm,
                 layer_scale_init_value=0.,
                 ):
        super().__init__()
        # 统一使用BCHW格式

        
        self.norm1 = norm_layer(d_model)  # 替代LayerNorm
        self.mamba = MambaRFlayer(**mamba_config)
        
        # 1x1卷积替代MLP
        self.conv_mlp = nn.Sequential(
            nn.Conv2d(d_model, int(d_model * mlp_ratio), 1, 1),  # 1x1卷积
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Conv2d(int(d_model* mlp_ratio), d_model, 1, 1)   # 1x1卷积
        )
        self.norm2 = norm_layer(d_model)  # 替代LayerNorm
        
        # 层缩放参数
        self.layer_scale1 = nn.Parameter(
            layer_scale_init_value * torch.ones(1, d_model, 1, 1), 
            requires_grad=True) if layer_scale_init_value > 0 else 1
        
        self.layer_scale2 = nn.Parameter(
            layer_scale_init_value * torch.ones(1, d_model, 1, 1), 
            requires_grad=True) if layer_scale_init_value > 0 else 1
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        #self.norm3=norm_layer(dim)
    def forward(self, x):
        x=x+self.drop_path(self.layer_scale1*self.mamba(self.norm1(x)))
        x=x+self.drop_path(self.layer_scale2*self.conv_mlp(self.norm2(x)))
        return x
    
class Stage(nn.Module):
    def __init__(self,
                 depth,
                 downsample,
                 #==============================
                 mlp_ratio=4.0, 
                 drop_rate=0., 
                 drop_path=0.,
                 norm_layer=ChannelLayerNorm,
                 layer_scale_init_value=0.,
                 #================================
                 d_model=0,
                 d_state=16,
                 d_conv=4,
                 expand=2,
                 dt_rank='auto',
                 share_fusion=False,
                 fusion_type='mean',
                 first='r',
                 drop_out=0,
                 share_mamba=False
                 ):
        super().__init__()
        
        d=d_model if first=='r' else (expand * d_model)//2
        fusion = None if not share_fusion  else RotationAwareFusion(d, fusion_type)
        
        self.blocks = nn.ModuleList()
        mamba_config={
            'd_model':d_model,
            'd_state':d_state,
            'd_conv':d_conv,
            'expand':expand,
            'dt_rank':dt_rank,
            'fusion_type':fusion_type,
            'fusion':fusion,
            'first':first,
            'drop_out':drop_out,
            'share_mamba':share_mamba
        }
        for i in range(depth):
            Block_i=Block( 
                d_model=d_model,
                 mamba_config=mamba_config,
                 mlp_ratio=mlp_ratio, 
                 drop_rate=drop_rate, 
                 drop_path=drop_path,
                 norm_layer=norm_layer,
                 layer_scale_init_value=layer_scale_init_value)
            self.blocks.append(Block_i)
        self.downsample = Downsample(dim=d_model) if downsample else nn.Identity()
    
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)
        Returns:
            x: (B, C', H', W') 其中 C' = C * 2 或者 C
        """
        for block in self.blocks:
            x = block(x)
        x = self.downsample(x)
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

class MambaRFVision(nn.Module):
    """旋转增强Mamba视觉模型，整合ViT架构与Mamba序列建模能力"""
    def __init__(self,
                 in_chans=3,
                 embed_dim=96,
                 depths=[2,2,8,2],
                 num_classes=1000,
                 mlp_ratio=4.0,
                 d_state=16,
                 d_conv=4,
                 expand=1,
                 fusion_type="mean",
                 share_mamba=False,
                 drop_rate=0.1,
                 drop_path_rate=0.2,
                 layer_scale_init_value=1e-6,
                 norm_layer=ChannelLayerNorm,
                 use_position=True,
                 share_fusion=False,
                 first='l',
                 radius_bins=64,
                 dt_rank='auto'
                 ):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.use_position=use_position
        
        
        # 1. 补丁嵌入层
        self.patch_embed = PatchEmbed(
            in_channels=in_chans,
            dim=embed_dim,
        )
        if use_position:
            self.position_embedding = RelativePositionEncoding(channel=embed_dim,radius_bins=radius_bins)
        if share_fusion:
            if first=='r':
                fusion=RotationAwareFusion(d_model=embed_dim,fusion_type=fusion_type)
            if first=='l':
                fusion=RotationAwareFusion(d_model=embed_dim*expand,fusion_type=fusion_type)
        else:
            fusion=None
        # 3. Mamba配置

        drop_path = [x.item() for x in torch.linspace(0, drop_path_rate, len(depths))]  # 随机深度衰减
        # 4. 堆叠Block（包含Mamba和MLP）
        self.stages = nn.ModuleList([
            Stage(
                depth=depths[i],
                downsample=(i < len(depths) - 1),
                mlp_ratio=mlp_ratio,
                drop_rate=drop_rate,
                drop_path=drop_path[i],
                norm_layer=norm_layer,
                layer_scale_init_value=layer_scale_init_value,
                #====================
                d_model= embed_dim * (2 ** i) ,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                dt_rank=dt_rank,
                fusion_type=fusion_type,
                share_fusion=share_fusion,
                first=first,
                drop_out=drop_path[i],
                share_mamba=share_mamba
                #====================
            )
            for i in range(len(depths))
        ])
        
        final_dim = embed_dim * (2 ** (len(depths) - 1))
        # 5. 归一化和分类头
        self.norm = norm_layer(final_dim)  # 
        self.pooling = nn.AdaptiveAvgPool2d(1)  # 对序列维度进行平均池化
        self.head = nn.Linear(final_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_module)
    
    def _init_module(self, m):
        if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Embedding):
            trunc_normal_(m.weight, std=.02)
    
    def forward(self, x):
        # 1. 补丁嵌入
        x = self.patch_embed(x)  # [B, C, H, W]
        B, C, H, W = x.shape
        
        # 2. 添加位置编码
        positions = self.position_embedding(H, W,x.device)
        if self.use_position:
            x = x + positions.unsqueeze(0)  # [B, C, H, W]
        
        # 3. 通过所有Block
        for stage in self.stages:
            x = stage(x)  # 保持[B, C, H, W]格式

        # 4. 归一化
        x = self.norm(x)
        
        # 5. 全局池化和展平
        x = self.pooling(x)  # [B, C, 1, 1]
        x = x.flatten(1)  # [B, C]
        """前向传播"""
        x = self.head(x)
        return x

if __name__=='__main__':
    model=MambaRFVision(in_chans=3,
                 embed_dim=96,
                 depths=[2,2,8,2],
                 num_classes=1000,
                 mlp_ratio=4.0,
                 d_state=2,
                 d_conv=3,
                 expand=1,
                 fusion_type="mean",
                 share_mamba=False,
                 drop_rate=0.1,
                 drop_path_rate=0.2,
                 layer_scale_init_value=1e-6,
                 norm_layer=ChannelLayerNorm,
                 use_position=True,
                 share_fusion=False,
                 first='r',
                 radius_bins=64,
                 dt_rank='auto')

    for name,p in model.named_parameters():
        print(name,p.numel())
    total_params=sum(p.numel() for p in model.parameters())
    trainable_params=sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total Parameters: {total_params:,}')
    print(f'Trainable Parameters: {trainable_params:,}')
    model.cuda()
    x=torch.rand((10,3,224,224)).cuda()
# %%
    y=model(x)
    print(y.shape)
# %%
