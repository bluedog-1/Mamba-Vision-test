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
        local window features (num_windows*B, window_size,window_size, C)
    """
    B,C,H,W=x.shape
    x=x.view(B,C,H//window_size,window_size,W//window_size,window_size).contiguous()
    x=rearrange(x,'b c hw hs ww ws -> (b hw ww)  (hs ws) c' )
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
    ws=hs=window_size

    hw,ww=H/window_size,W/window_size
    b=x.shape[0]//hw//ww
    x=rearrange(x,'(b hw ww) (hs ws) c -> b c (hw ws) (ww ws)',hw=hw,ww=ww,ws=ws,hs=hs,b=b)
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
   
class Block(nn.Module):
    def __init__(self, 

                 #================
                 atten_config,
                 mamba_config,
                 #======================
                 dim,
                 mlp_ratio=4,
                 reduction_ratio=4,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 layer_scale=None,
                 window_size=7,
                 channel_scaler=True
                 ):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError('dim must be oven')
        self.half_dim = dim // 2
        self.window_size=window_size
        self.atten = Attention(dim=self.half_dim,**atten_config)
        self.mamba = MambaVisionMixer(d_model=self.half_dim,**mamba_config) 
        self.channel_scaler = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),                           # (B, dim, 1, 1) - Squeeze
            nn.Conv2d(dim, dim // reduction_ratio, 1, 1),      # 第一个1x1卷积，降低维度
            act_layer(),                                         # 非线性激活
            nn.Conv2d(dim // reduction_ratio, dim, 1, 1),      # 第二个1x1卷积，恢复维度
            nn.Sigmoid()                                       # Sigmoid确保输出在0-1之间，作为缩放因子
        ) if channel_scaler else None
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, dim*mlp_ratio, kernel_size=1),
            act_layer(),
            nn.Conv2d(dim*mlp_ratio, dim, kernel_size=1)
        )
        self.atten_norm=norm_layer(self.half_dim) if norm_layer is not None else nn.Identity()
        self.mamba_norm=norm_layer(self.half_dim) if norm_layer is not None else nn.Identity()
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        use_layer_scale = layer_scale is not None and type(layer_scale) in [int, float]
        self.gamma_1 = nn.Parameter(layer_scale * torch.ones(dim))  if use_layer_scale else 1
        self.gamma_2 = nn.Parameter(layer_scale * torch.ones(dim))  if use_layer_scale else 1
    
    
    def forward(self, x):
        # 增加输入张量通道维度检查，提高健壮性
        if x.shape[1] != self.half_dim * 2: 
            raise ValueError(f"输入张量的通道维度 ({x.shape[1]}) 与预期总维度 ({self.half_dim * 2}) 不匹配。")

        B, C, H, W = x.shape
        x_atten=x[:, :self.half_dim, :, :]
        x_mamba=x[:, self.half_dim:, :, :]
               
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        if pad_r > 0 or pad_b > 0:
            x_atten = torch.nn.functional.pad(x_atten, (0,pad_r,0,pad_b))
            _, _, Hp, Wp = x_atten.shape
        else:
            Hp, Wp = H, W
        
        x_atten=windows_partition(x_atten,self.window_size)
        x_mamba=rearrange(x_mamba,'b c h w -> b (h w) c')
        x_atten=self.atten_norm(x_atten)
        x_mamba=self.mamba_norm(x_mamba)
        
        x_atten = self.atten(x_atten)
        x_mamba = self.mamba(x_mamba)
        
        x_atten=self.atten_norm(x_atten)
        x_mamba=self.mamba_norm(x_mamba)
        
        
        x_atten=windows_reverse(x_atten,self.window_size,Hp,Wp)
        if pad_r > 0 or pad_b > 0:
            x_atten = x_atten[:, :, :H, :W].contiguous()
            
        x_mamba=rearrange(x_mamba,'b (h w) c -> b c h w',h=H,w=W)
        y = torch.cat((x_atten, x_mamba), dim=1) 
        if self.channel_scaler:
            y = y * self.channel_scaler(x)
        y=x+self.drop_path(self.gamma_1*y)
        y =y+self.drop_path(self.gamma_2*self.mlp(y))
        return y
    
class GlobalLocalLayer(nn.Module):
    def __init__(self, dim, depth, 
                 window_size,
                 atten_config,
                 mamba_config,
                 conv=False,
                 mlp_ratio=4,
                 drop_path=0.,
                 layer_scale_conv=None,
                layer_scale=None,
                 downsample=False,
                 mamba_partition=False,
                 attention_partition=True,
                 num_experts=2,
                 ):
        super().__init__()
        self.conv_blocks=nn.ModuleList()
        self.mix_blocks=nn.ModuleList()
        if conv:
            for i in range(depth):
                self.conv_blocks.append = ConvBlock(dim=dim,
                                                   drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                                   layer_scale=layer_scale_conv)
                                                   
            for i in range(depth):

                self.mix_blocks.append(
                Block(dim=dim, 
                      atten_config=atten_config,
                      mamba_config=mamba_config,
                        window_size=window_size,
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
                 #==============
                d_state=16,
                d_conv=4,
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
            atten_config={
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
                #==============
                atten_config=atten_config,
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
