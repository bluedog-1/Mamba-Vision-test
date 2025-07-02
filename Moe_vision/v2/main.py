#%% ddp_imagenet.py
from ast import arg
import os
import argparse
import logging
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import LambdaLR
import models  # 自定义的 MoeMambavision 模型定义
import math
import timm.data
from timm.data.mixup import Mixup
from torch.optim.lr_scheduler import CosineAnnealingLR
# 配置Mixup
mixup_fn = Mixup(
    mixup_alpha=0.8,
    cutmix_alpha=1.0,
    cutmix_minmax=None,
    prob=0.8,
    switch_prob=0.5,
    mode='batch',
    label_smoothing=0.1,
    num_classes=1000
)

logging.basicConfig(filename='main.log', level=logging.INFO, format='%(asctime)s %(message)s')





def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='/media/shulab/wyc_zhongri1/imageNet', type=str, help='ImageNet 数据根目录')
    parser.add_argument('--batch_size', type=int, default=128)  # 每GPU的batch size
    parser.add_argument('--epochs', type=int, default=150)      # ViT通常需要更长的训练
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--lr', default=5e-4, type=float)      # 基准学习率，调整为ViT常用值
    parser.add_argument('--min_lr', default=1e-7, type=float)
    parser.add_argument('--max_lr', default=0.0005, type=float, help='最大学习率上限')
    parser.add_argument('--weight_decay', type=float, default=0.05)  # ViT常用的权重衰减
    parser.add_argument('--gradient_clip', type=float, default=1.0)
    parser.add_argument('--accumulation_steps', type=int, default=1, help='梯度累积步数')
    parser.add_argument('--amp', action='store_true', default=True, help='启用混合精度训练')
    parser.add_argument('--ema', action='store_true', default=True, help='启用EMA')
    parser.add_argument('--ema_decay', type=float, default=0.9999, help='EMA衰减率')
    parser.add_argument('--resume', type=str, default='', help='恢复训练的检查点路径')
    parser.add_argument('--warmup_epochs', type=int, default=5, help='学习率warmup的轮数')
    return parser.parse_args()


def get_cosine_schedule_with_warmup(
    optimizer, 
    num_warmup_steps, 
    num_training_steps, 
    min_lr_ratio=0.01, 
    max_lr=0.005,
    base_lr=0.005
):
    """
    创建带线性预热和余弦衰减的学习率调度器（带安全上限）
    
    Args:
        optimizer: 优化器对象
        num_warmup_steps: 预热步数
        num_training_steps: 总训练步数
        min_lr_ratio: 最小学习率与基础学习率的比值
        max_lr: 最大学习率绝对上限
    """
    # 获取基础学习率（从优化器的第一个参数组）
    
    # 计算最大学习率乘数（相对基础学习率）
    base_lr=base_lr
    max_multiplier = max_lr / base_lr
    def lr_lambda(current_step):
        # 1. 线性预热阶段
        if current_step < num_warmup_steps:
            warmup_ratio = float(current_step) / float(max(1, num_warmup_steps))
            # 预热阶段也应用上限
            return min(warmup_ratio, max_multiplier)
        
        # 2. 余弦退火阶段
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        # 确保进度在[0,1]范围内
        progress = min(max(progress, 0.0), 1.0)
        
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        
        # 计算最终的学习率乘数，使其从1平滑下降到min_lr_ratio
        final_multiplier = min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay
        
        # 应用学习率上限
        return min(final_multiplier, max_multiplier)

    return LambdaLR(optimizer, lr_lambda)

class EMA:
    """ 适用于PyTorch DDP的指数移动平均实现
    核心功能：
    - 参数影子副本维护
    - 分布式多卡参数同步
    - 动态衰减（训练初期更友好）
    - 模型参数应用与恢复
    """
    def __init__(self, model, decay=0.999):
        self.decay = decay
        # 处理DDP包装，获取原始模型参数
        self.model = model.module if hasattr(model, 'module') else model
        # 初始化影子参数副本（保留设备信息）
        self.shadow = {
            name: param.detach().clone()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }
        self.device = next(self.model.parameters()).device  # 记录设备
        self.updates = 0  # 记录更新次数

    def update(self, model):
        """ 更新EMA参数（带动态衰减）
        Args:
            model: DDP包装的模型，内部自动处理model.module
        """
        with torch.no_grad():
            self.updates += 1
            # 动态衰减：训练初期衰减更小，提升稳定性
            effective_decay = min(
                self.decay,  # 最大衰减上限
                (1 + self.updates) / (10 + self.updates)  # 初始衰减从0.1开始平滑上升
            )
            # 处理DDP包装
            target_model = model.module if hasattr(model, 'module') else model
            for name, param in target_model.named_parameters():
                if name in self.shadow:
                    # 确保影子参数与当前参数在同一设备
                    shadow_param = self.shadow[name].to(param.device)
                    # EMA更新公式：shadow = decay * shadow + (1-decay) * param
                    shadow_param.mul_(effective_decay).add_(param.data, alpha=1 - effective_decay)
                    # 同步回影子副本
                    self.shadow[name] = shadow_param

    def apply_to(self, model):
        """ 将EMA参数应用到模型（覆盖当前参数）
        Args:
            model: DDP包装的模型，内部自动处理model.module
        """
        target_model = model.module if hasattr(model, 'module') else model
        for name, param in target_model.named_parameters():
            if name in self.shadow:
                param.data.copy_(self.shadow[name].to(param.device))

    def sync_params(self):
        if not dist.is_initialized():
            return
        for name, param in self.shadow.items():
            # 确保参数在正确设备上
            param = param.to(self.device)
            # 主进程发送，其他进程接收
            dist.broadcast(param, src=0)
        torch.cuda.synchronize()


                
def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = torch.zeros(1, device=device)
    total_correct = torch.zeros(1, device=device)
    total_samples = torch.zeros(1, device=device)
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            preds = outputs.argmax(dim=1)
            correct = preds.eq(labels).sum()
            
            total_loss += loss.item() * labels.size(0)
            total_correct += correct
            total_samples += labels.size(0)
    torch.cuda.synchronize()
    dist.barrier()
    dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_correct, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_samples, op=dist.ReduceOp.SUM)
    avg_loss = total_loss / total_samples
    acc = total_correct / total_samples * 100
    

    
    
    return avg_loss, acc

def save_checkpoint(state, filename='checkpoint.pth'):
    torch.save(state, filename)
    logging.info(f"Saved checkpoint to {filename}")
def get_transform(is_train):
    if is_train:
        # 使用timm的高级数据增强
        return transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.08, 1.0)), 
            transforms.RandomHorizontalFlip(),
            timm.data.auto_augment.rand_augment_transform(config_str='rand-m9-mstd0.5', hparams={}),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.1)),
            transforms.RandomGrayscale(p=0.1)
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

def main_worker(local_rank, args):
 
    # 1) 初始化进程组
    torch.cuda.set_device(local_rank)
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
    master_port = os.environ.get("MASTER_PORT", "12355")

    dist.init_process_group(
      backend='nccl',
      init_method=f"tcp://{master_addr}:{master_port}",
      rank=rank,
      world_size=world_size
    )
    total_batch_size = args.batch_size * world_size * args.accumulation_steps
    base_lr = args.lr * total_batch_size / 512  # ViT学习率缩放规则
    
    if rank == 0:
        logging.info(f"Effective batch size: {total_batch_size}, Adjusted LR: {base_lr:.6f}")
    # 2) 构建模型 + DDP 包装
    model = models.M3Avision( 
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
    top_k=2,
    proj_drop=0.1,
    mamba_partition=[False, False, True, False],
    attention_partition=[True, True, True, False]
    )
    model.cuda(local_rank)
    start_epoch = 1
    best_acc = 0.0
    ema = None
    if args.ema: 
        ema = EMA(model, decay=args.ema_decay)
    if args.resume and os.path.isfile(args.resume):
        checkpoint = torch.load(args.resume, map_location=f'cuda:{local_rank}',weights_only=False)
        if hasattr(model, 'module'):
            model.module.load_state_dict(checkpoint['model'],strict=False)
        else:
            model.load_state_dict(checkpoint['model'],strict=False)
        # if rank == 0:
        
        best_acc = checkpoint.get('best_acc', 0)
        start_epoch = checkpoint['epoch'] + 1

        # 新增：恢复EMA影子参数
        if args.ema and 'ema_shadow' in checkpoint:
            # 确保影子参数设备与当前设备一致
            ema.shadow = {}
            for name, param in checkpoint['ema_shadow'].items():
                if param.device != ema.device:
                    param = param.to(ema.device)
                ema.shadow[name] = param
            if 'ema_updates' in checkpoint:
                ema.updates = checkpoint['ema_updates']
        if rank == 0:
            logging.info(f"Resumed EMA state with {ema.updates} updates")
        if rank == 0:
            logging.info(f"Loaded checkpoint '{args.resume}' bect_acc {best_acc} (epoch {checkpoint['epoch']})")
            print(f"Loaded checkpoint '{args.resume}' bect_acc {best_acc} (epoch {checkpoint['epoch']})")        
    if args.ema and dist.is_initialized():
        ema.sync_params()   
              
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=False, broadcast_buffers=True)

    # 3) 数据加载 + DistributedSampler
    train_dataset = datasets.ImageFolder(os.path.join(args.data_path, 'train'), get_transform(is_train=True))
    val_dataset = datasets.ImageFolder(os.path.join(args.data_path, 'val'), get_transform(is_train=False))

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank,shuffle=True)
    val_sampler   = DistributedSampler(val_dataset,   num_replicas=world_size, rank=rank)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=train_sampler,
        num_workers=args.num_workers, pin_memory=True,persistent_workers=True,prefetch_factor=4,drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, sampler=val_sampler,
        num_workers=args.num_workers, pin_memory=True,persistent_workers=True,prefetch_factor=4
    )

    # 4) 优化器、调度器、混合精度
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
    )
    
    steps_per_epoch = len(train_loader) // args.accumulation_steps
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = steps_per_epoch * args.warmup_epochs
    min_lr_ratio = args.min_lr / args.lr # 计算最小学习率比率
    if rank == 0:
        logging.info(f"Total training steps: {total_steps}")
        logging.info(f"Warmup steps: {int(args.warmup_epochs * total_steps / args.epochs)}")
        
        
    # scheduler = CosineAnnealingLR(
    #     optimizer,
    #     T_max=total_steps,      # 调度周期（step 数）
    #     eta_min=args.min_lr     # 最低 lr
    # )    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
        min_lr_ratio=min_lr_ratio,
        max_lr=args.max_lr,
        base_lr=args.lr
    )
    
    if args.resume and os.path.isfile(args.resume):
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])

        if 'scheduler' in checkpoint:
            # 推算 resume_step
            scheduler.load_state_dict(checkpoint['scheduler'])
            resume_epoch = checkpoint.get('epoch', 0)
            resume_step = resume_epoch * steps_per_epoch
            
            scheduler.last_epoch = resume_step-1
            scheduler.step()

            if rank == 0:
                logging.info(f"Resume from epoch {resume_epoch}, global step {resume_step}")
                logging.info(f"Learning rate after resume: {scheduler.get_last_lr()}")

        else:
            logging.warning("No scheduler state in checkpoint. Starting scheduler from scratch.")
                
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1).cuda(local_rank)
    
    val_loss, val_acc = validate(model, val_loader, criterion, local_rank)
    # if rank == 0:
    #     logging.info(f"Resumed model validation: Loss={val_loss.item():.4f}, Acc={val_acc.item():.2f}%")
    #     print(f"Resumed model validation: Loss={val_loss.item():.4f}, Acc={val_acc.item():.2f}%")
    
    scaler = GradScaler() if args.amp else None
    if args.amp and hasattr(torch, 'set_float32_matmul_precision'):
        torch.set_float32_matmul_precision('high')
    if args.amp and 'scaler' in checkpoint and checkpoint['scaler'] is not None:
        scaler.load_state_dict(checkpoint['scaler'])
        
    # 5) 训练 & 验证循环
    current_step = 0
    logging.info("Training started...")
    try:
        for epoch in range(start_epoch, args.epochs + 1):           
            total_loss = torch.tensor(0.0, device=local_rank)
            total_correct = torch.tensor(0, device=local_rank)
            total_samples = torch.tensor(0, device=local_rank)

            for batch_idx, (images, labels) in enumerate(train_loader):
                
                images = images.cuda(local_rank, non_blocking=True)
                labels = labels.cuda(local_rank, non_blocking=True)

                if mixup_fn is not None:
                    images, labels = mixup_fn(images, labels)
                    
                if args.amp:
                    with autocast(device_type='cuda'):
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                    if args.accumulation_steps > 1:
                        loss = loss / args.accumulation_steps
                else:
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    if args.accumulation_steps > 1:
                        loss = loss / args.accumulation_steps
                    
                    
                    
                # —— 新增：检查 loss 是否为 NaN —— #
                if torch.isnan(loss):
                    logging.warning(f"NaN loss at epoch {epoch} batch {batch_idx}, skipping batch")
                    print(outputs.mean(), outputs.std(), outputs.max(), outputs.min())
                    logging(outputs.mean(), outputs.std(), outputs.max(), outputs.min())
                    for name, param in model.named_parameters():
                        if torch.isnan(param).any():
                            print(f"NaN in {name}")
                            logging.warning(f"NaN in {name}")

                    # 1) 清空梯度，避免把 NaN 累积下去
                    # optimizer.zero_grad()
                    # # 2) 将学习率砍半，减少不稳定
                    # for pg in optimizer.param_groups:
                    #     pg['lr'] *= 0.5
                    # logging.info(f"Reduced LR to {optimizer.param_groups[0]['lr']:.6e}")
                    # 3) 如果用了 EMA，就把模型恢复到上一次稳定状态
                    # if args.ema:
                    #     ema.apply_to(model)
                    checkpoint = torch.load(args.resume, map_location=f'cuda:{local_rank}')
                    if hasattr(model, 'module'):
                        model.module.load_state_dict(checkpoint['model'],strict=False)
                    else:
                        model.load_state_dict(checkpoint['model'],strict=False)
                        optimizer.zero_grad()
                        optimizer.load_state_dict(checkpoint['optimizer'])
                        for pg in optimizer.param_groups:
                            pg['lr'] *= 0.5
                    # 跳过这次 batch 的 backward、step、metric 计算
                    continue
                # —— 结束 NaN 检测 —— #

                # —— 正常 backward & step —— #
                if args.amp:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                # 1) 累积梯度

                if (batch_idx + 1) % args.accumulation_steps == 0:
                    # 梯度裁剪
                    if args.amp:
                        scaler.unscale_(optimizer)
                    if args.gradient_clip > 0:
                            nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
                    if args.amp: 
                        # 更新参数
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad()
                    # 更新学习率
                    current_step += 1
                    scheduler.step()
                        
                    
                    # 更新EMA
                    if args.ema:
                        ema.update(model)
                
                # 更新统计信息
                # 检查是否使用了Mixup，处理标签格式
                if mixup_fn is not None:
                    # 如果使用了Mixup，labels是平滑的概率分布，需要转换为类别索引进行准确率计算
                    # 注意：Mixup可能会降低训练阶段的准确率计算意义，但有助于模型泛化
                    _, labels_indices = labels.max(dim=1)
                    preds = outputs.argmax(dim=1)
                    correct = preds.eq(labels_indices).sum()
                else:
                    # 正常情况：labels是类别索引
                    preds = outputs.argmax(dim=1)
                    correct = preds.eq(labels).sum()

                total_loss += loss.item() * args.accumulation_steps * labels.size(0)
                total_correct += correct
                total_samples += labels.size(0)
                
                if batch_idx % 50 == 0:
                    stats = torch.tensor(
                    [ total_loss,
                      float(total_correct),
                      float(total_samples) ],
                    dtype=torch.float32,
                    device=local_rank
                    )
                    dist.all_reduce(stats, op=dist.ReduceOp.SUM)
                    # 仅在主进程计算并打印
                    if rank == 0:
                        global_loss_sum, global_correct_sum, global_samples_sum = stats.tolist()
                        avg_loss = global_loss_sum / global_samples_sum
                        acc = global_correct_sum / global_samples_sum * 100

                        total_norm = torch.nn.utils.clip_grad_norm_(
                            model.parameters(), max_norm=1e10
                        )
                        current_lr = optimizer.param_groups[0]['lr']

                        msg = (
                            f"Epoch: {epoch}/{args.epochs} | "
                            f"Batch: {batch_idx}/{len(train_loader)} | "
                            f"Loss: {avg_loss:.4f} | "
                            f"Acc: {acc:.2f}% | "
                            f"LR: {current_lr:.7f} | "
                            #f"Grad Norm: {total_norm:.4f}"
                        )
                        logging.info(msg)
                        print(msg)
                # 重置统计数据
                    total_loss = torch.tensor(0.0, device=local_rank)
                    total_correct = torch.tensor(0, device=local_rank)
                    total_samples = torch.tensor(0, device=local_rank)
            
            val_loss, val_acc = validate(model, val_loader, criterion, local_rank)
            ema_val_loss, ema_val_acc = None, None
            if args.ema and (epoch % 5 == 0 or epoch == args.epochs):
                # 保存原始参数
                original_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
                
                # 所有进程同步应用EMA参数
                ema.apply_to(model)
                
                # 所有进程执行EMA验证
                ema_val_loss, ema_val_acc = validate(model, val_loader, criterion, local_rank)
                
                # 所有进程恢复原始参数
                if hasattr(model, 'module'):
                    model.module.load_state_dict(original_state)
                else:
                    model.load_state_dict(original_state)
                    
            if rank == 0:
                logging.info(f"Val Loss: {val_loss.item():.4f} | Val Acc: {val_acc.item():.2f}%")
                print(f"Val Loss: {val_loss.item():.4f} | Val Acc: {val_acc.item():.2f}%")
                            
                # 打印EMA验证结果（如果有）
                if ema_val_acc is not None:
                    logging.info(f"EMA Val Loss: {ema_val_loss.item():.4f} | EMA Val Acc: {ema_val_acc.item():.2f}%")
                    print(f"EMA Val Loss: {ema_val_loss.item():.4f} | EMA Val Acc: {ema_val_acc.item():.2f}%")
                
                current_acc = ema_val_acc if ema_val_acc is not None else val_acc
                
                def save_model(epoch, is_best=False):
                    filename = 'best_model.pth' if is_best else f'checkpoint_epoch_{epoch}.pth'
                    save_checkpoint({
                        'epoch': epoch,
                        'model': model.module.state_dict(),
                        'ema_shadow': ema.shadow if args.ema else None,
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'ema_updates': ema.updates if args.ema else 0,  # 保存更新计数
                        'scaler': scaler.state_dict() if args.amp else None, 
                        'best_acc': best_acc,
                    }, filename)
                    
                    if is_best:
                        logging.info(f"Saved best model with acc: {best_acc.item():.2f}%")
                
                    # 保存最佳模型
                if current_acc > best_acc:
                    best_acc = current_acc
                    save_model(epoch, is_best=True)
                
                # 定期保存
                if epoch % 10 == 0:
                    save_model(epoch)


        if rank == 0:
            logging.info(f"Training completed. Best accuracy: {best_acc:.2f}%")
    except KeyboardInterrupt:
        if rank == 0:
            logging.info("Training interrupted by user")
            save_checkpoint({
                    'epoch': epoch,
                    'model': model.module.state_dict(),
                    'ema_shadow': ema.shadow if args.ema else None,
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'ema_updates': ema.updates if args.ema else 0,  # 保存更新计数
                    'scaler': scaler.state_dict() if args.amp else None,  # ✅ 加这一句
                    'best_acc': best_acc,
                }, 'interrupted_model.pth')
    finally:
        # 确保资源正确释放
        dist.destroy_process_group()
        if hasattr(train_loader, 'sampler') and hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(0)
        if hasattr(val_loader, 'sampler') and hasattr(val_loader.sampler, 'set_epoch'):
            val_loader.sampler.set_epoch(0)
if __name__ == "__main__":
    args = parse_args()
        # torchrun 自动提供 LOCAL_RANK 环境变量
    local_rank = int(os.environ["LOCAL_RANK"])
    # 直接执行
    main_worker(local_rank, args)
