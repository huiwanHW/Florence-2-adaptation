import argparse
import os
import sys
# Add project root directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.insert(0, project_root)

import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict
import numpy as np
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import logging
import time
import segmentation_models_pytorch.losses as smp_losses

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('train_florence_no_lora.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Setup epoch results logger (only records epoch results)
epoch_logger = logging.getLogger('epoch_logger')
epoch_logger.setLevel(logging.INFO)
epoch_logger.addHandler(logging.FileHandler('output_train_florence_no_lora.log'))
epoch_logger.propagate = False

from dataset import MultiTaskDataset, MultiTaskUniformSampler
from model_factory import MultiTaskModelFactory, TASK_CONFIGURATIONS
from utils import multi_task_collate_fn, evaluate, set_seed, DetectionLoss, generate_gaussian_heatmap_with_dynamic_radius

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

# Override model_factory to use no lora adapter
# Replace the import in model_factory
import model_factory
model_factory.FlorenceEncoderAdapter = None

# Import the no lora adapter
from ablation_study.florence_adapter_no_lora import FlorenceEncoderAdapter
model_factory.FlorenceEncoderAdapter = FlorenceEncoderAdapter

# Update the encoder import in model_factory
model_factory.FlorenceEncoderAdapter = FlorenceEncoderAdapter

def get_transforms():
    # Base transforms that apply to all tasks
    base_train_transforms = [
        # 固定大小调整 - 所有任务都需要
        A.Resize(256, 256),
        # 归一化
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        # 转换为Tensor
        ToTensorV2(),
    ]
    
    # 基础验证变换
    val_transforms = A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'], min_visibility=0.1))
    
    # 针对不同任务的增强策略
    train_transforms = A.Compose([
        # 通用增强
        A.RandomResizedCrop(height=256, width=256, scale=(0.8, 1.0), ratio=(0.9, 1.1), p=0.5),
        A.Rotate(limit=30, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
        A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=20, val_shift_limit=10, p=0.3),
        A.Blur(blur_limit=(3, 7), p=0.2),
        
        # 针对fetal_abdomen_multi的特殊增强
        A.ElasticTransform(alpha=1, sigma=50, p=0.3),  # 增加弹性变形概率
        A.GridDistortion(distort_limit=0.2, p=0.3),  # 增加网格变形
        A.OpticalDistortion(distort_limit=0.1, p=0.2),  # 增加光学变形
        
        # 针对回归任务的特殊增强
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.3),  # 更温和的变换
        
        # 高斯噪声 - 增加回归任务的鲁棒性
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
        
        # 固定大小调整和归一化
        A.Resize(256, 256),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'], min_visibility=0.1))

    return train_transforms, val_transforms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--unfreeze', action='store_true', help='Unfreeze Florence encoder for fine-tuning')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--train_only', type=str, default=None, help='Train only specified task type (e.g., detection)')
    args = parser.parse_args()

    logger.info(f"Starting training with parameters: {args}")
    logger.info("=== ABLATION: w/o LoRA (LoRA disabled) ===")
    
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    train_transforms, val_transforms = get_transforms()
    logger.info("Transforms initialized")

    # Datasets (use all data for training, no validation split)
    temp_dataset = MultiTaskDataset(data_root=args.data_root, transforms=train_transforms)
    dataset_size = len(temp_dataset)
    train_size = dataset_size  # Use all data for training
    val_size = 0  # No validation set

    logger.info(f"Dataset size: {dataset_size}, Train size: {train_size}, Val size: {val_size}")

    # Use all indices for training
    indices = list(range(dataset_size))
    
    train_dataset = MultiTaskDataset(data_root=args.data_root, transforms=train_transforms)
    val_dataset = MultiTaskDataset(data_root=args.data_root, transforms=val_transforms)

    # Use all data for training
    train_subset = torch.utils.data.Subset(train_dataset, indices)
    train_subset.dataframe = train_dataset.dataframe.reset_index(drop=True)
    
    # Create empty validation set for compatibility
    val_subset = torch.utils.data.Subset(val_dataset, [])
    val_subset.dataframe = val_dataset.dataframe.iloc[:0].reset_index(drop=True)

    train_sampler = MultiTaskUniformSampler(train_subset, batch_size=args.batch_size)
    train_loader = torch.utils.data.DataLoader(train_subset, batch_sampler=train_sampler, num_workers=4, pin_memory=True, collate_fn=multi_task_collate_fn)
    val_loader = torch.utils.data.DataLoader(val_subset, batch_size=8, shuffle=False, num_workers=4, pin_memory=True, collate_fn=multi_task_collate_fn)

    logger.info("Data loaders initialized")

    # Model
    # Pass freeze_stages=0 when --unfreeze is specified (no freezing), else freeze_stages=4 (full freezing)
    freeze_stages = 0 if args.unfreeze else 4
    logger.info(f"Initializing model with freeze_stages: {freeze_stages}")
    model = MultiTaskModelFactory(
        encoder_name='florence', 
        encoder_weights=None, 
        task_configs=TASK_CONFIGURATIONS,
        freeze_stages=freeze_stages
    ).to(device)

    logger.info(f"Model initialized with {len(model.heads)} task heads")

    # Apply single-task training if specified
    if args.train_only:
        logger.info(f"Training only {args.train_only} tasks, freezing other heads")
        for task_id, head in model.heads.items():
            task_config = next((c for c in TASK_CONFIGURATIONS if c['task_id'] == task_id), None)
            if task_config:
                task_name = task_config['task_name'].lower()
                if args.train_only.lower() not in task_name:
                    # Freeze this head
                    for p in head.parameters():
                        p.requires_grad = False

    # Setup parameter groups for different optimizers
    # 分割任务参数组
    seg_param_groups = []
    # 其他任务参数组
    other_param_groups = []
    
    # 提取backbone后30%的可训练参数
    backbone_params = []
    if any(p.requires_grad for p in model.encoder.parameters()):
        backbone_params = [p for p in model.encoder.parameters() if p.requires_grad]
    
    # 为分割任务添加参数
    seg_lr = 1e-4  # 分割任务的学习率
    other_lr = 5e-5  # 其他任务的学习率
    
    print(f"设置学习率：分割任务 = {seg_lr}, 其他任务 = {other_lr}")
    
    # 添加分割任务头部参数
    for task_id, head in model.heads.items():
        task_config = next((c for c in TASK_CONFIGURATIONS if c['task_id'] == task_id), None)
        task_name = task_config['task_name'] if task_config else 'classification'
        
        if task_name == 'segmentation':
            seg_param_groups.append({'params': head.parameters(), 'lr': seg_lr * 10})
        else:
            other_param_groups.append({'params': head.parameters(), 'lr': other_lr * 5})
    
    # 添加分割专用解码器参数
    if hasattr(model, 'segmentation_decoder') and model.segmentation_decoder is not None:
        seg_param_groups.append({'params': model.segmentation_decoder.parameters(), 'lr': seg_lr * 5})
    
    # 添加FPN解码器参数
    if hasattr(model, 'fpn_decoder') and model.fpn_decoder is not None:
        seg_param_groups.append({'params': model.fpn_decoder.parameters(), 'lr': seg_lr * 5})
        other_param_groups.append({'params': model.fpn_decoder.parameters(), 'lr': other_lr * 5})
    
    # 添加backbone参数到两个优化器，但使用不同的学习率
    if backbone_params:
        seg_param_groups.append({'params': backbone_params, 'lr': seg_lr})
        other_param_groups.append({'params': backbone_params, 'lr': other_lr})
    
    # 创建两个优化器
    seg_optimizer = optim.AdamW(seg_param_groups, weight_decay=0.01)
    other_optimizer = optim.AdamW(other_param_groups, weight_decay=0.01)
    
    # 创建两个调度器
    seg_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(seg_optimizer, T_0=10, T_mult=2, eta_min=5e-7)
    other_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(other_optimizer, T_0=10, T_mult=2, eta_min=5e-7)
    
    # Store original learning rates for warmup
    seg_original_lrs = [pg['lr'] for pg in seg_param_groups]
    other_original_lrs = [pg['lr'] for pg in other_param_groups]
    
    # Add warmup phase configuration
    warmup_epochs = 3
    warmup_steps = len(train_loader) * warmup_epochs
    current_step = 0
    
    # 动态loss weight配置
    initial_seg_weight = 2.0  # 初始分割权重
    final_seg_weight = 1.0  # 最终分割权重
    weight_decay_epochs = 10  # 权重衰减的epoch数

    # Fine-grained task weights based on task difficulty and importance
    task_weights = {
        # Segmentation tasks
        'breast_lesion': 4.0,  # Increased weight for poor performer
        'thyroid_nodule': 4.0,  # Increased weight for poor performer
        'ovary_tumor': 3.0,  # Increased weight
        'fetal_heart': 3.0,  # Increased weight
        'lung': 2.0,
        'head_symphysis_multi': 2.0,
        'cardiac_multi': 1.5,  # Decreased weight for good performer
        'carotid_artery': 2.5,  # Increased weight
        'cervix': 2.5,  # Increased weight
        'fetal_abdomen_multi': 0.8,  # Decreased weight as recommended
        'fetal_head': 1.5,  # Decreased weight for good performer
        'cervix_multi': 1.5,
        
        # Classification tasks
        'fetal_head_pos_cls': 3.0,  # Increased weight
        'fetal_sacral_pos_cls': 3.0,  # Increased weight
        'fetal_plane_cls': 2.5,  # Increased weight
        'organ_cls': 3.0,  # Increased weight for poor performer
        'breast_3cls': 2.5,  # Increased weight
        'breast_2cls': 2.5,  # Increased weight
        'liver_lesion_2cls': 2.0,
        'lung_2cls': 2.0,
        'lung_disease_3cls': 2.5,  # Increased weight
        
        # Detection tasks - increase all weights significantly
        'spinal_cord_injury_loc': 5.0,  # Increased weight
        'thyroid_nodule_det': 5.0,  # Increased weight
        'uterine_fibroid_det': 5.0,  # Increased weight
        
        # Regression tasks - significantly increase weights
        'FUGC': 3.0,  # Increased weight
        'IUGC': 3.0,  # Increased weight
        'fetal_femur': 8.0  # Dramatically increased weight for worst performer
    }

    # 定义Focal Loss
    class FocalLoss(nn.Module):
        def __init__(self, gamma=2.0, alpha=None, ignore_index=-1):
            super().__init__()
            self.gamma = gamma
            self.alpha = alpha
            self.ignore_index = ignore_index
            self.ce_loss = nn.CrossEntropyLoss(weight=alpha, ignore_index=ignore_index)
        
        def forward(self, inputs, targets):
            logpt = -self.ce_loss(inputs, targets)
            pt = torch.exp(logpt)
            loss = -((1 - pt) ** self.gamma) * logpt
            return loss.mean()
    
    # 定义Boundary Loss (简化版，基于边缘距离)
    class BoundaryLoss(nn.Module):
        def __init__(self, ignore_index=-1):
            super().__init__()
            self.ignore_index = ignore_index
        
        def forward(self, inputs, targets):
            # 获取预测结果
            preds = torch.argmax(inputs, dim=1)
            
            # 过滤掉ignore_index
            valid_mask = (targets != self.ignore_index).float()
            
            # 计算边界
            def compute_boundary(mask):
                # 使用Sobel算子计算梯度
                sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=mask.device).unsqueeze(0).unsqueeze(0)
                sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=mask.device).unsqueeze(0).unsqueeze(0)
                
                mask = mask.float().unsqueeze(1)
                grad_x = torch.abs(torch.nn.functional.conv2d(mask, sobel_x, padding=1))
                grad_y = torch.abs(torch.nn.functional.conv2d(mask, sobel_y, padding=1))
                boundary = torch.clamp(grad_x + grad_y, 0, 1)
                return boundary.squeeze(1)
            
            # 计算预测边界和真实边界
            pred_boundary = compute_boundary(preds)
            true_boundary = compute_boundary(targets.float())
            
            # 计算边界损失 - 对于多类别分割，使用交叉熵损失
            # 计算每个像素的预测概率
            input_softmax = torch.nn.functional.softmax(inputs, dim=1)
            
            # 为每个像素创建one-hot编码的目标
            targets_one_hot = torch.nn.functional.one_hot(targets.long(), num_classes=inputs.shape[1]).permute(0, 3, 1, 2).float()
            
            # 计算交叉熵损失
            loss = -targets_one_hot * torch.log(input_softmax + 1e-10)
            loss = loss.sum(dim=1)
            
            # 应用边界权重
            loss = loss * true_boundary
            
            # 应用有效掩码
            loss = loss * valid_mask
            
            # 平均损失
            if valid_mask.sum() > 0:
                loss = loss.sum() / valid_mask.sum()
            else:
                loss = loss.sum()
            
            return loss
    
    # 定义组合损失函数
    class SegmentationLoss(nn.Module):
        def __init__(self, ignore_index=-1):
            super().__init__()
            self.dice_loss = smp_losses.DiceLoss(mode='multiclass', ignore_index=ignore_index)
            self.focal_loss = FocalLoss(gamma=2.0, ignore_index=ignore_index)
            self.boundary_loss = BoundaryLoss(ignore_index=ignore_index)
        
        def forward(self, inputs, targets):
            dice = self.dice_loss(inputs, targets)
            focal = self.focal_loss(inputs, targets)
            boundary = self.boundary_loss(inputs, targets)
            
            # 组合损失
            total_loss = dice + focal + boundary
            return total_loss

    loss_functions = {
        'segmentation': SegmentationLoss(ignore_index=-1),
        'classification': nn.CrossEntropyLoss(),
        'Regression': nn.SmoothL1Loss(beta=0.5),  # Use SmoothL1Loss for better robustness
        'detection': DetectionLoss()  # Use new DetectionLoss with heatmap, offset, and size components
    }

    # Initialize best metrics for each task type
    best_metrics = {
        'segmentation': {'Dice': -float('inf')},
        'classification': {'AUC': -float('inf')},
        'detection': {'IoU': -float('inf')},
        'regression': {'MRE': float('inf')}  # Lower is better
    }
    
    best_val = -float('inf')  # For overall best model (deprecated, kept for compatibility)
    best_train_loss = float('inf')  # Track best training loss for model saving
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        batch_count = 0
        
        # 计算当前epoch的分割任务权重（动态调整）
        if epoch < weight_decay_epochs:
            # 线性衰减从initial_seg_weight到final_seg_weight
            seg_weight_factor = initial_seg_weight - (initial_seg_weight - final_seg_weight) * (epoch / weight_decay_epochs)
        else:
            seg_weight_factor = final_seg_weight
        
        print(f"Epoch {epoch+1}/{args.epochs}: 分割任务权重因子 = {seg_weight_factor:.4f}")
        
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs} [Train]', disable=False, leave=True):
            images = batch['image'].to(device)
            task_ids = batch['task_id']
            labels = torch.stack(batch['label']).to(device)
            current_task = task_ids[0]
            outputs = model(images, task_id=current_task)

            # Task-specific loss handling
            task_config = next((c for c in TASK_CONFIGURATIONS if c['task_id'] == current_task), None)
            task_name = task_config['task_name'] if task_config else None

            loss = None
            if task_name == 'detection':
                """
                outputs: (B, 5, H, W)
                    - Channel 0: Heatmap
                    - Channel 1-2: Center offset
                    - Channel 3-4: Box size
                labels:  (B, 4)  -> [x1, y1, x2, y2] normalized to [0,1]
                """

                B, C, H, W = outputs.shape
                assert C >= 5, "Detection head output dim must be 5 (heatmap + offset + size)"
                
                # --- 1. 生成动态半径的高斯heatmap ---
                gt_heatmaps = generate_gaussian_heatmap_with_dynamic_radius(
                    labels, (H, W), stride=4
                )
                
                # --- 2. 计算 GT box 的中心点和相关参数 ---
                gt_x1, gt_y1, gt_x2, gt_y2 = labels[:, 0], labels[:, 1], labels[:, 2], labels[:, 3]
                
                # 计算中心点归一化坐标
                gt_center_x = (gt_x1 + gt_x2) / 2.0
                gt_center_y = (gt_y1 + gt_y2) / 2.0
                
                # 计算中心点在heatmap上的坐标
                gt_center_h = gt_center_y * H
                gt_center_w = gt_center_x * W
                
                # 计算整数坐标（用于索引）和偏移量
                center_indices_h = torch.clamp(gt_center_h.long(), 0, H - 1)
                center_indices_w = torch.clamp(gt_center_w.long(), 0, W - 1)
                center_indices = torch.stack([center_indices_h, center_indices_w], dim=1)
                
                # 计算偏移量 (dx, dy)
                offset_dx = gt_center_w - center_indices_w.float()
                offset_dy = gt_center_h - center_indices_h.float()
                offsets = torch.stack([offset_dx, offset_dy], dim=1)
                
                # 计算box大小 (w, h)
                box_w = gt_x2 - gt_x1
                box_h = gt_y2 - gt_y1
                sizes = torch.stack([box_w, box_h], dim=1)
                
                # --- 3. 准备target字典 ---
                valid = (labels[:, 0] >= 0)  # Valid box mask
                detection_targets = {
                    'heatmap': gt_heatmaps,
                    'center_indices': center_indices,
                    'offset': offsets,
                    'size': sizes,
                    'valid': valid
                }
                
                # --- 4. 计算检测loss ---
                loss = loss_functions['detection'](outputs, detection_targets)

            elif task_name == 'segmentation':
                # Handle segmentation labels correctly
                if labels.dim() == 1:
                    labels = labels.unsqueeze(1)  # Add channel dimension
                loss = loss_functions['segmentation'](outputs, labels.long())

            elif task_name == 'classification':
                # Ensure labels are in the correct format
                if labels.dim() == 2 and labels.size(1) == 1:
                    labels = labels.squeeze(1)
                loss = loss_functions['classification'](outputs, labels.long())

            elif task_name == 'Regression':
                loss = loss_functions['Regression'](outputs, labels)

            else:
                loss = loss_functions.get(task_name, nn.MSELoss())(outputs, labels)

            # Apply task-specific weight
            task_weight = task_weights.get(current_task, task_weights.get(task_name, 1.0))
            
            # 对分割任务应用动态权重因子
            if task_name == 'segmentation':
                loss *= task_weight * seg_weight_factor
            else:
                loss *= task_weight

            # 根据任务类型选择优化器
            if task_name == 'segmentation':
                optimizer = seg_optimizer
                scheduler = seg_scheduler
                original_lrs = seg_original_lrs
            else:
                optimizer = other_optimizer
                scheduler = other_scheduler
                original_lrs = other_original_lrs

            optimizer.zero_grad()
            loss.backward()
            # 添加梯度裁剪来稳定训练
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Update current step for warmup
            current_step += 1
            
            # Learning rate warmup
            if warmup_steps > 0 and current_step <= warmup_steps:
                # Linear warmup from 10% of base LR to full LR
                warmup_factor = 0.1 + (1.0 - 0.1) * (current_step / warmup_steps)
                for i, param_group in enumerate(optimizer.param_groups):
                    # Apply warmup factor to original LR
                    param_group['lr'] = original_lrs[i] * warmup_factor
            
            total_loss += loss.item()
            batch_count += 1
        
        # Calculate and print average epoch loss
        avg_epoch_loss = total_loss / batch_count if batch_count > 0 else 0
        logger.info(f'[Train] Epoch {epoch+1}/{args.epochs}, Average Loss: {avg_epoch_loss:.6f}')
        logger.info(f'[Train] Segmentation LR: {seg_scheduler.get_last_lr()[0]:.8f}')
        logger.info(f'[Train] Other tasks LR: {other_scheduler.get_last_lr()[0]:.8f}')
        
        # Log epoch results to separate file
        epoch_logger.info(f'[Train] Epoch {epoch+1}/{args.epochs}, Average Loss: {avg_epoch_loss:.6f}')
        epoch_logger.info(f'[Train] Segmentation LR: {seg_scheduler.get_last_lr()[0]:.8f}')
        epoch_logger.info(f'[Train] Other tasks LR: {other_scheduler.get_last_lr()[0]:.8f}')

        # 更新两个调度器
        seg_scheduler.step()
        other_scheduler.step()

        # Save best model based on training loss
        if avg_epoch_loss < best_train_loss:
            best_train_loss = avg_epoch_loss
            # Save model with filename best_model_no_lora.pth
            torch.save(model.state_dict(), 'best_model_no_lora.pth')
            logger.info(f'\nNew best model saved (train loss = {best_train_loss:.6f}) as best_model_no_lora.pth')
            epoch_logger.info(f'\nNew best model saved (train loss = {best_train_loss:.6f}) as best_model_no_lora.pth')

        # Skip validation during training as requested
        # To enable validation, uncomment the following code block
        # Validation: compute and display group-specific metrics (no blind averaging)
        try:
            grouped_eval = evaluate(model, val_loader, device, grouped=True)

            # Print per-group metrics requested by convention
            # Segmentation: Dice (DSC), Hausdorff (HD)
            if not grouped_eval['segmentation'].empty:
                logger.info('\n[Validation] Segmentation metrics:')
                logger.info(grouped_eval['segmentation'].to_string(index=False))

            # Classification: AUC, F1, MCC
            if not grouped_eval['classification'].empty:
                logger.info('\n[Validation] Classification metrics:')
                logger.info(grouped_eval['classification'].to_string(index=False))

            # Detection: IoU
            if not grouped_eval['detection'].empty:
                logger.info('\n[Validation] Detection metrics:')
                logger.info(grouped_eval['detection'].to_string(index=False))

            # Regression: MRE
            if not grouped_eval['regression'].empty:
                logger.info('\n[Validation] Regression metrics:')
                logger.info(grouped_eval['regression'].to_string(index=False))

            # 综合计算四个任务的评分，平衡各任务表现
            # 1) segmentation Dice (higher better)
            # 2) classification AUC (higher better)
            # 3) detection IoU (higher better)
            # 4) regression MRE (lower better -> we negate to make higher better)
            scores = []
            weights = []
            
            # 分割任务评分 (Dice)
            if not grouped_eval['segmentation'].empty and 'Dice' in grouped_eval['segmentation'].columns:
                seg_score = np.nanmean(grouped_eval['segmentation']['Dice'].values)
                if not np.isnan(seg_score):
                    scores.append(seg_score)
                    weights.append(1.0)  # 分割任务权重
            
            # 分类任务评分 (AUC)
            if not grouped_eval['classification'].empty and 'AUC' in grouped_eval['classification'].columns:
                cls_score = np.nanmean(grouped_eval['classification']['AUC'].values)
                if not np.isnan(cls_score):
                    scores.append(cls_score)
                    weights.append(1.0)  # 分类任务权重
            
            # 检测任务评分 (IoU)
            if not grouped_eval['detection'].empty and 'IoU' in grouped_eval['detection'].columns:
                det_score = np.nanmean(grouped_eval['detection']['IoU'].values)
                if not np.isnan(det_score):
                    scores.append(det_score)
                    weights.append(1.0)  # 检测任务权重
            
            # 回归任务评分 (负MRE，因为MRE越低越好)
            if not grouped_eval['regression'].empty and 'MRE (pixels)' in grouped_eval['regression'].columns:
                mre_val = np.nanmean(grouped_eval['regression']['MRE (pixels)'].values)
                if not np.isnan(mre_val):
                    # 将MRE归一化到[0,1]范围，假设合理的MRE上限为100像素
                    max_mre = 100.0
                    norm_mre = min(mre_val / max_mre, 1.0)
                    reg_score = 1.0 - norm_mre  # 转换为越高越好
                    scores.append(reg_score)
                    weights.append(1.0)  # 回归任务权重
            
            # 计算加权平均分作为综合评分
            new_score = None
            if scores and weights:
                # 如果所有权重相同，可以使用简单平均
                if all(w == weights[0] for w in weights):
                    new_score = np.mean(scores)
                else:
                    # 否则使用加权平均
                    new_score = np.average(scores, weights=weights)
            else:
                # 如果没有可用的任务评分，使用优先级选择方式作为备选
                new_score = None
                # Segmentation primary
                if not grouped_eval['segmentation'].empty and 'Dice' in grouped_eval['segmentation'].columns:
                    new_score = np.nanmean(grouped_eval['segmentation']['Dice'].values)

                # Classification primary
                if new_score is None and not grouped_eval['classification'].empty and 'AUC' in grouped_eval['classification'].columns:
                    new_score = np.nanmean(grouped_eval['classification']['AUC'].values)

                # Detection primary
                if new_score is None and not grouped_eval['detection'].empty and 'IoU' in grouped_eval['detection'].columns:
                    new_score = np.nanmean(grouped_eval['detection']['IoU'].values)

                # Regression primary (lower better -> negate)
                if new_score is None and not grouped_eval['regression'].empty and 'MRE (pixels)' in grouped_eval['regression'].columns:
                    mre_val = np.nanmean(grouped_eval['regression']['MRE (pixels)'].values)
                    if not np.isnan(mre_val):
                        new_score = -mre_val

            # If we couldn't compute any primary metric, fallback to legacy mixed table mean
            if new_score is None:
                val_df = evaluate(model, val_loader, device)
                if not val_df.empty:
                    numeric_cols = [c for c in val_df.columns if c not in ['Task ID', 'Task Name']]
                    if numeric_cols:
                        new_score = val_df[numeric_cols].mean().mean()

            if new_score is None:
                new_score = -float('inf')

        except Exception as e:
            logger.error('Validation failed, falling back to legacy evaluate:', exc_info=True)
            val_df = evaluate(model, val_loader, device)
            if not val_df.empty:
                numeric_cols = [c for c in val_df.columns if c not in ['Task ID', 'Task Name']]
                if numeric_cols:
                    new_score = val_df[numeric_cols].mean().mean()
            if new_score is None:
                new_score = -float('inf')

        # Save overall best model (平衡四个任务的表现)
        if new_score > best_val:
            best_val = new_score
            torch.save(model.state_dict(), 'best_florence_no_lora.pth')
            print(f'New best overall model saved (综合评分 = {best_val:.6f}) as best_florence_no_lora.pth')
            # 记录各任务的最佳表现
            try:
                if not grouped_eval['segmentation'].empty and 'Dice' in grouped_eval['segmentation'].columns:
                    avg_dice = np.nanmean(grouped_eval['segmentation']['Dice'].values)
                    if avg_dice > best_metrics['segmentation']['Dice']:
                        best_metrics['segmentation']['Dice'] = avg_dice
                        print(f'  当前最佳分割 Dice: {avg_dice:.6f}')
                
                if not grouped_eval['classification'].empty and 'AUC' in grouped_eval['classification'].columns:
                    avg_auc = np.nanmean(grouped_eval['classification']['AUC'].values)
                    if avg_auc > best_metrics['classification']['AUC']:
                        best_metrics['classification']['AUC'] = avg_auc
                        print(f'  当前最佳分类 AUC: {avg_auc:.6f}')
                
                if not grouped_eval['detection'].empty and 'IoU' in grouped_eval['detection'].columns:
                    avg_iou = np.nanmean(grouped_eval['detection']['IoU'].values)
                    if avg_iou > best_metrics['detection']['IoU']:
                        best_metrics['detection']['IoU'] = avg_iou
                        print(f'  当前最佳检测 IoU: {avg_iou:.6f}')
                
                if not grouped_eval['regression'].empty and 'MRE (pixels)' in grouped_eval['regression'].columns:
                    avg_mre = np.nanmean(grouped_eval['regression']['MRE (pixels)'].values)
                    if avg_mre < best_metrics['regression']['MRE']:
                        best_metrics['regression']['MRE'] = avg_mre
                        print(f'  当前最佳回归 MRE: {avg_mre:.6f}')
            except Exception as e:
                logger.error('记录任务最佳表现失败:', exc_info=True)


if __name__ == '__main__':
    main()