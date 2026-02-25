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
        logging.FileHandler('train_florence_freeze_all.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Setup epoch results logger (only records epoch results)
epoch_logger = logging.getLogger('epoch_logger')
epoch_logger.setLevel(logging.INFO)
epoch_logger.addHandler(logging.FileHandler('output_train_florence_freeze_all.log'))
epoch_logger.propagate = False

from dataset import MultiTaskDataset, MultiTaskUniformSampler
from model_factory import MultiTaskModelFactory, TASK_CONFIGURATIONS
from utils import multi_task_collate_fn, evaluate, set_seed, DetectionLoss, generate_gaussian_heatmap_with_dynamic_radius

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

# Override model_factory to use freeze all adapter
# Replace the import in model_factory
import model_factory
model_factory.FlorenceEncoderAdapter = None

# Import the freeze all adapter
from ablation_study.florence_adapter_freeze_all import FlorenceEncoderAdapter
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
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--unfreeze', action='store_true', help='Unfreeze Florence encoder for fine-tuning')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--train_only', type=str, default=None, help='Train only specified task type (e.g., detection)')
    args = parser.parse_args()

    logger.info(f"Starting training with parameters: {args}")
    logger.info("=== ABLATION: Freeze All (only task heads trainable) ===")
    
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
    # Pass freeze_stages=4 for Freeze All configuration (only task heads trainable)
    logger.info("Initializing model with freeze_stages: 4 (Freeze All)")
    model = MultiTaskModelFactory(
        encoder_name='florence',
        encoder_weights=None,
        task_configs=TASK_CONFIGURATIONS,
        freeze_stages=4  # Freeze all encoder parameters
    )

    model = model.to(device)
    logger.info("Model initialized")

    # 优化器和学习率调度器
    # Freeze All 配置只优化任务 heads 的参数
    # 找出所有可训练的参数
    trainable_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params.append(param)
            # logger.info(f"Trainable parameter: {name}")

    # 使用较大的学习率，因为只训练少量参数
    optimizer = optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)

    # 余弦退火学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    logger.info("Optimizer and scheduler initialized")

    # 训练循环
    best_metrics = {}
    best_epoch = 0
    patience = 10
    patience_counter = 0
    best_train_loss = float('inf')

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        task_losses = defaultdict(float)
        task_counts = defaultdict(int)
        epoch_start = time.time()

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs}')
        for batch in pbar:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            images = batch['image']
            labels = torch.stack(batch['label']).to(device)
            task_ids = batch['task_id']
            
            # Get current task from batch
            current_task = task_ids[0]
            
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(images, task_id=current_task)
            
            # 计算损失
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
                loss = DetectionLoss()(outputs, detection_targets)

            elif task_name == 'segmentation':
                # Handle segmentation labels correctly
                if labels.dim() == 1:
                    labels = labels.unsqueeze(1)  # Add channel dimension
                loss = nn.CrossEntropyLoss()(outputs, labels.long())

            elif task_name == 'classification':
                # Ensure labels are in the correct format
                if labels.dim() == 2 and labels.size(1) == 1:
                    labels = labels.squeeze(1)
                loss = nn.CrossEntropyLoss()(outputs, labels.long())

            elif task_name == 'Regression':
                loss = nn.MSELoss()(outputs, labels)

            else:
                loss = nn.MSELoss()(outputs, labels)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            
            optimizer.step()
            
            # 累计损失
            epoch_loss += loss.item()
            task_losses[current_task] += loss.item()
            task_counts[current_task] += 1
            
            # 更新进度条
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # 学习率调度
        scheduler.step()
        
        # 计算平均损失
        avg_epoch_loss = epoch_loss / len(train_loader)
        avg_task_losses = {task: task_losses[task] / task_counts[task] if task_counts[task] > 0 else 0 for task in task_losses}
        
        epoch_time = time.time() - epoch_start
        
        # 记录训练信息
        logger.info(f"Epoch {epoch+1}/{args.epochs}, Time: {epoch_time:.2f}s")
        logger.info(f"Train Loss: {avg_epoch_loss:.4f}")
        for task, loss_val in avg_task_losses.items():
            logger.info(f"{task} Loss: {loss_val:.4f}")
        logger.info(f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        logger.info('-' * 50)
        
        # Log epoch results to separate file
        epoch_logger.info(f"Epoch {epoch+1}/{args.epochs}, Time: {epoch_time:.2f}s")
        epoch_logger.info(f"Train Loss: {avg_epoch_loss:.4f}")
        for task, loss_val in avg_task_losses.items():
            epoch_logger.info(f"{task} Loss: {loss_val:.4f}")
        epoch_logger.info(f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        epoch_logger.info('-' * 50)
        
        # 模型保存
        if avg_epoch_loss < best_train_loss:
            best_train_loss = avg_epoch_loss
            patience_counter = 0
            # 保存最佳模型
            torch.save(model.state_dict(), 'best_model_freeze_all.pth')
            logger.info('Saved best model!')
            epoch_logger.info('Saved best model!')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f'Early stopping at epoch {epoch+1}')
                break

    logger.info('Training completed!')
    logger.info(f'Best training loss: {best_train_loss:.4f}')

if __name__ == '__main__':
    main()