import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import pandas as pd
import cv2
from tqdm import tqdm
from collections import defaultdict
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, matthews_corrcoef
from sklearn.preprocessing import label_binarize

from model_factory import TASK_CONFIGURATIONS


# =========================================================
# Reproducibility
# =========================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# =========================================================
# Multi-task collate fn (保持接口不变)
# =========================================================

def multi_task_collate_fn(batch):
    """
    Custom collate function for multi-task learning.
    - images are stacked
    - labels & task_ids stay as lists
    """
    images = torch.stack([item['image'] for item in batch], dim=0)
    labels = [item['label'] for item in batch]
    task_ids = [item['task_id'] for item in batch]

    return {
        'image': images,
        'label': labels,
        'task_id': task_ids
    }


# =========================================================
# Gaussian heatmap utilities
# =========================================================

def generate_gaussian_heatmap(bboxes, heatmap_size, sigma=5.0):
    """
    Generate Gaussian heatmap from bounding boxes.
    
    Args:
        bboxes: Tensor of shape (B, 4) with normalized coordinates (x1, y1, x2, y2)
        heatmap_size: Tuple (H, W) of the heatmap size
        sigma: Standard deviation of Gaussian kernel
        
    Returns:
        Tensor of shape (B, 1, H, W) with Gaussian heatmaps
    """
    B, _ = bboxes.shape
    H, W = heatmap_size
    heatmaps = torch.zeros((B, 1, H, W), device=bboxes.device)
    
    for i in range(B):
        bbox = bboxes[i]
        if bbox[0] < 0:
            continue  # Skip invalid boxes
            
        # Calculate center in normalized coordinates
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        
        # Convert to heatmap coordinates
        hx = int(cx * W)
        hy = int(cy * H)
        
        # Ensure coordinates are within heatmap bounds
        hx = max(0, min(hx, W - 1))
        hy = max(0, min(hy, H - 1))
        
        # Generate Gaussian distribution
        y, x = torch.meshgrid(
            torch.arange(H, device=bboxes.device),
            torch.arange(W, device=bboxes.device),
            indexing='ij'
        )
        
        dist_sq = (x - hx) ** 2 + (y - hy) ** 2
        gaussian = torch.exp(-dist_sq / (2 * sigma ** 2))
        
        # Normalize to ensure the peak is 1
        gaussian = gaussian / gaussian.max()
        
        heatmaps[i, 0] = gaussian
    
    return heatmaps


def generate_gaussian_heatmap_with_dynamic_radius(bboxes, heatmap_size, stride=4, 
                                                 min_radius=1, sigma_factor=0.3):
    """
    Generate Gaussian heatmap with dynamic radius based on box size.
    
    Args:
        bboxes: Tensor of shape (B, 4) with normalized coordinates (x1, y1, x2, y2)
        heatmap_size: Tuple (H, W) of the heatmap size
        stride: Downsampling stride of the feature map
        min_radius: Minimum radius for small boxes
        sigma_factor: Factor to calculate sigma from radius
        
    Returns:
        Tensor of shape (B, 1, H, W) with Gaussian heatmaps
    """
    B, _ = bboxes.shape
    H, W = heatmap_size
    heatmaps = torch.zeros((B, 1, H, W), device=bboxes.device)
    
    for i in range(B):
        bbox = bboxes[i]
        if bbox[0] < 0:
            continue  # Skip invalid boxes
            
        # Calculate center and box size in normalized coordinates
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        box_w = bbox[2] - bbox[0]
        box_h = bbox[3] - bbox[1]
        
        # Convert to heatmap coordinates
        hx = int(cx * W)
        hy = int(cy * H)
        
        # Ensure coordinates are within heatmap bounds
        hx = max(0, min(hx, W - 1))
        hy = max(0, min(hy, H - 1))
        
        # Calculate dynamic radius based on box size and stride
        box_size = max(box_w * W, box_h * H)
        radius = max(min_radius, box_size / stride / 2)
        sigma = radius * sigma_factor
        
        # Generate Gaussian distribution
        y, x = torch.meshgrid(
            torch.arange(H, device=bboxes.device),
            torch.arange(W, device=bboxes.device),
            indexing='ij'
        )
        
        dist_sq = ((x - hx) ** 2 + (y - hy) ** 2) / (2 * sigma ** 2)
        gaussian = torch.exp(-dist_sq)
        
        # Normalize to ensure the peak is 1
        gaussian = gaussian / gaussian.max() if gaussian.max() > 0 else gaussian
        
        heatmaps[i, 0] = gaussian
    
    return heatmaps


class FocalLoss(nn.Module):
    """Focal Loss for heatmap regression."""
    def __init__(self, alpha=2.0, beta=4.0, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, pred_heatmap, gt_heatmap):
        """Compute focal loss between predicted and ground truth heatmaps.
        
        Args:
            pred_heatmap: Predicted heatmap (B, 1, H, W)
            gt_heatmap: Ground truth heatmap (B, 1, H, W)
        """
        positive_mask = gt_heatmap > 0
        negative_mask = gt_heatmap == 0
        
        # Positive loss
        pos_loss = -self.alpha * gt_heatmap * torch.pow(1 - pred_heatmap, self.gamma) * torch.log(pred_heatmap + 1e-6)
        
        # Negative loss
        neg_loss = -self.beta * (1 - gt_heatmap) * torch.pow(pred_heatmap, self.gamma) * torch.log(1 - pred_heatmap + 1e-6)
        
        # Combine losses
        loss = pos_loss * positive_mask.float() + neg_loss * negative_mask.float()
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


# =========================================================
# Detection loss (未改接口)
# =========================================================

def dice_loss(pred, target, smooth=1e-6):
    """
    Dice Loss for binary segmentation/heatmap tasks.
    """
    pred = pred.contiguous()
    target = target.contiguous()
    
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    
    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))
    
    return loss.mean()

class DetectionLoss(nn.Module):
    def __init__(self, heatmap_weight=1.0, offset_weight=1.0, size_weight=1.0):
        super().__init__()
        self.heatmap_loss = FocalLoss()  # Use Focal Loss for heatmap
        self.offset_loss = nn.SmoothL1Loss(reduction='none')
        self.size_loss = nn.SmoothL1Loss(reduction='none')
        self.heatmap_weight = heatmap_weight
        self.offset_weight = offset_weight
        self.size_weight = size_weight

    def forward(self, predictions, targets):
        """
        Compute detection loss with heatmap, offset, and size components.
        
        Args:
            predictions: Tensor of shape (B, 5, H, W)
                - Channel 0: Heatmap
                - Channel 1-2: Center offset
                - Channel 3-4: Box size
            targets: Dict containing:
                - heatmap: Tensor of shape (B, 1, H, W) - Ground truth heatmap
                - center_indices: Tensor of shape (B, 2) - [y, x] indices of centers
                - offset: Tensor of shape (B, 2) - Ground truth offset
                - size: Tensor of shape (B, 2) - Ground truth size
                - valid: Tensor of shape (B,) - Valid box mask
        """
        batch_size = predictions.shape[0]
        B, _, H, W = predictions.shape
        
        # 1. Heatmap loss
        heatmap_loss = self.heatmap_loss(predictions[:, 0:1, :, :], targets['heatmap'])
        
        # 2. Offset loss and Size loss - only compute at positive locations
        offset_loss = 0.0
        size_loss = 0.0
        
        for i in range(B):
            if not targets['valid'][i]:
                continue  # Skip invalid boxes
            
            y, x = targets['center_indices'][i]
            
            # Ensure coordinates are within bounds
            y = max(0, min(y, H - 1))
            x = max(0, min(x, W - 1))
            
            # Calculate offset loss
            pred_offset = predictions[i, 1:3, y, x]
            gt_offset = targets['offset'][i]
            offset_loss += self.offset_loss(pred_offset, gt_offset).sum()
            
            # Calculate size loss
            pred_size = predictions[i, 3:5, y, x]
            gt_size = targets['size'][i]
            size_loss += self.size_loss(pred_size, gt_size).sum()
        
        # Normalize by batch size
        offset_loss = offset_loss / batch_size
        size_loss = size_loss / batch_size
        
        # Total loss
        total_loss = (
            self.heatmap_weight * heatmap_loss +
            self.offset_weight * offset_loss +
            self.size_weight * size_loss
        )
        
        return total_loss


# =========================================================
# Classification metrics
# =========================================================

def calculate_accuracy(y_true, y_pred_logits):
    y_pred = torch.argmax(y_pred_logits, dim=1).cpu().numpy()
    y_true = y_true.cpu().numpy()
    return accuracy_score(y_true, y_pred)


def calculate_f1_score(y_true, y_pred_logits):
    y_pred = torch.argmax(y_pred_logits, dim=1).cpu().numpy()
    y_true = y_true.cpu().numpy()
    return f1_score(y_true, y_pred, average='macro', zero_division=0)


def calculate_auc(y_true, y_pred_logits):
    """
    Skip batches where AUC is undefined (single-class batch)
    """
    try:
        y_true_np = y_true.cpu().numpy()
        if len(np.unique(y_true_np)) < 2:
            return float('nan')

        probs = torch.softmax(y_pred_logits, dim=1).cpu().numpy()
        num_classes = probs.shape[1]

        if num_classes == 2:
            return float(roc_auc_score(y_true_np, probs[:, 1]))
        else:
            y_true_bin = label_binarize(y_true_np, classes=list(range(num_classes)))
            return float(roc_auc_score(y_true_bin, probs, multi_class='ovr'))
    except Exception:
        return float('nan')


def calculate_mcc(y_true, y_pred_logits):
    try:
        y_pred = torch.argmax(y_pred_logits, dim=1).cpu().numpy()
        y_true_np = y_true.cpu().numpy()
        return float(matthews_corrcoef(y_true_np, y_pred))
    except Exception:
        return float('nan')


# =========================================================
# Segmentation metrics（核心修正点）
# =========================================================

def calculate_dice_coefficient(y_true, y_pred_logits):
    """
    Foreground Dice:
    - merge all foreground classes
    - avoid per-class harsh penalty in early multi-task training
    """
    with torch.no_grad():
        probs = torch.softmax(y_pred_logits, dim=1)
        fg_prob = probs[:, 1:].sum(dim=1)
        pred_fg = (fg_prob > 0.5).float()
        gt_fg = (y_true > 0).float()

        intersection = (pred_fg * gt_fg).sum()
        union = pred_fg.sum() + gt_fg.sum()

        if union == 0:
            return 1.0

        return (2. * intersection / union).item()


def calculate_hausdorff(y_true, y_pred_logits):
    y_pred = torch.argmax(y_pred_logits, dim=1).cpu().numpy()
    y_true = y_true.cpu().numpy()

    batch_hd = []
    for gt, pred in zip(y_true, y_pred):
        gt_bin = (gt > 0).astype('uint8')
        pred_bin = (pred > 0).astype('uint8')

        if gt_bin.sum() == 0 or pred_bin.sum() == 0:
            continue   # ← 关键：直接跳过，不 append nan

        try:
            dt_gt = cv2.distanceTransform(
                (gt_bin == 0).astype('uint8'),
                cv2.DIST_L2, 5
            )
            dt_pred = cv2.distanceTransform(
                (pred_bin == 0).astype('uint8'),
                cv2.DIST_L2, 5
            )

            d1 = dt_gt[pred_bin.astype(bool)].max()
            d2 = dt_pred[gt_bin.astype(bool)].max()
            batch_hd.append(float(max(d1, d2)))
        except Exception:
            continue

    if len(batch_hd) == 0:
        return float('nan')   # ← 显式返回 NaN

    return float(np.mean(batch_hd))



# =========================================================
# Regression metrics
# =========================================================

def calculate_mae(y_true, y_pred, image_size=(256, 256)):
    """
    Calculate Mean Absolute Error for regression tasks.
    
    Args:
        y_true: Ground truth coordinates (normalized)
        y_pred: Predicted coordinates (normalized)
        image_size: Original image size (H, W)
        
    Returns:
        Mean Absolute Error in pixels
    """
    h, w = image_size
    y_true = y_true.cpu().numpy().copy()
    y_pred = y_pred.cpu().numpy().copy()

    # Convert normalized coordinates to pixel coordinates
    y_true[:, 0::2] *= w
    y_true[:, 1::2] *= h
    y_pred[:, 0::2] *= w
    y_pred[:, 1::2] *= h

    return np.mean(np.abs(y_true - y_pred))


def calculate_mre(y_true, y_pred, image_size=(256, 256)):
    """
    Calculate Mean Radial Error for regression tasks.
    
    Args:
        y_true: Ground truth coordinates (normalized)
        y_pred: Predicted coordinates (normalized)
        image_size: Original image size (H, W)
        
    Returns:
        Mean Radial Error in pixels
    """
    h, w = image_size
    y_true = y_true.cpu().numpy().copy()
    y_pred = y_pred.cpu().numpy().copy()

    # Convert normalized coordinates to pixel coordinates
    y_true[:, 0::2] *= w
    y_true[:, 1::2] *= h
    y_pred[:, 0::2] *= w
    y_pred[:, 1::2] *= h

    # Calculate Euclidean distance for each point pair
    diffs = y_true - y_pred
    dists = np.sqrt(diffs[:, 0::2]**2 + diffs[:, 1::2]**2)
    return np.mean(dists)


# =========================================================
# Detection metric
# =========================================================

def calculate_iou(y_true, y_pred):
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()

    ious = []
    for i in range(y_true.shape[0]):
        gt = y_true[i]
        pr = y_pred[i]

        xA = max(gt[0], pr[0])
        yA = max(gt[1], pr[1])
        xB = min(gt[2], pr[2])
        yB = min(gt[3], pr[3])

        inter = max(0, xB - xA) * max(0, yB - yA)
        area_gt = (gt[2] - gt[0]) * (gt[3] - gt[1])
        area_pr = (pr[2] - pr[0]) * (pr[3] - pr[1])
        union = area_gt + area_pr - inter

        ious.append(inter / (union + 1e-6))

    return float(np.mean(ious))


def heatmap_to_bbox(heatmap, box_size=0.1):
    """
    Convert heatmap to bounding box coordinates with proper coordinate transformation.
    
    Args:
        heatmap: Tensor of shape (B, 5, H, W) or (B, 1, H, W)
            - For 5-channel input: [heatmap, dx, dy, w, h]
            - For 1-channel input: [heatmap]
        box_size: Relative size of the bounding box (0-1) (only used for 1-channel input)
        
    Returns:
        Tensor of shape (B, 4) with normalized coordinates (x1, y1, x2, y2)
    """
    # Handle different input dimensions
    if heatmap.dim() == 3:
        heatmap = heatmap.unsqueeze(0)
    
    B, C, H, W = heatmap.shape
    bboxes = torch.zeros((B, 4), device=heatmap.device)
    
    for i in range(B):
        if C == 5:
            # New format: (heatmap, dx, dy, w, h)
            # Get heatmap channel
            heatmap_channel = heatmap[i, 0:1, :, :]
            
            # Find coordinates of maximum response in heatmap
            max_val, max_idx = heatmap_channel.reshape(-1).max(dim=0)
            h_idx = max_idx // W
            w_idx = max_idx % W
            
            # Get offset and size predictions at the max location
            dx = heatmap[i, 1, h_idx, w_idx]
            dy = heatmap[i, 2, h_idx, w_idx]
            predicted_w = heatmap[i, 3, h_idx, w_idx]
            predicted_h = heatmap[i, 4, h_idx, w_idx]
            
            # ==================== FIXED: 与训练时一致的坐标转换 ====================
            # 训练时：gt_center_h = gt_center_y * H
            # 推理时：orig_center_y = feat_center_y / H
            # 1. 计算特征图上的中心点（带偏移）
            feat_center_x = w_idx.float() + dx
            feat_center_y = h_idx.float() + dy
            
            # 2. 转换为归一化坐标（直接除以特征图尺寸，与训练转换方向相反）
            orig_center_x = feat_center_x / W  # 转换回归一化x坐标
            orig_center_y = feat_center_y / H  # 转换回归一化y坐标
            
            # 3. 处理预测的size - 确保size为正且合理
            predicted_w = torch.abs(predicted_w)
            predicted_h = torch.abs(predicted_h)
            
            # 确保size在合理范围内（0.01到0.5，避免过大或过小的框）
            predicted_w = torch.clamp(predicted_w, 0.01, 0.5)
            predicted_h = torch.clamp(predicted_h, 0.01, 0.5)
            
            half_w = predicted_w / 2.0
            half_h = predicted_h / 2.0
        else:
            # Old format: only heatmap channel
            # Find coordinates of maximum response
            max_val, max_idx = heatmap[i].reshape(-1).max(dim=0)
            h_idx = max_idx // W
            w_idx = max_idx % W
            
            # Convert heatmap coordinates to normalized coordinates
            orig_center_x = w_idx.float() / W
            orig_center_y = h_idx.float() / H
            
            # Use fixed box size
            half_w = box_size / 2
            half_h = box_size / 2
        
        # Calculate bbox coordinates
        x1 = torch.clamp(orig_center_x - half_w, 0.0, 1.0)
        y1 = torch.clamp(orig_center_y - half_h, 0.0, 1.0)
        x2 = torch.clamp(orig_center_x + half_w, 0.0, 1.0)
        y2 = torch.clamp(orig_center_y + half_h, 0.0, 1.0)
        
        bboxes[i] = torch.tensor([x1, y1, x2, y2], device=heatmap.device)
    
    return bboxes


# =========================================================
# Evaluation loop（接口名 evaluate 保持不变）
# =========================================================

def evaluate(model, val_loader, device, grouped: bool = False):
    model.eval()
    task_metrics = defaultdict(lambda: defaultdict(list))
    task_id_to_name = {cfg['task_id']: cfg['task_name'] for cfg in TASK_CONFIGURATIONS}

    with torch.no_grad():
        for batch in tqdm(val_loader, desc='[Validation]', disable=False, leave=True):
            images = batch['image'].to(device)
            labels = batch['label']
            task_ids = batch['task_id']

            for task_id in set(task_ids):
                idx = [i for i, t in enumerate(task_ids) if t == task_id]
                task_images = images[idx]
                task_labels = torch.stack([labels[i] for i in idx], dim=0)

                outputs = model(task_images, task_id=task_id)
                task_name = task_id_to_name[task_id]

                if task_name == 'classification':
                    task_metrics[task_id]['Accuracy'].append(
                        calculate_accuracy(task_labels, outputs))
                    task_metrics[task_id]['F1-Score'].append(
                        calculate_f1_score(task_labels, outputs))
                    task_metrics[task_id]['AUC'].append(
                        calculate_auc(task_labels, outputs))
                    task_metrics[task_id]['MCC'].append(
                        calculate_mcc(task_labels, outputs))

                elif task_name == 'segmentation':
                    task_metrics[task_id]['Dice'].append(
                        calculate_dice_coefficient(task_labels.to(device), outputs))
                    task_metrics[task_id]['Hausdorff (px)'].append(
                        calculate_hausdorff(task_labels.to(device), outputs))

                elif task_name == 'Regression':
                    task_metrics[task_id]['MAE (pixels)'].append(
                        calculate_mae(task_labels, outputs))
                    task_metrics[task_id]['MRE (pixels)'].append(
                        calculate_mre(task_labels, outputs))

                elif task_name == 'detection':
                    # Heatmap post-processing: convert heatmap to bbox
                    final_boxes = heatmap_to_bbox(outputs, box_size=0.1)
                    
                    task_metrics[task_id]['IoU'].append(
                        calculate_iou(task_labels, final_boxes))

    grouped_results = {
        'segmentation': [],
        'classification': [],
        'detection': [],
        'regression': []
    }

    for task_id, metrics in task_metrics.items():
        task_name = task_id_to_name[task_id]
        row = {'Task ID': task_id, 'Task Name': task_name}

        for k, v in metrics.items():
            row[k] = float(np.nanmean(v))

        grouped_results[task_name.lower()].append(row)

    for k in grouped_results:
        grouped_results[k] = pd.DataFrame(grouped_results[k])

    if grouped:
        return grouped_results

    return pd.concat(grouped_results.values(), ignore_index=True, sort=False)