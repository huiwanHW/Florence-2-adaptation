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
    
    # 添加index字段
    indices = [item.get('index', i) for i, item in enumerate(batch)]
    
    # 添加original_size字段
    original_sizes = [item.get('original_size', (256, 256)) for item in batch]

    return {
        'image': images,
        'label': labels,
        'task_id': task_ids,
        'index': indices,
        'original_size': original_sizes
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
# Post-processing utilities
# =========================================================

def process_segmentation_output(pred, original_size, task_id=None):
    """
    Process segmentation prediction results with post-processing.
    
    Args:
        pred: Model output (C, H, W) or (H, W)
        original_size: Original image size (H, W)
        task_id: Task ID for task-specific processing
        
    Returns:
        Processed segmentation mask
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    
    # For multi-class segmentation (C, H, W), take argmax
    if pred.ndim == 3:
        mask = np.argmax(pred, axis=0).astype(np.uint8)
    else:
        mask = pred.astype(np.uint8)
    
    # Resize back to original size
    h, w = original_size
    interpolation = cv2.INTER_NEAREST
    mask = cv2.resize(mask, (w, h), interpolation=interpolation)
    
    # Check if this is fetal_abdomen_multi task
    is_fetal_abdomen = False
    if task_id and 'fetal_abdomen_multi' in task_id:
        is_fetal_abdomen = True
    
    # Special processing for fetal_abdomen_multi
    if is_fetal_abdomen:
        # Step 1: Keep only the largest connected component
        # Convert to binary mask first
        binary_mask = (mask > 0).astype(np.uint8)
        
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
        
        if num_labels > 1:
            # Find the largest component (excluding background which is label 0)
            largest_label = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
            mask = (labels == largest_label).astype(np.uint8) * mask.max()
        
        # Step 2: Morphological closing to fill holes and connect gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Step 3: Ellipse fitting (if possible)
        binary_mask = (mask > 0).astype(np.uint8)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find the longest contour
            contour = max(contours, key=cv2.contourArea)
            
            # Check if contour has enough points to fit ellipse (at least 5 points)
            if len(contour) >= 5:
                try:
                    # Fit ellipse
                    ellipse = cv2.fitEllipse(contour)
                    
                    # Create empty mask for ellipse
                    ellipse_mask = np.zeros_like(mask)
                    
                    # Draw ellipse on mask
                    cv2.ellipse(ellipse_mask, ellipse, color=mask.max(), thickness=-1)  # -1 fills the ellipse
                    
                    # Use ellipse mask as final mask
                    mask = ellipse_mask
                except:
                    # If ellipse fitting fails, use the previous mask
                    pass
    
    return mask

def process_classification_output(pred):
    """
    Process classification prediction results.
    
    Args:
        pred: Model output logits
        
    Returns:
        Dict with predicted class and probability distribution
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    
    # Get predicted class
    pred_class = int(np.argmax(pred))
    
    # Calculate probability distribution (using softmax)
    # Stable softmax to avoid numerical overflow
    pred_exp = np.exp(pred - np.max(pred))
    pred_probs = pred_exp / np.sum(pred_exp)
    
    return {
        'predicted_class': pred_class,
        'predicted_probs': pred_probs.tolist()
    }

def process_regression_output(pred, original_size):
    """
    Process regression task prediction results with engineering stop-loss.
    
    Args:
        pred: Model output (normalized coordinates)
        original_size: Original image size (H, W)
        
    Returns:
        Dict with normalized and pixel coordinates
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    
    # Get image dimensions
    img_h, img_w = original_size
    
    # Normalized coordinates from regression output
    coords = pred.flatten().tolist()
    
    normalized_coords = []
    pixel_coords = []
    
    # Process each point with engineering stop-loss
    for i in range(0, len(coords), 2):
        x_norm, y_norm = coords[i], coords[i+1]
        
        # ====== Engineering Stop-Loss Techniques ======
        
        # 1. Basic clamp: ensure coordinates are within [0, 1] range
        x_norm = np.clip(x_norm, 0.0, 1.0)
        y_norm = np.clip(y_norm, 0.0, 1.0)
        
        # 2. Reasonable range constraint based on medical prior
        # For femur and other regression tasks, limit to central region
        # Assume the target is within central 80% of the image
        center_x, center_y = 0.5, 0.5
        max_distance = 0.4  # 80% of image (40% from center)
        
        # Calculate distance from center
        dx = x_norm - center_x
        dy = y_norm - center_y
        distance = np.sqrt(dx**2 + dy**2)
        
        # If distance exceeds max, clamp to the edge of the reasonable region
        if distance > max_distance:
            # Normalize direction vector
            dx_norm = dx / distance
            dy_norm = dy / distance
            
            # Clamp to max_distance from center
            x_norm = center_x + dx_norm * max_distance
            y_norm = center_y + dy_norm * max_distance
        
        # 3. Additional constraint: ensure coordinates are not too close to edges
        edge_margin = 0.05  # 5% margin from edges
        x_norm = np.clip(x_norm, edge_margin, 1.0 - edge_margin)
        y_norm = np.clip(y_norm, edge_margin, 1.0 - edge_margin)
        
        # ====== End of Stop-Loss Techniques ======
        
        # Convert to pixel coordinates
        x_pixel = x_norm * img_w
        y_pixel = y_norm * img_h
        
        normalized_coords.extend([x_norm, y_norm])
        pixel_coords.extend([x_pixel, y_pixel])
    
    return {
        'predicted_points_normalized': normalized_coords,
        'predicted_points_pixels': pixel_coords
    }

# =========================================================
# Classification metrics
# =========================================================

def calculate_accuracy(y_true, y_pred_logits):
    """
    Calculate accuracy with error handling
    """
    try:
        # Check for valid logits shape
        if y_pred_logits.dim() != 2:
            print(f"Accuracy skipped: Invalid logits dimension: {y_pred_logits.dim()}")
            return float('nan')
        
        y_pred = torch.argmax(y_pred_logits, dim=1).cpu().numpy()
        y_true = y_true.cpu().numpy()
        return accuracy_score(y_true, y_pred)
    except Exception as e:
        print(f"Accuracy calculation error: {str(e)}")
        return float('nan')


def calculate_f1_score(y_true, y_pred_logits):
    """
    Calculate F1 score with error handling
    """
    try:
        # Check for valid logits shape
        if y_pred_logits.dim() != 2:
            print(f"F1 score skipped: Invalid logits dimension: {y_pred_logits.dim()}")
            return float('nan')
        
        y_pred = torch.argmax(y_pred_logits, dim=1).cpu().numpy()
        y_true = y_true.cpu().numpy()
        return f1_score(y_true, y_pred, average='macro', zero_division=0)
    except Exception as e:
        print(f"F1 score calculation error: {str(e)}")
        return float('nan')


def calculate_auc(y_true, y_pred_logits):
    """
    Calculate AUC with detailed error handling
    """
    try:
        y_true_np = y_true.cpu().numpy()
        
        # Check for single-class batch
        unique_classes = np.unique(y_true_np)
        if len(unique_classes) < 2:
            print(f"AUC skipped: Only {len(unique_classes)} class(es) in batch: {unique_classes}")
            return float('nan')

        # Check for valid logits shape
        if y_pred_logits.dim() != 2:
            print(f"AUC skipped: Invalid logits dimension: {y_pred_logits.dim()}")
            return float('nan')

        probs = torch.softmax(y_pred_logits, dim=1).cpu().numpy()
        num_classes = probs.shape[1]
        
        # Check for class mismatch
        if num_classes < 2:
            print(f"AUC skipped: Insufficient number of classes: {num_classes}")
            return float('nan')

        # Calculate AUC based on number of classes
        if num_classes == 2:
            auc_score = roc_auc_score(y_true_np, probs[:, 1])
        else:
            y_true_bin = label_binarize(y_true_np, classes=list(range(num_classes)))
            auc_score = roc_auc_score(y_true_bin, probs, multi_class='ovr')
        
        return float(auc_score)
        
    except Exception as e:
        print(f"AUC calculation error: {str(e)}")
        return float('nan')


def calculate_mcc(y_true, y_pred_logits):
    """
    Calculate Matthews Correlation Coefficient with error handling
    """
    try:
        # Check for valid logits shape
        if y_pred_logits.dim() != 2:
            print(f"MCC skipped: Invalid logits dimension: {y_pred_logits.dim()}")
            return float('nan')
        
        y_pred = torch.argmax(y_pred_logits, dim=1).cpu().numpy()
        y_true_np = y_true.cpu().numpy()
        
        # Check for single-class batch
        unique_classes = np.unique(y_true_np)
        if len(unique_classes) < 2:
            print(f"MCC skipped: Only {len(unique_classes)} class(es) in batch: {unique_classes}")
            return float('nan')
        
        mcc_score = matthews_corrcoef(y_true_np, y_pred)
        return float(mcc_score)
    except Exception as e:
        print(f"MCC calculation error: {str(e)}")
        return float('nan')


# =========================================================
# Segmentation metrics（核心修正点）
# =========================================================

def calculate_dice_coefficient(y_true, y_pred):
    """
    Foreground Dice:
    - merge all foreground classes
    - avoid per-class harsh penalty in early multi-task training
    - Support both logits and post-processed masks
    """
    with torch.no_grad():
        # Check if input is logits (C, H, W) or post-processed mask (H, W)
        if y_pred.dim() == 4 and y_pred.size(1) > 1:
            # Input is logits, apply softmax
            probs = torch.softmax(y_pred, dim=1)
            fg_prob = probs[:, 1:].sum(dim=1)
            pred_fg = (fg_prob > 0.5).float()
        else:
            # Input is post-processed mask, directly use it
            # Ensure mask is float and binary
            pred_fg = (y_pred > 0).float()
        
        gt_fg = (y_true > 0).float()

        intersection = (pred_fg * gt_fg).sum()
        union = pred_fg.sum() + gt_fg.sum()

        if union == 0:
            return 1.0

        return (2. * intersection / union).item()


def calculate_hausdorff(y_true, y_pred):
    # Check if input is logits (C, H, W) or post-processed mask (H, W)
    if y_pred.dim() == 4 and y_pred.size(1) > 1:
        # Input is logits, apply argmax
        y_pred = torch.argmax(y_pred, dim=1).cpu().numpy()
    else:
        # Input is post-processed mask, directly use it
        y_pred = y_pred.cpu().numpy()
    y_true = y_true.cpu().numpy()

    batch_hd = []
    skipped_empty = 0
    skipped_error = 0
    error_messages = []

    for gt, pred in zip(y_true, y_pred):
        gt_bin = (gt > 0).astype('uint8')
        pred_bin = (pred > 0).astype('uint8')

        if gt_bin.sum() == 0 or pred_bin.sum() == 0:
            skipped_empty += 1
            continue   # ← 关键：直接跳过，不 append nan

        try:
            # Check if input arrays are valid
            if gt_bin.ndim != 2 or pred_bin.ndim != 2:
                skipped_error += 1
                error_messages.append(f"Invalid dimension: gt={gt_bin.ndim}, pred={pred_bin.ndim}")
                continue

            # Check if arrays are not empty
            if gt_bin.size == 0 or pred_bin.size == 0:
                skipped_error += 1
                error_messages.append("Empty array")
                continue

            # Calculate distance transforms
            dt_gt = cv2.distanceTransform(
                (gt_bin == 0).astype('uint8'),
                cv2.DIST_L2, 5
            )
            dt_pred = cv2.distanceTransform(
                (pred_bin == 0).astype('uint8'),
                cv2.DIST_L2, 5
            )

            # Check if distance transforms are valid
            if dt_gt is None or dt_pred is None:
                skipped_error += 1
                error_messages.append("Distance transform returned None")
                continue

            # Extract non-zero pixels and calculate max distance
            gt_positions = pred_bin.astype(bool)
            pred_positions = gt_bin.astype(bool)

            if gt_positions.any() and pred_positions.any():
                d1 = dt_gt[gt_positions].max()
                d2 = dt_pred[pred_positions].max()
                batch_hd.append(float(max(d1, d2)))
            else:
                skipped_error += 1
                error_messages.append("No valid positions for distance calculation")
                continue

        except Exception as e:
            skipped_error += 1
            error_messages.append(str(e))
            continue

    # Log debugging information
    if len(batch_hd) == 0:
        print(f"Hausdorff calculation: All {len(y_true)} samples skipped")
        print(f"  - Empty masks: {skipped_empty}")
        print(f"  - Calculation errors: {skipped_error}")
        if error_messages:
            print(f"  - Error examples: {error_messages[:3]}")  # Show first 3 errors
        return float('nan')   # ← 显式返回 NaN
    else:
        print(f"Hausdorff calculation: Processed {len(batch_hd)} samples")
        print(f"  - Skipped empty masks: {skipped_empty}")
        print(f"  - Skipped due to errors: {skipped_error}")

    return float(np.mean(batch_hd))



# =========================================================
# Regression metrics
# =========================================================

def calculate_mae(y_true, y_pred, image_size=(256, 256)):
    """
    Calculate Mean Absolute Error for regression tasks with error handling
    
    Args:
        y_true: Ground truth coordinates (normalized)
        y_pred: Predicted coordinates (normalized)
        image_size: Original image size (H, W)
        
    Returns:
        Mean Absolute Error in pixels
    """
    try:
        h, w = image_size
        y_true = y_true.cpu().numpy().copy()
        y_pred = y_pred.cpu().numpy().copy()

        # Check for valid shape
        if y_true.ndim != 2 or y_pred.ndim != 2:
            print(f"MAE skipped: Invalid dimensions - true: {y_true.ndim}, pred: {y_pred.ndim}")
            return float('nan')

        # Check for empty arrays
        if y_true.size == 0 or y_pred.size == 0:
            print("MAE skipped: Empty arrays")
            return float('nan')

        # Convert normalized coordinates to pixel coordinates
        y_true[:, 0::2] *= w
        y_true[:, 1::2] *= h
        y_pred[:, 0::2] *= w
        y_pred[:, 1::2] *= h

        return np.mean(np.abs(y_true - y_pred))
    except Exception as e:
        print(f"MAE calculation error: {str(e)}")
        return float('nan')


def calculate_mre(y_true, y_pred, image_size=(256, 256)):
    """
    Calculate Mean Radial Error for regression tasks with error handling
    
    Args:
        y_true: Ground truth coordinates (normalized)
        y_pred: Predicted coordinates (normalized)
        image_size: Original image size (H, W)
        
    Returns:
        Mean Radial Error in pixels
    """
    try:
        h, w = image_size
        y_true = y_true.cpu().numpy().copy()
        y_pred = y_pred.cpu().numpy().copy()

        # Check for valid shape
        if y_true.ndim != 2 or y_pred.ndim != 2:
            print(f"MRE skipped: Invalid dimensions - true: {y_true.ndim}, pred: {y_pred.ndim}")
            return float('nan')

        # Check for empty arrays
        if y_true.size == 0 or y_pred.size == 0:
            print("MRE skipped: Empty arrays")
            return float('nan')

        # Convert normalized coordinates to pixel coordinates
        y_true[:, 0::2] *= w
        y_true[:, 1::2] *= h
        y_pred[:, 0::2] *= w
        y_pred[:, 1::2] *= h

        # Calculate Euclidean distance for each point pair
        diffs = y_true - y_pred
        dists = np.sqrt(diffs[:, 0::2]**2 + diffs[:, 1::2]**2)
        return np.mean(dists)
    except Exception as e:
        print(f"MRE calculation error: {str(e)}")
        return float('nan')


# =========================================================
# Detection metric
# =========================================================

def calculate_iou(y_true, y_pred):
    """
    Calculate IoU with error handling
    """
    try:
        y_true = y_true.cpu().numpy()
        y_pred = y_pred.cpu().numpy()

        # Check for valid shape
        if y_true.ndim != 2 or y_pred.ndim != 2:
            print(f"IoU skipped: Invalid dimensions - true: {y_true.ndim}, pred: {y_pred.ndim}")
            return float('nan')

        # Check for empty arrays
        if y_true.size == 0 or y_pred.size == 0:
            print("IoU skipped: Empty arrays")
            return float('nan')

        ious = []
        for i in range(y_true.shape[0]):
            gt = y_true[i]
            pr = y_pred[i]

            # Check for valid box coordinates
            if len(gt) < 4 or len(pr) < 4:
                print(f"IoU skipped: Invalid box coordinates at index {i}")
                continue

            # Check for valid box (x2 > x1 and y2 > y1)
            if gt[2] <= gt[0] or gt[3] <= gt[1] or pr[2] <= pr[0] or pr[3] <= pr[1]:
                print(f"IoU skipped: Invalid box at index {i}")
                continue

            xA = max(gt[0], pr[0])
            yA = max(gt[1], pr[1])
            xB = min(gt[2], pr[2])
            yB = min(gt[3], pr[3])

            inter = max(0, xB - xA) * max(0, yB - yA)
            area_gt = (gt[2] - gt[0]) * (gt[3] - gt[1])
            area_pr = (pr[2] - pr[0]) * (pr[3] - pr[1])
            union = area_gt + area_pr - inter

            ious.append(inter / (union + 1e-6))

        if not ious:
            print("IoU skipped: No valid boxes to calculate")
            return float('nan')

        return float(np.mean(ious))
    except Exception as e:
        print(f"IoU calculation error: {str(e)}")
        return float('nan')


def process_detection_output(pred, original_size):
    """
    Process detection task prediction results.
    
    Args:
        pred: Model output (5, H, W) - grid predictions
        original_size: Original image size (H, W)
        
    Returns:
        Dict with bounding box in normalized and pixel coordinates
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    
    # pred shape: (5, H, W) - grid predictions
    # Channel 0: Heatmap (objectness score)
    # Channel 1-2: Center offset (dx, dy)
    # Channel 3-4: Box size (w, h)
    _, h, w = pred.shape
    
    # Find location with highest confidence score in heatmap (Channel 0)
    scores = pred[0, :, :].flatten()
    best_idx = np.argmax(scores)
    best_h = best_idx // w
    best_w = best_idx % w
    
    # Extract offset and size at the best location
    dx = pred[1, best_h, best_w]
    dy = pred[2, best_h, best_w]
    predicted_w = pred[3, best_h, best_w]
    predicted_h = pred[4, best_h, best_w]
    
    # Calculate center coordinates in feature map space (with offset)
    feat_center_x = best_w + dx
    feat_center_y = best_h + dy
    
    # Convert to normalized coordinates (0-1 range)
    center_x_norm = feat_center_x / w
    center_y_norm = feat_center_y / h
    
    # Ensure predicted size is positive and reasonable
    predicted_w = abs(predicted_w)
    predicted_h = abs(predicted_h)
    
    # Clamp size to reasonable range (0.01 to 0.5 of image)
    predicted_w = max(0.01, min(predicted_w, 0.5))
    predicted_h = max(0.01, min(predicted_h, 0.5))
    
    # Calculate bbox coordinates in normalized space
    half_w = predicted_w / 2.0
    half_h = predicted_h / 2.0
    
    x1_norm = max(0.0, min(1.0, center_x_norm - half_w))
    y1_norm = max(0.0, min(1.0, center_y_norm - half_h))
    x2_norm = max(0.0, min(1.0, center_x_norm + half_w))
    y2_norm = max(0.0, min(1.0, center_y_norm + half_h))
    
    bbox_norm_list = [x1_norm, y1_norm, x2_norm, y2_norm]
    
    # Convert to pixel coordinates
    img_h, img_w = original_size
    bbox_pixel = [
        x1_norm * img_w,
        y1_norm * img_h,
        x2_norm * img_w,
        y2_norm * img_h
    ]
    
    return {
        'bbox_normalized': bbox_norm_list,
        'bbox_pixels': bbox_pixel
    }

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

def evaluate(model, val_loader, device, grouped: bool = False, use_post_processing: bool = False):
    model.eval()
    task_metrics = defaultdict(lambda: defaultdict(list))
    task_id_to_name = {cfg['task_id']: cfg['task_name'] for cfg in TASK_CONFIGURATIONS}

    with torch.no_grad():
        for batch in tqdm(val_loader, desc='[Validation]', disable=False, leave=True):
            images = batch['image'].to(device)
            labels = batch['label']
            task_ids = batch['task_id']
            # Get original sizes if available, default to (256, 256)
            original_sizes = batch.get('original_size', [(256, 256)] * len(images))

            for task_id in set(task_ids):
                idx = [i for i, t in enumerate(task_ids) if t == task_id]
                task_images = images[idx]
                task_labels = torch.stack([labels[i] for i in idx], dim=0)
                task_original_sizes = [original_sizes[i] for i in idx]

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
                    # Apply post-processing if enabled
                    if use_post_processing:
                        # Process each sample individually due to possible different original sizes
                        processed_outputs = []
                        for i in range(len(outputs)):
                            # Get original size for this sample
                            original_size = task_original_sizes[i]
                            # Process segmentation output
                            processed_mask = process_segmentation_output(outputs[i], original_size, task_id)
                            # Convert back to tensor and add batch dimension
                            # Use float32 dtype to avoid ByteTensor issues with softmax
                            processed_output = torch.tensor(processed_mask, dtype=torch.float32).unsqueeze(0).to(device)
                            processed_outputs.append(processed_output)
                        # Stack processed outputs
                        processed_outputs = torch.stack(processed_outputs, dim=0)
                        # Calculate metrics on processed outputs
                        task_metrics[task_id]['Dice'].append(
                            calculate_dice_coefficient(task_labels.to(device), processed_outputs))
                        task_metrics[task_id]['Hausdorff (px)'].append(
                            calculate_hausdorff(task_labels.to(device), processed_outputs))
                    else:
                        # Original evaluation without post-processing
                        task_metrics[task_id]['Dice'].append(
                            calculate_dice_coefficient(task_labels.to(device), outputs))
                        task_metrics[task_id]['Hausdorff (px)'].append(
                            calculate_hausdorff(task_labels.to(device), outputs))

                elif task_name == 'Regression':
                    # Use original size for regression metrics if available
                    if use_post_processing and task_original_sizes:
                        # For regression, use the first original size as representative
                        # (assuming all samples in batch have similar sizes)
                        original_size = task_original_sizes[0]
                        task_metrics[task_id]['MAE (pixels)'].append(
                            calculate_mae(task_labels, outputs, image_size=original_size))
                        task_metrics[task_id]['MRE (pixels)'].append(
                            calculate_mre(task_labels, outputs, image_size=original_size))
                    else:
                        # Fallback to default image size
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
            # Calculate mean, handling NaN values
            mean_val = np.nanmean(v)
            # Check if mean is still NaN
            if np.isnan(mean_val):
                print(f"Warning: All values for {task_id} - {k} are NaN")
            row[k] = float(mean_val)

        grouped_results[task_name.lower()].append(row)

    for k in grouped_results:
        grouped_results[k] = pd.DataFrame(grouped_results[k])

    if grouped:
        return grouped_results

    return pd.concat(grouped_results.values(), ignore_index=True, sort=False)
