# model.py - Docker Version - Foundation Model Challenge for Ultrasound Image Analysis (FMC_UIA)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import cv2
import json
import numpy as np
import pandas as pd
import glob
import time
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Optional

# Import local modules
from model_factory import MultiTaskModelFactory


class InferenceDataset(Dataset):
    """Inference dataset class"""
    
    def __init__(self, data_root: str, transforms: Optional[A.Compose] = None):
        super().__init__()
        self.data_root = data_root
        self.transforms = transforms
        self.csv_path = os.path.join(self.data_root, 'csv_files')
        
        if not os.path.isdir(self.csv_path):
            raise FileNotFoundError(f"CSV path not found: {self.csv_path}")
            
        all_csv_files = glob.glob(os.path.join(self.csv_path, '*.csv'))
        if not all_csv_files:
            raise FileNotFoundError(f"No CSV files found in {self.csv_path}")
            
        df_list = [pd.read_csv(csv_file) for csv_file in all_csv_files]
        self.dataframe = pd.concat(df_list, ignore_index=True).reset_index(drop=True)
        print(f"Data loading complete. Total samples: {len(self.dataframe)}")

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, idx: int) -> dict:
        record = self.dataframe.iloc[idx]
        task_id = record['task_id']
        task_name = record['task_name']
        
        # Load image
        image_rel_path = record['image_path']
        # 直接使用image_rel_path，因为它已经包含了train/前缀
        image_abs_path = os.path.normpath(image_rel_path)
        image = cv2.imread(image_abs_path)
        
        if image is None:
            print(f"Warning: Unable to load image {image_abs_path}")
            # Return next sample
            return self.__getitem__((idx + 1) % len(self))
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_height, original_width = image.shape[:2]
        
        # Get mask_path (if segmentation task)
        mask_path = None
        if task_name == 'segmentation' and 'mask_path' in record and pd.notna(record['mask_path']):
            mask_path = record['mask_path']
        
        # Apply transforms
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented['image']
        
        # Return data including metadata
        return {
            'image': image,
            'task_id': task_id,
            'task_name': task_name,
            'image_path': image_rel_path,
            'mask_path': mask_path,
            'original_size': (original_height, original_width),
            'index': idx
        }


def inference_collate_fn(batch):
    """Inference collate function that preserves metadata"""
    images = torch.stack([item['image'] for item in batch], 0)
    task_ids = [item['task_id'] for item in batch]
    task_names = [item['task_name'] for item in batch]
    image_paths = [item['image_path'] for item in batch]
    mask_paths = [item['mask_path'] for item in batch]
    original_sizes = [item['original_size'] for item in batch]
    indices = [item['index'] for item in batch]
    
    return {
        'image': images,
        'task_id': task_ids,
        'task_name': task_names,
        'image_path': image_paths,
        'mask_path': mask_paths,
        'original_size': original_sizes,
        'index': indices
    }


class Model:
    """
    Foundation Model for Ultrasound Image Analysis
    Supports four task types: segmentation, classification, Regression, detection
    """
    
    def __init__(self):
        """Initialize model and load pretrained weights"""
        print("Initializing model...")
        
        # Set compute device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Model variables, will be initialized in predict()
        self.model = None
        self.task_configs = None
        self.task_id_to_name = None
        
        # Define data preprocessing transforms (no augmentation for inference)
        self.transforms = A.Compose([
            A.Resize(256, 256),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
        
        print("Model initialization complete!\n")
    
    def predict(self, data_root: str, output_dir: str, batch_size: int = 8, model_path: str = 'best_model.pth'):
        """
        Perform prediction on input data
        
        Args:
            data_root: Data root directory containing csv_files subdirectory
            output_dir: Output results root directory
            batch_size: Batch size, default is 8
            model_path: Path to the model weights file, default is 'best_model.pth'
        
        Output:
            - Segmentation tasks: Save predicted masks as image files
            - Classification tasks: Save to classification_predictions.json
            - Detection tasks: Save to detection_predictions.json
            - Regression tasks: Save to regression_predictions.json
        """
        print(f"{'='*60}")
        print(f"Starting prediction...")
        print(f"Data directory: {data_root}")
        print(f"Output directory: {output_dir}")
        print(f"Batch size: {batch_size}")
        print(f"Model path: {model_path}")
        print(f"{'='*60}\n")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Load dataset
        print(f"Loading dataset...")
        dataset = InferenceDataset(data_root=data_root, transforms=self.transforms)
        
        # Build task_configs from dataset (dynamic construction)
        print(f"\nBuilding task configurations from dataset...")
        self.task_configs = []
        task_config_map = {}
        
        for _, row in dataset.dataframe.iterrows():
            task_id = row['task_id']
            if task_id not in task_config_map:
                task_config = {
                    'task_id': task_id,
                    'task_name': row['task_name'],
                    'num_classes': int(row['num_classes'])
                }
                task_config_map[task_id] = task_config
                self.task_configs.append(task_config)
        
        print(f"Detected {len(self.task_configs)} task configurations")
        for cfg in sorted(self.task_configs, key=lambda x: x['task_id']):
            print(f"  - {cfg['task_id']}: {cfg['task_name']}, num_classes={cfg['num_classes']}")
        
        # Build task_id to task_name mapping
        self.task_id_to_name = {cfg['task_id']: cfg['task_name'] for cfg in self.task_configs}
        
        # Create and load model
        print(f"\nLoading model...")
        
        # Determine if model uses LoRA and prompt based on model path
        use_lora = 'no_lora' not in model_path
        use_prompt = 'no_prompt' not in model_path
        
        print(f"Model configuration:")
        print(f"  - Use LoRA: {use_lora}")
        print(f"  - Use prompt: {use_prompt}")
        
        # Create encoder with appropriate configuration
        from florence_adapter import FlorenceEncoderAdapter
        encoder = FlorenceEncoderAdapter(
            model_path='./model/florence-2-base',
            pretrained=True,
            feature_dim=256,
            freeze_stages=4,
            use_lora=use_lora,
            use_prompt=use_prompt
        )
        
        # Create model with custom encoder
        self.model = MultiTaskModelFactory(
            encoder_name='florence',
            encoder_weights=None,
            task_configs=self.task_configs,
            custom_encoder=encoder
        ).to(self.device)
        
        # Load trained model weights
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        # 设置 strict=False 以允许加载包含 LoRA 相关参数的模型权重
        self.model.load_state_dict(checkpoint, strict=False)
        self.model.eval()
        print("Model weights loaded successfully!")
        
        # Create data loader (batch processing for faster inference)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=inference_collate_fn
        )
        
        # Batch inference
        print(f"\nStarting inference...")
        classification_results = []
        detection_results = []
        regression_results = []
        task_counts = {}
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Prediction progress"):
                images = batch['image'].to(self.device)
                task_ids = batch['task_id']
                task_names = batch['task_name']
                image_paths = batch['image_path']
                mask_paths = batch['mask_path']
                original_sizes = batch['original_size']
                
                # Process each task in batch
                unique_tasks = list(set(task_ids))
                
                for task_id in unique_tasks:
                    # Get indices for all samples of current task
                    task_indices = [i for i, tid in enumerate(task_ids) if tid == task_id]
                    task_images = images[task_indices]
                    
                    # Model inference
                    outputs = self.model(task_images, task_id=task_id)
                    task_name = task_names[task_indices[0]]
                    
                    # Save prediction results for each sample
                    for i, batch_idx in enumerate(task_indices):
                        pred = outputs[i]
                        image_path = image_paths[batch_idx]
                        mask_path = mask_paths[batch_idx]
                        original_size = original_sizes[batch_idx]
                        
                        # Statistics
                        task_counts[task_id] = task_counts.get(task_id, 0) + 1
                        
                        # Process results by task type
                        if task_name == 'segmentation':
                            self._save_segmentation(pred, image_path, mask_path, output_dir, original_size)
                        
                        elif task_name == 'classification':
                            result = self._process_classification(pred, task_id, image_path)
                            classification_results.append(result)
                        
                        elif task_name == 'Regression':
                            result = self._process_regression(pred, task_id, image_path, original_size)
                            regression_results.append(result)
                        
                        elif task_name == 'detection':
                            result = self._process_detection(pred, task_id, image_path, original_size)
                            detection_results.append(result)
        
        # Save aggregated JSON results
        print("\nSaving prediction results...")
        
        if classification_results:
            json_path = os.path.join(output_dir, 'classification_predictions.json')
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(classification_results, f, indent=2, ensure_ascii=False)
            print(f"  - Classification results: {json_path} ({len(classification_results)} samples)")
        
        if detection_results:
            json_path = os.path.join(output_dir, 'detection_predictions.json')
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(detection_results, f, indent=2, ensure_ascii=False)
            print(f"  - Detection results: {json_path} ({len(detection_results)} samples)")
        
        if regression_results:
            json_path = os.path.join(output_dir, 'regression_predictions.json')
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(regression_results, f, indent=2, ensure_ascii=False)
            print(f"  - Regression results: {json_path} ({len(regression_results)} samples)")
        
        # Print statistics
        print(f"\n{'='*60}")
        print("Prediction complete!")
        print(f"{'='*60}")
        print("\nPrediction count by task:")
        for task_id in sorted(task_counts.keys()):
            task_name_str = self.task_id_to_name.get(task_id, 'unknown')
            count = task_counts[task_id]
            print(f"  - {task_id:<25} ({task_name_str:<15}): {count:>5} samples")
        print(f"\nTotal: {sum(task_counts.values())} samples")
        print()
    
    def _save_segmentation(self, pred, image_path, mask_path, output_dir, original_size):
        """Save segmentation prediction results as image file"""
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
        
        # Determine if this is fetal_abdomen_multi task
        is_fetal_abdomen = False
        if mask_path and 'fetal_abdomen_multi' in mask_path:
            is_fetal_abdomen = True
        elif 'fetal_abdomen_multi' in image_path:
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
        
        # Determine output path
        if mask_path:
            # Use mask path specified in CSV, remove leading '../' and 'train/'
            mask_path_clean = mask_path.replace('../', '').replace('train/', '')
            output_path = os.path.join(output_dir, mask_path_clean)
        else:
            # Default: replace keywords in image_path and remove 'train/'
            default_mask_path = image_path.replace('img', 'mask').replace('IMG', 'MASK').replace('train/', '')
            output_path = os.path.join(output_dir, default_mask_path)
        
        # Create output directory
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save mask image
        cv2.imwrite(output_path, mask)
    
    def _process_classification(self, pred, task_id, image_path):
        """Process classification task prediction results"""
        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().numpy()
        
        # Get predicted class
        pred_class = int(np.argmax(pred))
        
        # Calculate probability distribution (using softmax)
        # Stable softmax to avoid numerical overflow
        pred_exp = np.exp(pred - np.max(pred))
        pred_probs = pred_exp / np.sum(pred_exp)
        
        return {
            'image_path': image_path,
            'task_id': task_id,
            'predicted_class': pred_class,
            'predicted_probs': pred_probs.tolist()
        }
    
    def _process_regression(self, pred, task_id, image_path, original_size):
        """Process regression task prediction results (keypoint localization) with engineering stop-loss"""
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
            'image_path': image_path,
            'task_id': task_id,
            'predicted_points_normalized': normalized_coords,
            'predicted_points_pixels': pixel_coords
        }
    
    def _process_detection(self, pred, task_id, image_path, original_size):
        """Process detection task prediction results"""
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
        # Using same logic as in utils.py heatmap_to_bbox function
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
            'image_path': image_path,
            'task_id': task_id,
            'bbox_normalized': bbox_norm_list,
            'bbox_pixels': bbox_pixel
        }


# Docker entry point
if __name__ == '__main__':
    """
    Docker environment entry point
    """
    import argparse
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Foundation Model Challenge for Ultrasound Image Analysis (FMC_UIA) - Docker Inference')
    parser.add_argument('--data_root', type=str, default='valdataset_for_participants', help='验证集目录（相对于/app）')
    parser.add_argument('--output_dir', type=str, default='output', help='输出目录（相对于/app）')
    parser.add_argument('--batch_size', type=int, default=8, help='批处理大小')
    parser.add_argument('--model_path', type=str, default='best_model.pth', help='模型文件路径')
    args = parser.parse_args()
    
    # 容器路径配置
    data_root = args.data_root
    output_dir = args.output_dir
    batch_size = args.batch_size
    model_path = args.model_path
    
    print('='*60)
    print('Foundation Model Challenge for Ultrasound Image Analysis (FMC_UIA) - Docker Inference')
    print('='*60)
    print(f"Data root: {os.path.abspath(data_root)}")
    print(f"Output dir: {os.path.abspath(output_dir)}")
    print(f"Batch size: {batch_size}")
    print(f"Model path: {os.path.abspath(model_path)}")
    print('='*60)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    start_time = time.time()
    
    # Create model and perform prediction
    model = Model()
    model.predict(data_root, output_dir, batch_size=batch_size, model_path=model_path)
    
    elapsed_time = time.time() - start_time
    print(f"\nTotal time: {elapsed_time:.2f} seconds")
    print("Inference complete!")
