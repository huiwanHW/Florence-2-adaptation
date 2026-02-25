import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
try:
    from florence_adapter import FlorenceEncoderAdapter
    _HAS_FLORENCE = True
except Exception:
    _HAS_FLORENCE = False
from typing import List, Dict

# Import task configurations from config
from config import TASK_CONFIGURATIONS

class SegmentationTaskDecoder(nn.Module):
    """轻量级分割任务专用解码器"""
    def __init__(self, in_channels, out_channels=256):
        super().__init__()
        
        # 解码器结构：2-3层Conv + 上采样
        self.decoder = nn.Sequential(
            # 第一层卷积
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            
            # 上采样
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            
            # 第二层卷积
            nn.Conv2d(out_channels, out_channels // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True),
            
            # 上采样
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            
            # 第三层卷积
            nn.Conv2d(out_channels // 2, out_channels // 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True)
        )
        
        # 输出通道数
        self.out_channels = out_channels // 4

    def forward(self, x):
        return self.decoder(x)

# Task specific heads

class SmpClassificationHead(nn.Module):
    """Wrapper for SMP Classification Head."""
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.head = smp.base.ClassificationHead(
            in_channels=in_channels,
            classes=num_classes,
            pooling="avg",
            dropout=0.2,
            activation=None,
        )
        
    def forward(self, features: list):
        # Use the last feature map from encoder
        return self.head(features[-1])

class RegressionHead(nn.Module):
    """Custom head for regression tasks."""
    def __init__(self, in_channels: int, num_points: int):
        super().__init__()
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        # Output dimension is num_points * 2 (x, y)
        self.linear = nn.Linear(in_channels, num_points * 2)

    def forward(self, features: list):
        x = self.pooling(features[-1])
        x = self.flatten(x)
        return self.linear(x)

class FPNGridDetectionHead(nn.Module):
    """Detection head designed for FPN outputs using Gaussian heatmap.
    
    Output format:
    - Channel 0: Heatmap (objectness score)
    - Channel 1-2: Center offset (dx, dy)
    - Channel 3-4: Box size (w, h)
    """
    def __init__(self, fpn_out_channels: int, num_classes: int = 1, num_anchors: int = 1):
        super().__init__()
        mid_channels = 128
        num_outputs = 5  # 1 heatmap + 2 offset + 2 size
        
        self.conv_block = nn.Sequential(
            nn.Conv2d(fpn_out_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
            nn.Conv2d(mid_channels, num_outputs, kernel_size=1)
        )

    def forward(self, fpn_features: torch.Tensor):
        # Input fpn_features is already a single fused tensor from FPN Decoder
        predictions_map = self.conv_block(fpn_features)
        
        # Apply sigmoid to heatmap channel only
        predictions_map[:, 0:1, :, :] = torch.sigmoid(predictions_map[:, 0:1, :, :])
        
        return predictions_map

# ====================================================================
# --- 2. Multi-Task Model Factory ---
# ====================================================================

class MultiTaskModelFactory(nn.Module):
    def __init__(self, encoder_name: str, encoder_weights: str, task_configs: List[Dict], freeze_stages: int = 4):
        super().__init__()
        
        # Initialize shared encoder
        print(f"Initializing shared encoder: {encoder_name}")
        if encoder_name.lower().startswith('florence') and _HAS_FLORENCE:
            # Use Florence adapter as frozen encoder            
            if encoder_name == "florence":
                model_name = "./model/florence-2-base"
            else:
                print(f"Florence model '{encoder_name}' not implemented.")
                return NotImplementedError(f"Florence model '{encoder_name}' not implemented.")
            self.encoder = FlorenceEncoderAdapter(
                model_path=model_name,
                pretrained=True,
                feature_dim=256,
                freeze_stages=freeze_stages
            )

            # Create a proper FPN decoder for Florence encoder
            class FlorenceFPNDecoder(nn.Module):
                """FPN decoder designed to work with Florence encoder outputs."""
                def __init__(self, in_channels, out_channels=256):
                    super().__init__()
                    self.out_channels = out_channels
                    
                    # Simple FPN-like structure
                    self.lateral_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
                    self.output_conv = nn.Sequential(
                        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(inplace=True)
                    )

                def forward(self, features):
                    """
                    Args:
                        features: List of feature maps from encoder. For Florence, it's a single feature map.
                    Returns:
                        Fused feature map.
                    """
                    if isinstance(features, list):
                        x = features[-1]  # Use the last feature map
                    else:
                        x = features
                    
                    # Apply lateral convolution
                    x = self.lateral_conv(x)
                    
                    # Apply output convolution
                    x = self.output_conv(x)
                    
                    return x

            self.fpn_decoder = FlorenceFPNDecoder(self.encoder.out_channels[0], out_channels=256)
        else:
            # For SMP encoders, convert freeze_stages to boolean (freeze all if freeze_stages > 0)
            freeze_encoder = freeze_stages > 0
            self.encoder = smp.encoders.get_encoder(
                name=encoder_name,
                in_channels=3,
                depth=5,
                weights=encoder_weights,
            )
            
            # Freeze encoder if requested
            if freeze_encoder:
                for param in self.encoder.parameters():
                    param.requires_grad = False
            
            # Initialize shared FPN decoder
            temp_fpn_model = smp.FPN(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=3,
                classes=1, 
            )
            self.fpn_decoder = temp_fpn_model.decoder
        
        # 为分割任务添加专用解码器
        self.segmentation_decoder = None
        if hasattr(self.fpn_decoder, 'out_channels'):
            fpn_out_channels = self.fpn_decoder.out_channels
        else:
            fpn_out_channels = 256  # 默认值
        
        self.segmentation_decoder = SegmentationTaskDecoder(
            in_channels=fpn_out_channels,
            out_channels=256
        )
        print(f"Created segmentation task decoder with output channels: {self.segmentation_decoder.out_channels}")

        # Initialize task heads
        self.heads = nn.ModuleDict()
        
        print(f"Creating heads for {len(task_configs)} tasks...")
        for config in task_configs:
            task_id = config['task_id']
            task_name = config['task_name']
            num_classes = config['num_classes']
            
            head_module = None
            if task_name == 'segmentation':
                head_module = smp.base.SegmentationHead(
                    in_channels=self.segmentation_decoder.out_channels, 
                    out_channels=num_classes, 
                    kernel_size=1,
                    upsampling=8  # 调整上采样倍数，因为解码器已经做了部分上采样
                )

            elif task_name == 'classification':
                head_module = SmpClassificationHead(
                    in_channels=self.encoder.out_channels[-1],
                    num_classes=num_classes
                )

            elif task_name == 'Regression':
                num_points = config['num_classes']
                head_module = RegressionHead(
                    in_channels=self.encoder.out_channels[-1],
                    num_points=num_points
                )

            elif task_name == 'detection':
                head_module = FPNGridDetectionHead(
                    fpn_out_channels=fpn_out_channels,
                    num_classes=num_classes
                )

            if head_module:
                self.heads[task_id] = head_module
            else:
                print(f"Warning: Unknown task type '{task_name}' for {task_id}")

    def forward(self, x: torch.Tensor, task_id: str) -> torch.Tensor:
        if task_id not in self.heads:
            raise ValueError(f"Task ID '{task_id}' not found.")

        task_config = next((item for item in TASK_CONFIGURATIONS if item["task_id"] == task_id), None)
        task_name = task_config['task_name'] if task_config else None

        # Special handling when using Florence adapter
        if hasattr(self.encoder, 'get_fpn_feature'):
            if task_name == 'segmentation':
                # For segmentation task, use dedicated decoder
                features = self.encoder(x, task_id=task_id)
                fpn_features = self.fpn_decoder(features)
                # 使用分割专用解码器
                seg_features = self.segmentation_decoder(fpn_features)
                output = self.heads[task_id](seg_features)
            elif task_name == 'detection':
                # For detection task, use regular FPN path
                features = self.encoder(x, task_id=task_id)
                fpn_features = self.fpn_decoder(features)
                output = self.heads[task_id](fpn_features)
            elif task_name in ['classification', 'Regression']:
                # For global tasks, use forward to get the feature list
                features = self.encoder(x, task_id=task_id)
                # Ensure we're passing a list as expected by the heads
                if not isinstance(features, list):
                    features = [features]
                output = self.heads[task_id](features)
            else:
                # Default fallback
                features = self.encoder(x, task_id=task_id)
                if not isinstance(features, list):
                    features = [features]
                output = self.heads[task_id](features)
            return output

        # Default path for SMP encoders + FPN
        features = self.encoder(x)
        if task_name == 'segmentation':
            # For segmentation task, use dedicated decoder
            fpn_features = self.fpn_decoder(features)
            seg_features = self.segmentation_decoder(fpn_features)
            output = self.heads[task_id](seg_features)
        elif task_name == 'detection':
            # For detection task, use regular FPN path
            fpn_features = self.fpn_decoder(features)
            output = self.heads[task_id](fpn_features)
        elif task_name in ['classification', 'Regression']:
            # For global tasks, pass the feature list
            output = self.heads[task_id](features)
        else:
            # Default fallback
            output = self.heads[task_id](features)

        return output

# Example usage

if __name__ == '__main__':
    model = MultiTaskModelFactory(
        encoder_name='resnet34',
        encoder_weights='imagenet',
        task_configs=TASK_CONFIGURATIONS
    )

    print("\n--- Forward Pass Test ---")
    dummy_image_batch = torch.randn(2, 3, 256, 256) # Reduced batch size for test

    # Test specific tasks
    test_tasks = ['cardiac_multi', 'fetal_plane_cls', 'FUGC', 'thyroid_nodule_det']
    
    for t_id in test_tasks:
        try:
            out = model(dummy_image_batch, task_id=t_id)
            print(f"Task: {t_id:<25} | Output Shape: {out.shape}")
        except Exception as e:
            print(f"Task: {t_id:<25} | Error: {e}")