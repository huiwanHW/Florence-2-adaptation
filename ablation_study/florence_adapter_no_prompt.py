import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

# Import task configurations from config
from config import TASK_CONFIGURATIONS


class FlorenceEncoderAdapter(nn.Module):
    """Adapter to use Florence-style local model as a frozen feature extractor WITHOUT prompt support.

    Behavior:
    - Loads local DaViT model from ./model/florence-2-base
    - Exposes `out_channels` and `out_channels_list` to mimic SMP encoder API.
    - `forward(x, task_id)` returns a list of features; last element is a 4D tensor used by heads.
    - `get_fpn_feature(x, task_id)` returns a single fused 4D feature map for dense tasks.
    - NO prompt mechanism (w/o Prompt ablation).
    """

    def __init__(self, model_path: str = './model/florence-2-base', pretrained: bool = True, feature_dim: int = 256, freeze_stages: int = 4):
        super().__init__()
        self.model_path = model_path
        self.feature_dim = feature_dim
        self.pretrained = pretrained
        self.freeze_stages = freeze_stages  # 0: no freeze, 1-4: freeze first N stages

        # Define Florence-2-base configuration directly in code
        vision_config = {
            'depths': [1, 1, 9, 1],
            'patch_size': [7, 3, 3, 3],
            'patch_stride': [4, 2, 2, 2],
            'patch_padding': [3, 1, 1, 1],
            'patch_prenorm': [False, True, True, True],
            'embed_dims': [128, 256, 512, 1024],
            'num_heads': [8, 16, 32, 64],
            'num_groups': [8, 16, 32, 64],
            'window_size': 12,
            'enable_checkpoint': False
        }

        # Try to load the model and handle exceptions
        try:
            # Import DaViT from local module without Hugging Face dependencies
            import sys
            sys.path.append(os.path.abspath(model_path))
            
            # Dynamically load only necessary modules from modeling_florence2.py
            from modeling_florence2 import DaViT

            # Initialize DaViT model
            self.model = DaViT(
                in_chans=3,
                num_classes=0,
                depths=tuple(vision_config['depths']),
                patch_size=tuple(vision_config['patch_size']),
                patch_stride=tuple(vision_config['patch_stride']),
                patch_padding=tuple(vision_config['patch_padding']),
                patch_prenorm=tuple(vision_config['patch_prenorm']),
                embed_dims=tuple(vision_config['embed_dims']),
                num_heads=tuple(vision_config['num_heads']),
                num_groups=tuple(vision_config['num_groups']),
                window_size=vision_config['window_size'],
                enable_checkpoint=vision_config['enable_checkpoint'],
            )

            # Load pretrained weights if available
            if pretrained:
                weights_path = os.path.join(model_path, 'pytorch_model.bin')
                if os.path.exists(weights_path):
                    state_dict = torch.load(weights_path, map_location='cpu', weights_only=True)
                    # Extract vision tower weights
                    vision_state_dict = {}
                    for k, v in state_dict.items():
                        if k.startswith('vision_tower.'):
                            # Remove 'vision_tower.' prefix
                            vision_state_dict[k[13:]] = v
                    # Load weights
                    self.model.load_state_dict(vision_state_dict, strict=False)
                else:
                    print(f"警告: 未找到模型权重文件 {weights_path}")
                    print("请确保您已从Hugging Face下载了完整的模型文件")
        except (ImportError, FileNotFoundError) as e:
            print("错误: 无法加载Florence-2模型")
            print(f"详细错误: {e}")
            print("\n请按照以下步骤操作:")
            print("1. 访问 Hugging Face 模型页面: https://huggingface.co/microsoft/Florence-2-base")
            print("2. 下载完整的模型文件（包括 modeling_florence2.py 和 pytorch_model.bin）")
            print(f"3. 将文件放置到目录: {model_path}")
            print("4. 确保目录结构如下:")
            print(f"   {model_path}/")
            print("   ├── modeling_florence2.py")
            print("   └── pytorch_model.bin")
            raise

        # Get output dimension of model
        self.model_dim = vision_config['embed_dims'][-1]

        # Projection layer to match feature_dim
        self.proj = nn.Conv2d(self.model_dim, self.feature_dim, kernel_size=1)
        
        # NO soft prompt mechanism (w/o Prompt ablation)
        
        # Add LoRA to the vision encoder's later layers
        self.add_lora_to_encoder()

        # expose SMP-compatible attributes
        self.out_channels = [self.feature_dim]
        self.out_channels_list = self.out_channels

        if self.freeze_stages > 0:
            self._freeze()

    def add_lora_to_encoder(self):
        # Add LoRA to later layers of the vision encoder
        # DaViT has 4 stages with different depths: [1, 1, 9, 1] blocks
        # We'll add LoRA to last 30% of the blocks
        
        # Calculate total number of blocks
        total_blocks = sum([len(block) for block in self.model.blocks])
        print(f"Total blocks in DaViT: {total_blocks}")
        
        # Calculate number of blocks to add LoRA to (last 30%)
        num_lora_blocks = int(total_blocks * 0.3)
        print(f"Number of blocks to add LoRA: {num_lora_blocks}")
        
        # Identify the blocks to add LoRA to
        lora_block_indices = []
        current_block_idx = 0
        for stage_idx, stage_blocks in enumerate(self.model.blocks):
            for block_idx, block in enumerate(stage_blocks):
                if current_block_idx >= (total_blocks - num_lora_blocks):
                    lora_block_indices.append((stage_idx, block_idx))
                current_block_idx += 1
        
        print(f"Blocks to add LoRA: {lora_block_indices}")
        
        # Create LoRA config
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["qkv", "proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="feature_extraction"
        )
        
        # Apply LoRA to selected blocks
        for stage_idx, block_idx in lora_block_indices:
            # Get block
            block = self.model.blocks[stage_idx][block_idx]
            
            # Apply LoRA to spatial_block's window_attn
            if hasattr(block, 'spatial_block') and hasattr(block.spatial_block, 'window_attn'):
                self.model.blocks[stage_idx][block_idx].spatial_block.window_attn = get_peft_model(
                    block.spatial_block.window_attn, lora_config
                )
            
            # Apply LoRA to channel_block's channel_attn
            if hasattr(block, 'channel_block') and hasattr(block.channel_block, 'channel_attn'):
                self.model.blocks[stage_idx][block_idx].channel_block.channel_attn = get_peft_model(
                    block.channel_block.channel_attn, lora_config
                )
    
    def _freeze(self):
        # Freeze 70% of the vision encoder's layers
        # DaViT has 4 stages with different depths: [1, 1, 9, 1] blocks
        
        # Calculate total number of blocks
        total_blocks = sum([len(block) for block in self.model.blocks])
        print(f"Total blocks in DaViT: {total_blocks}")
        
        # Calculate number of blocks to freeze (first 70%)
        num_freeze_blocks = int(total_blocks * 0.7)
        print(f"Number of blocks to freeze: {num_freeze_blocks}")
        
        # Freeze the first N blocks
        current_block_idx = 0
        for stage_idx, stage_blocks in enumerate(self.model.blocks):
            for block_idx, block in enumerate(stage_blocks):
                if current_block_idx < num_freeze_blocks:
                    # Freeze the entire block
                    for p in block.parameters():
                        p.requires_grad = False
                current_block_idx += 1
        
        # Keep normalization layer trainable
        if hasattr(self.model, 'norms'):
            for p in self.model.norms.parameters():
                p.requires_grad = True
        
        # Keep projection layer trainable
        for p in self.proj.parameters():
            p.requires_grad = True

    def forward(self, x: torch.Tensor, task_id: str = None):
        """Return a list of feature tensors. Last element is (B, C, H, W)."""
        feat = self._extract_feature_map(x, task_id)
        # return as list like SMP encoder
        return [feat]

    def get_fpn_feature(self, x: torch.Tensor, task_id: str = None) -> torch.Tensor:
        """Return a single fused feature map suitable for FPN-based heads."""
        return self._extract_feature_map(x, task_id)

    def _extract_feature_map(self, x: torch.Tensor, task_id: str = None) -> torch.Tensor:
        B, C, H, W = x.shape
        
        # NO prompt mechanism (w/o Prompt ablation)
        # Direct visual feature extraction for all tasks
        
        def forward_without_prompt(x):
            # Check if input is 4D (B, C, H, W)
            if x.dim() == 4:
                # If input is 4D, extract H and W from shape
                input_size = (x.size(2), x.size(3))
                original_H, original_W = x.size(2), x.size(3)
            else:
                raise ValueError(f"Input must be 4D, got {x.dim()}D")
            
            # Process through each stage: conv layer followed by corresponding blocks
            # This matches the original DaViT architecture's forward pass
            for i, (conv, blocks) in enumerate(zip(self.model.convs, self.model.blocks)):
                # Process through the convolutional layer
                x, input_size = conv(x, input_size)
                
                # Process through all blocks in the current stage
                for block in blocks:
                    x, input_size = block(x, input_size)
            
            # Now x is in sequence format (B, N, C) after all stages
            return x, original_H, original_W
        
        # Get visual features without prompt tokens
        last_hidden, H, W = forward_without_prompt(x)  # (B, H*W, hidden)
        
        # Calculate target feature map size based on input resolution
        target_size = (H // 32, W // 32)  # DaViT reduces input size by factor of 32
        
        # If grid reshape works perfectly, use it
        B, N, hidden = last_hidden.shape
        grid = int(math.sqrt(N))
        if grid * grid == N:
            # Reshape to feature map
            feat_map = last_hidden.transpose(1, 2).reshape(B, hidden, grid, grid)
            
            # Resize to target size if needed
            if feat_map.shape[2] != target_size[0] or feat_map.shape[3] != target_size[1]:
                feat_map = F.interpolate(feat_map, size=target_size, mode='bilinear', align_corners=False)
        else:
            # Handle non-square cases by directly resizing the feature sequence
            # Reshape to 1x1 feature map and then interpolate
            feat_map = last_hidden.transpose(1, 2).reshape(B, hidden, 1, N)
            # First interpolate to square size based on width
            mid_size = (target_size[1], target_size[1])
            feat_map = F.interpolate(feat_map, size=mid_size, mode='bilinear', align_corners=False)
            # Then interpolate to final target size
            if mid_size != target_size:
                feat_map = F.interpolate(feat_map, size=target_size, mode='bilinear', align_corners=False)
        
        # Apply projection
        feat_map = self.proj(feat_map)
        return feat_map


if __name__ == '__main__':
    # Quick initialization test (offline friendly)
    try:
        adapter = FlorenceEncoderAdapter(pretrained=False, feature_dim=128, freeze_stages=0)
        print('✅ FlorenceEncoderAdapter (w/o Prompt) initialized successfully!')
        print(f'   - Model dimensions: {adapter.model_dim}')
        print(f'   - Feature dimensions after projection: {adapter.feature_dim}')
        print(f'   - NO prompt mechanism (w/o Prompt ablation)')
        print(f'   - Output channels: {adapter.out_channels}')
        print('\nThe adapter is ready for use in the training pipeline.')
    except Exception as e:
        print(f'❌ Error initializing FlorenceEncoderAdapter: {e}')
        import traceback
        traceback.print_exc()