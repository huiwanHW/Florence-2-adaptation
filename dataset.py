import os
import cv2
import pandas as pd
import numpy as np
import torch
import glob
import random
import json
import logging
from torch.utils.data import Dataset, Sampler
from tqdm import tqdm
from typing import Optional, Iterator, List
import albumentations as A

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultiTaskDataset(Dataset):
    def __init__(self, data_root: str, transforms: Optional[A.Compose] = None):
        super().__init__()
        self.data_root = data_root
        self.transforms = transforms
        # Allow common layouts:
        # - data_root/csv_files/ (expected)
        # - data_root/train/csv_files/ (user passed repo root)
        # - data_root/<any_subdir>/csv_files/ (scan immediate children)
        # First check if CSV files are directly in data_root
        all_csv_files = glob.glob(os.path.join(self.data_root, '*.csv'))
        
        if all_csv_files:
            # CSV files are directly in data_root
            self.csv_path = self.data_root
        else:
            # Look for csv_files subdirectory
            self.csv_path = os.path.join(self.data_root, 'csv_files')

            if not os.path.isdir(self.csv_path):
                # check for train/ subfolder
                candidate = os.path.join(self.data_root, 'train')
                if os.path.isdir(os.path.join(candidate, 'csv_files')):
                    self.data_root = candidate
                    self.csv_path = os.path.join(self.data_root, 'csv_files')
                else:
                    # scan immediate children for a directory that contains csv_files
                    found = False
                    try:
                        for child in os.listdir(self.data_root):
                            childpath = os.path.join(self.data_root, child)
                            if os.path.isdir(childpath) and os.path.isdir(os.path.join(childpath, 'csv_files')):
                                self.data_root = childpath
                                self.csv_path = os.path.join(self.data_root, 'csv_files')
                                found = True
                                break
                    except Exception:
                        pass

                    if not found and not os.path.isdir(self.csv_path):
                        # Try a wider search to give helpful error messages
                        candidates = []
                        for root, dirs, files in os.walk(self.data_root):
                            if 'csv_files' in dirs:
                                candidates.append(os.path.join(root, 'csv_files'))

                        if candidates:
                            raise FileNotFoundError(
                                f"CSV path not found at expected location: {self.csv_path}.\nHowever, found csv_files at: {candidates}.\nPlease pass the parent of that path as --data_root (e.g., {os.path.dirname(candidates[0])})."
                            )
                        else:
                            raise FileNotFoundError(f"CSV path not found: {self.csv_path}. Expected structure: data_root/csv_files/ with subfolders Classification, Segmentation, Detection, Regression.")
            
            all_csv_files = glob.glob(os.path.join(self.csv_path, '*.csv'))
        if not all_csv_files:
            raise FileNotFoundError(f"No CSV files found in {self.csv_path}")
            
        df_list = []
        for csv_file in all_csv_files:
            for enc in ["utf-8", "gbk", "cp936"]:
                try:
                    df = pd.read_csv(csv_file, encoding=enc)
                    df_list.append(df)
                    print(f"[OK] Loaded {csv_file} with encoding={enc}")
                    break
                except UnicodeDecodeError:
                    continue
            else:
                raise RuntimeError(f"Failed to read {csv_file} with common encodings")

        self.dataframe = pd.concat(df_list, ignore_index=True).reset_index(drop=True)
        print(f"Data loaded. Total samples: {len(self.dataframe)}")

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, idx: int) -> dict:
        record = self.dataframe.iloc[idx]
        task_id = record['task_id']
        task_name = record['task_name']
        num_classes = int(record['num_classes'])  # 获取num_classes
        
        # Load image
        image_rel_path = self.normalize_path_string(record['image_path'])

        def _resolve_path(rel_path: str):
            # Try several candidate roots to be tolerant to different CSV conventions
            candidates = []
            # 1) relative to csv_files
            candidates.append(os.path.normpath(os.path.join(self.csv_path, rel_path)))
            # 2) relative to data_root
            candidates.append(os.path.normpath(os.path.join(self.data_root, rel_path)))
            # 3) relative to parent of data_root (in case csv sits in subdir)
            candidates.append(os.path.normpath(os.path.join(os.path.dirname(self.data_root), rel_path)))
            # 4) treat rel_path as absolute/relative to cwd
            candidates.append(os.path.normpath(rel_path))

            for p in candidates:
                if os.path.exists(p):
                    return p

            # 5) fallback: try to find by basename anywhere under data_root
            basename = os.path.basename(rel_path)
            matches = glob.glob(os.path.join(self.data_root, '**', basename), recursive=True)
            if matches:
                return matches[0]

            return None

        image_abs_path = _resolve_path(image_rel_path)
        if image_abs_path is None:
            # warn and return next sample (robustness)
            import warnings
            warnings.warn(f"Image file not found for record {idx}: tried '{image_rel_path}' under {self.csv_path} and {self.data_root}. Skipping sample.")
            return self.__getitem__((idx + 1) % len(self))

        # Robust image read: OpenCV sometimes fails on Windows with non-ASCII paths.
        def _imread_fallback(p):
            # Try cv2.imread first
            try:
                img = cv2.imread(p)
                if img is not None:
                    return img
            except Exception:
                pass

            # Try reading bytes + cv2.imdecode
            try:
                with open(p, 'rb') as f:
                    data = f.read()
                arr = np.frombuffer(data, dtype=np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if img is not None:
                    return img
            except Exception:
                pass

            # Try PIL as a last resort
            try:
                from PIL import Image
                pil = Image.open(p).convert('RGB')
                img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
                return img
            except Exception:
                pass

            return None

        image = _imread_fallback(image_abs_path)

        # Robustness check: retry next index if image load fails
        if image is None:
            import warnings
            warnings.warn(f"Image load failed for path: {image_abs_path}. Skipping sample {idx}.")
            return self.__getitem__((idx + 1) % len(self))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Save original image size BEFORE any transforms (for Regression coordinate normalization)
        original_height, original_width = image.shape[:2]

        # Load raw labels based on task
        label = None
        mask = None
        bboxes = []
        class_labels = []

        if task_name == 'segmentation':
            if pd.notna(record.get('mask_path')):
                mask_rel_path = self.normalize_path_string(record['mask_path'])
                mask_path = _resolve_path(mask_rel_path)
                if mask_path and os.path.exists(mask_path):
                    # Mask reading fallback similar to image read
                    try:
                        m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                        if m is None:
                            with open(mask_path, 'rb') as f:
                                data = f.read()
                            arr = np.frombuffer(data, dtype=np.uint8)
                            m = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
                        if m is not None:
                            # 确保mask是numpy数组
                            if not isinstance(m, np.ndarray):
                                m = np.asarray(m)
                            # 确保mask不为空且是2D数组
                            if m.size > 0 and len(m.shape) == 2:
                                mask = m
                            else:
                                mask = None
                        else:
                            mask = None
                    except Exception:
                        mask = None
                else:
                    mask = None
        
        elif task_name == 'classification':
            # 更健壮的分类标签处理，支持不同字段名
            if pd.notna(record.get('mask')):
                label = int(record['mask'])
            elif pd.notna(record.get('label')):
                label = int(record['label'])
            elif pd.notna(record.get('class')):
                label = int(record['class'])
            else:
                # 如果所有可能的标签字段都不存在，使用默认值0
                label = 0

        elif task_name == 'Regression':
            num_points = record['num_classes']
            coords = []
            for i in range(1, num_points + 1):
                col = f'point_{i}_xy'
                if col in record and pd.notna(record[col]):
                    coords.extend(json.loads(record[col]))
                else:
                    coords.extend([0, 0])
            label = np.array(coords, dtype=np.float32)

        elif task_name == 'detection':
            cols = ['x_min', 'y_min', 'x_max', 'y_max']
            if all(c in record and pd.notna(record[c]) for c in cols):
                box_coords = [float(record[c]) for c in cols]
                x_min, y_min, x_max, y_max = box_coords
                
                # 确保bbox坐标有效
                if x_max <= x_min or y_max <= y_min:
                    bboxes = []
                    class_labels = []
                else:
                    # 确保bbox坐标在图像边界内
                    x_min = max(0, min(x_min, original_width - 1))
                    y_min = max(0, min(y_min, original_height - 1))
                    x_max = max(0, min(x_max, original_width - 1))
                    y_max = max(0, min(y_max, original_height - 1))
                    
                    # 再次检查坐标有效性
                    if x_max <= x_min or y_max <= y_min:
                        bboxes = []
                        class_labels = []
                    else:
                        # 将bbox格式化为[min_x, min_y, max_x, max_y, class_id]
                        bboxes = [[x_min, y_min, x_max, y_max, 0]]
                        class_labels = [0]

        # Apply augmentations
        if self.transforms:
            # Ensure mask is numpy array or None (albumentations requires numpy)
            if mask is not None:
                try:
                    if not isinstance(mask, np.ndarray):
                        if hasattr(mask, 'cpu'):
                            mask = mask.cpu().numpy()
                        else:
                            mask = np.asarray(mask)
                    # 确保mask不为空且维度正确
                    if mask.size == 0 or len(mask.shape) != 2:
                        mask = None
                except Exception:
                    mask = None
            # 确保bbox坐标在合理范围内（对检测任务）
            if task_name == 'detection' and bboxes:
                for i, box in enumerate(bboxes):
                    if len(box) >= 5:
                        x_min, y_min, x_max, y_max = box[:4]
                        # 确保x_max >= x_min且y_max >= y_min
                        if x_max < x_min or y_max < y_min:
                            continue  # 跳过无效bbox
                        # 确保归一化后的坐标在[0, 1]范围内
                        # 注意：这里的坐标还没有归一化，我们将在后续步骤中进行归一化和clipping

            # Ensure bboxes/class_labels are lists
            if bboxes is None:
                bboxes = []
            if class_labels is None:
                class_labels = []

            # 对检测任务的bbox进行额外检查
            if task_name == 'detection' and bboxes:
                # 确保bbox坐标有效且在合理范围内
                valid_bboxes = []
                valid_class_labels = []
                for box, label in zip(bboxes, class_labels):
                    if len(box) >= 5:
                        x_min, y_min, x_max, y_max = box[:4]
                        # 确保坐标是有效的数字
                        if all(isinstance(coord, (int, float)) for coord in [x_min, y_min, x_max, y_max]):
                            # 确保x_max >= x_min且y_max >= y_min
                            if x_max >= x_min and y_max >= y_min:
                                # 对超出范围的坐标进行截断（这里假设坐标是相对于图像尺寸的绝对坐标）
                                # 在后续归一化时还会再次检查范围
                                valid_bboxes.append(box)
                                valid_class_labels.append(label)
                bboxes = valid_bboxes
                class_labels = valid_class_labels

            try:
                # 最终检查：确保mask是单个numpy数组且尺寸正确
                use_mask = False
                if task_name == 'segmentation' and mask is not None:
                    # 确保mask是numpy数组
                    if isinstance(mask, np.ndarray):
                        # 确保mask是2D数组
                        if len(mask.shape) == 2:
                            # 确保mask不为空
                            if mask.size > 0:
                                # 确保mask尺寸与图像尺寸匹配
                                if image.shape[:2] == mask.shape[:2]:
                                    use_mask = True
                                else:
                                    # 调整mask尺寸以匹配图像
                                    try:
                                        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
                                        if image.shape[:2] == mask.shape[:2]:
                                            use_mask = True
                                    except Exception:
                                        use_mask = False
                
                # 根据是否使用mask选择合适的参数
                if use_mask:
                    augmented = self.transforms(image=image, mask=mask, bboxes=bboxes, class_labels=class_labels)
                else:
                    # 对于非分割任务或mask无效的情况，不传递mask参数
                    augmented = self.transforms(image=image, bboxes=bboxes, class_labels=class_labels)
            except Exception as e:
                # 任何错误都尝试不传递mask参数
                msg = str(e)
                logger.warning(f"Transforms应用失败: {msg}，尝试不传递mask参数")
                augmented = self.transforms(image=image, bboxes=bboxes, class_labels=class_labels)
            image = augmented['image']
            
            if task_name == 'segmentation':
                label = augmented.get('mask')
            elif task_name == 'detection':
                if augmented['bboxes']:
                    label = np.array(augmented['bboxes'][0][:4], dtype=np.float32)
                else:
                    label = np.array([-1.0, -1.0, -1.0, -1.0], dtype=np.float32)

        # Format conversion & normalization
        final_label = None
        h, w = image.shape[1], image.shape[2]

        # Ensure label is numpy for processing
        if isinstance(label, torch.Tensor):
            label = label.cpu().numpy()

        if task_name == 'segmentation':
            if label is None:
                label = np.zeros((h, w), dtype=np.int64)
            
            # 针对fetal_abdomen_multi的特殊处理
            if task_id == 'fetal_abdomen_multi':
                # 确保mask的类标签是连续的，避免训练时出现问题
                unique_labels = np.unique(label)
                if len(unique_labels) > 0:
                    # 确保所有标签都在有效范围内
                    label = np.clip(label, 0, num_classes - 1)
            
            final_label = torch.from_numpy(label).long()

        elif task_name == 'classification':
            final_label = torch.tensor(label).long()

        elif task_name in ['Regression', 'detection']:
            if not isinstance(label, np.ndarray):
                if task_name == 'Regression':
                    # 根据num_classes调整回归标签长度
                    label = np.zeros(num_classes * 2, dtype=np.float32)
                else:
                    label = np.array([-1.0, -1.0, -1.0, -1.0], dtype=np.float32)

            # Normalize coordinates to [0, 1]
            if task_name == 'detection' and np.all(label >= 0):
                # 归一化坐标到[0, 1]范围
                label[[0, 2]] /= w
                label[[1, 3]] /= h
                # 确保归一化后的坐标在[0, 1]范围内
                label = np.clip(label, 0.0, 1.0)
            elif task_name == 'Regression':
                # 针对fetal_femur的特殊处理
                if task_id == 'fetal_femur':
                    # 确保fetal_femur的坐标在合理范围内
                    label[0::2] = np.clip(label[0::2], 0.0, original_width)
                    label[1::2] = np.clip(label[1::2], 0.0, original_height)
                
                # 归一化坐标
                label[0::2] /= original_width
                label[1::2] /= original_height
                # 确保归一化后的坐标在[0, 1]范围内
                label = np.clip(label, 0.0, 1.0)
            
            final_label = torch.from_numpy(label).float()
        
        return {'image': image, 'label': final_label, 'task_id': task_id, 'original_size': (original_height, original_width)}

    def normalize_path_string(self, s: str) -> str:
        # 常见异常字符 → 空格
        BAD_CHARS = [
            '\u807d',  # 聽
            '\u00a0',  # NBSP
            '\u3000',  # 全角空格
        ]
        for ch in BAD_CHARS:
            s = s.replace(ch, ' ')
        return s


class MultiTaskUniformSampler(Sampler[List[int]]):
    def __init__(self, dataset: MultiTaskDataset, batch_size: int, steps_per_epoch: Optional[int] = None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.indices_by_task = {}

        # Group indices by task_id
        print("\n--- Initializing Sampler ---")
        for idx, task_id in enumerate(tqdm(dataset.dataframe['task_id'], desc="Grouping indices")):
            if task_id not in self.indices_by_task:
                self.indices_by_task[task_id] = []
            self.indices_by_task[task_id].append(idx)
            
        self.task_ids = list(self.indices_by_task.keys())
        
        # Calculate task weights based on number of samples (sample-level uniform sampling)
        task_sizes = [len(indices) for indices in self.indices_by_task.values()]
        total_samples = sum(task_sizes)
        self.task_weights = [size / total_samples for size in task_sizes] if total_samples > 0 else [1/len(self.task_ids)]*len(self.task_ids)
        
        # Initial shuffle
        for task_id in self.task_ids:
            random.shuffle(self.indices_by_task[task_id])

        # Determine epoch length
        if steps_per_epoch is None:
            self.steps_per_epoch = len(self.dataset) // self.batch_size
        else:
            self.steps_per_epoch = steps_per_epoch

    def __iter__(self) -> Iterator[List[int]]:
        task_cursors = {task_id: 0 for task_id in self.task_ids}

        for _ in range(self.steps_per_epoch):
            # Select task based on sample size weights (sample-level uniform sampling)
            task_id = random.choices(self.task_ids, weights=self.task_weights)[0]
            indices = self.indices_by_task[task_id]
            cursor = task_cursors[task_id]
            
            start_idx = cursor
            end_idx = start_idx + self.batch_size
            
            if end_idx > len(indices):
                # Wrap around
                batch_indices = indices[start_idx:]
                random.shuffle(indices)
                remaining = self.batch_size - len(batch_indices)
                batch_indices.extend(indices[:remaining])
                task_cursors[task_id] = remaining
            else:
                batch_indices = indices[start_idx:end_idx]
                task_cursors[task_id] = end_idx
            
            yield batch_indices
            
    def __len__(self) -> int:
        return self.steps_per_epoch