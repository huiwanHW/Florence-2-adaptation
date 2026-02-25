import os
import json
import pandas as pd
import numpy as np
import cv2
from sklearn.metrics import roc_auc_score, f1_score, matthews_corrcoef
from scipy.spatial.distance import directed_hausdorff
from tqdm import tqdm

class ResultEvaluator:
    """
    评估模型推导结果的各项指标
    """
    
    def __init__(self, output_dir, csv_dir):
        """
        初始化评估器
        
        Args:
            output_dir: 模型输出结果目录
            csv_dir: 包含真实标签的CSV文件目录
        """
        self.output_dir = output_dir
        self.csv_dir = csv_dir
        self.results = {
            'segmentation': {},
            'classification': {},
            'detection': {},
            'regression': {}
        }
    
    def load_results(self):
        """
        加载模型输出结果
        """
        # 加载分类结果
        classification_path = os.path.join(self.output_dir, 'classification_predictions.json')
        if os.path.exists(classification_path):
            with open(classification_path, 'r', encoding='utf-8') as f:
                self.results['classification'] = json.load(f)
        
        # 加载检测结果
        detection_path = os.path.join(self.output_dir, 'detection_predictions.json')
        if os.path.exists(detection_path):
            with open(detection_path, 'r', encoding='utf-8') as f:
                self.results['detection'] = json.load(f)
        
        # 加载回归结果
        regression_path = os.path.join(self.output_dir, 'regression_predictions.json')
        if os.path.exists(regression_path):
            with open(regression_path, 'r', encoding='utf-8') as f:
                self.results['regression'] = json.load(f)
        
        print(f"Loaded results:")
        print(f"- Classification: {len(self.results['classification'])} samples")
        print(f"- Detection: {len(self.results['detection'])} samples")
        print(f"- Regression: {len(self.results['regression'])} samples")
    
    def load_ground_truth(self):
        """
        加载真实标签
        """
        self.ground_truth = {}
        
        # 遍历所有CSV文件
        for csv_file in os.listdir(self.csv_dir):
            if not csv_file.endswith('.csv'):
                continue
            
            csv_path = os.path.join(self.csv_dir, csv_file)
            df = pd.read_csv(csv_path)
            
            # 按任务类型分类
            for _, row in df.iterrows():
                task_id = row['task_id']
                image_path = row['image_path']
                
                if task_id not in self.ground_truth:
                    self.ground_truth[task_id] = {}
                
                self.ground_truth[task_id][image_path] = row
    
    def evaluate_segmentation(self):
        """
        评估分割任务指标：DSC, HD95
        """
        segmentation_metrics = {}
        
        # 收集所有分割结果文件
        seg_files = []
        for root, dirs, files in os.walk(os.path.join(self.output_dir, 'Segmentation')):
            for file in files:
                if file.endswith('.png') or file.endswith('.jpg'):
                    seg_files.append(os.path.join(root, file))
        
        # 遍历所有分割结果文件，添加进度条
        for seg_file in tqdm(seg_files, desc="Evaluating segmentation", unit="files"):
            # 获取相对路径
            relative_path = os.path.relpath(seg_file, self.output_dir)
            # 统一路径分隔符为正斜杠
            relative_path = relative_path.replace('\\', '/')
            # 转换为与CSV中一致的路径格式
            csv_path = 'train/' + relative_path
            
            # 尝试多种路径格式匹配
            possible_paths = []
            
            # 基础路径
            base_path = csv_path
            possible_paths.append(base_path)
            
            # 替换不同的目录名组合
            dir_replacements = [
                ('labels/', 'images/'),
                ('Labels/', 'Images/'),
                ('LABELS/', 'IMAGES/'),
                ('masks/', 'images/'),
                ('Masks/', 'Images/'),
                ('MASKS/', 'IMAGES/'),
                ('Mask/', 'Image/'),
                ('mask/', 'image/')
            ]
            
            for old_dir, new_dir in dir_replacements:
                if old_dir in base_path:
                    possible_paths.append(base_path.replace(old_dir, new_dir))
                if new_dir in base_path:
                    possible_paths.append(base_path.replace(new_dir, old_dir))
            
            # 去重
            possible_paths = list(set(possible_paths))
            
            # 查找对应的真实标签
            found = False
            for task_id, images in self.ground_truth.items():
                for path in possible_paths:
                    if path in images:
                        # 计算DSC和HD95
                        # 加载预测掩码
                        pred_mask = cv2.imread(seg_file, cv2.IMREAD_GRAYSCALE)
                        if pred_mask is None:
                            continue
                        
                        # 加载真实掩码
                        gt_row = images[path]
                        gt_mask_path = gt_row.get('mask_path', None)
                        if not gt_mask_path:
                            continue
                        
                        # 修正真实掩码路径：检查是否存在，如果不存在尝试绝对路径
                        if not os.path.exists(gt_mask_path):
                            # 尝试相对于项目根目录的路径
                            project_root = os.path.dirname(os.path.abspath(__file__))
                            gt_mask_path_abs = os.path.join(project_root, gt_mask_path)
                            if os.path.exists(gt_mask_path_abs):
                                gt_mask_path = gt_mask_path_abs
                            else:
                                # 尝试相对于train目录的路径
                                gt_mask_path_train = os.path.join(project_root, 'train', gt_mask_path.replace('train/', ''))
                                if os.path.exists(gt_mask_path_train):
                                    gt_mask_path = gt_mask_path_train
                                else:
                                    continue
                        
                        gt_mask = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)
                        if gt_mask is None:
                            continue
                        
                        # 确保尺寸一致
                        if pred_mask.shape != gt_mask.shape:
                            pred_mask = cv2.resize(pred_mask, (gt_mask.shape[1], gt_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
                        
                        # 计算DSC
                        intersection = np.sum((pred_mask > 0) & (gt_mask > 0))
                        union = np.sum(pred_mask > 0) + np.sum(gt_mask > 0)
                        dsc = 2 * intersection / union if union > 0 else 0
                        
                        # 计算HD95
                        # 只考虑前景像素
                        pred_pts = np.argwhere(pred_mask > 0)
                        gt_pts = np.argwhere(gt_mask > 0)
                        
                        if len(pred_pts) == 0 or len(gt_pts) == 0:
                            hd95 = 0
                        else:
                            # 优化HD95计算：使用采样点减少计算量
                            # 对于大型图像，随机采样最多1000个点
                            max_points = 1000
                            if len(pred_pts) > max_points:
                                pred_pts = pred_pts[np.random.choice(len(pred_pts), max_points, replace=False)]
                            if len(gt_pts) > max_points:
                                gt_pts = gt_pts[np.random.choice(len(gt_pts), max_points, replace=False)]
                            
                            # 双向豪斯多夫距离
                            d1 = directed_hausdorff(pred_pts, gt_pts)[0]
                            d2 = directed_hausdorff(gt_pts, pred_pts)[0]
                            hd = max(d1, d2)
                            # 计算95%分位数
                            distances = []
                            # 使用向量化操作计算距离，减少循环
                            for pt in pred_pts:
                                min_dist = np.min(np.sqrt(np.sum((gt_pts - pt) ** 2, axis=1)))
                                distances.append(min_dist)
                            for pt in gt_pts:
                                min_dist = np.min(np.sqrt(np.sum((pred_pts - pt) ** 2, axis=1)))
                                distances.append(min_dist)
                            hd95 = np.percentile(distances, 95)
                        
                        # 按任务ID存储指标
                        if task_id not in segmentation_metrics:
                            segmentation_metrics[task_id] = {'dsc': [], 'hd95': []}
                        segmentation_metrics[task_id]['dsc'].append(dsc)
                        segmentation_metrics[task_id]['hd95'].append(hd95)
                        found = True
                        break
                if found:
                    break
                    

                    
        # 计算每个任务的平均值
        for task_id, metrics in segmentation_metrics.items():
            avg_dsc = np.mean(metrics['dsc'])
            avg_hd95 = np.mean(metrics['hd95'])
            segmentation_metrics[task_id]['avg_dsc'] = avg_dsc
            segmentation_metrics[task_id]['avg_hd95'] = avg_hd95
        
        # 计算所有任务的平均值
        all_dsc = []
        all_hd95 = []
        for metrics in segmentation_metrics.values():
            all_dsc.extend(metrics['dsc'])
            all_hd95.extend(metrics['hd95'])
        
        overall_metrics = {
            'avg_dsc': np.mean(all_dsc) if all_dsc else 0,
            'avg_hd95': np.mean(all_hd95) if all_hd95 else 0,
            'per_task': segmentation_metrics
        }
        
        return overall_metrics
    
    def evaluate_classification(self):
        """
        评估分类任务指标：AUC, F1-score, MCC
        """
        classification_metrics = {}
        
        # 按任务ID分组
        task_results = {}
        for result in self.results['classification']:
            task_id = result['task_id']
            if task_id not in task_results:
                task_results[task_id] = []
            task_results[task_id].append(result)
        
        # 计算每个任务的指标
        for task_id, results in task_results.items():
            y_true = []
            y_pred = []
            y_score = []
            
            for result in results:
                image_path = result['image_path']
                # 查找真实标签
                if task_id in self.ground_truth and image_path in self.ground_truth[task_id]:
                    gt_row = self.ground_truth[task_id][image_path]
                    # 真实标签在'mask'列中
                    if 'mask' in gt_row:
                        y_true.append(int(gt_row['mask']))
                        y_pred.append(result['predicted_class'])
                        y_score.append(result['predicted_probs'])
            
            if len(y_true) > 0:
                # 计算F1-score
                f1 = f1_score(y_true, y_pred, average='macro') if len(set(y_true)) > 1 else 0
                # 计算MCC
                mcc = matthews_corrcoef(y_true, y_pred) if len(set(y_true)) > 1 else 0
                # 计算AUC（需要多分类处理）
                auc = 0
                if len(set(y_true)) > 1:
                    try:
                        # 多分类AUC
                        from sklearn.preprocessing import label_binarize
                        y_true_bin = label_binarize(y_true, classes=list(set(y_true)))
                        if y_true_bin.shape[1] > 1:
                            auc = roc_auc_score(y_true_bin, y_score, average='macro', multi_class='ovo')
                    except:
                        pass
                
                classification_metrics[task_id] = {
                    'auc': auc,
                    'f1_score': f1,
                    'mcc': mcc
                }
        
        # 计算所有任务的平均值
        all_auc = [m['auc'] for m in classification_metrics.values()]
        all_f1 = [m['f1_score'] for m in classification_metrics.values()]
        all_mcc = [m['mcc'] for m in classification_metrics.values()]
        
        overall_metrics = {
            'avg_auc': np.mean(all_auc) if all_auc else 0,
            'avg_f1_score': np.mean(all_f1) if all_f1 else 0,
            'avg_mcc': np.mean(all_mcc) if all_mcc else 0,
            'per_task': classification_metrics
        }
        
        return overall_metrics
    
    def calculate_iou(self, box1, box2):
        """
        计算两个边界框的IoU
        box1和box2的格式：[x1, y1, x2, y2]
        """
        # 计算交集
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        # 计算交集面积
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        
        # 计算并集面积
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        # 计算IoU
        iou = intersection / union if union > 0 else 0
        return iou
    
    def evaluate_detection(self):
        """
        评估检测任务指标：mIoU
        """
        detection_metrics = {}
        
        # 按任务ID分组
        task_results = {}
        for result in self.results['detection']:
            task_id = result['task_id']
            if task_id not in task_results:
                task_results[task_id] = []
            task_results[task_id].append(result)
        
        # 计算每个任务的mIoU
        for task_id, results in task_results.items():
            ious = []
            
            for result in results:
                image_path = result['image_path']
                # 查找真实标签
                if task_id in self.ground_truth and image_path in self.ground_truth[task_id]:
                    gt_row = self.ground_truth[task_id][image_path]
                    # 检查是否有边界框坐标
                    if all(col in gt_row for col in ['x_min', 'y_min', 'x_max', 'y_max']):
                        # 真实边界框
                        gt_box = [
                            gt_row['x_min'],
                            gt_row['y_min'],
                            gt_row['x_max'],
                            gt_row['y_max']
                        ]
                        # 预测边界框
                        pred_box = result['bbox_pixels']
                        # 计算IoU
                        iou = self.calculate_iou(gt_box, pred_box)
                        ious.append(iou)
            
            if ious:
                detection_metrics[task_id] = {
                    'mIoU': np.mean(ious)
                }
        
        # 计算所有任务的平均值
        all_miou = [m['mIoU'] for m in detection_metrics.values()]
        
        overall_metrics = {
            'avg_mIoU': np.mean(all_miou) if all_miou else 0,
            'per_task': detection_metrics
        }
        
        return overall_metrics
    
    def evaluate_regression(self):
        """
        评估回归任务指标：MRE
        """
        regression_metrics = {}
        
        # 按任务ID分组
        task_results = {}
        for result in self.results['regression']:
            task_id = result['task_id']
            if task_id not in task_results:
                task_results[task_id] = []
            task_results[task_id].append(result)
        
        # 计算每个任务的MRE
        for task_id, results in task_results.items():
            mres = []
            
            for result in results:
                image_path = result['image_path']
                # 查找真实标签
                if task_id in self.ground_truth and image_path in self.ground_truth[task_id]:
                    gt_row = self.ground_truth[task_id][image_path]
                    # 检查是否有坐标点
                    if all(col in gt_row for col in ['point_1_xy', 'point_2_xy']):
                        try:
                            # 解析真实点坐标
                            import ast
                            gt_point1 = ast.literal_eval(gt_row['point_1_xy'])
                            gt_point2 = ast.literal_eval(gt_row['point_2_xy'])
                            
                            # 计算真实距离
                            gt_distance = np.sqrt(
                                (gt_point2[0] - gt_point1[0])**2 + 
                                (gt_point2[1] - gt_point1[1])**2
                            )
                            
                            # 预测点坐标
                            pred_points = result['predicted_points_pixels']
                            if len(pred_points) >= 4:
                                pred_point1 = pred_points[:2]
                                pred_point2 = pred_points[2:4]
                                
                                # 计算预测距离
                                pred_distance = np.sqrt(
                                    (pred_point2[0] - pred_point1[0])**2 + 
                                    (pred_point2[1] - pred_point1[1])**2
                                )
                                
                                # 计算MRE（使用绝对误差，单位为像素）
                                mre = abs(pred_distance - gt_distance)
                                mres.append(mre)
                        except:
                            pass
            
            if mres:
                regression_metrics[task_id] = {
                    'mre': np.mean(mres)
                }
        
        # 计算所有任务的平均值
        all_mre = [m['mre'] for m in regression_metrics.values()]
        
        overall_metrics = {
            'avg_mre': np.mean(all_mre) if all_mre else 0,
            'per_task': regression_metrics
        }
        
        return overall_metrics
    
    def evaluate_all(self):
        """
        评估所有任务的指标
        """
        self.load_results()
        self.load_ground_truth()
        
        print("\nEvaluating segmentation...")
        segmentation_metrics = self.evaluate_segmentation()
        
        print("\nEvaluating classification...")
        classification_metrics = self.evaluate_classification()
        
        print("\nEvaluating detection...")
        detection_metrics = self.evaluate_detection()
        
        print("\nEvaluating regression...")
        regression_metrics = self.evaluate_regression()
        
        # 汇总所有指标
        all_metrics = {
            'segmentation': segmentation_metrics,
            'classification': classification_metrics,
            'detection': detection_metrics,
            'regression': regression_metrics
        }
        
        # 保存评估结果
        with open(os.path.join(self.output_dir, 'evaluation_results.json'), 'w', encoding='utf-8') as f:
            json.dump(all_metrics, f, indent=2, ensure_ascii=False)
        
        # 生成结果文本
        result_text = """
================================================================================
Evaluation Results
================================================================================

Segmentation Metrics:
Avg DSC: {:.4f}
Avg HD95: {:.4f}

Classification Metrics:
Avg AUC: {:.4f}
Avg F1-score: {:.4f}
Avg MCC: {:.4f}

Detection Metrics:
Avg mIoU: {:.4f}

Regression Metrics:
Avg MRE: {:.4f}

================================================================================
Evaluation completed! Results saved to evaluation_results.json
""".format(
            segmentation_metrics['avg_dsc'],
            segmentation_metrics['avg_hd95'],
            classification_metrics['avg_auc'],
            classification_metrics['avg_f1_score'],
            classification_metrics['avg_mcc'],
            detection_metrics['avg_mIoU'],
            regression_metrics['avg_mre']
        )
        
        # 打印结果
        print(result_text)
        
        # 保存结果到文本文件
        output_txt_path = os.path.join(self.output_dir, 'evaluate_out.txt')
        with open(output_txt_path, 'w', encoding='utf-8') as f:
            f.write(result_text)
        
        print(f"Results also saved to {output_txt_path}")
        
        return all_metrics

if __name__ == '__main__':
    # 示例用法
    output_dir = 'output'
    csv_dir = 'train/valdataset_8_csv/csv_files'
    
    evaluator = ResultEvaluator(output_dir, csv_dir)
    evaluator.evaluate_all()
