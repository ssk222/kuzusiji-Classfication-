#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""
YOLOX Model Evaluation Utilities
評価用ユーティリティ関数
"""

import cv2
import numpy as np
import os
import json
import torch
from loguru import logger

__all__ = [
    "vis_evaluation", 
    "match_predictions_for_vis", 
    "calculate_iou", 
    "yolov5_to_xyxy",
    "load_yolov5_annotation",
    "get_yolov5_image_list",
    "evaluate_predictions",
    "EvaluationMetrics"
]

# 色定義
_COLORS = np.array([
    0.000, 0.447, 0.741,
    0.850, 0.325, 0.098,
    0.929, 0.694, 0.125,
    0.494, 0.184, 0.556,
    0.466, 0.674, 0.188,
    0.301, 0.745, 0.933,
    0.635, 0.078, 0.184,
    0.300, 0.300, 0.300,
    0.600, 0.600, 0.600,
    1.000, 0.000, 0.000,
    1.000, 0.500, 0.000,
    0.749, 0.749, 0.000,
    0.000, 1.000, 0.000,
    0.000, 0.000, 1.000,
    0.667, 0.000, 1.000,
    0.333, 0.333, 0.000,
    0.333, 0.667, 0.000,
    0.333, 1.000, 0.000,
    0.667, 0.333, 0.000,
    0.667, 0.667, 0.000,
    0.667, 1.000, 0.000,
    1.000, 0.333, 0.000,
    1.000, 0.667, 0.000,
    1.000, 1.000, 0.000,
    0.000, 0.333, 0.500,
    0.000, 0.667, 0.500,
    0.000, 1.000, 0.500,
    0.333, 0.000, 0.500,
    0.333, 0.333, 0.500,
    0.333, 0.667, 0.500,
    0.333, 1.000, 0.500,
    0.667, 0.000, 0.500,
    0.667, 0.333, 0.500,
    0.667, 0.667, 0.500,
    0.667, 1.000, 0.500,
    1.000, 0.000, 0.500,
    1.000, 0.333, 0.500,
    1.000, 0.667, 0.500,
    1.000, 1.000, 0.500,
    0.000, 0.333, 1.000,
    0.000, 0.667, 1.000,
    0.000, 1.000, 1.000,
    0.333, 0.000, 1.000,
    0.333, 0.333, 1.000,
    0.333, 0.667, 1.000,
    0.333, 1.000, 1.000,
    0.667, 0.000, 1.000,
    0.667, 0.333, 1.000,
    0.667, 0.667, 1.000,
    0.667, 1.000, 1.000,
    1.000, 0.000, 1.000,
    1.000, 0.333, 1.000,
    1.000, 0.667, 1.000,
    0.333, 0.000, 0.000,
    0.500, 0.000, 0.000,
    0.667, 0.000, 0.000,
    0.833, 0.000, 0.000,
    1.000, 0.000, 0.000,
    0.000, 0.167, 0.000,
    0.000, 0.333, 0.000,
    0.000, 0.500, 0.000,
    0.000, 0.667, 0.000,
    0.000, 0.833, 0.000,
    0.000, 1.000, 0.000,
    0.000, 0.000, 0.167,
    0.000, 0.000, 0.333,
    0.000, 0.000, 0.500,
    0.000, 0.000, 0.667,
    0.000, 0.000, 0.833,
    0.000, 0.000, 1.000,
    0.000, 0.000, 0.000,
    0.143, 0.143, 0.143,
    0.286, 0.286, 0.286,
    0.429, 0.429, 0.429,
    0.571, 0.571, 0.571,
    0.714, 0.714, 0.714,
    0.857, 0.857, 0.857,
    0.000, 0.447, 0.741,
    0.314, 0.717, 0.741,
    0.50, 0.5, 0
]).astype(np.float32).reshape(-1, 3)

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]

class EvaluationMetrics:
    """評価指標を管理するクラス"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """統計をリセット"""
        self.total_tp = 0
        self.total_fp = 0
        self.total_fn = 0
        self.total_gt = 0
        self.total_pred = 0
        self.processed_images = 0
        self.per_image_results = []
    
    def update(self, eval_result):
        """画像ごとの評価結果を追加"""
        self.total_tp += eval_result['tp']
        self.total_fp += eval_result['fp']
        self.total_fn += eval_result['fn']
        self.total_gt += eval_result['gt_count']
        self.total_pred += eval_result['pred_count']
        self.processed_images += 1
        self.per_image_results.append(eval_result)
    
    def get_overall_metrics(self):
        """全体の評価指標を計算"""
        precision = self.total_tp / (self.total_tp + self.total_fp) if (self.total_tp + self.total_fp) > 0 else 0.0
        recall = self.total_tp / (self.total_tp + self.total_fn) if (self.total_tp + self.total_fn) > 0 else 0.0
        accuracy = self.total_tp / self.total_gt if self.total_gt > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'processed_images': self.processed_images,
            'total_gt': self.total_gt,
            'total_pred': self.total_pred,
            'total_tp': self.total_tp,
            'total_fp': self.total_fp,
            'total_fn': self.total_fn,
            'precision': precision,
            'recall': recall,
            'accuracy': accuracy,
            'f1_score': f1_score
        }
    
    def print_summary(self, iou_thresh=0.5):
        """評価結果のサマリーを表示"""
        metrics = self.get_overall_metrics()
        
        logger.info("\n" + "="*60)
        logger.info("EVALUATION RESULTS")
        logger.info("="*60)
        logger.info(f"Total Images Processed: {metrics['processed_images']}")
        logger.info(f"Total Ground Truth Boxes: {metrics['total_gt']}")
        logger.info(f"Total Predicted Boxes: {metrics['total_pred']}")
        logger.info(f"True Positives: {metrics['total_tp']}")
        logger.info(f"False Positives: {metrics['total_fp']}")
        logger.info(f"False Negatives: {metrics['total_fn']}")
        logger.info("-"*60)
        logger.info(f"Overall Precision: {metrics['precision']:.4f}")
        logger.info(f"Overall Recall: {metrics['recall']:.4f}")
        logger.info(f"Overall Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"F1 Score: {metrics['f1_score']:.4f}")
        logger.info(f"IoU Threshold: {iou_thresh}")
        logger.info("="*60)
    
    def save_to_json(self, save_path, iou_thresh=0.5):
        """評価結果をJSONファイルに保存"""
        overall_metrics = self.get_overall_metrics()
        overall_metrics['iou_threshold'] = iou_thresh
        
        evaluation_summary = {
            'overall_stats': overall_metrics,
            'per_image_results': self.per_image_results
        }
        
        with open(save_path, 'w') as f:
            json.dump(evaluation_summary, f, indent=2)
        logger.info(f"Evaluation results saved to: {save_path}")

def get_yolov5_image_list(dataset_path):
    """YOLOV5フォルダ構造から画像リストを取得"""
    images_folder = os.path.join(dataset_path, "images")
    if not os.path.exists(images_folder):
        raise ValueError(f"Images folder not found: {images_folder}")
    
    image_names = []
    for filename in os.listdir(images_folder):
        if any(filename.lower().endswith(ext) for ext in IMAGE_EXT):
            image_path = os.path.join(images_folder, filename)
            image_names.append(image_path)
    return sorted(image_names)

def load_yolov5_annotation(image_path, dataset_path):
    """YOLOV5形式のアノテーションを読み込み"""
    basename = os.path.splitext(os.path.basename(image_path))[0]
    label_path = os.path.join(dataset_path, "labels", f"{basename}.txt")
    
    annotations = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        annotations.append({
                            'class_id': class_id,
                            'x_center': x_center,
                            'y_center': y_center,
                            'width': width,
                            'height': height
                        })
    return annotations

def yolov5_to_xyxy(annotation, img_width, img_height):
    """YOLOV5形式(正規化済み)をxyxy形式に変換"""
    x_center = annotation['x_center'] * img_width
    y_center = annotation['y_center'] * img_height
    width = annotation['width'] * img_width
    height = annotation['height'] * img_height
    
    x1 = x_center - width / 2
    y1 = y_center - height / 2
    x2 = x_center + width / 2
    y2 = y_center + height / 2
    
    return [x1, y1, x2, y2]

def calculate_iou(box1, box2):
    """IoU計算"""
    x1, y1, x2, y2 = box1
    x1_gt, y1_gt, x2_gt, y2_gt = box2
    
    # 交差領域
    xi1 = max(x1, x1_gt)
    yi1 = max(y1, y1_gt)
    xi2 = min(x2, x2_gt)
    yi2 = min(y2, y2_gt)
    
    if xi2 <= xi1 or yi2 <= yi1:
        return 0.0
    
    intersection = (xi2 - xi1) * (yi2 - yi1)
    
    # 各ボックスの面積
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_gt - x1_gt) * (y2_gt - y1_gt)
    
    # 和集合
    union = box1_area + box2_area - intersection
    
    return intersection / union if union > 0 else 0.0

def match_predictions_for_vis(pred_boxes, gt_boxes, iou_thresh=0.5):
    """可視化用の予測-正解マッチング"""
    if pred_boxes is None or len(pred_boxes) == 0:
        return []
    
    matched_pairs = []
    gt_matched = [False] * len(gt_boxes)
    
    for pred_idx, pred_box in enumerate(pred_boxes):
        best_iou = 0.0
        best_gt_idx = -1
        
        for gt_idx, gt_box in enumerate(gt_boxes):
            if not gt_matched[gt_idx]:
                iou = calculate_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
        
        if best_iou >= iou_thresh and best_gt_idx != -1:
            matched_pairs.append((pred_idx, best_gt_idx))
            gt_matched[best_gt_idx] = True
    
    return matched_pairs

def evaluate_predictions(predictions, ground_truths, img_width, img_height, iou_thresh=0.5, target_class_id=0):
    """予測結果と正解アノテーションを比較評価"""
    if predictions is None or len(predictions) == 0:
        gt_count = len([gt for gt in ground_truths if gt['class_id'] == target_class_id])
        return {
            'tp': 0,
            'fp': 0,
            'fn': gt_count,
            'precision': 0.0,
            'recall': 0.0,
            'accuracy': 0.0,
            'gt_count': gt_count,
            'pred_count': 0
        }
    
    # 正解ボックスをxyxy形式に変換
    gt_boxes = []
    for gt in ground_truths:
        if gt['class_id'] == target_class_id:  # 指定クラスのみ
            gt_box = yolov5_to_xyxy(gt, img_width, img_height)
            gt_boxes.append(gt_box)
    
    if len(gt_boxes) == 0:
        return {
            'tp': 0,
            'fp': len(predictions),
            'fn': 0,
            'precision': 0.0,
            'recall': 0.0,
            'accuracy': 0.0,
            'gt_count': 0,
            'pred_count': len(predictions)
        }
    
    # 予測ボックス
    if torch.is_tensor(predictions):
        pred_boxes = predictions[:, :4].cpu().numpy()
    else:
        pred_boxes = predictions[:, :4]
    
    # マッチング
    gt_matched = [False] * len(gt_boxes)
    tp = 0
    fp = 0
    
    for pred_box in pred_boxes:
        best_iou = 0.0
        best_gt_idx = -1
        
        for gt_idx, gt_box in enumerate(gt_boxes):
            if not gt_matched[gt_idx]:
                iou = calculate_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
        
        if best_iou >= iou_thresh and best_gt_idx != -1:
            tp += 1
            gt_matched[best_gt_idx] = True
        else:
            fp += 1
    
    fn = len(gt_boxes) - tp
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    accuracy = tp / len(gt_boxes) if len(gt_boxes) > 0 else 0.0
    
    return {
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'precision': precision,
        'recall': recall,
        'accuracy': accuracy,
        'gt_count': len(gt_boxes),
        'pred_count': len(pred_boxes)
    }

def vis_evaluation(img, pred_boxes, pred_scores, pred_cls_ids, gt_boxes, 
                  conf=0.35, class_names=None, matched_pairs=None, 
                  iou_thresh=0.5, save_path=None):
    """
    評価用可視化関数: 正解bbox（緑）と予測bbox（赤/青）を同時に表示
    
    Args:
        img: 入力画像
        pred_boxes: 予測bbox [N, 4] (x1, y1, x2, y2)
        pred_scores: 予測スコア [N]
        pred_cls_ids: 予測クラスID [N]
        gt_boxes: 正解bbox リスト, 各要素は [x1, y1, x2, y2]
        conf: 信頼度閾値
        class_names: クラス名リスト
        matched_pairs: マッチしたペアのインデックス [(pred_idx, gt_idx), ...]
        iou_thresh: IoU閾値
        save_path: 保存パス
    
    Returns:
        可視化済み画像
    """
    result_img = img.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # 統計情報
    total_gt = len(gt_boxes)
    total_pred = 0
    tp_count = 0
    
    # 1. 正解bboxを緑色で描画
    for i, gt_box in enumerate(gt_boxes):
        x1, y1, x2, y2 = map(int, gt_box)
        # 緑色の太い線で正解bbox
        cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        # GT ラベル
        gt_text = f"GT_{i}"
        gt_txt_size = cv2.getTextSize(gt_text, font, 0.5, 1)[0]
        cv2.rectangle(result_img, (x1, y1-25), (x1 + gt_txt_size[0] + 5, y1), (0, 255, 0), -1)
        cv2.putText(result_img, gt_text, (x1+2, y1-5), font, 0.5, (0, 0, 0), 1)
    
    # 2. 予測bboxを赤色/青色で描画
    if pred_boxes is not None and len(pred_boxes) > 0:
        for i, (box, score, cls_id) in enumerate(zip(pred_boxes, pred_scores, pred_cls_ids)):
            if score < conf:
                continue
                
            total_pred += 1
            x1, y1, x2, y2 = map(int, box)
            
            # マッチしたかどうかを確認
            is_matched = False
            matched_gt_idx = -1
            if matched_pairs:
                for pred_idx, gt_idx in matched_pairs:
                    if pred_idx == i:
                        is_matched = True
                        matched_gt_idx = gt_idx
                        tp_count += 1
                        break
            
            # 色を決定: マッチした場合は青、しなかった場合は赤
            bbox_color = (255, 0, 0) if not is_matched else (0, 0, 255)  # 赤 or 青
            
            # 予測bbox描画
            cv2.rectangle(result_img, (x1, y1), (x2, y2), bbox_color, 2)
            
            # 予測ラベル
            # class_name = class_names[cls_id] if class_names and cls_id < len(class_names) else f"class_{cls_id}"
            # pred_text = f"{class_name}:{score*100:.1f}%"
            # if is_matched:
            #     pred_text += f" (GT_{matched_gt_idx})"
            
            # pred_txt_size = cv2.getTextSize(pred_text, font, 0.4, 1)[0]
            
            # # テキスト背景
            # txt_bg_color = (128, 0, 0) if not is_matched else (0, 0, 128)
            # cv2.rectangle(result_img, 
            #              (x1, y2), 
            #              (x1 + pred_txt_size[0] + 5, y2 + pred_txt_size[1] + 8), 
            #              txt_bg_color, -1)
            
            # # テキスト描画
            # cv2.putText(result_img, pred_text, (x1+2, y2 + pred_txt_size[1] + 3), 
            #            font, 0.4, (255, 255, 255), 1)
    
    # 3. 統計情報を画像上部に描画
    fp_count = total_pred - tp_count
    fn_count = total_gt - tp_count
    
    precision = tp_count / total_pred if total_pred > 0 else 0.0
    recall = tp_count / total_gt if total_gt > 0 else 0.0
    accuracy = tp_count / total_gt if total_gt > 0 else 0.0
    
    # 統計テキスト
    stats_lines = [
        f"GT: {total_gt}, Pred: {total_pred}",
        f"TP: {tp_count}, FP: {fp_count}, FN: {fn_count}",
        f"Precision: {precision:.3f}, Recall: {recall:.3f}",
        f"Accuracy: {accuracy:.3f} (IoU>={iou_thresh})"
    ]
    
    # 統計情報の背景
    max_text_width = max([cv2.getTextSize(line, font, 0.6, 2)[0][0] for line in stats_lines])
    bg_height = len(stats_lines) * 25 + 10
    cv2.rectangle(result_img, (10, 10), (max_text_width + 20, bg_height), (0, 0, 0), -1)
    
    # 統計テキスト描画
    for i, line in enumerate(stats_lines):
        y_pos = 30 + i * 25
        cv2.putText(result_img, line, (15, y_pos), font, 0.6, (255, 255, 255), 2)
    
    # 4. 凡例
    legend_y_start = bg_height + 30
    legend_items = [
        ("GT (Ground Truth)", (0, 255, 0)),
        ("TP (True Positive)", (0, 0, 255)),
        ("FP (False Positive)", (255, 0, 0))
    ]
    
    for i, (label, color) in enumerate(legend_items):
        y_pos = legend_y_start + i * 25
        # 色見本の矩形
        cv2.rectangle(result_img, (15, y_pos-15), (35, y_pos-5), color, -1)
        cv2.rectangle(result_img, (15, y_pos-15), (35, y_pos-5), (255, 255, 255), 1)
        # ラベル
        cv2.putText(result_img, label, (45, y_pos-8), font, 0.5, (255, 255, 255), 1)
    
    # 5. 保存
    if save_path:
        cv2.imwrite(save_path, result_img)
        logger.info(f"Evaluation visualization saved: {save_path}")
    
    return result_img