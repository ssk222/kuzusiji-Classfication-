#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import cv2
import numpy as np

__all__ = ["vis"]


import os
import logging
def vis(img, boxes, scores, cls_ids, conf=None, class_names=None, save_crops=True, save_dir=None, image_name=None):
    # 元の画像をコピー（切り取り用）
    original_img = img.copy()
    
    # 切り取り保存用のディレクトリを作成
    if save_crops and save_dir:
        crops_dir = os.path.join(save_dir, "crops")
        os.makedirs(crops_dir, exist_ok=True)
    
    crop_count = 0
    
    for i in range(len(boxes)):
        box = boxes[i]
        cls_id = int(cls_ids[i])
        score = scores[i]
        if score < conf:
            continue
        x0 = int(box[0])
        y0 = int(box[1])
        x1 = int(box[2])
        y1 = int(box[3])
        
        # 最初に切り取り画像を保存（描画前の元画像から）
        if save_crops and save_dir:
            # 座標の境界チェックと正規化
            img_height, img_width = original_img.shape[:2]
            
            x0_crop = max(0, min(x0, img_width - 1))
            y0_crop = max(0, min(y0, img_height - 1))
            x1_crop = max(x0_crop + 1, min(x1, img_width))
            y1_crop = max(y0_crop + 1, min(y1, img_height))
            
            # 切り取り領域のサイズチェック
            crop_width = x1_crop - x0_crop
            crop_height = y1_crop - y0_crop
            
            if crop_width > 0 and crop_height > 0:
                # 切り取り（元の画像から）
                crop = original_img[y0_crop:y1_crop, x0_crop:x1_crop]
                
                # 切り取り画像が空でないかチェック
                if crop.size > 0:
                    # ファイル名の生成
                    if image_name:
                        base_name = os.path.splitext(os.path.basename(image_name))[0]
                    else:
                        base_name = "image"
                    
                    class_name = class_names[cls_id] if class_names else f"class_{cls_id}"
                    crop_filename = f"{base_name}_{class_name}_{crop_count:03d}.jpg"
                    crop_path = os.path.join(crops_dir, crop_filename)
                    
                    # 保存
                    try:
                        cv2.imwrite(crop_path, crop)
                        crop_count += 1
                        print(f"Saved crop: {crop_path} (size: {crop.shape})")
                    except Exception as e:
                        print(f"Error saving crop {crop_path}: {e}")
                        print(f"Crop shape: {crop.shape}, bbox: ({x0}, {y0}, {x1}, {y1})")
                else:
                    print(f"Empty crop detected: bbox ({x0}, {y0}, {x1}, {y1})")
            else:
                print(f"Invalid crop size: width={crop_width}, height={crop_height}")

        # 次にbboxの描画（imgに直接描画）
        color = (_COLORS[cls_id] * 255).astype(np.uint8).tolist()
        text = '{}:{:.1f}%'.format(class_names[cls_id], score * 100)
        txt_color = (0, 0, 0) if np.mean(_COLORS[cls_id]) > 0.5 else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX

        txt_size = cv2.getTextSize(text, font, 0.3, 1)[0]
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

        txt_bk_color = (_COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
        cv2.rectangle(
            img,
            (x0, y0 + 1),
            (x0 + txt_size[0] + 1, y0 + int(1.5*txt_size[1])),
            txt_bk_color,
            -1
        )
        cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.3, txt_color, thickness=1)

    return img






###正解と予測の可視化関数
# def vis(img, boxes, scores, cls_ids, conf=0.5, class_names=None):
#     used_scores = []  # ★ スコア記録リスト

#     for i in range(len(boxes)):
#         box = boxes[i]
#         cls_id = int(cls_ids[i])
#         score = scores[i]
#         if score < conf:
#             continue
#         used_scores.append(score)  # ★ スコアを記録

#         x0 = int(box[0])
#         y0 = int(box[1])
#         x1 = int(box[2])
#         y1 = int(box[3])

#         color = (_COLORS[cls_id] * 255).astype(np.uint8).tolist()
#         text = '{}:{:.1f}%'.format(class_names[cls_id], score * 100)
#         txt_color = (0, 0, 0) if np.mean(_COLORS[cls_id]) > 0.5 else (255, 255, 255)
#         font = cv2.FONT_HERSHEY_SIMPLEX

#         txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
#         cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

#         txt_bk_color = (_COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
#         cv2.rectangle(
#             img,
#             (x0, y0 + 1),
#             (x0 + txt_size[0] + 1, y0 + int(1.5 * txt_size[1])),
#             txt_bk_color,
#             -1
#         )
#         cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)

#     # ★ 平均スコアの表示
#     if used_scores:
#         mean_score = np.mean(used_scores)
#         avg_text = f"Avg Score: {mean_score * 100:.2f}%"
#         cv2.putText(img, avg_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

#     return img

# def vis(img, boxes, scores, cls_ids, conf=0.5, class_names=None):
#     cropped_images = []  # クロップした画像を保存するリスト
#     for i in range(len(boxes)):
#         box = boxes[i]
#         cls_id = int(cls_ids[i])
#         score = scores[i]
#         if score < conf:
#             continue

#         img_height, img_width = img.shape[:2]

#         # 左上と右下の座標を取得
#         x0, y0, x1, y1 = map(int, box)
        
#         # バウンディングボックスの中点を取得
#         g_x, g_y = (x0 + x1) // 2, (y0 + y1) // 2
#         width, height = x1 - x0, y1 - y0
#         max_length = max(width, height)

#         # 新しい正方形のバウンディングボックスの範囲を計算
#         x0_new = g_x - max_length // 2
#         y0_new = g_y - max_length // 2
#         x1_new = g_x + max_length // 2
#         y1_new = g_y + max_length // 2

#         # パディングの計算
#         pad_x0 = max(0, -x0_new)
#         pad_y0 = max(0, -y0_new)
#         pad_x1 = max(0, x1_new - img_width)
#         pad_y1 = max(0, y1_new - img_height)

#         # バウンディングボックスの修正
#         x0_new = max(0, x0_new)
#         y0_new = max(0, y0_new)
#         x1_new = min(img_width, x1_new)
#         y1_new = min(img_height, y1_new)

#         # クロップした領域を取得
#         cropped = img[y0_new:y1_new, x0_new:x1_new]

#         # 必要に応じてパディングを追加
#         cropped_padded = cv2.copyMakeBorder(
#             cropped, pad_y0, pad_y1, pad_x0, pad_x1, 
#             borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0)
#         )
        
#         # 正方形のサイズを保証
#         # cropped_images.append(cropped_padded)

#     return cropped_images


_COLORS = np.array(
    [
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
    ]
).astype(np.float32).reshape(-1, 3)
