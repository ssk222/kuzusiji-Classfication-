print(">>> [1] プログラム起動: 最終結果テーブル表示機能付き")
import os
import sys

# ★★★ GPU設定 ★★★
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
import torch.nn as nn
from torchvision.models import efficientnet_v2_l
from torchvision import transforms
from PIL import Image, ImageStat
import json
import cv2
import csv
import numpy as np
import timm 
import glob
import gc 
import statistics
from collections import defaultdict

# 1. 設定エリア
# 評価対象のBook IDリスト
TARGET_BOOK_IDS = [
    "200003803", "200010454", "200015843", "200017458", 
    "200018243", "200019865", "200021063", "200021071", 
    "200004107", "200005798", "200008003"
]

DATA_ROOT_DIR = "../data" # 画像・CSVの親フォルダ
CLASS_JSON = "trained_class_map.json"
WEIGHT_EFF  = 0.5
WEIGHT_CONV = 0.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# モデルパス
EFF_PATH = "/home/ihpc_double/Documents/sasaki/vgg/output/model_Effi/best_model.pth"
EFF_SIZE = 384
FORCE_CLASSES_EFF = 4329

CONV_PATH = "/home/ihpc_double/Documents/sasaki/vgg/output/model_conv2/best_model.pth"
CONV_NAME = 'convnextv2_large'
CONV_SIZE = 384
FORCE_CLASSES_CONV = 3962

# YOLOX設定
YOLOX_EXP_PATH = "nano.py"; YOLOX_CKPT_PATH = "best_ckpt.pth"
YOLOX_CONF = 0.35; YOLOX_NMS = 0.45; YOLOX_SIZE = 640

# CSVカラム
COL_UNICODE = 0; COL_IMAGE = 1; COL_X = 2; COL_Y = 3; COL_W = 6; COL_H = 7

# 2. 関数・クラス定義
def calculate_iou(box1, box2):
    ix1 = max(box1[0], box2[0]); iy1 = max(box1[1], box2[1])
    ix2 = min(box1[2], box2[2]); iy2 = min(box1[3], box2[3])
    inter_area = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter_area == 0: return 0
    box1_area = (box1[2]-box1[0])*(box1[3]-box1[1])
    box2_area = (box2[2]-box2[0])*(box2[3]-box2[1])
    return inter_area / (box1_area + box2_area - inter_area)

def unicode_to_char(u): return chr(int(u.replace("U+", ""), 16)) if u.startswith("U+") else u

# --- YOLOX ---
try: from yolox_detector import get_exp, Predictor
except: print("[ERROR] yolox_detector.py が見つかりません"); sys.exit()

def load_yolox():
    exp = get_exp(YOLOX_EXP_PATH, None); exp.test_conf=YOLOX_CONF; exp.nmsthre=YOLOX_NMS; exp.test_size=(YOLOX_SIZE, YOLOX_SIZE)
    m = exp.get_model(); m.to(DEVICE); m.half(); m.eval()
    m.load_state_dict(torch.load(YOLOX_CKPT_PATH, map_location=DEVICE)["model"])
    predictor = Predictor(m, exp, None, None, None, "gpu", True, False)
    return predictor, exp

def run_detector(predictor, exp, img_path):
    img = cv2.imread(img_path)
    if img is None: return [], None
    from yolox_detector import split_image, merge_outputs
    ts, cs = split_image(img, 640, 320)
    outs = []
    for t in ts:
        outputs, _ = predictor.inference(t)
        outs.append(outputs[0] if outputs is not None else None)
    merged = merge_outputs(outs, cs, exp.nmsthre)
    if merged is None: return [], img
    res = merged.cpu().numpy()
    if res.ndim == 1: res = res[np.newaxis, :]
    return res, img

# --- Recognition ---
class SmartPadResize:
    def __init__(self, s): self.s = s
    def __call__(self, img):
        try: fill = tuple(ImageStat.Stat(img).median)
        except: fill = (255, 255, 255)
        w, h = img.size; r = self.s/max(w,h); nw, nh = int(w*r), int(h*r)
        img = img.resize((nw,nh), Image.BICUBIC)
        new = Image.new("RGB", (self.s, self.s), fill); new.paste(img, ((self.s-nw)//2, (self.s-nh)//2))
        return new

def load_recognition_model(name):
    if name=='eff':
        m=efficientnet_v2_l(); m.classifier[1]=nn.Linear(m.classifier[1].in_features, FORCE_CLASSES_EFF); p=EFF_PATH; s=EFF_SIZE
    else:
        m=timm.create_model(CONV_NAME, num_classes=FORCE_CLASSES_CONV); p=CONV_PATH; s=CONV_SIZE
    ckpt = torch.load(p, map_location=DEVICE)
    if isinstance(ckpt, dict) and "state_dict" in ckpt: sd = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and "model" in ckpt: sd = ckpt["model"]
    else: sd = ckpt
    m.load_state_dict(sd); m.to(DEVICE).eval()
    return m, s

def run_recognizer_batch(model, size, crops):
    t = transforms.Compose([SmartPadResize(size), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    res = []
    BATCH_SIZE = 32 
    with torch.no_grad():
        for i in range(0, len(crops), BATCH_SIZE):
            batch_crops = crops[i : i+BATCH_SIZE]
            tensors = [t(c) for c in batch_crops]
            input_tensor = torch.stack(tensors).to(DEVICE)
            output = model(input_tensor)
            probs = torch.nn.functional.softmax(output, dim=1)
            res.extend(probs.cpu().numpy())
    return np.array(res)

# ★★★ 画像1枚の処理 ★★★
def process_single_image(img_path, book_gt_data, models, class_map):
    yolo_p, yolo_e, mod_eff, sz_eff, mod_conv, sz_conv = models
    filename = os.path.basename(img_path)
    name_without_ext = os.path.splitext(filename)[0]

    # GT検索
    current_gt = []
    if name_without_ext in book_gt_data:
        current_gt = book_gt_data[name_without_ext]
    else:
        for k, v in book_gt_data.items():
            if k == name_without_ext: current_gt = v; break
    
    if len(current_gt) == 0: return None # GTなし

    # 推論
    det, img_cv2 = run_detector(yolo_p, yolo_e, img_path)
    predictions = []
    
    if len(det) > 0:
        img_pil = Image.fromarray(cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB))
        crops = []; boxes = []
        for d in det:
            x1,y1,x2,y2 = map(int, d[:4])
            if x2-x1<5 or y2-y1<5: continue
            crops.append(img_pil.crop((x1,y1,x2,y2)))
            boxes.append([x1,y1,x2,y2])

        if len(crops) > 0:
            probs_eff = run_recognizer_batch(mod_eff, sz_eff, crops)
            probs_conv = run_recognizer_batch(mod_conv, sz_conv, crops)
            
            for i in range(len(crops)):
                pe = probs_eff[i]; pc = probs_conv[i]
                if len(pc) < len(pe): pc = np.pad(pc, (0, len(pe)-len(pc)))
                elif len(pe) < len(pc): pe = np.pad(pe, (0, len(pc)-len(pe)))
                
                ens = (pe * WEIGHT_EFF) + (pc * WEIGHT_CONV)
                top_idx = np.argmax(ens)
                char = unicode_to_char(class_map.get(top_idx, "?"))
                predictions.append({'box': boxes[i], 'char': char})

    # スコア計算
    correct_count = 0
    for gt in current_gt:
        best_iou = 0; best_pred_idx = -1
        for j, pred in enumerate(predictions):
            iou = calculate_iou(gt['box'], pred['box'])
            if iou > 0.1 and iou > best_iou:
                best_iou = iou; best_pred_idx = j
        if best_pred_idx != -1 and predictions[best_pred_idx]['char'] == gt['char']:
            correct_count += 1
    
    total_gt = len(current_gt)
    total_preds = len(predictions)
    
    recall = (correct_count / total_gt) * 100 if total_gt > 0 else 0
    precision = (correct_count / total_preds) * 100 if total_preds > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "filename": filename,
        "f1": f1,
        "recall": recall,
        "precision": precision
    }

# メイン処理
if __name__ == "__main__":
    print(f">>> Weights: Eff={WEIGHT_EFF}, Conv={WEIGHT_CONV}")

    print(">>> Loading Models...")
    with open(CLASS_JSON) as f: cm = {int(k): v for k,v in json.load(f).items()}
    yolo_p, yolo_e = load_yolox()
    mod_eff, sz_eff = load_recognition_model('eff')
    mod_conv, sz_conv = load_recognition_model('conv')
    models = (yolo_p, yolo_e, mod_eff, sz_eff, mod_conv, sz_conv)
    print(">>> All Models Loaded.\n")

    book_stats = defaultdict(list)
    
    print("Processing Books...")

    # 書籍ごとにループ
    for book_id in TARGET_BOOK_IDS:
        print(f" -> Processing {book_id} ...")
        
        search_pattern = os.path.join(DATA_ROOT_DIR, book_id, "images", "*.jpg")
        image_files = sorted(glob.glob(search_pattern))
        
        if not image_files:
            print(f"    (No images found in {search_pattern})")
            continue

        # GTロード
        csv_path = os.path.join(DATA_ROOT_DIR, book_id, f"{book_id}_coordinate.csv")
        book_gt_data = {}
        if os.path.exists(csv_path):
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                for row in reader:
                    try:
                        fname = row[COL_IMAGE].strip()
                        if fname not in book_gt_data: book_gt_data[fname] = []
                        x, y, w, h = int(row[COL_X]), int(row[COL_Y]), int(row[COL_W]), int(row[COL_H])
                        if w<=0 or h<=0: continue
                        book_gt_data[fname].append({
                            'box': [x, y, x+w, y+h], 
                            'char': unicode_to_char(row[COL_UNICODE].strip())
                        })
                    except: pass
        
        # 画像ループ
        results = []
        for img_path in image_files:
            res = process_single_image(img_path, book_gt_data, models, cm)
            if res:
                results.append(res)
        
        book_stats[book_id] = results
        print(f"    Done. ({len(results)} evaluated)")


    # ★★★ 最終結果テーブル表示 ★★★

    print("\n" + "="*90)
    print(f" FINAL RESULT TABLE (Ensemble)")
    print("="*90)
    print(f"{'Book ID':<15} | {'Count':<6} | {'F1 (Avg)':<10} | {'Recall':<10} | {'Precision':<10} | {'Min F1':<8} | {'Max F1':<8}")
    print("-" * 90)
    
    all_f1_total = []

    # BookID順に表示
    for book_id in TARGET_BOOK_IDS:
        results = book_stats.get(book_id, [])
        if not results:
            print(f"{book_id:<15} | {'0':<6} | {'-':<10} | {'-':<10} | {'-':<10} | {'-':<8} | {'-':<8}")
            continue

        f1_list = [r["f1"] for r in results]
        rec_list = [r["recall"] for r in results]
        pre_list = [r["precision"] for r in results]

        avg_f1 = statistics.mean(f1_list)
        avg_rec = statistics.mean(rec_list)
        avg_pre = statistics.mean(pre_list)
        min_f1 = min(f1_list)
        max_f1 = max(f1_list)
        
        all_f1_total.extend(f1_list)
        
        print(f"{book_id:<15} | {len(results):<6} | {avg_f1:>6.2f}%    | {avg_rec:>6.2f}%    | {avg_pre:>6.2f}%    | {min_f1:>6.2f}%  | {max_f1:>6.2f}%")

    print("="*90)
    if all_f1_total:
        print(f" OVERALL AVERAGE F1 ({len(all_f1_total)} images): {statistics.mean(all_f1_total):.2f}%")
    print("="*90)