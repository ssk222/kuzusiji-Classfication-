import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageStat
import os
import sys
import cv2
import csv
import numpy as np
import timm
import glob
from collections import defaultdict
import statistics
from tqdm import tqdm
import json
from loguru import logger
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

logger.remove()  # デフォルトのハンドラを削除
logger.add(sys.stderr, level="WARNING")  # WARNING以上のみ表示
# ==========================================
# 1. 設定エリア
# ==========================================
DATA_ROOT_DIR = "/home/ihpc_double/Documents/sasaki/data"
FOLDER_IDS    = ["200003803", "200010454", "200015843", "200017458", 
    "200018243", "200019865", "200021063", "200021071", 
    "200004107", "200005798", "200008003"]  # ← 評価したいフォルダの数字を追加

MODEL_PATH         = "/home/ihpc_double/Documents/sasaki/vgg/output/model_conv_mae2/best_model.pth"
CLASS_JSON         = "trained_class_map.json"
MODEL_NAME         = 'convnextv2_large'
INPUT_SIZE         = 384
FORCE_NUM_CLASSES  = 3962
EMBED_DIM          = 512
ARCFACE_S          = 16.0

# YOLOX設定
try:
    from yolox_detector import get_exp, fuse_model, Predictor, split_image, merge_outputs
    from yolox.data.datasets import COCO_CLASSES
except ImportError:
    print("Error: 'yolox_detector.py' not found."); sys.exit()

YOLOX_CONFIG = {
    "exp_file": "nano.py", "ckpt": "best_ckpt.pth",
    "conf": 0.35, "nms": 0.45, "tsize": 640,
    "device": "gpu", "fp16": True, "legacy": False,
    "fuse": False, "trt": False,
}

# CSV列定義
COL_UNICODE = 0; COL_IMAGE = 1; COL_X = 2; COL_Y = 3; COL_W = 6; COL_H = 7

# ==========================================
# 2. ユーティリティ
# ==========================================
def calculate_iou(box1, box2):
    ix1 = max(box1[0], box2[0]); iy1 = max(box1[1], box2[1])
    ix2 = min(box1[2], box2[2]); iy2 = min(box1[3], box2[3])
    inter_area = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter_area == 0: return 0
    box1_area = (box1[2]-box1[0]) * (box1[3]-box1[1])
    box2_area = (box2[2]-box2[0]) * (box2[3]-box2[1])
    return inter_area / (box1_area + box2_area - inter_area)

def unicode_to_char(u_code):
    if u_code and u_code.startswith("U+"):
        try: return chr(int(u_code.replace("U+", ""), 16))
        except: return u_code
    return u_code

# ==========================================
# 3. ArcFace付きConvNeXtモデル定義
#    学習コードと完全に一致させる
# ==========================================
class ArcFaceHead(nn.Module):
    def __init__(self, in_features, num_classes, s=16.0, m=0.3):
        super().__init__()
        self.s      = s
        self.m      = m
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, features, labels=None):
        features = features.float()
        weight   = self.weight.float()
        cosine   = F.linear(F.normalize(features), F.normalize(weight))
        if labels is None:
            return cosine * self.s
        theta        = torch.acos(cosine.clamp(-1 + 1e-7, 1 - 1e-7))
        target_logit = torch.cos(theta + self.m)
        one_hot      = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)
        output = cosine * (1 - one_hot) + target_logit * one_hot
        return output * self.s


class KuzushijiModel(nn.Module):
    def __init__(self, model_name, num_classes, embed_dim=512):
        super().__init__()
        self.encoder = timm.create_model(
            model_name, pretrained=False,
            num_classes=0, global_pool='avg'
        )
        in_features = self.encoder.num_features
        self.embed = nn.Sequential(
            nn.Linear(in_features, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.arcface = ArcFaceHead(embed_dim, num_classes, s=ARCFACE_S, m=0.3)

    def forward(self, x, labels=None):
        features = self.encoder(x)
        embed    = self.embed(features)
        logits   = self.arcface(embed, labels)
        return logits


# ==========================================
# 4. YOLOXラッパー
# ==========================================
class YOLOXWrapper:
    def __init__(self, config):
        self.exp = get_exp(config["exp_file"], None)
        self.exp.test_conf = config["conf"]
        self.exp.nmsthre   = config["nms"]
        self.exp.test_size = (config["tsize"], config["tsize"])
        self.model = self.exp.get_model()
        if config["device"] == "gpu":
            self.model.cuda()
            if config["fp16"]: self.model.half()
        self.model.eval()
        ckpt = torch.load(config["ckpt"], map_location="cuda", weights_only=False)
        self.model.load_state_dict(ckpt["model"])
        if config["fuse"]: self.model = fuse_model(self.model)
        self.predictor = Predictor(
            self.model, self.exp, COCO_CLASSES, None, None,
            config["device"], config["fp16"], config["legacy"]
        )

    def detect(self, img_path):
        img = cv2.imread(img_path)
        if img is None: return []
        tiles, coords = split_image(img, tile_size=640, stride=320)
        outputs_list = []
        for tile in tiles:
            outputs, _ = self.predictor.inference(tile)
            outputs_list.append(outputs[0] if outputs is not None else None)
        merged = merge_outputs(outputs_list, coords, nms_thresh=self.exp.nmsthre)
        if merged is None: return []

        # 形状チェック（1件の場合に2次元に変換）
        merged_np = merged.cpu().numpy()
        if merged_np.ndim == 1:
            merged_np = merged_np[np.newaxis, :]

        bboxes = []
        for det in merged_np:
            if len(det) < 7: continue
            bboxes.append([det[0], det[1], det[2], det[3],
                           det[4]*det[5], int(det[6])])
        return bboxes


# ==========================================
# 5. ConvNeXt+ArcFace認識クラス
# ==========================================
class ConvNeXtArcFaceRecognizer:
    def __init__(self, model_path, class_json, model_name, num_classes, embed_dim):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        with open(class_json, "r", encoding="utf-8") as f:
            self.class_map = {int(k): v for k, v in json.load(f).items()}

        self.model = KuzushijiModel(model_name, num_classes, embed_dim)

        ckpt = torch.load(model_path, map_location=self.device, weights_only=False)
        state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        self.model.load_state_dict(state_dict)
        self.model.to(self.device).eval()
        print(f"✅ ConvNeXtV2+ArcFace loaded from {model_path}")

        self.transform = transforms.Compose([
            self.SmartPadResize(INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def predict(self, img_pil):
        with torch.no_grad():
            logits = self.model(
                self.transform(img_pil).unsqueeze(0).to(self.device),
                labels=None
            )
            probs = torch.softmax(logits[0], dim=0)
        val, idx = torch.topk(probs, 1)
        char = unicode_to_char(self.class_map.get(idx[0].item(), "?"))
        return char, val[0].item()

    class SmartPadResize:
        def __init__(self, size): self.size = size
        def __call__(self, img):
            try: fill = tuple(int(v) for v in ImageStat.Stat(img).median)
            except: fill = (255, 255, 255)
            w, h = img.size
            r = self.size / max(w, h)
            nw, nh = int(w * r), int(h * r)
            new_img = Image.new("RGB", (self.size, self.size), fill)
            new_img.paste(img.resize((nw, nh), Image.BICUBIC),
                          ((self.size-nw)//2, (self.size-nh)//2))
            return new_img


# ==========================================
# 6. 1枚の画像を評価する関数
# ==========================================
def process_single_image(img_path, yolox, recognizer):
    filename        = os.path.basename(img_path)
    name_without_ext = os.path.splitext(filename)[0]
    book_id         = name_without_ext.split("_")[0]

    # CSVロード
    csv_filename    = f"{book_id}_coordinate.csv"
    coordinate_csv  = os.path.join(DATA_ROOT_DIR, book_id, csv_filename)
    gt_boxes = []

    if os.path.exists(coordinate_csv):
        with open(coordinate_csv, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                try:
                    if (row[COL_IMAGE] not in name_without_ext) and \
                       (name_without_ext not in row[COL_IMAGE]): continue
                    x, y, w, h = int(row[COL_X]), int(row[COL_Y]), \
                                 int(row[COL_W]), int(row[COL_H])
                    if w<=0 or h<=0: continue
                    gt_boxes.append({
                        'box' : [x, y, x+w, y+h],
                        'char': unicode_to_char(row[COL_UNICODE].strip())
                    })
                except: continue

    if not gt_boxes: return None

    # 推論
    det_boxes = yolox.detect(img_path)
    img_pil   = Image.open(img_path).convert("RGB")

    predictions = []
    for box in det_boxes:
        x1, y1, x2, y2 = map(int, box[:4])
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(img_pil.width, x2); y2 = min(img_pil.height, y2)
        if x2-x1 < 5 or y2-y1 < 5: continue
        crop       = img_pil.crop((x1, y1, x2, y2))
        char, prob = recognizer.predict(crop)
        predictions.append({'box': [x1, y1, x2, y2], 'char': char, 'prob': prob})

    # スコア計算
    correct_count       = 0
    matched_pred_indices = set()

    for gt in gt_boxes:
        best_iou     = 0
        best_pred_idx = -1
        for j, pred in enumerate(predictions):
            iou = calculate_iou(gt['box'], pred['box'])
            if iou > 0.5 and iou > best_iou:
                best_iou      = iou
                best_pred_idx = j
        if best_pred_idx != -1:
            matched_pred_indices.add(best_pred_idx)
            if predictions[best_pred_idx]['char'] == gt['char']:
                correct_count += 1

    total_gt    = len(gt_boxes)
    total_preds = len(predictions)
    tp          = correct_count
    fp          = total_preds - len(matched_pred_indices)
    fn          = total_gt - len(matched_pred_indices)

    recall    = (tp / total_gt    * 100) if total_gt    > 0 else 0
    precision = (tp / total_preds * 100) if total_preds > 0 else 0
    f1        = 2 * (precision * recall) / (precision + recall) \
                if (precision + recall) > 0 else 0

    return {
        "filename" : filename,
        "book_id"  : book_id,
        "f1"       : f1,
        "recall"   : recall,
        "precision": precision,
        "tp"       : tp,
        "fp"       : fp,
        "fn"       : fn,
        "gt_count" : total_gt,
        "pred_count": total_preds,
    }


# ==========================================
# 7. メイン処理
# ==========================================
if __name__ == "__main__":
    print("Loading Models...")
    yolox      = YOLOXWrapper(YOLOX_CONFIG)
    recognizer = ConvNeXtArcFaceRecognizer(
        MODEL_PATH, CLASS_JSON, MODEL_NAME,
        FORCE_NUM_CLASSES, EMBED_DIM
    )

    # 指定フォルダの画像を収集
    image_files = []
    for folder_id in FOLDER_IDS:
        images_dir = os.path.join(DATA_ROOT_DIR, folder_id, "images")
        if not os.path.exists(images_dir):
            print(f"Warning: {images_dir} not found. Skipping.")
            continue
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            image_files.extend(glob.glob(os.path.join(images_dir, ext)))

    if not image_files:
        print("No images found."); sys.exit()

    print(f"Found {len(image_files)} images. Starting evaluation...")
    book_stats = defaultdict(list)

    for img_path in tqdm(image_files):
        result = process_single_image(img_path, yolox, recognizer)
        if result:
            book_stats[result["book_id"]].append(result)

    # ==========================================
    # 集計・表示
    # ==========================================
    print("\n" + "="*90)
    print(f"{'Book ID':<20} | {'Count':<5} | {'F1 (Avg)':<10} | {'Recall':<10} | {'Precision':<10} | {'F1 (Min)':<10} | {'F1 (Max)':<10}")
    print("-"*90)

    all_f1        = []
    all_tp        = 0
    all_fp        = 0
    all_fn        = 0
    all_gt        = 0
    all_pred      = 0

    for book_id, results in sorted(book_stats.items()):
        f1_list   = [r["f1"]        for r in results]
        rec_list  = [r["recall"]    for r in results]
        pre_list  = [r["precision"] for r in results]

        avg_f1  = statistics.mean(f1_list)
        avg_rec = statistics.mean(rec_list)
        avg_pre = statistics.mean(pre_list)

        all_f1.extend(f1_list)
        all_tp   += sum(r["tp"]         for r in results)
        all_fp   += sum(r["fp"]         for r in results)
        all_fn   += sum(r["fn"]         for r in results)
        all_gt   += sum(r["gt_count"]   for r in results)
        all_pred += sum(r["pred_count"] for r in results)

        print(f"{book_id:<20} | {len(results):<5} | {avg_f1:.2f}%     | "
              f"{avg_rec:.2f}%     | {avg_pre:.2f}%      | "
              f"{min(f1_list):.2f}%     | {max(f1_list):.2f}%")

    # 全体集計
    overall_precision = (all_tp / all_pred * 100) if all_pred > 0 else 0
    overall_recall    = (all_tp / all_gt   * 100) if all_gt   > 0 else 0
    overall_f1        = 2 * (overall_precision * overall_recall) / \
                        (overall_precision + overall_recall) \
                        if (overall_precision + overall_recall) > 0 else 0

    print("="*90)
    print(f"\n{'全体結果':}")
    print(f"  総画像数    : {len(all_f1)}")
    print(f"  総GT文字数  : {all_gt}")
    print(f"  総予測文字数: {all_pred}")
    print(f"  TP          : {all_tp}")
    print(f"  FP          : {all_fp}")
    print(f"  FN          : {all_fn}")
    print(f"  Precision   : {overall_precision:.2f}%")
    print(f"  Recall      : {overall_recall:.2f}%")
    print(f"  F1 Score    : {overall_f1:.2f}%")
    print(f"  F1 (画像平均): {statistics.mean(all_f1):.2f}%")

    # ワースト5
    all_results    = [r for results in book_stats.values() for r in results]
    sorted_results = sorted(all_results, key=lambda x: x['f1'])
    print(f"\n[最低スコア画像 Top5]")
    for res in sorted_results[:5]:
        print(f"  {res['filename']} : F1={res['f1']:.2f}% "
              f"(Recall={res['recall']:.1f}%, Precision={res['precision']:.1f}%)")
    
    # 現在のワースト5の下に追加
    print(f"\n[最高スコア画像 Top5]")
    for res in sorted_results[-5:][::-1]:  # 末尾5件を逆順（高い順）
        print(f"  {res['filename']} : F1={res['f1']:.2f}% "
              f"(Recall={res['recall']:.1f}%, Precision={res['precision']:.1f}%)")