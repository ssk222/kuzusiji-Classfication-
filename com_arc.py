import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import efficientnet_v2_l
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
import builtins
import warnings

# ログ・print抑制
warnings.filterwarnings("ignore")
logger.remove()
logger.add(sys.stderr, level="WARNING")

_original_print = builtins.print
def _silent_print(*args, **kwargs):
    msg = " ".join(str(a) for a in args)
    if any(x in msg for x in ["Added padding", "Split into", "Infer time"]):
        return
    _original_print(*args, **kwargs)
builtins.print = _silent_print

# ==========================================
# 1. 設定エリア
# ==========================================
DATA_ROOT_DIR = "/home/ihpc_double/Documents/sasaki/data"
TARGET_IMAGE = "/home/ihpc_double/Documents/sasaki/data/200010454/images/200010454_00016_1.jpg"

# ConvNeXtV2+ArcFace
CONV_MODEL_PATH   = "/home/ihpc_double/Documents/sasaki/vgg/output/model_conv_mae/best_model.pth"
CONV_MODEL_NAME   = 'convnextv2_large'
CONV_NUM_CLASSES  = 3962
CONV_EMBED_DIM    = 512
CONV_INPUT_SIZE   = 384

# EfficientNetV2+ArcFace
EFFI_MODEL_PATH   = "/home/ihpc_double/Documents/sasaki/vgg/output/model_effi_arc/best_model.pth"
EFFI_NUM_CLASSES  = 3962
EFFI_EMBED_DIM    = 512
EFFI_INPUT_SIZE   = 384

# 共通設定
ARCFACE_S  = 16.0
ARCFACE_M  = 0.3
IOU_THRESH = 0.5

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

class SmartPadResize:
    def __init__(self, target_size): self.target_size = target_size
    def __call__(self, img):
        try: fill = tuple(int(v) for v in ImageStat.Stat(img).median)
        except: fill = (255, 255, 255)
        w, h = img.size
        ratio = self.target_size / max(w, h)
        new_w, new_h = int(w * ratio), int(h * ratio)
        img = img.resize((new_w, new_h), Image.BICUBIC)
        new_img = Image.new("RGB", (self.target_size, self.target_size), fill)
        new_img.paste(img, ((self.target_size-new_w)//2,
                            (self.target_size-new_h)//2))
        return new_img

# ==========================================
# 3. ArcFace Head（共通）
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

# ==========================================
# 4. ConvNeXtV2+ArcFaceモデル
# ==========================================
class ConvNeXtArcFaceModel(nn.Module):
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
        self.arcface = ArcFaceHead(embed_dim, num_classes,
                                   s=ARCFACE_S, m=ARCFACE_M)

    def forward(self, x, labels=None):
        features = self.encoder(x)
        embed    = self.embed(features)
        logits   = self.arcface(embed, labels)
        return logits

# ==========================================
# 5. EfficientNetV2+ArcFaceモデル
# ==========================================
class EfficientNetArcFaceModel(nn.Module):
    def __init__(self, num_classes, embed_dim=512):
        super().__init__()
        # EfficientNetV2-Lのエンコーダ（分類頭なし）
        base = efficientnet_v2_l(weights='IMAGENET1K_V1')
        # 分類頭を除いた特徴抽出部分
        self.features = base.features   # 特徴抽出
        self.avgpool  = base.avgpool 
        in_features  = base.classifier[1].in_features  # 1280

        self.embed = nn.Sequential(
            nn.Linear(in_features, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.arcface = ArcFaceHead(embed_dim, num_classes,
                                   s=ARCFACE_S, m=ARCFACE_M)

    def forward(self, x, labels=None):
        features = self.features(x)        # (B, 1280, H, W)
        features = self.avgpool(features)  # (B, 1280, 1, 1)
        features = features.flatten(1)     # (B, 1280)
        embed    = self.embed(features)
        logits   = self.arcface(embed, labels)
        return logits

# ==========================================
# 6. YOLOXラッパー
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
# 7. 認識クラス（共通インターフェース）
# ==========================================
class ArcFaceRecognizer:
    def __init__(self, model, class_json, input_size, device):
        self.model   = model
        self.device  = device
        self.model.to(device).eval()
        with open(class_json, "r", encoding="utf-8") as f:
            self.class_map = {int(k): v for k, v in json.load(f).items()}
        self.transform = transforms.Compose([
            SmartPadResize(input_size),
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

# ==========================================
# 8. 1枚の画像を評価する関数
# ==========================================
def process_single_image(img_path, yolox, recognizer_conv, recognizer_effi):
    filename         = os.path.basename(img_path)
    name_without_ext = os.path.splitext(filename)[0]
    book_id          = name_without_ext.split("_")[0]

    # GTロード
    csv_filename   = f"{book_id}_coordinate.csv"
    coordinate_csv = os.path.join(DATA_ROOT_DIR, book_id, csv_filename)
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

    # 推論（YOLOXは1回だけ）
    det_boxes = yolox.detect(img_path)
    img_pil   = Image.open(img_path).convert("RGB")

    predictions = []
    for box in det_boxes:
        x1, y1, x2, y2 = map(int, box[:4])
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(img_pil.width, x2); y2 = min(img_pil.height, y2)
        if x2-x1 < 5 or y2-y1 < 5: continue
        crop            = img_pil.crop((x1, y1, x2, y2))
        char_c, prob_c  = recognizer_conv.predict(crop)
        char_e, prob_e  = recognizer_effi.predict(crop)
        predictions.append({
            'box' : [x1, y1, x2, y2],
            'conv': {'char': char_c, 'prob': prob_c},
            'effi': {'char': char_e, 'prob': prob_e},
        })

    # スコア計算（両モデル）
    def calc_score(model_key):
        correct_count        = 0
        matched_pred_indices = set()
        for gt in gt_boxes:
            best_iou      = 0
            best_pred_idx = -1
            for j, pred in enumerate(predictions):
                iou = calculate_iou(gt['box'], pred['box'])
                if iou > IOU_THRESH and iou > best_iou:
                    best_iou      = iou
                    best_pred_idx = j
            if best_pred_idx != -1:
                matched_pred_indices.add(best_pred_idx)
                if predictions[best_pred_idx][model_key]['char'] == gt['char']:
                    correct_count += 1
        total_gt    = len(gt_boxes)
        total_preds = len(predictions)
        tp          = correct_count
        fp          = total_preds - len(matched_pred_indices)
        fn          = total_gt    - len(matched_pred_indices)
        recall    = (tp / total_gt    * 100) if total_gt    > 0 else 0
        precision = (tp / total_preds * 100) if total_preds > 0 else 0
        f1        = 2 * (precision * recall) / (precision + recall) \
                    if (precision + recall) > 0 else 0
        return {"f1": f1, "recall": recall, "precision": precision,
                "tp": tp, "fp": fp, "fn": fn,
                "gt_count": total_gt, "pred_count": total_preds}

    conv_score = calc_score('conv')
    effi_score = calc_score('effi')

    return {
        "filename": filename,
        "book_id" : book_id,
        "conv"    : conv_score,
        "effi"    : effi_score,
    }

# ==========================================
# 9. 集計表示関数
# ==========================================
def print_results(book_stats, model_key, model_name):
    print(f"\n{'='*90}")
    print(f"  {model_name}")
    print(f"{'='*90}")
    print(f"{'Book ID':<20} | {'Count':<5} | {'F1 (Avg)':<10} | {'Recall':<10} | {'Precision':<10} | {'F1 (Min)':<10} | {'F1 (Max)':<10}")
    print(f"{'-'*90}")

    all_f1   = []
    all_tp   = all_fp = all_fn = all_gt = all_pred = 0

    for book_id, results in sorted(book_stats.items()):
        scores   = [r[model_key] for r in results]
        f1_list  = [s["f1"]        for s in scores]
        rec_list = [s["recall"]    for s in scores]
        pre_list = [s["precision"] for s in scores]

        all_f1.extend(f1_list)
        all_tp   += sum(s["tp"]          for s in scores)
        all_fp   += sum(s["fp"]          for s in scores)
        all_fn   += sum(s["fn"]          for s in scores)
        all_gt   += sum(s["gt_count"]    for s in scores)
        all_pred += sum(s["pred_count"]  for s in scores)

        print(f"{book_id:<20} | {len(results):<5} | "
              f"{statistics.mean(f1_list):.2f}%     | "
              f"{statistics.mean(rec_list):.2f}%     | "
              f"{statistics.mean(pre_list):.2f}%      | "
              f"{min(f1_list):.2f}%     | {max(f1_list):.2f}%")

    overall_p  = (all_tp / all_pred * 100) if all_pred > 0 else 0
    overall_r  = (all_tp / all_gt   * 100) if all_gt   > 0 else 0
    overall_f1 = 2 * (overall_p * overall_r) / (overall_p + overall_r) \
                 if (overall_p + overall_r) > 0 else 0

    print(f"{'='*90}")
    print(f"  総画像数    : {len(all_f1)}")
    print(f"  総GT文字数  : {all_gt}")
    print(f"  TP          : {all_tp}  FP: {all_fp}  FN: {all_fn}")
    print(f"  Precision   : {overall_p:.2f}%")
    print(f"  Recall      : {overall_r:.2f}%")
    print(f"  F1 Score    : {overall_f1:.2f}%")
    print(f"  F1 (画像平均): {statistics.mean(all_f1):.2f}%")

    all_results    = [r[model_key] for results in book_stats.values()
                      for r in results]
    sorted_results = sorted(zip(all_results,
                                [r["filename"] for results in book_stats.values()
                                 for r in results]),
                            key=lambda x: x[0]["f1"])

    print(f"\n[最低スコア Top5]")
    for score, fname in sorted_results[:5]:
        print(f"  {fname} : F1={score['f1']:.2f}% "
              f"(R={score['recall']:.1f}%, P={score['precision']:.1f}%)")

    print(f"\n[最高スコア Top5]")
    for score, fname in sorted_results[-5:][::-1]:
        print(f"  {fname} : F1={score['f1']:.2f}% "
              f"(R={score['recall']:.1f}%, P={score['precision']:.1f}%)")

# ==========================================
# 10. メイン処理
# ==========================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading Models...")

    # YOLOX
    yolox = YOLOXWrapper(YOLOX_CONFIG)

    # ConvNeXtV2+ArcFace
    conv_model = ConvNeXtArcFaceModel(
        CONV_MODEL_NAME, CONV_NUM_CLASSES, CONV_EMBED_DIM
    )
    ckpt = torch.load(CONV_MODEL_PATH, map_location=device, weights_only=False)
    sd   = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    conv_model.load_state_dict(sd)
    recognizer_conv = ArcFaceRecognizer(
        conv_model, "trained_class_map.json", CONV_INPUT_SIZE, device
    )
    print("  ✅ ConvNeXtV2+ArcFace loaded.")

    # EfficientNetV2+ArcFace
    effi_model = EfficientNetArcFaceModel(EFFI_NUM_CLASSES, EFFI_EMBED_DIM)
    ckpt = torch.load(EFFI_MODEL_PATH, map_location=device, weights_only=False)
    sd   = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    effi_model.load_state_dict(sd)
    recognizer_effi = ArcFaceRecognizer(
        effi_model, "trained_class_map.json", EFFI_INPUT_SIZE, device
    )
    print("  ✅ EfficientNetV2+ArcFace loaded.")

    # 画像収集
    image_files = [TARGET_IMAGE]

    if not image_files:
        print("No images found."); sys.exit()

    print(f"Found {len(image_files)} images. Starting evaluation...")
    book_stats = defaultdict(list)

    for img_path in tqdm(image_files):
        result = process_single_image(
            img_path, yolox, recognizer_conv, recognizer_effi
        )
        if result:
            book_stats[result["book_id"]].append(result)

    # 結果表示
    print_results(book_stats, "conv", "ConvNeXtV2 + ArcFace")
    print_results(book_stats, "effi", "EfficientNetV2 + ArcFace")

    # 最終比較
    all_conv_f1 = [r["conv"]["f1"] for results in book_stats.values() for r in results]
    all_effi_f1 = [r["effi"]["f1"] for results in book_stats.values() for r in results]

    print(f"\n{'='*50}")
    print(f"  最終比較（F1画像平均）")
    print(f"{'='*50}")
    print(f"  ConvNeXtV2+ArcFace    : {statistics.mean(all_conv_f1):.2f}%")
    print(f"  EfficientNetV2+ArcFace: {statistics.mean(all_effi_f1):.2f}%")
    winner = "ConvNeXtV2+ArcFace" if statistics.mean(all_conv_f1) > statistics.mean(all_effi_f1) \
             else "EfficientNetV2+ArcFace"
    print(f"  勝者: {winner}")
    print(f"{'='*50}")