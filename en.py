import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import efficientnet_v2_l
from PIL import Image, ImageDraw, ImageFont, ImageStat
import os
import sys
import cv2
import csv
import numpy as np
import timm
import glob
import json

# ==========================================
# 1. 設定エリア
# ==========================================
TARGET_FILENAME = "200021071_00075_1.jpg"  # ← 対象画像

IMAGE_DIR     = "./data"
DATA_ROOT_DIR = "/home/ihpc_double/Documents/sasaki/data"
OUTPUT_FOLDER = "./output_en"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

CLASS_JSON = "trained_class_map.json"
FONT_PATH  = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ConvNeXtV2+ArcFace
CONV_MODEL_PATH  = "/home/ihpc_double/Documents/sasaki/vgg/output/model_conv_mae/best_model.pth"
CONV_MODEL_NAME  = 'convnextv2_large'
CONV_NUM_CLASSES = 3962
CONV_EMBED_DIM   = 512
ARCFACE_S        = 16.0
ARCFACE_M        = 0.3

# EfficientNetV2単体
EFFI_MODEL_PATH   = "/home/ihpc_double/Documents/sasaki/vgg/output/model_Effi/best_model.pth"
EFFI_NUM_CLASSES  = 4329
EFFI_INPUT_SIZE   = 384

INPUT_SIZE = 384
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

def is_kanji(char):
    if not char or len(char) != 1: return False
    code = ord(char)
    return (0x4E00 <= code <= 0x9FFF or
            0x3400 <= code <= 0x4DBF or
            0xF900 <= code <= 0xFAFF)

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
# 3. ArcFace Head
# ==========================================
class ArcFaceHead(nn.Module):
    def __init__(self, in_features, num_classes, s=16.0, m=0.3):
        super().__init__()
        self.s = s; self.m = m
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
        return (cosine * (1 - one_hot) + target_logit * one_hot) * self.s

# ==========================================
# 4. ConvNeXtV2+ArcFaceモデル
# ==========================================
class KuzushijiModel(nn.Module):
    def __init__(self, model_name, num_classes, embed_dim=512):
        super().__init__()
        self.encoder = timm.create_model(
            model_name, pretrained=False, num_classes=0, global_pool='avg'
        )
        in_features = self.encoder.num_features
        self.embed = nn.Sequential(
            nn.Linear(in_features, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.arcface = ArcFaceHead(embed_dim, num_classes, s=ARCFACE_S, m=ARCFACE_M)

    def forward(self, x, labels=None):
        return self.arcface(self.embed(self.encoder(x)), labels)

# ==========================================
# 5. YOLOXラッパー
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
        if merged_np.ndim == 1: merged_np = merged_np[np.newaxis, :]
        bboxes = []
        for det in merged_np:
            if len(det) < 7: continue
            bboxes.append([det[0], det[1], det[2], det[3], det[4]*det[5], int(det[6])])
        return bboxes

# ==========================================
# 6. 認識クラス（共通）
# ==========================================
class ArcFaceRecognizer:
    def __init__(self, model, class_map, input_size):
        self.model     = model.to(DEVICE).eval()
        self.class_map = class_map
        self.transform = transforms.Compose([
            SmartPadResize(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def predict(self, img_pil):
        with torch.no_grad():
            logits = self.model(
                self.transform(img_pil).unsqueeze(0).to(DEVICE), labels=None
            )
            probs = torch.softmax(logits[0], dim=0)
        val, idx = torch.topk(probs, 1)
        return unicode_to_char(self.class_map.get(idx[0].item(), "?")), val[0].item(), probs

class EfficientNetRecognizer:
    def __init__(self, model_path, class_map, num_classes, input_size):
        self.class_map = class_map
        self.model = efficientnet_v2_l(weights=None)
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, num_classes)
        ckpt = torch.load(model_path, map_location=DEVICE, weights_only=False)
        sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        self.model.load_state_dict(sd)
        self.model.to(DEVICE).eval()
        self.transform = transforms.Compose([
            SmartPadResize(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def predict(self, img_pil):
        with torch.no_grad():
            output = self.model(self.transform(img_pil).unsqueeze(0).to(DEVICE))
            probs  = torch.softmax(output[0], dim=0)
        val, idx = torch.topk(probs, 1)
        return unicode_to_char(self.class_map.get(idx[0].item(), "?")), val[0].item(), probs

# ==========================================
# 7. アンサンブル予測
# ==========================================
# 修正版（pred_convをchar_convに統一）
def ensemble_predict(crop, recognizer_conv, recognizer_effi, class_map):
    char_conv, conf_conv, prob_conv = recognizer_conv.predict(crop)
    char_effi, conf_effi, prob_effi = recognizer_effi.predict(crop)

    # 両モデルが一致
    if char_conv == char_effi:
        return char_conv, conf_conv, "agree"

    # ConvNeXtが漢字 かつ EfficientNetが非漢字の時だけEfficientNetを優先
    if is_kanji(char_conv) and not is_kanji(char_effi):
        return char_effi, conf_effi, "effi_priority"

    # それ以外はConvNeXtをそのまま使う
    return char_conv, conf_conv, "conv_only"
# ==========================================
# 8. メイン処理
# ==========================================
if __name__ == "__main__":
    name_without_ext = os.path.splitext(TARGET_FILENAME)[0]
    book_id          = name_without_ext.split("_")[0]
    SAVE_DIR         = os.path.join(OUTPUT_FOLDER, name_without_ext)
    os.makedirs(SAVE_DIR, exist_ok=True)

    # 画像検索
    target_path = None
    for p in glob.glob(os.path.join(IMAGE_DIR, "*")):
        if os.path.basename(p) == TARGET_FILENAME:
            target_path = p; break
    if target_path is None:
        print(f"Error: '{TARGET_FILENAME}' not found."); sys.exit()

    # クラスマップ
    with open(CLASS_JSON, "r") as f:
        class_map = {int(k): v for k, v in json.load(f).items()}

    # モデルロード
    print("Loading Models...")
    yolox = YOLOXWrapper(YOLOX_CONFIG)

    conv_model = KuzushijiModel(CONV_MODEL_NAME, CONV_NUM_CLASSES, CONV_EMBED_DIM)
    ckpt = torch.load(CONV_MODEL_PATH, map_location=DEVICE, weights_only=False)
    sd   = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    conv_model.load_state_dict(sd)
    recognizer_conv = ArcFaceRecognizer(conv_model, class_map, INPUT_SIZE)
    print("  ✅ ConvNeXtV2+ArcFace loaded.")

    recognizer_effi = EfficientNetRecognizer(
        EFFI_MODEL_PATH, class_map, EFFI_NUM_CLASSES, EFFI_INPUT_SIZE
    )
    print("  ✅ EfficientNetV2 loaded.")

    # GTロード
    csv_filename   = f"{book_id}_coordinate.csv"
    COORDINATE_CSV = os.path.join(DATA_ROOT_DIR, book_id, csv_filename)
    gt_boxes = []
    if os.path.exists(COORDINATE_CSV):
        with open(COORDINATE_CSV, "r", encoding="utf-8") as f:
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

    # 推論
    print("Running Inference...")
    det_boxes  = yolox.detect(target_path)
    img_origin = Image.open(target_path).convert("RGB")

    predictions = []
    for box in det_boxes:
        x1, y1, x2, y2 = map(int, box[:4])
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(img_origin.width, x2); y2 = min(img_origin.height, y2)
        if x2-x1 < 5 or y2-y1 < 5: continue
        crop = img_origin.crop((x1, y1, x2, y2))

        char_conv, conf_conv, _ = recognizer_conv.predict(crop)
        char_effi, conf_effi, _ = recognizer_effi.predict(crop)
        char_ens,  conf_ens, strategy = ensemble_predict(
            crop, recognizer_conv, recognizer_effi, class_map
        )

        predictions.append({
            'box'     : [x1, y1, x2, y2],
            'conv'    : {'char': char_conv, 'conf': conf_conv},
            'effi'    : {'char': char_effi, 'conf': conf_effi},
            'ensemble': {'char': char_ens,  'conf': conf_ens, 'strategy': strategy},
        })

    # スコア計算
    def calc_score(key):
        correct = 0
        matched = set()
        for gt in gt_boxes:
            best_iou = 0; best_idx = -1
            for j, pred in enumerate(predictions):
                iou = calculate_iou(gt['box'], pred['box'])
                if iou > IOU_THRESH and iou > best_iou:
                    best_iou = iou; best_idx = j
            if best_idx != -1:
                matched.add(best_idx)
                if predictions[best_idx][key]['char'] == gt['char']:
                    correct += 1
        total_gt   = len(gt_boxes)
        total_pred = len(predictions)
        r  = (correct / total_gt   * 100) if total_gt   > 0 else 0
        p  = (correct / total_pred * 100) if total_pred > 0 else 0
        f1 = 2 * (p * r) / (p + r) if (p + r) > 0 else 0
        return correct, r, p, f1

    c_conv, r_conv, p_conv, f1_conv = calc_score('conv')
    c_effi, r_effi, p_effi, f1_effi = calc_score('effi')
    c_ens,  r_ens,  p_ens,  f1_ens  = calc_score('ensemble')

    print(f"\n GT Count: {len(gt_boxes)} | Pred Count: {len(predictions)}")
    print(f"{'='*60}")
    print(f" Model                | Correct | Recall  | Prec    | F1")
    print(f"{'-'*60}")
    print(f" ConvNeXtV2+ArcFace   | {c_conv:<7} | {r_conv:.2f}%  | {p_conv:.2f}%  | {f1_conv:.2f}%")
    print(f" EfficientNetV2       | {c_effi:<7} | {r_effi:.2f}%  | {p_effi:.2f}%  | {f1_effi:.2f}%")
    print(f" Ensemble             | {c_ens:<7} | {r_ens:.2f}%  | {p_ens:.2f}%  | {f1_ens:.2f}%")
    print(f"{'='*60}")

    # ==========================================
    # 可視化（4枚）
    # ==========================================
    def draw_result(img, key):
        draw = ImageDraw.Draw(img)
        try: font = ImageFont.truetype(FONT_PATH, max(18, int(img.height / 60)))
        except: font = ImageFont.load_default()

        for pred in predictions:
            x1, y1, x2, y2 = pred['box']
            pred_char = pred[key]['char']
            pred_conf = pred[key]['conf']

            matched_gt_char = None
            best_iou_vis    = 0
            for gt in gt_boxes:
                iou = calculate_iou([x1, y1, x2, y2], gt['box'])
                if iou > IOU_THRESH and iou > best_iou_vis:
                    best_iou_vis    = iou
                    matched_gt_char = gt['char']

            if matched_gt_char:
                is_correct = (pred_char == matched_gt_char)
                box_color  = "blue" if is_correct else "red"
                draw.rectangle([x1, y1, x2, y2], outline=box_color, width=2)
                bbox = draw.textbbox((0, 0), pred_char, font=font)
                th = bbox[3] - bbox[1]; tw = bbox[2] - bbox[0]
                draw.text((x1, y1-th-5), pred_char, fill=box_color, font=font)
                if not is_correct:
                    draw.text((x1+tw+10, y1-th-5), matched_gt_char,
                              fill="green", font=font)
            else:
                draw.rectangle([x1, y1, x2, y2], outline="magenta", width=2)
                draw.text((x1, y1-20),
                          f"Pred:{pred_char} {pred_conf*100:.0f}%",
                          fill="magenta", font=font)
        return img

    # アンサンブル画像（採用モデルも表示）
    def draw_ensemble(img):
        draw = ImageDraw.Draw(img)
        try: font = ImageFont.truetype(FONT_PATH, max(18, int(img.height / 60)))
        except: font = ImageFont.load_default()

        strategy_color = {
            "agree"        : "blue",    # 両モデル一致
            "effi_priority": "cyan",    # EfficientNet優先
            "avg"          : "orange",  # 平均で決定
        }

        for pred in predictions:
            x1, y1, x2, y2 = pred['box']
            pred_char = pred['ensemble']['char']
            pred_conf = pred['ensemble']['conf']
            strategy  = pred['ensemble']['strategy']

            matched_gt_char = None
            best_iou_vis    = 0
            for gt in gt_boxes:
                iou = calculate_iou([x1, y1, x2, y2], gt['box'])
                if iou > IOU_THRESH and iou > best_iou_vis:
                    best_iou_vis    = iou
                    matched_gt_char = gt['char']

            if matched_gt_char:
                is_correct = (pred_char == matched_gt_char)
                if is_correct:
                    box_color = strategy_color.get(strategy, "blue")
                else:
                    box_color = "red"
                draw.rectangle([x1, y1, x2, y2], outline=box_color, width=2)
                bbox = draw.textbbox((0, 0), pred_char, font=font)
                th = bbox[3] - bbox[1]; tw = bbox[2] - bbox[0]
                draw.text((x1, y1-th-5), pred_char, fill=box_color, font=font)
                if not is_correct:
                    draw.text((x1+tw+10, y1-th-5), matched_gt_char,
                              fill="green", font=font)
            else:
                draw.rectangle([x1, y1, x2, y2], outline="magenta", width=2)
                draw.text((x1, y1-20),
                          f"Pred:{pred_char} {pred_conf*100:.0f}%",
                          fill="magenta", font=font)
        return img

    img_conv = draw_result(img_origin.copy(), 'conv')
    img_effi = draw_result(img_origin.copy(), 'effi')
    img_ens  = draw_ensemble(img_origin.copy())

    img_conv.save(os.path.join(SAVE_DIR, "conv_arcface.jpg"))
    img_effi.save(os.path.join(SAVE_DIR, "efficientnet.jpg"))
    img_ens.save(os.path.join(SAVE_DIR,  "ensemble.jpg"))

    print(f"\n Images saved in: {SAVE_DIR}")
    print(f"\n[アンサンブル色の凡例]")
    print(f"  青   : 両モデル一致で正解")
    print(f"  シアン: EfficientNet優先で正解")
    print(f"  橙   : 平均で正解")
    print(f"  赤   : 不正解")
    print(f"  マゼンタ: 過検出")