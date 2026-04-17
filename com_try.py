import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16, efficientnet_v2_l
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont, ImageStat
import os
import sys
import json
import cv2
import csv
import numpy as np
import timm
import glob
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ==========================================
# 1. 設定エリア
# ==========================================
TARGET_FILENAME = "200021071_00075_1.jpg"

IMAGE_DIR       = "./data"
DATA_ROOT_DIR   = "/home/ihpc_double/Documents/sasaki/data"
OUTPUT_BASE_FOLDER = "./output"

CLASS_JSON  = "trained_class_map.json"
FONT_PATH   = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- モデル設定 ---
VGG_PATH  = "/home/ihpc_double/Documents/sasaki/vgg/output/model_vgg/best_model.pth"
VGG_SIZE  = 224

EFF_PATH         = "/home/ihpc_double/Documents/sasaki/vgg/output/model_Effi/best_model.pth"
EFF_SIZE         = 384
FORCE_CLASSES_EFF = 4329

# 現在学習中のArcFace付きConvNeXtモデル
CONV_PATH         = "/home/ihpc_double/Documents/sasaki/vgg/output/model_conv_mae/best_model.pth"
CONV_NAME         = 'convnextv2_large'
CONV_SIZE         = 384
FORCE_CLASSES_CONV = 3962
CONV_EMBED_DIM    = 512   # 学習時のembed_dim
ARCFACE_S         = 16.0  # 学習時のARCFACE_S

# --- YOLOX設定 ---
YOLOX_EXP_PATH  = "nano.py"
YOLOX_CKPT_PATH = "best_ckpt.pth"
YOLOX_CONF = 0.35
YOLOX_NMS  = 0.45
YOLOX_SIZE = 640

# ==========================================
# 2. 準備
# ==========================================
try:
    from yolox_detector import get_exp, Predictor
    from yolox.data.datasets import COCO_CLASSES
except ImportError:
    print("Error: 'yolox_detector.py' が見つかりません。"); sys.exit()

COL_UNICODE = 0; COL_IMAGE = 1; COL_X = 2; COL_Y = 3; COL_W = 6; COL_H = 7

def unicode_to_char(u_code):
    if u_code and u_code.startswith("U+"):
        try: return chr(int(u_code.replace("U+", ""), 16))
        except: return u_code
    return u_code

def calculate_iou(box1, box2):
    ix1 = max(box1[0], box2[0]); iy1 = max(box1[1], box2[1])
    ix2 = min(box1[2], box2[2]); iy2 = min(box1[3], box2[3])
    inter_area = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter_area == 0: return 0
    box1_area = (box1[2]-box1[0])*(box1[3]-box1[1])
    box2_area = (box2[2]-box2[0])*(box2[3]-box2[1])
    return inter_area / (box1_area + box2_area - inter_area)

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
        new_img.paste(img, ((self.target_size - new_w)//2, (self.target_size - new_h)//2))
        return new_img

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
        self.arcface = ArcFaceHead(embed_dim, num_classes,
                                   s=ARCFACE_S, m=0.3)

    def forward(self, x, labels=None):
        features = self.encoder(x)
        embed    = self.embed(features)
        logits   = self.arcface(embed, labels)
        return logits


# ==========================================
# 4. YOLOXラッパー
# ==========================================
class YOLOXWrapper:
    def __init__(self):
        self.exp = get_exp(YOLOX_EXP_PATH, None)
        self.exp.test_conf = YOLOX_CONF
        self.exp.nmsthre   = YOLOX_NMS
        self.exp.test_size = (YOLOX_SIZE, YOLOX_SIZE)
        self.model = self.exp.get_model()
        if DEVICE.type == "cuda":
            self.model.cuda(); self.model.half()
        self.model.eval()
        ckpt = torch.load(YOLOX_CKPT_PATH, map_location="cuda", weights_only=False)
        self.model.load_state_dict(ckpt["model"])
        self.predictor = Predictor(
            self.model, self.exp, COCO_CLASSES, None, None,
            "gpu" if DEVICE.type == "cuda" else "cpu",
            True if DEVICE.type == "cuda" else False, False
        )

    def detect(self, img_path):
        img = cv2.imread(img_path)
        if img is None: return []
        from yolox_detector import split_image, merge_outputs
        tiles, coords = split_image(img, tile_size=640, stride=320)
        outputs_list = []
        for tile in tiles:
            outputs, _ = self.predictor.inference(tile)
            outputs_list.append(outputs[0] if outputs is not None else None)
        merged = merge_outputs(outputs_list, coords, nms_thresh=self.exp.nmsthre)
        if merged is None: return []
        bboxes = []
        for det in merged.cpu().numpy():
            bboxes.append([det[0], det[1], det[2], det[3], det[4]*det[5], int(det[6])])
        return bboxes


# ==========================================
# 5. 認識モデルの基底クラスと各モデル
# ==========================================
class BasePredictor:
    def _load_weights(self, path):
        try:
            ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
            sd   = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
            self.model.load_state_dict(sd)
            self.model.to(DEVICE).eval()
        except Exception as e:
            print(f"Error loading {path}: {e}"); sys.exit()

    def _get_transform(self, size):
        return transforms.Compose([
            SmartPadResize(size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def predict(self, img_pil):
        with torch.no_grad():
            output = self.model(self.transform(img_pil).unsqueeze(0).to(DEVICE))
            probs  = torch.nn.functional.softmax(output[0], dim=0)
        val, idx = torch.topk(probs, 1)
        return unicode_to_char(self.class_map.get(idx[0].item(), "?")), val[0].item()


class VGGPredictor(BasePredictor):
    def __init__(self, model_path, class_map, input_size):
        self.class_map = class_map
        self.model = vgg16(weights=None)
        self.model.classifier = nn.Sequential(
            nn.Linear(512*7*7, 1024), nn.ReLU(True), nn.Dropout(0.5),
            nn.Linear(1024, 1024),    nn.ReLU(True), nn.Dropout(0.5),
            nn.Linear(1024, len(self.class_map))
        )
        self._load_weights(model_path)
        self.transform = self._get_transform(input_size)


class EfficientNetPredictor(BasePredictor):
    def __init__(self, model_path, class_map, input_size, force_classes):
        self.class_map = class_map
        self.model = efficientnet_v2_l(weights=None)
        self.model.classifier[1] = nn.Linear(
            self.model.classifier[1].in_features, force_classes
        )
        self._load_weights(model_path)
        self.transform = self._get_transform(input_size)


class ConvNeXtArcFacePredictor(BasePredictor):
    """ArcFace頭付きConvNeXtの推論クラス"""
    def __init__(self, model_path, class_map, model_name, input_size, num_classes):
        self.class_map = class_map
        self.model     = KuzushijiModel(model_name, num_classes, embed_dim=CONV_EMBED_DIM)
        self._load_weights(model_path)
        self.transform = self._get_transform(input_size)

    def predict(self, img_pil):
        with torch.no_grad():
            # ArcFaceはlabels=Noneで推論
            logits = self.model(self.transform(img_pil).unsqueeze(0).to(DEVICE),
                                labels=None)
            probs  = torch.softmax(logits[0], dim=0)
        val, idx = torch.topk(probs, 1)
        return unicode_to_char(self.class_map.get(idx[0].item(), "?")), val[0].item()


# ==========================================
# 6. メイン処理
# ==========================================
if __name__ == "__main__":
    name_without_ext = os.path.splitext(TARGET_FILENAME)[0]
    book_id          = name_without_ext.split("_")[0]
    SAVE_DIR         = os.path.join(OUTPUT_BASE_FOLDER, name_without_ext)
    os.makedirs(SAVE_DIR, exist_ok=True)

    # 画像検索
    target_path = None
    for p in glob.glob(os.path.join(IMAGE_DIR, "*")):
        if os.path.basename(p) == TARGET_FILENAME:
            target_path = p; break
    if target_path is None:
        print(f"Error: '{TARGET_FILENAME}' not found in {IMAGE_DIR}"); sys.exit()

    # モデルロード
    with open(CLASS_JSON, "r") as f:
        class_map = {int(k): v for k, v in json.load(f).items()}

    yolox      = YOLOXWrapper()
    model_vgg  = VGGPredictor(VGG_PATH, class_map, VGG_SIZE)
    model_eff  = EfficientNetPredictor(EFF_PATH, class_map, EFF_SIZE, FORCE_CLASSES_EFF)
    model_conv = ConvNeXtArcFacePredictor(CONV_PATH, class_map, CONV_NAME,
                                          CONV_SIZE, FORCE_CLASSES_CONV)

    # GTロード
    csv_filename    = f"{book_id}_coordinate.csv"
    COORDINATE_CSV  = os.path.join(DATA_ROOT_DIR, book_id, csv_filename)
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

    # 推論実行
    print("Running Inference...")
    det_boxes  = yolox.detect(target_path)
    img_origin = Image.open(target_path).convert("RGB")

    predictions = []
    for box in det_boxes:
        x1, y1, x2, y2 = map(int, box[:4])
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(img_origin.width, x2); y2 = min(img_origin.height, y2)
        if x2-x1 < 5 or y2-y1 < 5: continue

        crop       = img_origin.crop((x1, y1, x2, y2))
        char_v, prob_v = model_vgg.predict(crop)
        char_e, prob_e = model_eff.predict(crop)
        char_c, prob_c = model_conv.predict(crop)

        predictions.append({
            'box' : [x1, y1, x2, y2],
            'vgg' : {'char': char_v, 'prob': prob_v},
            'eff' : {'char': char_e, 'prob': prob_e},
            'conv': {'char': char_c, 'prob': prob_c}
        })

    # スコア計算
    stats     = {k: {'correct': 0, 'precision': 0, 'recall': 0, 'f1': 0}
                 for k in ['vgg', 'eff', 'conv']}
    total_gt  = len(gt_boxes)
    total_pred = len(predictions)

    for model_key in ['vgg', 'eff', 'conv']:
        correct = 0
        for gt in gt_boxes:
            best_iou = 0; best_idx = -1
            for i, pred in enumerate(predictions):
                iou = calculate_iou(gt['box'], pred['box'])
                if iou > 0.5 and iou > best_iou:
                    best_iou = iou; best_idx = i
            if best_idx != -1 and predictions[best_idx][model_key]['char'] == gt['char']:
                correct += 1

        r  = (correct / total_gt   * 100) if total_gt   > 0 else 0
        p  = (correct / total_pred * 100) if total_pred > 0 else 0
        f1 = 2 * (p * r) / (p + r) if (p + r) > 0 else 0
        stats[model_key] = {'correct': correct, 'recall': r, 'precision': p, 'f1': f1}

    # 結果表示
    print(f"\n COMPARISON RESULT: {TARGET_FILENAME}")
    print("="*60)
    print(f" GT Count: {total_gt} | Pred Count: {total_pred}")
    print("-"*60)
    print(f" Model           | Correct | Recall | Precision | F1 Score")
    print("-"*60)
    for key, name in [('vgg','VGG16'), ('eff','EfficientNetV2'), ('conv','ConvNeXtV2+ArcFace')]:
        s = stats[key]
        print(f" {name:<16} | {s['correct']:<7} | {s['recall']:.2f}%  | {s['precision']:.2f}%     | {s['f1']:.2f}%")
    print("="*60)

    # 可視化
    def draw_result(img, model_key):
        draw = ImageDraw.Draw(img)
        try: font = ImageFont.truetype(FONT_PATH, max(18, int(img.height / 60)))
        except: font = ImageFont.load_default()

        for pred in predictions:
            x1, y1, x2, y2 = pred['box']
            pred_char = pred[model_key]['char']
            pred_prob = pred[model_key]['prob']

            matched_gt_char = None
            best_iou_vis    = 0
            for gt in gt_boxes:
                iou = calculate_iou([x1, y1, x2, y2], gt['box'])
                if iou > 0.5 and iou > best_iou_vis:
                    best_iou_vis    = iou
                    matched_gt_char = gt['char']

            if matched_gt_char:
                is_correct  = (pred_char == matched_gt_char)
                box_color   = "blue" if is_correct else "red"
                pred_color  = "blue" if is_correct else "red"
                draw.rectangle([x1, y1, x2, y2], outline=box_color, width=2)
                bbox = draw.textbbox((0, 0), pred_char, font=font)
                th   = bbox[3] - bbox[1]
                tw   = bbox[2] - bbox[0]
                draw.text((x1, y1 - th - 5), pred_char, fill=pred_color, font=font)
                if not is_correct:
                    draw.text((x1 + tw + 10, y1 - th - 5),
                              matched_gt_char, fill="green", font=font)
            else:
                draw.rectangle([x1, y1, x2, y2], outline="magenta", width=2)
                draw.text((x1, y1 - 20),
                          f"Pred:{pred_char} {pred_prob*100:.0f}%",
                          fill="magenta", font=font)
        return img

    img_vgg  = draw_result(img_origin.copy(), 'vgg')
    img_eff  = draw_result(img_origin.copy(), 'eff')
    img_conv = draw_result(img_origin.copy(), 'conv')

    # all_compare
    img_all  = img_origin.copy()
    draw_all = ImageDraw.Draw(img_all)
    try: font = ImageFont.truetype(FONT_PATH, max(16, int(img_all.height / 70)))
    except: font = ImageFont.load_default()
    for pred in predictions:
        x1, y1, x2, y2 = pred['box']
        label = f"V:{pred['vgg']['char']} E:{pred['eff']['char']} C:{pred['conv']['char']}"
        draw_all.rectangle([x1, y1, x2, y2], outline="cyan", width=2)
        bbox = draw_all.textbbox((x1, y1-25), label, font=font)
        draw_all.rectangle(bbox, fill="black")
        draw_all.text((x1, y1-25), label, fill="white", font=font)

    img_vgg.save(os.path.join(SAVE_DIR, "vgg.jpg"))
    img_eff.save(os.path.join(SAVE_DIR, "eff.jpg"))
    img_conv.save(os.path.join(SAVE_DIR, "conv.jpg"))
    img_all.save(os.path.join(SAVE_DIR, "all_compare.jpg"))
    print(f"\n Images saved in: {SAVE_DIR}")