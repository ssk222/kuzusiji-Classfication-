import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageStat, ImageDraw, ImageFont
import timm
import numpy as np
import os
import sys
import json
import csv
import glob
import cv2
from tqdm import tqdm

# 1. 設定（元コードに準拠）
CONV_NAME         = 'convnextv2_large'
CONV_PATH         = "../vgg/output/model_conv_mae/best_model.pth"
CONV_SIZE         = 384
FORCE_CLASSES_CONV = 3962
CONV_EMBED_DIM    = 512
ARCFACE_S         = 16.0

CLASS_JSON        = "./trained_class_map.json"
IMAGE_DIR         = "./data_demo"          # 推論対象フォルダ
DATA_ROOT_DIR     = "/home/ihpc_double/Documents/sasaki/data"            # CSVルート
OUTPUT_BASE_FOLDER = "./output/conv_arc"
FONT_PATH         = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

# YOLOX設定
YOLOX_EXP_PATH  = "nano.py"
YOLOX_CKPT_PATH = "best_ckpt.pth"
YOLOX_CONF      = 0.35
YOLOX_NMS       = 0.45
YOLOX_SIZE      = 640

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. 準備
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
        new_img.paste(img, ((self.target_size - new_w)//2,
                             (self.target_size - new_h)//2))
        return new_img

# 3. ArcFace付きConvNeXtモデル定義（元コードと完全一致）
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

# 4. YOLOXラッパー
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
        ckpt = torch.load(YOLOX_CKPT_PATH, map_location="cuda",
                          weights_only=False)
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
            bboxes.append([det[0], det[1], det[2], det[3],
                           det[4]*det[5], int(det[6])])
        return bboxes

# 5. 認識モデル
class BasePredictor:
    def _load_weights(self, path):
        try:
            ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
            sd   = ckpt["state_dict"] if isinstance(ckpt, dict) \
                   and "state_dict" in ckpt else ckpt
            self.model.load_state_dict(sd)
            self.model.to(DEVICE).eval()
        except Exception as e:
            print(f"Error loading {path}: {e}"); sys.exit()

    def _get_transform(self, size):
        return transforms.Compose([
            SmartPadResize(size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

class ConvNeXtArcFacePredictor(BasePredictor):
    """ArcFace頭付きConvNeXtの推論クラス（Top5対応）"""
    def __init__(self, model_path, class_map, model_name,
                 input_size, num_classes):
        self.class_map = class_map
        self.model     = KuzushijiModel(model_name, num_classes,
                                        embed_dim=CONV_EMBED_DIM)
        self._load_weights(model_path)
        self.transform = self._get_transform(input_size)

    def predict(self, img_pil):
        """Top1文字・conf、Top5リストを返す"""
        with torch.no_grad():
            logits = self.model(
                self.transform(img_pil).unsqueeze(0).to(DEVICE),
                labels=None
            )
            probs = torch.softmax(logits[0], dim=0)
        top5_probs, top5_idx = torch.topk(probs, k=5)
        pred_char = unicode_to_char(
            self.class_map.get(top5_idx[0].item(), "?"))
        pred_prob = top5_probs[0].item()
        top5 = [
            (unicode_to_char(self.class_map.get(idx.item(), "?")),
             prob.item())
            for idx, prob in zip(top5_idx, top5_probs)
        ]
        return pred_char, pred_prob, top5

# 6. 可視化
def draw_result(img, predictions, gt_boxes, font):
    draw = ImageDraw.Draw(img)
    for pred in predictions:
        x1, y1, x2, y2 = pred['box']
        pred_char = pred['conv']['char']
        pred_prob = pred['conv']['prob']

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
            draw.text((x1, y1 - th - 5), pred_char,
                      fill=pred_color, font=font)
            if not is_correct:
                draw.text((x1 + tw + 10, y1 - th - 5),
                          matched_gt_char, fill="green", font=font)
        else:
            draw.rectangle([x1, y1, x2, y2], outline="magenta", width=2)
            draw.text((x1, y1 - 20),
                      f"Pred:{pred_char} {pred_prob*100:.0f}%",
                      fill="magenta", font=font)
    return img

# 7. スコア計算（元コードと同じ方法）
def calc_scores(predictions, gt_boxes):
    total_gt   = len(gt_boxes)
    total_pred = len(predictions)
    correct    = 0
    for gt in gt_boxes:
        best_iou = 0; best_idx = -1
        for i, pred in enumerate(predictions):
            iou = calculate_iou(gt['box'], pred['box'])
            if iou > 0.5 and iou > best_iou:
                best_iou = iou; best_idx = i
        if best_idx != -1 and \
           predictions[best_idx]['conv']['char'] == gt['char']:
            correct += 1
    r  = correct / total_gt   * 100 if total_gt   > 0 else 0
    p  = correct / total_pred * 100 if total_pred > 0 else 0
    f1 = 2*p*r/(p+r) if (p+r) > 0 else 0
    return {"correct": correct, "total_gt": total_gt,
            "total_pred": total_pred, "recall": r, "precision": p, "f1": f1}

# 8. メイン処理（フォルダ一括）
if __name__ == "__main__":
    os.makedirs(OUTPUT_BASE_FOLDER, exist_ok=True)

    # 画像一覧取得
    image_paths = sorted([
        p for p in glob.glob(os.path.join(IMAGE_DIR, "*"))
        if os.path.splitext(p)[1].lower() in IMAGE_EXTS
    ])
    if not image_paths:
        print(f"Error: no images found in {IMAGE_DIR}"); sys.exit()
    print(f"Found {len(image_paths)} images in {IMAGE_DIR}")

    # クラスマップロード（元コードと同じ形式）
    with open(CLASS_JSON, "r") as f:
        data = json.load(f)
    class_map = {int(k): v for k, v in data.items()}

    # モデルロード
    yolox      = YOLOXWrapper()
    model_conv = ConvNeXtArcFacePredictor(
        CONV_PATH, class_map, CONV_NAME, CONV_SIZE, FORCE_CLASSES_CONV
    )
    print(f"Models loaded.")

    # 集計用
    all_detail_rows  = []
    summary_rows     = []
    total_correct_all = 0
    total_gt_all      = 0
    total_pred_all    = 0
    correct_confs_all   = []
    incorrect_confs_all = []


    # 全画像ループ

    for target_path in tqdm(image_paths, desc="Processing"):
        filename         = os.path.basename(target_path)
        name_without_ext = os.path.splitext(filename)[0]
        book_id          = name_without_ext.split("_")[0]
        SAVE_DIR         = os.path.join(OUTPUT_BASE_FOLDER, name_without_ext)
        os.makedirs(SAVE_DIR, exist_ok=True)

        # GTロード（元コードと同じ）
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

        # YOLOX検出
        print(f"Running Inference... {filename}")
        det_boxes  = yolox.detect(target_path)
        img_origin = Image.open(target_path).convert("RGB")

        # 認識
        predictions = []
        for box in det_boxes:
            x1, y1, x2, y2 = map(int, box[:4])
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(img_origin.width, x2)
            y2 = min(img_origin.height, y2)
            if x2-x1 < 5 or y2-y1 < 5: continue
            crop = img_origin.crop((x1, y1, x2, y2))
            char_c, prob_c, top5_c = model_conv.predict(crop)
            predictions.append({
                'box' : [x1, y1, x2, y2],
                'conv': {'char': char_c, 'prob': prob_c, 'top5': top5_c},
            })

        # スコア計算（元コードと同じ）
        scores = calc_scores(predictions, gt_boxes)
        total_correct_all += scores["correct"]
        total_gt_all      += scores["total_gt"]
        total_pred_all    += scores["total_pred"]

        # 結果表示（元コードと同じ形式）
        print(f"\n COMPARISON RESULT: {filename}")
        print("="*60)
        print(f" GT Count: {scores['total_gt']} | "
              f"Pred Count: {scores['total_pred']}")
        print("-"*60)
        print(f" Model               | Correct | Recall "
              f"| Precision | F1 Score")
        print("-"*60)
        print(f" {'ConvNeXtV2+ArcFace':<20}| "
              f"{scores['correct']:<7} | "
              f"{scores['recall']:.2f}%  | "
              f"{scores['precision']:.2f}%     | "
              f"{scores['f1']:.2f}%")
        print("="*60)

        # confidence集計
        for gt in gt_boxes:
            best_iou = 0; best_idx = -1
            for i, pred in enumerate(predictions):
                iou = calculate_iou(gt['box'], pred['box'])
                if iou > 0.5 and iou > best_iou:
                    best_iou = iou; best_idx = i
            if best_idx != -1:
                prob  = predictions[best_idx]['conv']['prob']
                is_ok = predictions[best_idx]['conv']['char'] == gt['char']
                (correct_confs_all if is_ok else incorrect_confs_all).append(prob)

        # 可視化（元コードと同じ色仕様）
        try:
            font = ImageFont.truetype(
                FONT_PATH, max(18, int(img_origin.height / 60)))
        except:
            font = ImageFont.load_default()

        img_conv = draw_result(img_origin.copy(), predictions, gt_boxes, font)
        img_conv.save(os.path.join(SAVE_DIR, "conv.jpg"))
        print(f" Images saved in: {SAVE_DIR}")

        # サマリー行
        summary_rows.append([
            name_without_ext,
            scores["total_gt"], scores["total_pred"], scores["correct"],
            f"{scores['recall']:.2f}",
            f"{scores['precision']:.2f}",
            f"{scores['f1']:.2f}",
        ])

        # 詳細CSV行（予測box単位・conf付き）
        for pred in predictions:
            matched_gt_char = None
            best_iou        = 0
            for gt in gt_boxes:
                iou = calculate_iou(pred['box'], gt['box'])
                if iou > 0.5 and iou > best_iou:
                    best_iou = iou; matched_gt_char = gt['char']

            is_ok = (matched_gt_char is not None and
                     pred['conv']['char'] == matched_gt_char)
            top5  = pred['conv']['top5']

            row = [
                name_without_ext,
                pred['box'][0], pred['box'][1],
                pred['box'][2], pred['box'][3],
                matched_gt_char if matched_gt_char else "",
                pred['conv']['char'],
                f"{pred['conv']['prob']*100:.1f}",
                "O" if is_ok else ("X" if matched_gt_char else ""),
            ]
            for k in range(1, 5):
                if k < len(top5):
                    row += [top5[k][0], f"{top5[k][1]*100:.1f}"]
                else:
                    row += ["", ""]
            all_detail_rows.append(row)


    # 全体サマリー

    overall_r  = total_correct_all / total_gt_all   * 100 \
                 if total_gt_all   > 0 else 0
    overall_p  = total_correct_all / total_pred_all * 100 \
                 if total_pred_all > 0 else 0
    overall_f1 = 2*overall_p*overall_r/(overall_p+overall_r) \
                 if (overall_p+overall_r) > 0 else 0

    print(f"\n{'='*60}")
    print(f" OVERALL RESULT  ({len(image_paths)} images)")
    print(f"{'='*60}")
    print(f" GT total  : {total_gt_all}")
    print(f" Pred total: {total_pred_all}")
    print(f" Correct   : {total_correct_all}")
    print(f" Recall    : {overall_r:.2f}%")
    print(f" Precision : {overall_p:.2f}%")
    print(f" F1 Score  : {overall_f1:.2f}%")
    print(f"{'='*60}")

    if correct_confs_all:
        print(f"\n【正解時のconfidence（全体）】")
        print(f"  平均:    {np.mean(correct_confs_all)*100:.1f}%")
        print(f"  中央値:  {np.median(correct_confs_all)*100:.1f}%")
        print(f"  90%tile: {np.percentile(correct_confs_all, 90)*100:.1f}%")
        print(f"  conf<50%の正解: "
              f"{sum(1 for c in correct_confs_all if c < 0.5)}件")

    if incorrect_confs_all:
        print(f"\n【不正解時のconfidence（全体）】")
        print(f"  平均:    {np.mean(incorrect_confs_all)*100:.1f}%")
        print(f"  中央値:  {np.median(incorrect_confs_all)*100:.1f}%")
        print(f"  conf>80%の不正解（過信）: "
              f"{sum(1 for c in incorrect_confs_all if c > 0.8)}件")


    # CSV保存

    detail_csv = os.path.join(OUTPUT_BASE_FOLDER, "results_all.csv")
    with open(detail_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "image", "x1", "y1", "x2", "y2",
            "gt_char", "pred_char", "pred_conf(%)", "correct",
            "top2_char", "top2_conf(%)",
            "top3_char", "top3_conf(%)",
            "top4_char", "top4_conf(%)",
            "top5_char", "top5_conf(%)",
        ])
        writer.writerows(all_detail_rows)

    summary_csv = os.path.join(OUTPUT_BASE_FOLDER, "summary.csv")
    with open(summary_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "gt_count", "pred_count", "correct",
                         "recall(%)", "precision(%)", "f1(%)"])
        writer.writerows(summary_rows)
        writer.writerow(["TOTAL", total_gt_all, total_pred_all,
                         total_correct_all,
                         f"{overall_r:.2f}", f"{overall_p:.2f}",
                         f"{overall_f1:.2f}"])

    print(f"\n詳細CSV  : {detail_csv}")
    print(f"サマリーCSV: {summary_csv}")