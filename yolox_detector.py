# #!/usr/bin/env python3
# # -*- coding:utf-8 -*-

###分割推論用プログラム
##可視化or切り取りが可能
###文字切り取り
###分割画像の真ん中5割でx,yがあれば検出


# python demo_bunkatu_cut.py image -f nano.py -c best_ckpt.pth  --path /home/yoshizu/document/1_YOLOX/DEMO_YOLOX_charactercut/demo_data/200017197_00008_1.png  --save_result --device gpu


import argparse
import os
import time
from loguru import logger
import cv2
import torch
import numpy as np

from yolox.data.data_augment import ValTransform
from yolox.data.datasets import COCO_CLASSES
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess, vis

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]

def make_parser():
    parser = argparse.ArgumentParser("YOLOX Demo Bunkatu!")
    parser.add_argument("demo", default="image", help="demo type, eg. image, video and webcam")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")
    parser.add_argument("--path", default="./assets/dog.jpg", help="path to images or video")
    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
    parser.add_argument("--save_result", action="store_true", help="whether to save the inference result of image/video")
    parser.add_argument("-f", "--exp_file", default=None, type=str, help="please input your experiment description file")
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument("--device", default="cpu", type=str, help="device to run our model, can either be cpu or gpu")
    parser.add_argument("--conf", default=0.35, type=float, help="test conf")
    parser.add_argument("--nms", default=0.45, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=640, type=int, help="test img size")
    parser.add_argument("--fp16", dest="fp16", default=False, action="store_true", help="Adopting mix precision evaluating.")
    parser.add_argument("--legacy", dest="legacy", default=False, action="store_true", help="To be compatible with older versions")
    parser.add_argument("--fuse", dest="fuse", default=False, action="store_true", help="Fuse conv and bn for testing.")
    parser.add_argument("--trt", dest="trt", default=False, action="store_true", help="Using TensorRT model for testing.")
    return parser

def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names

def split_image(img, tile_size=640, stride=320, padding=320):
    """
    画像を分割する前に、周囲にパディングを追加
    
    Args:
        img: 入力画像
        tile_size: タイルサイズ (デフォルト: 640)
        stride: ストライド (デフォルト: 320)
        padding: 周囲に追加するパディングサイズ (デフォルト: 320)
    """
    h, w = img.shape[:2]
    
    # 画像の周囲にパディングを追加
    if padding > 0:
        padded_img = np.zeros((h + 2 * padding, w + 2 * padding, 3), dtype=img.dtype)
        padded_img[padding:padding + h, padding:padding + w] = img
        print(f"Added padding: {padding}px around image. New size: {padded_img.shape[:2]}")
    else:
        padded_img = img
    
    # パディング後の画像サイズ
    padded_h, padded_w = padded_img.shape[:2]
    
    tiles = []
    coords = []
    
    for y in range(0, padded_h, stride):
        for x in range(0, padded_w, stride):
            x_end = min(x + tile_size, padded_w)
            y_end = min(y + tile_size, padded_h)
            tile = padded_img[y:y_end, x:x_end]
            
            # タイルサイズにパディング
            pad_tile = np.zeros((tile_size, tile_size, 3), dtype=img.dtype)
            pad_tile[:tile.shape[0], :tile.shape[1]] = tile
            tiles.append(pad_tile)
            
            # 座標は元画像基準に調整（パディング分を差し引く）
            original_x = x - padding
            original_y = y - padding
            coords.append((original_x, original_y, tile.shape[1], tile.shape[0]))
            
            if x_end == padded_w:
                break
        if y_end == padded_h:
            break
    
    print(f"Split into {len(tiles)} tiles with padding={padding}px")
    return tiles, coords

def merge_outputs(outputs_list, coords_list, nms_thresh=0.3, original_img_size=None):
    """
    パディングを考慮したマージ処理
    
    Args:
        outputs_list: 各タイルの推論結果
        coords_list: 各タイルの座標情報
        nms_thresh: NMS閾値
        original_img_size: 元画像サイズ (h, w) - 境界チェック用
    """
    merged = []
    
    for outputs, (x0, y0, w, h) in zip(outputs_list, coords_list):
        if outputs is None or len(outputs) == 0:
            continue
        
        out = outputs.cpu().numpy()

        # パッチ内の中央領域（中心50% = 25%〜75%）の範囲を計算
        x_min_center = int(w * 0.25)
        x_max_center = int(w * 0.75)
        y_min_center = int(h * 0.25)
        y_max_center = int(h * 0.75)

        keep_indices = []
        for i, bbox in enumerate(out):
            x, y = bbox[0], bbox[1]  # 左上 (x1, y1)
            if x_min_center <= x <= x_max_center and y_min_center <= y <= y_max_center:
                keep_indices.append(i)

        if len(keep_indices) > 0:
            kept = out[keep_indices]
            # グローバル座標へ変換
            kept[:, 0] += x0  # x1
            kept[:, 2] += x0  # x2
            kept[:, 1] += y0  # y1
            kept[:, 3] += y0  # y2
            
            # 元画像の境界内にある検出のみを保持
            if original_img_size is not None:
                orig_h, orig_w = original_img_size
                valid_detections = []
                
                for detection in kept:
                    x1, y1, x2, y2 = detection[:4]
                    # 検出が元画像の境界内にあるかチェック
                    if (x1 >= 0 and y1 >= 0 and x2 <= orig_w and y2 <= orig_h and
                        x1 < x2 and y1 < y2):  # 有効なbboxかもチェック
                        valid_detections.append(detection)
                
                if valid_detections:
                    merged.append(np.array(valid_detections))
            else:
                merged.append(kept)

    if len(merged) == 0:
        return None

    merged = np.concatenate(merged, axis=0)
    import torchvision
    boxes = merged[:, :4]
    scores = merged[:, 4] * merged[:, 5]
    keep = torchvision.ops.nms(torch.tensor(boxes), torch.tensor(scores), nms_thresh)
    return torch.tensor(merged[keep])


class Predictor(object):
    def __init__(
        self,
        model,
        exp,
        cls_names=COCO_CLASSES,
        trt_file=None,
        decoder=None,
        device="cpu",
        fp16=False,
        legacy=False,
    ):
        self.model = model
        self.cls_names = cls_names
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        self.preproc = ValTransform(legacy=legacy)
        if trt_file is not None:
            from torch2trt import TRTModule
            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))
            x = torch.ones(1, 3, exp.test_size[0], exp.test_size[1]).cuda()
            self.model(x)
            self.model = model_trt

    def inference(self, img):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
        img_info["ratio"] = ratio

        img, _ = self.preproc(img, None, self.test_size)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()
        if self.device == "gpu":
            img = img.cuda()
            if self.fp16:
                img = img.half()  # to FP16

        with torch.no_grad():
            t0 = time.time()
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre,
                self.nmsthre, class_agnostic=True
            )
            logger.info("Infer time: {:.4f}s".format(time.time() - t0))
        return outputs, img_info

    def visual(self, output, img_info, cls_conf=0.35,save_folder=None):
        img = img_info["raw_img"]
        if output is None:
            return img
        output = output.cpu()
        bboxes = output[:, 0:4]
        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]
        # vis_res = vis(img, bboxes, scores, cls, cls_conf, self.cls_names)
        vis_res = vis(img, bboxes, scores, cls, conf=0.5, class_names=self.cls_names, 
                      save_crops=True, save_dir=save_folder, image_name=None)
        return vis_res


def image_demo_bunkatu(predictor, vis_folder, path, current_time, save_result, tile_size=640, stride=320):
    if os.path.isdir(path):
        files = get_image_list(path)
    else:
        files = [path]
    files.sort()
    
    # パス名からフォルダ名を取得
    path_name = os.path.basename(os.path.normpath(path)) if os.path.isdir(path) else os.path.splitext(os.path.basename(path))[0]
    
    for image_name in files:
        img = cv2.imread(image_name)
        tiles, coords = split_image(img, tile_size=tile_size, stride=stride)
        outputs_list = []

        # 保存ディレクトリの準備（時刻フォルダ/データセット名フォルダ/画像ファイル名フォルダ構造）
        save_folder = None
        patch_folder = None
        crops_folder = None
        if save_result:
            # 時刻フォルダ
            time_folder = os.path.join(vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time))
            # データセット名フォルダ
            dataset_folder = os.path.join(time_folder, path_name)
            
            # 画像ファイル名からフォルダ名を作成（拡張子なし）
            image_base_name = os.path.splitext(os.path.basename(image_name))[0]
            image_folder = os.path.join(dataset_folder, image_base_name)
            
            # 画像ファイルが複数のサブフォルダにある場合の処理
            relative_path = os.path.relpath(os.path.dirname(image_name), path if os.path.isdir(path) else os.path.dirname(path))
            if relative_path != '.' and relative_path != '':
                # サブフォルダがある場合
                save_folder = os.path.join(dataset_folder, relative_path, image_base_name)
            else:
                # サブフォルダがない場合
                save_folder = image_folder
            
            # 各画像ファイル専用のフォルダを作成
            patch_folder = os.path.join(save_folder, "patches")
            crops_folder = os.path.join(save_folder, "crops")
            os.makedirs(patch_folder, exist_ok=True)
            os.makedirs(crops_folder, exist_ok=True)

        for idx, tile in enumerate(tiles):
            outputs, _ = predictor.inference(tile)
            output = outputs[0] if outputs is not None else None
            outputs_list.append(output)

            # ★ パッチごとの結果画像を保存する処理 ★
            if save_result:
                img_info = {
                    "raw_img": tile,
                    "ratio": 1.0
                }
                vis_img = predictor.visual(output, img_info, predictor.confthre)
                patch_filename = os.path.join(patch_folder, f"{os.path.splitext(os.path.basename(image_name))[0]}_patch{idx}.jpg")
                cv2.imwrite(patch_filename, vis_img)
        
        # 全体画像に対する統合結果
        merged_outputs_tensor = merge_outputs(outputs_list, coords, nms_thresh=predictor.nmsthre)
        img_info = {
            "raw_img": img,
            "ratio": 1.0
        }
        result_image = predictor.visual(merged_outputs_tensor, img_info, predictor.confthre, save_folder=crops_folder)

        # 全体画像の保存（画像ファイル専用フォルダ内に保存）
        if save_result:
            save_file_name = os.path.join(save_folder, os.path.basename(image_name))
            logger.info("Saving detection result in {}".format(save_file_name))
            cv2.imwrite(save_file_name, result_image)


def main(exp, args):
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    file_name = os.path.join(exp.output_dir, args.experiment_name)
    os.makedirs(file_name, exist_ok=True)

    vis_folder = None
    if args.save_result:
        vis_folder = os.path.join(file_name, "vis_res")
        os.makedirs(vis_folder, exist_ok=True)

    if args.trt:
        args.device = "gpu"

    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model()
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))

    if args.device == "gpu":
        model.cuda()
        if args.fp16:
            model.half()
    model.eval()

    if not args.trt:
        if args.ckpt is None:
            ckpt_file = os.path.join(file_name, "best_ckpt.pth")
        else:
            ckpt_file = args.ckpt
        logger.info("loading checkpoint")
        ckpt = torch.load(ckpt_file, map_location="cuda",weights_only=False)
        model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    if args.trt:
        assert not args.fuse, "TensorRT model is not support model fusing!"
        trt_file = os.path.join(file_name, "model_trt.pth")
        assert os.path.exists(
            trt_file
        ), "TensorRT model is not found!\n Run python3 tools/trt.py first!"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
        logger.info("Using TensorRT to inference")
    else:
        trt_file = None
        decoder = None

    predictor = Predictor(
        model, exp, COCO_CLASSES, trt_file, decoder,
        args.device, args.fp16, args.legacy,
    )
    current_time = time.localtime()
    if args.demo == "image":
        image_demo_bunkatu(predictor, vis_folder, args.path, current_time, args.save_result)
    else:
        logger.info("Only image mode is supported in demo_bunkatu.py.")

if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)
    main(exp, args)
