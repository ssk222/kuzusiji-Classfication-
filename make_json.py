import json
from torchvision import datasets
DATA_DIR = "../kkanji" 

print("データセットを読み込んでいます...")
dataset = datasets.ImageFolder(DATA_DIR)

# PyTorchが自動で割り振った {'U+xxxx': 0, 'U+yyyy': 1...} という辞書を
# {0: 'U+xxxx', 1: 'U+yyyy'...} という逆向きに変換する
idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}

# 新しいJSONファイルとして保存
with open("kkanji_class_map.json", "w") as f:
    json.dump(idx_to_class, f, indent=4)

print("✅ ConvNeXt用の新しい辞書 (kkanji_class_map.json) を作成しました！")