import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms, datasets
from torchvision.transforms import RandAugment
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm
from collections import Counter
import numpy as np
import os
import random
from PIL import Image, ImageStat
import timm

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# 1. 設定パラメータ
BATCH_SIZE    = 16
INPUT_SIZE    = 384         
NUM_EPOCHS    = 100
LEARNING_RATE = 4e-5
MIXUP_ALPHA   = 0.8
MIX_PROB      = 0.5          

# Step3: ArcFace
ARCFACE_S     = 16.0
ARCFACE_M     = 0.3

# Step2: TTA
TTA_TEMPERATURE = 5.0        

TRAIN_DIR          = "./vgg_dataset1/train"
VAL_DIR            = "./vgg_dataset1/val"
PRETRAINED_WEIGHTS = "./output/mae/best_mae_encoder_ema.pth"
MODEL_SAVE_DIR     = "./output/model_conv_mae"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

MODEL_NAME = 'convnextv2_large'
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 2. ユーティリティ
class SmartPadResize:
    def __init__(self, target_size):
        self.target_size = target_size

    def __call__(self, img):
        try:
            fill_color = tuple(int(v) for v in ImageStat.Stat(img).median)
        except Exception:
            fill_color = (255, 255, 255)
        w, h = img.size
        ratio = self.target_size / max(w, h)
        new_w, new_h = int(w * ratio), int(h * ratio)
        img = img.resize((new_w, new_h), Image.BICUBIC)
        new_img = Image.new("RGB", (self.target_size, self.target_size), fill_color)
        paste_x = (self.target_size - new_w) // 2
        paste_y = (self.target_size - new_h) // 2
        new_img.paste(img, (paste_x, paste_y))
        return new_img


class EMA:
    def __init__(self, model, decay=0.9999):
        self.model  = model
        self.decay  = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if name not in self.shadow:
                    self.shadow[name] = param.data.clone()
                new_avg = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_avg.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                if name in self.shadow:
                    param.data.copy_(self.shadow[name])

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if name in self.backup:
                    param.data.copy_(self.backup[name])
        self.backup = {}


# 3. Step3: ArcFace 
class ArcFaceHead(nn.Module):
    def __init__(self, in_features, num_classes, s=16.0, m=0.3):
        super().__init__()
        self.s      = s
        self.m      = m
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, features, labels=None):
        # FP32で計算
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


# 4. モデル定義 
class KuzushijiModel(nn.Module):
    def __init__(self, model_name, num_classes, embed_dim=512, pretrained=False):
        super().__init__()
        self.encoder = timm.create_model(
            model_name, pretrained=pretrained,
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


# 5. Mixup / CutMix
def mixup_data(x, y, alpha=0.8):
    lam   = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    index = torch.randperm(x.size(0)).to(x.device)
    mixed = lam * x + (1 - lam) * x[index]
    return mixed, y, y[index], lam


def cutmix_data(x, y, alpha=1.0):
    lam     = np.random.beta(alpha, alpha)
    index   = torch.randperm(x.size(0)).to(x.device)
    B, C, W, H = x.shape
    cut_rat = np.sqrt(1.0 - lam)
    cut_w   = int(W * cut_rat)
    cut_h   = int(H * cut_rat)
    cx, cy  = np.random.randint(W), np.random.randint(H)
    x1 = np.clip(cx - cut_w // 2, 0, W)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    y2 = np.clip(cy + cut_h // 2, 0, H)
    x  = x.clone()
    x[:, :, x1:x2, y1:y2] = x[index, :, x1:x2, y1:y2]
    lam = 1 - (x2 - x1) * (y2 - y1) / (W * H)
    return x, y, y[index], lam


def mix_criterion(criterion, pred, ya, yb, lam):
    return lam * criterion(pred, ya) + (1 - lam) * criterion(pred, yb)


# 6. Step2: TTA（推論時のみ・学習コストゼロ）
@torch.no_grad()
def predict_with_tta(model, inputs):
    model.eval()
    preds   = []
    weights = []

    with torch.amp.autocast('cuda'):
        logits = model(inputs, labels=None)
        preds.append(torch.softmax(logits / TTA_TEMPERATURE, dim=1))
        weights.append(2.0)

        # ±5度回転
        for angle in [-5, 5]:
            rotated = transforms.functional.rotate(inputs, angle, fill=0)
            logits  = model(rotated, labels=None)
            preds.append(torch.softmax(logits / TTA_TEMPERATURE, dim=1))
            weights.append(1.0)

    w_tensor = torch.tensor(weights, device=inputs.device).float()
    w_tensor = w_tensor / w_tensor.sum()
    result   = torch.zeros_like(preds[0])
    for p, w in zip(preds, w_tensor):
        result += w * p
    return result


# 7. データセット構築
train_transform = transforms.Compose([
    SmartPadResize(INPUT_SIZE),
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    RandAugment(num_ops=2, magnitude=9),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

minority_transform = transforms.Compose([
    SmartPadResize(INPUT_SIZE),
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.85, 1.15)),
    RandAugment(num_ops=3, magnitude=15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    SmartPadResize(INPUT_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

print("Loading Datasets...")
train_dataset_raw  = datasets.ImageFolder(TRAIN_DIR)
val_dataset_raw    = datasets.ImageFolder(VAL_DIR, transform=val_transform)
train_class_to_idx = train_dataset_raw.class_to_idx
train_classes      = train_dataset_raw.classes

new_val_samples = []
for path, old_idx in val_dataset_raw.samples:
    class_name = val_dataset_raw.classes[old_idx]
    if class_name in train_class_to_idx:
        new_val_samples.append((path, train_class_to_idx[class_name]))
val_dataset_raw.samples      = new_val_samples
val_dataset_raw.targets      = [s[1] for s in new_val_samples]
val_dataset_raw.imgs         = new_val_samples
val_dataset_raw.classes      = train_classes
val_dataset_raw.class_to_idx = train_class_to_idx

train_labels    = [label for _, label in train_dataset_raw.samples]
label_counts    = Counter(train_labels)
minority_labels = {label for label, cnt in label_counts.items() if cnt < 10}

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, minority_labels, normal_tf, minority_tf):
        self.dataset         = dataset
        self.minority_labels = minority_labels
        self.normal_tf       = normal_tf
        self.minority_tf     = minority_tf

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        path, label = self.dataset.samples[idx]
        img = Image.open(path).convert('RGB')
        if label in self.minority_labels:
            return self.minority_tf(img), label
        return self.normal_tf(img), label

train_dataset = CustomDataset(
    train_dataset_raw, minority_labels, train_transform, minority_transform
)

weights = [1.0 / (label_counts[l] ** 0.5) for l in train_labels]
max_w   = np.percentile(weights, 95)
weights = torch.DoubleTensor([min(w, max_w) for w in weights])
sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler,
                          num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_dataset_raw, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=4, pin_memory=True)


# 8. モデル構築とMAE重みロード
num_classes = len(train_classes)
print(f"Model: {MODEL_NAME}, Num Classes: {num_classes}")

model = KuzushijiModel(MODEL_NAME, num_classes, embed_dim=512, pretrained=False)

if os.path.exists(PRETRAINED_WEIGHTS):
    print(f"Loading MAE weights from {PRETRAINED_WEIGHTS}...")
    state_dict = torch.load(PRETRAINED_WEIGHTS, map_location='cpu',
                            weights_only=True)
    encoder_state = {}
    for k, v in state_dict.items():
        new_k = k.replace("encoder.", "") if k.startswith("encoder.") else k
        encoder_state[new_k] = v

    # 凍結されていた層をImageNet重みで補完
    base_model   = timm.create_model(MODEL_NAME, pretrained=True,
                                     num_classes=0, global_pool='avg')
    merged_state = base_model.state_dict().copy()
    merged_state.update(encoder_state)
    missing, unexpected = model.encoder.load_state_dict(merged_state, strict=False)
    print(f"  Missing: {len(missing)}, Unexpected: {len(unexpected)}")
    print("  MAE weights loaded.")
    del base_model
else:
    print("  MAE weights not found. Using ImageNet pretrained.")
    model = KuzushijiModel(MODEL_NAME, num_classes, embed_dim=512, pretrained=True)

model.to(DEVICE)


# 9. Optimizer / Loss / Scheduler / EMA
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.05)

ema = EMA(model, decay=0.9999)
ema.register()

warmup_epochs   = 5
cosine_epochs   = NUM_EPOCHS - warmup_epochs
steps_per_epoch = len(train_loader)

scheduler_warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0,
                             total_iters=warmup_epochs * steps_per_epoch)
scheduler_cosine = CosineAnnealingLR(optimizer,
                                     T_max=cosine_epochs * steps_per_epoch,
                                     eta_min=1e-6)
scheduler = SequentialLR(optimizer,
                         schedulers=[scheduler_warmup, scheduler_cosine],
                         milestones=[warmup_epochs * steps_per_epoch])

scaler = torch.amp.GradScaler('cuda')


# 10. 学習ループ
best_val_acc = 0.0
print(f"Start Training {MODEL_NAME} for {NUM_EPOCHS} epochs...")
print(f"  Input    : {INPUT_SIZE}x{INPUT_SIZE}")
print(f"  ArcFace  : s={ARCFACE_S}, m={ARCFACE_M}")
print(f"  Mixup/CutMix prob={MIX_PROB}")
print(f"  TTA      : temperature={TTA_TEMPERATURE}")

for epoch in range(1, NUM_EPOCHS + 1):

    # ── Train ──
    model.train()
    total_loss = total_correct = total_samples = 0.0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS} [Train]")

    for inputs, labels in pbar:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()

        # MixupとCutMixをランダムに切り替え
        if random.random() < MIX_PROB:
            if random.random() < 0.5:
                inputs, ta, tb, lam = cutmix_data(inputs, labels, alpha=1.0)
            else:
                inputs, ta, tb, lam = mixup_data(inputs, labels, MIXUP_ALPHA)

            with torch.amp.autocast('cuda'):
                outputs = model(inputs, labels=None)
                loss    = mix_criterion(criterion, outputs, ta, tb, lam)
        else:
            ta, tb, lam = labels, labels, 1.0
            with torch.amp.autocast('cuda'):
                outputs = model(inputs, labels=labels)
                loss    = criterion(outputs, labels)

        scaler.scale(loss).backward()

        scale_before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()

        if scale_before <= scaler.get_scale():
            scheduler.step()

        ema.update()

        total_loss    += loss.item() * inputs.size(0)
        preds          = outputs.argmax(dim=1)
        correct        = (preds == ta).float() * lam + (preds == tb).float() * (1 - lam)
        total_correct += correct.sum().item()
        total_samples += inputs.size(0)

        pbar.set_postfix({
            'loss': f"{total_loss / total_samples:.4f}",
            'acc' : f"{total_correct / total_samples:.4f}",
            'lr'  : f"{optimizer.param_groups[0]['lr']:.2e}",
        })

    train_acc = total_correct / total_samples

    # ── Validation (EMA重み + TTA) ──
    model.eval()
    ema.apply_shadow()

    total_loss_val = total_correct_val = total_samples_val = 0.0

    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS} [Val]"):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            # TTA予測
            probs = predict_with_tta(model, inputs)
            preds = probs.argmax(dim=1)

            # val loss
            with torch.amp.autocast('cuda'):
                outputs = model(inputs, labels=None)
                loss    = criterion(outputs, labels)

            total_loss_val    += loss.item() * inputs.size(0)
            total_correct_val += (preds == labels).sum().item()
            total_samples_val += inputs.size(0)

    ema.restore()

    val_acc  = total_correct_val / total_samples_val
    val_loss = total_loss_val    / total_samples_val

    print(f"Ep {epoch}: Train Acc {train_acc:.4f} | Val Acc {val_acc:.4f} (Loss: {val_loss:.4f})")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        save_path = os.path.join(MODEL_SAVE_DIR, "best_model.pth")
        torch.save(model.state_dict(), save_path)
        print(f"  🌟 Best Model Saved! ({val_acc:.4f})")

    if epoch % 10 == 0:
        torch.save(model.state_dict(),
                   os.path.join(MODEL_SAVE_DIR, f"epoch_{epoch}.pth"))

print(f"\n✅ Training Finished! Best Val Acc: {best_val_acc:.4f}")