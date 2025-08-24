import os
import argparse
import random
from glob import glob
from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, models
from torchvision.models.video import r3d_18, R3D_18_Weights
from sklearn.metrics import roc_auc_score

# Optional SSIM
try:
    from pytorch_msssim import ssim as ms_ssim
    have_ms_ssim = True
except:
    have_ms_ssim = False


# ---------------- Utilities -----------------
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------- Dataset ------------------
class MultiVideoDataset(Dataset):
    def __init__(self, roots, seq_len=4, resize=(128, 128), cache_flow=True, transform=None):
        self.frame_paths = []  # tuples (seq_paths, target_path)
        self.seq_len = seq_len
        self.resize = resize
        self.transform = transform
        self.cache_flow = cache_flow

        for root in roots:
            root = Path(root)
            for video_dir in sorted(root.rglob('*')):
                if video_dir.is_dir():
                    imgs = []
                    for ext in ('*.tif', '*.png', '*.jpg', '*.jpeg'):
                        imgs.extend(video_dir.glob(ext))
                    imgs = sorted(imgs)
                    if len(imgs) < seq_len + 1:
                        continue
                    for i in range(len(imgs) - seq_len):
                        seq = imgs[i:i+seq_len]
                        target = imgs[i+seq_len]
                        self.frame_paths.append((seq, target))

    def __len__(self):
        return len(self.frame_paths)

    def read_img(self, p):
        img = cv2.imread(str(p))
        if img is None:
            raise RuntimeError(f"Failed to read image {p}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.resize)
        img = img.astype(np.float32) / 255.0
        return img

    def compute_optical_flow(self, prev, next, cache_path=None):
        prev_gray = cv2.cvtColor((prev * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        next_gray = cv2.cvtColor((next * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        tvl1 = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        flow = tvl1.calc(prev_gray, next_gray, None)
        flow = np.clip(flow, -20, 20) / 20.0
        if cache_path is not None:
            np.save(cache_path, flow)
        return flow

    def __getitem__(self, idx):
        seq_paths, target_path = self.frame_paths[idx]
        seq = [self.read_img(p) for p in seq_paths]
        target = self.read_img(target_path)

        flows = []
        for i in range(len(seq) - 1):
            p1, p2 = seq[i], seq[i + 1]
            cache_path = None
            if self.cache_flow:
                tmpdir = Path('flow_cache')
                tmpdir.mkdir(exist_ok=True)
                cache_name = str(seq_paths[i]).replace('/', '_').replace('\\', '_') + '_flow.npy'
                cache_path = tmpdir / Path(cache_name).name
                if cache_path.exists():
                    flows.append(np.load(cache_path))
                    continue
            flows.append(self.compute_optical_flow(p1, p2, str(cache_path) if cache_path else None))

        seq_np = np.stack(seq, axis=0)  # T,H,W,3
        seq_np = np.transpose(seq_np, (3, 0, 1, 2))  # 3,T,H,W

        flows_np = np.stack(flows, axis=0)
        last = flows_np[-1:]
        flows_np = np.concatenate([flows_np, last], axis=0)
        flows_np = np.transpose(flows_np, (3, 0, 1, 2))  # 2,T,H,W

        target_np = np.transpose(target, (2, 0, 1))  # 3,H,W

        return {
            'rgb_seq': torch.tensor(seq_np, dtype=torch.float32),
            'flow_seq': torch.tensor(flows_np, dtype=torch.float32),
            'target': torch.tensor(target_np, dtype=torch.float32)
        }


# ---------------- Model ------------------
class TwoStreamPredictor(nn.Module):
    def __init__(self, in_channels_rgb=3, in_channels_flow=2, feature_dim=512, seq_len=4):
        super().__init__()
        # RGB backbone
        self.rgb_backbone = r3d_18(weights=R3D_18_Weights.KINETICS400_V1)
        self.rgb_backbone.fc = nn.Identity()
        # Flow backbone
        self.flow_backbone = r3d_18(weights=None)
        self.flow_backbone.stem[0] = nn.Conv3d(
            in_channels_flow, 64, kernel_size=(3, 7, 7),
            stride=(1, 2, 2), padding=(1, 3, 3), bias=False
        )
        self.flow_backbone.fc = nn.Identity()

        # Project features to common dim
        self.rgb_proj = nn.Linear(512, feature_dim)
        self.flow_proj = nn.Linear(512, feature_dim)

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim, nhead=8,
            dim_feedforward=feature_dim*4, dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)

        # Decoder: use feature_dim as input channels
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(feature_dim, 256, kernel_size=(1,4,4), stride=(1,2,2), padding=(0,1,1)),  # 1->2
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(256, 128, kernel_size=(1,4,4), stride=(1,2,2), padding=(0,1,1)),         # 2->4
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(128, 64, kernel_size=(1,4,4), stride=(1,2,2), padding=(0,1,1)),          # 4->8
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(64, 32, kernel_size=(1,4,4), stride=(1,2,2), padding=(0,1,1)),           # 8->16
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(32, 16, kernel_size=(1,4,4), stride=(1,2,2), padding=(0,1,1)),           # 16->32
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(16, 8, kernel_size=(1,4,4), stride=(1,2,2), padding=(0,1,1)),            # 32->64
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(8, 3, kernel_size=(1,4,4), stride=(1,2,2), padding=(0,1,1)),             # 64->128
            nn.Sigmoid()
        )


        self.seq_len = seq_len
        self.feature_dim = feature_dim

    def forward(self, rgb_seq, flow_seq):
        B, C, T, H, W = rgb_seq.shape

        # Backbone features
        rgb_feat = self.rgb_backbone(rgb_seq)  # [B, 512]
        flow_feat = self.flow_backbone(flow_seq)  # [B, 512]

        # Project to common feature dim
        r = self.rgb_proj(rgb_feat)  # [B, feature_dim]
        f = self.flow_proj(flow_feat)  # [B, feature_dim]

        # Combine and feed transformer
        token = torch.stack([r, f], dim=1)  # [B, 2, feature_dim]
        token = token.permute(1, 0, 2)  # [2, B, feature_dim]
        transformed = self.transformer(token)  # [2, B, feature_dim]
        fused = transformed.permute(1, 0, 2).mean(dim=1)  # [B, feature_dim]

        # Reshape for decoder: [B, feature_dim, 1, 1, 1] -> allow ConvTranspose3d
        x = fused.view(B, self.feature_dim, 1, 1, 1)  # [B, feature_dim, 1, 1, 1]

        out = self.decoder(x)  # [B, 3, T, H, W]
        out = out.squeeze(2)  # remove depth dimension if T=1
        return out



# ---------------- Loss ------------------
class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features.eval()
        for p in vgg.parameters():
            p.requires_grad = False
        self.vgg = vgg

    def forward(self, pred, target):
        def preprocess(x):
            x = x * 255.0
            mean = torch.tensor([123.68, 116.779, 103.939], device=x.device).view(1, 3, 1, 1)
            return x - mean
        f_p = self.vgg[:16](preprocess(pred))
        f_t = self.vgg[:16](preprocess(target))
        return F.l1_loss(f_p, f_t)


# ---------------- Training ------------------
def train_one_epoch(model, loader, optim, device, scaler, loss_fns):
    model.train()
    total_loss = 0.0
    for batch in tqdm(loader, desc='train'):
        rgb = batch['rgb_seq'].to(device)
        flow = batch['flow_seq'].to(device)
        target = batch['target'].to(device)
        optim.zero_grad()
        with torch.amp.autocast(device_type=device):
            pred = model(rgb, flow)
            loss_mse = F.mse_loss(pred, target)
            loss_perc = loss_fns['perc'](pred, target) if loss_fns.get('perc') else 0.0
            loss_ssim = 1.0 - ms_ssim(pred, target, data_range=1.0) if have_ms_ssim else torch.tensor(0.0, device=device)
            loss = loss_mse + 0.1 * loss_perc + 0.5 * loss_ssim
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            loss.backward()
            optim.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def validate(model, loader, device):
    model.eval()
    all_scores = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(loader, desc='val'):
            rgb = batch['rgb_seq'].to(device)
            flow = batch['flow_seq'].to(device)
            target = batch['target'].to(device)
            pred = model(rgb, flow)
            score = F.mse_loss(pred, target, reduction='none').mean(dim=[1, 2, 3]).cpu().numpy()
            all_scores.extend(score.tolist())
            all_labels.extend([0] * len(score))
    return all_scores, all_labels


# ---------------- Main ------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ucsd_root', type=str, default=r'C:\Users\msaya\Downloads\survey\dataanomaly\UCSD_Anomaly_Dataset.v1p2')
    parser.add_argument('--avenue_root', type=str, default=r'C:\Users\msaya\Downloads\survey\dataanomaly\Avenue Dataset')
    parser.add_argument('--exp_dir', type=str, default='./exp')
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seq_len', type=int, default=4)
    parser.add_argument('--resize', type=int, default=128)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.exp_dir, exist_ok=True)

    # Collect dataset roots
    roots = []
    u_root = Path(args.ucsd_root)
    if u_root.exists():
        roots.append(str(u_root))
    a_root = Path(args.avenue_root)
    if a_root.exists():
        roots.append(str(a_root))
    if len(roots) == 0:
        raise RuntimeError('No dataset roots found. Check paths.')

    dataset = MultiVideoDataset(roots, seq_len=args.seq_len, resize=(args.resize, args.resize), cache_flow=True)
    n = len(dataset)
    idxs = list(range(n))
    random.shuffle(idxs)
    split = int(0.95 * n)
    train_ds = Subset(dataset, idxs[:split])
    val_ds = Subset(dataset, idxs[split:])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model = TwoStreamPredictor(seq_len=args.seq_len).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler() if device == 'cuda' else None
    loss_fns = {'perc': PerceptualLoss().to(device)}

    best_val_loss = 1e9
    for epoch in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, device, scaler, loss_fns)
        val_scores, val_labels = validate(model, val_loader, device)
        val_mean_score = float(np.mean(val_scores))
        print(f"Epoch {epoch}/{args.epochs} train_loss={tr_loss:.6f} val_mean_score={val_mean_score:.6f}")
        scheduler.step()
        ckpt_path = os.path.join(args.exp_dir, f'ckpt_epoch{epoch}.pth')
        torch.save({'epoch': epoch, 'model': model.state_dict(), 'optim': optimizer.state_dict()}, ckpt_path)
        if val_mean_score < best_val_loss:
            best_val_loss = val_mean_score
            torch.save({'epoch': epoch, 'model': model.state_dict(), 'optim': optimizer.state_dict()}, os.path.join(args.exp_dir, 'best.pth'))

    print('Training complete. Best val mean score:', best_val_loss)


if __name__ == '__main__':
    main()
