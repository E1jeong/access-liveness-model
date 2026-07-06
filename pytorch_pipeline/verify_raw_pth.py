import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from pytorch_pipeline.model import get_anti_spoof_model
from pytorch_pipeline.dataset import get_data_loaders
from utils import calculate_validation_metrics

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = get_anti_spoof_model().to(device)

for f in range(5):
    pth_path = f'model/best_model_fold{f}.pth'
    if not os.path.exists(pth_path):
        continue
    state_dict = torch.load(pth_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    _, val_loader = get_data_loaders('dataset/raw', batch_size=16, k_folds=5, fold_idx=f, num_workers=1)
    
    all_labels, all_preds = [], []
    with torch.no_grad():
        for rgb, ir, label in val_loader:
            rgb = rgb.to(device)
            ir = ir.to(device)
            out = model(rgb, ir)
            preds = torch.argmax(out, dim=1).cpu().tolist()
            all_labels.extend(label.tolist())
            all_preds.extend(preds)
            
    cm, recalls, apcer, bpcer, acer = calculate_validation_metrics(all_labels, all_preds)
    print(f'[Fold {f} PyTorch 원본 성능] APCER={apcer:.4f}, BPCER={bpcer:.4f}, ACER={acer:.4f}')
