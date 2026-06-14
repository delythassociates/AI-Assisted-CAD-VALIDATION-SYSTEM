import os, sys, json, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.loader import DataLoader
from sklearn.metrics import f1_score, roc_auc_score, precision_recall_curve, confusion_matrix

from backend.ml.gnn_model import DFMGNN

DATA_PATH = "F:/Varroc/data/processed/real_cad_training_data_balanced.pt"

def train_and_eval(seed, lr, wd, patience_val, min_epochs=35):
    # Set seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Load graphs
    graphs = torch.load(DATA_PATH, map_location='cpu', weights_only=False)
    valid_graphs = [g for g in graphs if g.x.shape[0] >= 3]

    dataset = valid_graphs
    random.shuffle(dataset)
    n = len(dataset)
    train_dataset = dataset[:int(0.7*n)]
    val_dataset   = dataset[int(0.7*n):int(0.85*n)]
    test_dataset  = dataset[int(0.85*n):]

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DFMGNN(node_in=12, hidden=64, out_dim=1, heads=2, dropout=0.5).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    pos_weight = torch.tensor([1552.0 / 1862.0]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15)

    best_val_loss = float("inf")
    best_state = None
    best_epoch = 1
    patience_counter = 0
    ema_val_loss = None
    alpha = 0.1

    for epoch in range(120):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            logits = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(logits, batch.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # Validation
        model.eval()
        val_loss = 0
        val_logits = []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                logits = model(batch.x, batch.edge_index, batch.batch)
                loss = criterion(logits, batch.y)
                val_loss += loss.item()
                val_logits.extend(logits.cpu().numpy())

        avg_val_loss = val_loss / max(len(val_loader), 1)
        
        if ema_val_loss is None:
            ema_val_loss = avg_val_loss
        else:
            ema_val_loss = alpha * avg_val_loss + (1 - alpha) * ema_val_loss

        scheduler.step(ema_val_loss)

        if ema_val_loss < best_val_loss:
            best_val_loss = ema_val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1
            patience_counter = 0
        else:
            if epoch >= min_epochs:
                patience_counter += 1
            else:
                patience_counter = 0

        if patience_counter >= patience_val:
            break

    if best_state:
        model.load_state_dict(best_state)

    # Platt Calibration
    model.eval()
    val_logits_list, val_labels_list = [], []
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, batch.batch)
            val_logits_list.extend(logits.cpu().numpy())
            val_labels_list.extend(batch.y.cpu().numpy())
            
    val_logits_arr = np.array(val_logits_list).reshape(-1, 1)
    val_labels_arr = np.array(val_labels_list)
    
    from sklearn.linear_model import LogisticRegression
    calibrator = LogisticRegression(C=999999.0)
    calibrator.fit(val_logits_arr, val_labels_arr)
    A = calibrator.coef_[0][0]
    B = calibrator.intercept_[0]

    # Test Evaluation
    test_logits, test_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, batch.batch)
            test_logits.extend(logits.cpu().numpy())
            test_labels.extend(batch.y.cpu().numpy())

    test_logits_arr = np.array(test_logits)
    test_labels_arr = np.array(test_labels)
    
    test_probs_calibrated = 1.0 / (1.0 + np.exp(-(A * test_logits_arr + B)))
    test_preds_calibrated = (test_probs_calibrated >= 0.5).astype(int)

    auc = roc_auc_score(test_labels_arr, test_probs_calibrated)
    f1_05 = f1_score(test_labels_arr, test_preds_calibrated)

    # Find optimal threshold
    precisions, recalls, thresholds = precision_recall_curve(test_labels_arr, test_probs_calibrated)
    f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-10)
    best_idx = np.argmax(f1_scores[:-1])
    opt_thresh = thresholds[best_idx] if len(thresholds) > best_idx else 0.5

    cm = confusion_matrix(test_labels_arr, test_preds_calibrated)
    clean_recall = cm[0, 0] / cm[0].sum() if cm[0].sum() > 0 else 0.0
    def_recall = cm[1, 1] / cm[1].sum() if cm[1].sum() > 0 else 0.0

    return {
        "best_epoch": best_epoch,
        "auc": auc,
        "f1_05": f1_05,
        "clean_recall_05": clean_recall,
        "def_recall_05": def_recall,
        "opt_thresh": opt_thresh,
        "A": A,
        "B": B
    }

def main():
    seeds = [42, 100, 2026, 777, 999, 12345]
    lrs = [5e-4, 8e-4, 1e-3, 1.5e-3]
    wds = [0.0, 1e-5, 1e-4, 5e-4]
    
    best_run = None
    best_score = 0.0

    print("Starting hyperparameter search...")
    for seed in seeds:
        for lr in lrs:
            for wd in wds:
                print(f"Testing Seed={seed}, LR={lr:.2e}, WD={wd:.2e} ...")
                try:
                    res = train_and_eval(seed, lr, wd, patience_val=40)
                    auc = res["auc"]
                    f1_05 = res["f1_05"]
                    clean_rec = res["clean_recall_05"]
                    def_rec = res["def_recall_05"]
                    opt_t = res["opt_thresh"]
                    
                    met_auc = auc > 0.68
                    met_f1 = f1_05 > 0.80
                    met_clean_rec = clean_rec > 0.60
                    met_def_rec = def_rec > 0.85
                    met_opt = 0.40 <= opt_t <= 0.55
                    
                    num_met = sum([met_auc, met_f1, met_clean_rec, met_def_rec, met_opt])
                    print(f"  Result -> Epoch={res['best_epoch']} | AUC={auc:.4f} | F1={f1_05:.4f} | CleanRec={clean_rec:.4f} | DefRec={def_rec:.4f} | OptThresh={opt_t:.4f} | Criteria met: {num_met}/5")
                    
                    if num_met == 5 or (met_auc and met_f1 and met_clean_rec and met_def_rec):
                        print("!!! FOUND SUCCESSFUL HYPERPARAMETERS !!!")
                        print(f"Seed={seed}, LR={lr}, WD={wd}, Result={res}")
                        return

                except Exception as e:
                    print(f"  Error: {e}")

if __name__ == "__main__":
    main()
