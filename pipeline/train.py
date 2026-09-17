"""
Stage 4 — Season-aware crop classifier with Multi-Scale Temporal Attention.

Key design decisions (v2):
  - SEPARATE Rabi / Kharif models (timestamps differ by season).
  - Per-fold normalization prevents data leakage.
  - Spatial CV option via --spatial-cv (District-grouped folds).
  - Multi-Scale Temporal Attention CNN (MSTACNN) as primary architecture.
  - Random Forest baseline via --model rf.
  - Improved augmentation: jitter + magnitude scaling for minority classes.

Usage:
    python run.py train                          # MSTACNN, both seasons
    python run.py train --season rabi            # Rabi only
    python run.py train --model rf               # Random Forest baseline
    python run.py train --spatial-cv             # District-based spatial CV
    python run.py train --test-size 0.15         # 15% held-out test set
    python pipeline/train.py --model lstm --season kharif
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import json
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import (
    StratifiedKFold, train_test_split, GroupShuffleSplit,
)
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, cohen_kappa_score,
    classification_report, confusion_matrix,
)

from config import (
    PROCESSED_DIR, PROCESSED_PIXEL_DIR, MODELS_DIR, RESULTS_DIR,
    SEASON_MONTH_LABELS, SEASON_DEKADAL_LABELS, SEED, PATCH_SIZE,
    RAW_FEATURE_IDX,
)
from models import build_model, ATTENTION_MODELS, TORCH_MODELS
from losses import build_loss
from utils import plot_temporal_attention, plot_confusion_matrix

# StratifiedGroupKFold requires scikit-learn >= 1.0
try:
    from sklearn.model_selection import StratifiedGroupKFold
except ImportError:
    StratifiedGroupKFold = None


# ------------------------------------------------------------------ #
#  Augmentation
# ------------------------------------------------------------------ #

def augment_minority(X: np.ndarray, y: np.ndarray,
                     min_samples: int = 50,
                     noise_std: float = 0.05,
                     scale_std: float = 0.1,
                     groups: np.ndarray | None = None):
    """Oversample minority classes via jitter + magnitude scaling.

    Works for both 3D (N, T, C) and 5D (N, T, P, P, C) tensors. When ``groups``
    is given (e.g. district ids for region-adversarial training), the synthetic
    samples inherit the group of their source sample and the augmented group
    array is returned alongside X and y.
    """
    classes, counts = np.unique(y, return_counts=True)
    X_aug, y_aug = [X.copy()], [y.copy()]
    g_aug = [groups.copy()] if groups is not None else None

    # Build broadcastable shape for noise/scale: (need, 1, 1) or (need, 1, 1, 1, 1)
    n_extra_dims = X.ndim - 1  # dims after the batch dimension

    for cls, cnt in zip(classes, counts):
        if cnt >= min_samples:
            continue
        need = min_samples - cnt
        idx = np.where(y == cls)[0]
        chosen = np.random.choice(idx, size=need, replace=True)

        noise_shape = (need,) + X.shape[1:]
        noise = np.random.normal(0, noise_std, size=noise_shape)
        scale_shape = (need,) + (1,) * n_extra_dims
        scale = np.random.normal(1.0, scale_std, size=scale_shape)

        X_aug.append((X[chosen] * scale + noise).astype(np.float32))
        y_aug.append(np.full(need, cls, dtype=y.dtype))
        if groups is not None:
            g_aug.append(groups[chosen])

    if groups is not None:
        return np.concatenate(X_aug), np.concatenate(y_aug), np.concatenate(g_aug)
    return np.concatenate(X_aug), np.concatenate(y_aug)


# ------------------------------------------------------------------ #
#  Per-fold normalization
# ------------------------------------------------------------------ #

def normalize_fold(X_train: np.ndarray,
                   X_val: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit StandardScaler on training fold, transform both.

    Handles both 3D (N, T, C) field-level and 5D (N, T, P, P, C) pixel-level data.
    """
    shape_train = X_train.shape
    shape_val = X_val.shape
    C = shape_train[-1]  # last dim is always feature channels

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train.reshape(-1, C)).reshape(shape_train)
    X_va = scaler.transform(X_val.reshape(-1, C)).reshape(shape_val)

    return X_tr.astype(np.float32), X_va.astype(np.float32)


# ------------------------------------------------------------------ #
#  Training helpers
# ------------------------------------------------------------------ #

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y_batch)
        correct += (logits.argmax(1) == y_batch).sum().item()
        total += len(y_batch)
    return total_loss / total, correct / total


def train_one_epoch_radv(model, loader, criterion, adv_criterion, optimizer,
                         device, lambd, adv_weight):
    """One epoch of region-invariant training: crop-classification loss plus an
    adversarial district-classification loss routed through gradient reversal.
    ``lambd`` scales the reversed gradient (ramped over training); ``adv_weight``
    scales the adversary loss term."""
    model.train()
    model.grl_lambda = lambd
    total_loss, correct, total = 0.0, 0, 0
    for X_batch, y_batch, g_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        g_batch = g_batch.to(device)
        optimizer.zero_grad()
        logits, region_logits = model(X_batch, return_region=True)
        loss = criterion(logits, y_batch) + adv_weight * adv_criterion(region_logits, g_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y_batch)
        correct += (logits.argmax(1) == y_batch).sum().item()
        total += len(y_batch)
    return total_loss / total, correct / total


def train_one_epoch_conftr(model, loader, criterion, optimizer, device,
                           alpha, size_weight, temp, kappa=1.0):
    """One epoch of conformal-efficiency-aware training (ConfTr; Stutz et al.,
    Learning Optimal Conformal Classifiers, 2022). Each batch is split into a
    calibration and a prediction half; a differentiable conformal threshold is
    taken from the calibration half's true-class scores, soft prediction sets are
    formed on the prediction half, and a smooth set-size penalty is added to the
    base loss. The base loss keeps the true class probable (coverage) while the
    size term tightens the sets the conformal layer will later certify."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        B = X_batch.size(0)
        if B >= 8:
            perm = torch.randperm(B, device=device)
            half = B // 2
            cal, pred = perm[:half], perm[half:]
            scores = 1.0 - torch.softmax(logits, dim=1)        # THR/LAC score (B, C)
            s_true_cal = scores[cal, y_batch[cal]]             # true-class scores
            n = s_true_cal.numel()
            level = min(1.0, float(np.ceil((n + 1) * (1 - alpha)) / n))
            tau = torch.quantile(s_true_cal, level)            # differentiable threshold
            soft_sets = torch.sigmoid((tau - scores[pred]) / temp)  # (P, C) soft membership
            size_loss = torch.relu(soft_sets.sum(dim=1) - kappa).mean()
            loss = loss + size_weight * size_loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y_batch)
        correct += (logits.argmax(1) == y_batch).sum().item()
        total += len(y_batch)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels, all_attn = [], [], []
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        total_loss += loss.item() * len(y_batch)
        preds = logits.argmax(1)
        correct += (preds == y_batch).sum().item()
        total += len(y_batch)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())
        if hasattr(model, '_attn_weights') and model._attn_weights is not None:
            all_attn.append(model._attn_weights.cpu().numpy())

    attn = np.concatenate(all_attn) if all_attn else None
    return (total_loss / total, correct / total,
            np.array(all_preds), np.array(all_labels), attn)


def enable_mc_dropout(model: nn.Module) -> None:
    """Put only Dropout layers in train mode (BatchNorm stays in eval) so we can
    sample stochastic forward passes for MC-dropout uncertainty."""
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d)):
            m.train()


@torch.no_grad()
def predict_proba_uncertainty(model, X_norm, device, batch_size=256, mc_samples=20):
    """MC-dropout inference: T stochastic passes (dropout on, BN frozen).

    Returns mean softmax probabilities (N, C) and predictive entropy (N,) —
    the latter is the epistemic-uncertainty signal used for label cleaning.
    """
    model.eval()
    if mc_samples and mc_samples > 1:
        enable_mc_dropout(model)
    X_t = torch.tensor(X_norm, dtype=torch.float32)
    N = len(X_t)
    passes = max(1, mc_samples)
    probs_sum = None
    for _ in range(passes):
        chunks = []
        for i in range(0, N, batch_size):
            logits = model(X_t[i:i + batch_size].to(device))
            chunks.append(torch.softmax(logits, dim=1).cpu())
        p = torch.cat(chunks, dim=0)
        probs_sum = p if probs_sum is None else probs_sum + p
    mean_p = (probs_sum / passes).numpy()
    entropy = -(mean_p * np.log(np.clip(mean_p, 1e-12, 1.0))).sum(axis=1)
    return mean_p, entropy


# ------------------------------------------------------------------ #
#  Synthetic label noise (ablation only)
# ------------------------------------------------------------------ #

def inject_label_noise(y, rate, num_classes, rng, mode="symmetric", cmap=None):
    """Flip a `rate` fraction of labels (TRAINING only; val/test untouched).

    mode="symmetric": flip to a uniformly random different class.
    mode="asymmetric": flip to the confusion target cmap[c] (a confusion-structured
    scheme), where cmap is a per-class list of target classes. Used by the
    label-noise ablation to test noise robustness causally.
    """
    if rate <= 0 or num_classes < 2:
        return y
    y = np.asarray(y).copy()
    n_flip = int(round(rate * len(y)))
    if n_flip == 0:
        return y
    flip_idx = rng.choice(len(y), size=n_flip, replace=False)
    for i in flip_idx:
        if mode == "asymmetric" and cmap is not None:
            t = int(cmap[int(y[i])])
            if t != int(y[i]):
                y[i] = t
        else:
            alt = rng.randint(num_classes - 1)      # uniform over the other classes
            y[i] = alt if alt < y[i] else alt + 1
    return y


# ------------------------------------------------------------------ #
#  Season training
# ------------------------------------------------------------------ #

def train_season(season: str, args, data_dir: Path, device) -> pd.DataFrame:
    """Train and evaluate models for one season."""
    print(f"\n{'#' * 70}")
    print(f"  SEASON: {season.upper()}")
    print(f"{'#' * 70}")

    # ---- Load data ----
    is_pixel = args.model == "spatial-ssm"

    if is_pixel:
        pixel_dir = Path(args.data_dir).parent / "processed_pixel"
        X = np.load(pixel_dir / f"patches_{season}.npy")    # (N, T, P, P, C)
        y = np.load(pixel_dir / f"labels_{season}.npy")
        meta = pd.read_csv(pixel_dir / f"meta_{season}.csv")
        label_map = pd.read_csv(pixel_dir / "label_map.csv")
    else:
        X = np.load(data_dir / f"X_{season}.npy")
        y = np.load(data_dir / f"y_{season}.npy")
        meta = pd.read_csv(data_dir / f"meta_{season}.csv")
        label_map = pd.read_csv(data_dir / "label_map.csv")

    if (not is_pixel) and getattr(args, "feature_set", "all") == "raw":
        X = X[:, :, RAW_FEATURE_IDX]
        print(f"  Feature set: RAW spectral bands only "
              f"({len(RAW_FEATURE_IDX)} of 28 features, indices dropped)")
    global_class_names = label_map["crop_type"].tolist()

    # Remap global labels to season-local labels
    unique_global = sorted(np.unique(y))
    remap = {old: new for new, old in enumerate(unique_global)}
    class_names = [global_class_names[g] for g in unique_global]
    y_local = np.array([remap[yi] for yi in y])
    num_classes = len(class_names)

    if is_pixel:
        N, T, P1, P2, C = X.shape
        print(f"  Data: {N} pixel samples, T={T}, patch={P1}x{P2}, C={C}, {num_classes} classes")
    else:
        N, T, C = X.shape
        print(f"  Data: {N} samples, {T} timesteps, {C} features, {num_classes} classes")
    print(f"  Classes: {dict(zip(class_names, [int((y_local == c).sum()) for c in range(num_classes)]))}")

    # ---- Filter ultra-rare classes (BEFORE test split to avoid stratify error) ----
    if args.min_class_size > 0:
        counts = np.bincount(y_local, minlength=num_classes)
        keep_cls = [c for c in range(num_classes) if counts[c] >= args.min_class_size]
        if len(keep_cls) < num_classes:
            dropped = [class_names[c] for c in range(num_classes) if c not in keep_cls]
            print(f"  Dropping classes with < {args.min_class_size} samples: {dropped}")
            sample_mask = np.isin(y_local, keep_cls)
            X = X[sample_mask]
            y_local = y_local[sample_mask]
            meta = meta[sample_mask].reset_index(drop=True)
            old_to_new = {old: new for new, old in enumerate(keep_cls)}
            y_local = np.array([old_to_new[yi] for yi in y_local])
            class_names = [class_names[c] for c in keep_cls]
            num_classes = len(class_names)
            N = len(X)
            print(f"  After filtering: {N} samples, {num_classes} classes")
            print(f"  Classes: {dict(zip(class_names, [int((y_local == c).sum()) for c in range(num_classes)]))}")

    # ---- Held-out test split ----
    X_test, y_test, meta_test = None, None, None
    if args.test_size > 0:
        if args.spatial_test and "District" in meta.columns:
            gss = GroupShuffleSplit(n_splits=1, test_size=args.test_size,
                                    random_state=args.seed)
            train_idx, test_idx = next(
                gss.split(X, y_local, groups=meta["District"].values)
            )
            n_test_districts = meta.iloc[test_idx]["District"].nunique()
            print(f"  SPATIALLY-DISJOINT test split: {n_test_districts} held-out districts")
        else:
            train_idx, test_idx = train_test_split(
                np.arange(N), test_size=args.test_size,
                stratify=y_local, random_state=args.seed,
            )
        X_test, y_test = X[test_idx], y_local[test_idx]
        meta_test = meta.iloc[test_idx].reset_index(drop=True)
        X, y_local = X[train_idx], y_local[train_idx]
        meta = meta.iloc[train_idx].reset_index(drop=True)
        N = len(X)
        print(f"  Held-out test: {len(X_test)} samples  |  CV pool: {N} samples")

    # ---- Output dirs ----
    model_dir = Path(args.model_dir) / season
    results_dir = Path(args.results_dir) / season
    model_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # ---- Save full-data scaler for inference ----
    full_scaler = StandardScaler()
    full_scaler.fit(X.reshape(-1, X.shape[-1]))  # last dim is always C
    with open(model_dir / "scaler.pkl", "wb") as fp:
        pickle.dump(full_scaler, fp)
    inference_meta = {
        "season": season, "num_classes": num_classes,
        "class_names": class_names, "in_channels": X.shape[-1],
        "timesteps": X.shape[1], "model_type": args.model,
    }
    with open(model_dir / "inference_meta.json", "w") as fp:
        json.dump(inference_meta, fp, indent=2)
    print(f"  Saved scaler & inference metadata to {model_dir}/")

    # ---- CV setup ----
    if is_pixel:
        # Field-grouped CV prevents data leakage: all pixels from one field
        # stay in the same fold (train or val, never split across).
        if StratifiedGroupKFold is None:
            sys.exit("ERROR: StratifiedGroupKFold requires scikit-learn >= 1.0.")
        field_ids = meta["field_id"].values
        cv = StratifiedGroupKFold(n_splits=args.folds)
        split_iter = list(cv.split(X, y_local, groups=field_ids))
        cv_label = f"Field-grouped ({args.folds} folds)"
    elif args.spatial_cv:
        if StratifiedGroupKFold is None:
            sys.exit("ERROR: StratifiedGroupKFold requires scikit-learn >= 1.0.")
        districts = meta["District"].values
        cv = StratifiedGroupKFold(n_splits=args.folds)
        split_iter = list(cv.split(X, y_local, groups=districts))
        cv_label = f"Spatial (District-grouped, {args.folds} folds)"
    else:
        cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
        split_iter = list(cv.split(X, y_local))
        cv_label = f"Stratified Random ({args.folds} folds)"

    print(f"  CV: {cv_label}")

    # ---- Loss (imbalance-aware / noise-robust) ----
    class_counts = np.bincount(y_local, minlength=num_classes).astype(np.float32)
    criterion = None
    if args.model != "rf":
        criterion = build_loss(
            args.loss, torch.tensor(class_counts, dtype=torch.float32), device,
            use_class_weights=not args.no_class_weights,
        )
        print(f"  Loss: {args.loss} (class_weights={not args.no_class_weights})")

    # ---- Training loop ----
    fold_metrics = []
    all_fold_preds = np.full(N, -1, dtype=np.int64)
    all_fold_attn  = np.zeros((N, T), dtype=np.float32)
    all_fold_probs = np.zeros((N, num_classes), dtype=np.float32)
    all_fold_unc   = np.zeros(N, dtype=np.float32)
    training_log = []

    for fold, (train_idx, val_idx) in enumerate(split_iter):
        print(f"\n{'=' * 60}")
        print(f"  FOLD {fold + 1}/{args.folds}")
        print(f"{'=' * 60}")

        X_train_raw, y_train = X[train_idx], y_local[train_idx]
        X_val_raw,   y_val   = X[val_idx],   y_local[val_idx]

        if getattr(args, "label_noise", 0.0) and args.label_noise > 0:
            _rng = np.random.RandomState(args.seed + fold)
            _cmap = ([int(x) for x in args.noise_confusion.split(",")]
                     if getattr(args, "noise_confusion", None) else None)
            y_train = inject_label_noise(
                y_train, args.label_noise, num_classes, _rng,
                mode=getattr(args, "label_noise_type", "symmetric"), cmap=_cmap,
            )
            print(f"  Injected {args.label_noise:.0%} "
                  f"{getattr(args,'label_noise_type','symmetric')} label noise "
                  f"into TRAIN only ({len(y_train)} labels)")

        X_train_norm, X_val_norm = normalize_fold(X_train_raw, X_val_raw)

        # Conformal-efficiency-aware training (PhenoSSM only).
        conftr = (getattr(args, "conftr", False)
                  and args.model in ("ms-s4", "mss4", "ms4"))
        # Region-invariant training (PhenoSSM only): district ids for this fold,
        # mapped to contiguous indices, then oversampled in lock-step with X/y.
        region_adv = (getattr(args, "region_adv", False)
                      and args.model in ("ms-s4", "mss4", "ms4"))
        g_train_aug, n_regions = None, 0
        if region_adv:
            districts_fold = meta["District"].astype(str).to_numpy()[train_idx]
            g_train_raw = pd.factorize(districts_fold)[0].astype(np.int64)
            X_train_aug, y_train_aug, g_train_aug = augment_minority(
                X_train_norm, y_train, min_samples=args.min_samples, groups=g_train_raw
            )
            n_regions = int(g_train_aug.max()) + 1
        else:
            X_train_aug, y_train_aug = augment_minority(
                X_train_norm, y_train, min_samples=args.min_samples
            )
        print(f"  Train: {len(X_train_raw)} -> {len(X_train_aug)} (after augmentation)")
        print(f"  Val  : {len(X_val_raw)}")
        if region_adv:
            print(f"  Region-adversarial: {n_regions} training districts as domains")
        if conftr:
            print(f"  Conformal-aware training: size_weight={args.conftr_weight}, "
                  f"alpha={args.conftr_alpha}, temp={args.conftr_temp}")

        if args.model == "rf":
            rf = RandomForestClassifier(
                n_estimators=300, class_weight="balanced",
                max_depth=None, min_samples_leaf=2,
                random_state=args.seed, n_jobs=-1,
            )
            rf.fit(X_train_aug.reshape(len(X_train_aug), -1), y_train_aug)
            val_preds = rf.predict(X_val_norm.reshape(len(X_val_norm), -1))
            val_attn = None
            val_probs, val_unc = None, None
            if args.dump_oof:
                proba = rf.predict_proba(X_val_norm.reshape(len(X_val_norm), -1))
                val_probs = np.zeros((len(X_val_norm), num_classes), dtype=np.float32)
                val_probs[:, rf.classes_] = proba
                val_unc = -(np.clip(val_probs, 1e-12, 1) *
                            np.log(np.clip(val_probs, 1e-12, 1))).sum(1)

        else:
            if region_adv:
                train_ds = TensorDataset(
                    torch.tensor(X_train_aug, dtype=torch.float32),
                    torch.tensor(y_train_aug, dtype=torch.long),
                    torch.tensor(g_train_aug, dtype=torch.long),
                )
            else:
                train_ds = TensorDataset(
                    torch.tensor(X_train_aug, dtype=torch.float32),
                    torch.tensor(y_train_aug, dtype=torch.long),
                )
            val_ds = TensorDataset(
                torch.tensor(X_val_norm, dtype=torch.float32),
                torch.tensor(y_val, dtype=torch.long),
            )
            train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
            val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

            model = build_model(args.model, C, num_classes,
                                region_adv=region_adv, n_regions=n_regions).to(device)
            optimizer = optim.Adam(model.parameters(), lr=args.lr)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=7
            )
            adv_criterion = nn.CrossEntropyLoss() if region_adv else None

            best_val_f1 = 0.0
            patience_counter = 0
            best_path = model_dir / f"best_model_fold{fold + 1}.pt"

            for epoch in range(1, args.epochs + 1):
                if region_adv:
                    # DANN gradient-reversal ramp: 0 -> region_adv_lambda over training.
                    p = epoch / args.epochs
                    lambd = args.region_adv_lambda * (2.0 / (1.0 + np.exp(-10 * p)) - 1.0)
                    train_loss, train_acc = train_one_epoch_radv(
                        model, train_loader, criterion, adv_criterion, optimizer,
                        device, lambd, args.region_adv_weight
                    )
                elif conftr:
                    train_loss, train_acc = train_one_epoch_conftr(
                        model, train_loader, criterion, optimizer, device,
                        args.conftr_alpha, args.conftr_weight, args.conftr_temp
                    )
                else:
                    train_loss, train_acc = train_one_epoch(
                        model, train_loader, criterion, optimizer, device
                    )
                val_loss, val_acc, _vp, _vl, _ = evaluate(
                    model, val_loader, criterion, device
                )
                val_f1 = f1_score(_vl, _vp, average="weighted")
                scheduler.step(val_loss)

                training_log.append({
                    "season": season, "fold": fold + 1, "epoch": epoch,
                    "train_loss": round(train_loss, 4),
                    "train_acc": round(train_acc, 4),
                    "val_loss": round(val_loss, 4),
                    "val_acc": round(val_acc, 4),
                    "val_f1": round(val_f1, 4),
                    "lr": optimizer.param_groups[0]["lr"],
                })

                if epoch % 10 == 0 or epoch == 1:
                    print(f"  Epoch {epoch:3d}  "
                          f"train_loss={train_loss:.4f}  train_acc={train_acc:.3f}  "
                          f"val_loss={val_loss:.4f}  val_f1={val_f1:.3f}")

                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    patience_counter = 0
                    torch.save(model.state_dict(), best_path)
                else:
                    patience_counter += 1
                    if patience_counter >= args.patience:
                        print(f"  Early stopping at epoch {epoch}")
                        break

            model.load_state_dict(torch.load(best_path, weights_only=True))
            _, _, val_preds, val_labels, val_attn = evaluate(
                model, val_loader, criterion, device
            )
            val_probs, val_unc = None, None
            if args.dump_oof:
                val_probs, val_unc = predict_proba_uncertainty(
                    model, X_val_norm, device, args.batch_size, args.mc_samples
                )

        all_fold_preds[val_idx] = val_preds
        if val_attn is not None:
            all_fold_attn[val_idx] = val_attn
        if args.dump_oof and val_probs is not None:
            all_fold_probs[val_idx] = val_probs
            all_fold_unc[val_idx] = val_unc

        oa  = accuracy_score(y_val, val_preds)
        f1  = f1_score(y_val, val_preds, average="weighted")
        f1m = f1_score(y_val, val_preds, average="macro")
        kap = cohen_kappa_score(y_val, val_preds)

        fold_metrics.append({
            "fold": fold + 1, "OA": round(oa, 4),
            "Weighted_F1": round(f1, 4), "Macro_F1": round(f1m, 4),
            "Kappa": round(kap, 4),
        })
        print(f"\n  Fold {fold + 1}: OA={oa:.4f}  wF1={f1:.4f}  "
              f"mF1={f1m:.4f}  Kappa={kap:.4f}")

    # ---- Aggregate results ----
    print(f"\n{'=' * 60}")
    print(f"  {season.upper()} CROSS-VALIDATION SUMMARY ({args.model.upper()})")
    print(f"{'=' * 60}")

    metrics_df = pd.DataFrame(fold_metrics)
    print(metrics_df.to_string(index=False))
    for col in ["OA", "Weighted_F1", "Macro_F1", "Kappa"]:
        print(f"  Mean {col:<12s}: {metrics_df[col].mean():.4f} +/- {metrics_df[col].std():.4f}")

    valid_mask = all_fold_preds >= 0
    report = classification_report(
        y_local[valid_mask], all_fold_preds[valid_mask],
        target_names=class_names, digits=4, zero_division=0,
    )
    print(f"\n{report}")

    # ---- Out-of-fold probabilities + uncertainty (for label cleaning & calibration) ----
    if args.dump_oof:
        np.save(results_dir / "oof_probs.npy", all_fold_probs[valid_mask])
        np.save(results_dir / "oof_uncertainty.npy", all_fold_unc[valid_mask])
        np.save(results_dir / "oof_labels.npy", y_local[valid_mask])
        keep_cols = [c for c in ["ID", "District", "Province"] if c in meta.columns]
        if keep_cols:
            meta[valid_mask][keep_cols].reset_index(drop=True).to_csv(
                results_dir / "oof_meta.csv", index=False)
        np.save(results_dir / "oof_index.npy", np.where(valid_mask)[0])
        print(f"  Saved OOF artefacts for {int(valid_mask.sum())} CV samples "
              f"-> {results_dir}/oof_*.npy")

    # ---- Field-level majority vote (pixel models only) ----
    if is_pixel and valid_mask.any():
        from scipy import stats as sp_stats
        field_ids_cv = meta["field_id"].values[valid_mask]
        preds_cv = all_fold_preds[valid_mask]
        labels_cv = y_local[valid_mask]

        unique_fields = np.unique(field_ids_cv)
        field_preds_list, field_labels_list = [], []
        for fid in unique_fields:
            mask_f = field_ids_cv == fid
            majority_pred = int(sp_stats.mode(preds_cv[mask_f], keepdims=False).mode)
            majority_label = int(sp_stats.mode(labels_cv[mask_f], keepdims=False).mode)
            field_preds_list.append(majority_pred)
            field_labels_list.append(majority_label)

        field_preds_arr = np.array(field_preds_list)
        field_labels_arr = np.array(field_labels_list)
        field_oa = accuracy_score(field_labels_arr, field_preds_arr)
        field_f1 = f1_score(field_labels_arr, field_preds_arr, average="weighted")
        field_kap = cohen_kappa_score(field_labels_arr, field_preds_arr)

        print(f"\n  FIELD-LEVEL (majority vote over {len(unique_fields)} fields):")
        print(f"    OA={field_oa:.4f}  F1={field_f1:.4f}  Kappa={field_kap:.4f}")

        field_report = classification_report(
            field_labels_arr, field_preds_arr,
            target_names=class_names, digits=4, zero_division=0,
        )
        print(f"\n{field_report}")

    # ---- Held-out test evaluation ----
    if X_test is not None:
        print(f"\n{'=' * 60}")
        print(f"  HELD-OUT TEST SET EVALUATION ({season.upper()})")
        print(f"{'=' * 60}")

        # Find best fold by weighted F1
        best_fold_idx = max(range(len(fold_metrics)),
                           key=lambda i: fold_metrics[i]["Weighted_F1"])
        best_fold_num = fold_metrics[best_fold_idx]["fold"]
        print(f"  Using best fold: {best_fold_num} "
              f"(CV F1={fold_metrics[best_fold_idx]['Weighted_F1']:.4f})")

        # Normalize test data using full-data scaler
        X_test_flat = full_scaler.transform(
            X_test.reshape(-1, X_test.shape[-1])
        ).reshape(X_test.shape).astype(np.float32)

        if args.model == "rf":
            # Retrain RF on all CV data with best hyperparams
            X_all_norm = full_scaler.transform(
                X.reshape(-1, X.shape[-1])
            ).reshape(X.shape).astype(np.float32)
            rf_final = RandomForestClassifier(
                n_estimators=300, class_weight="balanced",
                max_depth=None, min_samples_leaf=2,
                random_state=args.seed, n_jobs=-1,
            )
            rf_final.fit(X_all_norm.reshape(len(X_all_norm), -1), y_local)
            test_preds = rf_final.predict(X_test_flat.reshape(len(X_test_flat), -1))
        else:
            best_path = model_dir / f"best_model_fold{best_fold_num}.pt"
            test_model = build_model(args.model, C, num_classes).to(device)
            # region-adv checkpoints carry an extra district head; ignore it at test.
            test_model.load_state_dict(
                torch.load(best_path, weights_only=True),
                strict=not getattr(args, "region_adv", False),
            )

            test_ds = TensorDataset(
                torch.tensor(X_test_flat, dtype=torch.float32),
                torch.tensor(y_test, dtype=torch.long),
            )
            test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
            criterion_test = nn.CrossEntropyLoss()
            _, _, test_preds, _, _ = evaluate(
                test_model, test_loader, criterion_test, device
            )

        test_oa = accuracy_score(y_test, test_preds)
        test_f1 = f1_score(y_test, test_preds, average="weighted")
        test_f1m = f1_score(y_test, test_preds, average="macro")
        test_kap = cohen_kappa_score(y_test, test_preds)

        test_report = classification_report(
            y_test, test_preds,
            target_names=class_names, digits=4, zero_division=0,
        )
        print(f"\n  Test OA={test_oa:.4f}  wF1={test_f1:.4f}  "
              f"mF1={test_f1m:.4f}  Kappa={test_kap:.4f}")
        print(f"\n{test_report}")

        # Save the exact held-out test split (raw, un-normalised) so the
        # standalone evaluator (calibration, risk-coverage, McNemar) reuses it.
        np.save(results_dir / "X_test.npy", X_test)
        np.save(results_dir / "y_test.npy", y_test)
        if meta_test is not None:
            meta_test.to_csv(results_dir / "meta_test.csv", index=False)

        # Save test artefacts
        with open(results_dir / "test_report.txt", "w") as f:
            f.write(f"Test OA={test_oa:.4f}  F1={test_f1:.4f}  Kappa={test_kap:.4f}\n\n")
            f.write(test_report)

        test_cm = confusion_matrix(y_test, test_preds)
        plot_confusion_matrix(
            test_cm, class_names,
            f"Test Confusion Matrix — {season.upper()} ({args.model.upper()})",
            results_dir / "test_confusion_matrix.png",
        )

        # Add test metrics to metrics_df
        test_row = pd.DataFrame([{
            "fold": "TEST", "OA": round(test_oa, 4),
            "Weighted_F1": round(test_f1, 4), "Macro_F1": round(test_f1m, 4),
            "Kappa": round(test_kap, 4),
        }])
        metrics_df = pd.concat([metrics_df, test_row], ignore_index=True)

    # Save artefacts
    metrics_df.to_csv(results_dir / "fold_metrics.csv", index=False)
    pd.DataFrame(training_log).to_csv(results_dir / "training_log.csv", index=False)
    with open(results_dir / "classification_report.txt", "w") as f:
        f.write(report)

    cm = confusion_matrix(y_local[valid_mask], all_fold_preds[valid_mask])
    plot_confusion_matrix(
        cm, class_names,
        f"Confusion Matrix — {season.upper()} ({args.model.upper()}, {cv_label})",
        results_dir / "confusion_matrix.png",
    )

    if args.model in ATTENTION_MODELS and all_fold_attn.sum() > 0:
        # Use dekadal labels if T=21, otherwise monthly labels
        if T == 21:
            months = SEASON_DEKADAL_LABELS.get(season, [f"t{i}" for i in range(T)])
        else:
            months = SEASON_MONTH_LABELS.get(season, [f"t{i}" for i in range(T)])
        plot_temporal_attention(
            all_fold_attn[valid_mask], y_local[valid_mask],
            class_names, months, results_dir / "temporal_attention.png",
        )

    config = {
        "season": season, "model": args.model, "loss": args.loss,
        "class_weights": not args.no_class_weights,
        "cv": cv_label, "spatial_cv": args.spatial_cv,
        "spatial_test": args.spatial_test,
        "folds": args.folds, "epochs": args.epochs,
        "batch_size": args.batch_size, "lr": args.lr,
        "patience": args.patience, "min_samples": args.min_samples,
        "seed": args.seed, "device": str(device),
        "samples": N, "num_classes": num_classes, "class_names": class_names,
    }
    with open(results_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n  Results saved to {results_dir}/")
    return metrics_df


# ------------------------------------------------------------------ #
#  Main
# ------------------------------------------------------------------ #

def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 4 — Train crop classifier.")
    parser.add_argument("--data-dir", default=str(PROCESSED_DIR))
    parser.add_argument("--model-dir", default=str(MODELS_DIR))
    parser.add_argument("--results-dir", default=str(RESULTS_DIR))
    parser.add_argument("--model",
                        choices=["mstacnn", "cnn", "tempcnn", "lstm", "ltae",
                                 "rf", "s4d", "ms-s4", "spatial-ssm", "transformer",
                                 "ms-s4-noattn", "ms-s4-noms", "ms-s4-bb"],
                        default="mstacnn",
                        help="Model architecture (default: mstacnn). "
                             "ms-s4 = proposed Multi-Scale State-Space model. "
                             "transformer = temporal Transformer baseline. "
                             "ms-s4-{noattn,noms,bb} = component ablations.")
    parser.add_argument("--season", choices=["rabi", "kharif", "both"],
                        default="both",
                        help="Season to train (default: both sequentially).")
    parser.add_argument("--loss", choices=["ce", "focal", "logit-adj", "gce", "sce"],
                        default="ce",
                        help="Training objective: ce | focal | logit-adj (long-tail) "
                             "| gce/sce (noise-robust). Default: ce.")
    parser.add_argument("--no-class-weights", action="store_true",
                        help="Disable inverse-frequency class weights (ce/focal/gce).")
    parser.add_argument("--spatial-cv", action="store_true",
                        help="Use District-based spatial cross-validation.")
    parser.add_argument("--spatial-test", action="store_true",
                        help="Hold out whole DISTRICTS for the test set (spatially disjoint).")
    parser.add_argument("--dump-oof", action="store_true",
                        help="Save out-of-fold probabilities + MC-dropout uncertainty "
                             "(needed by clean_labels.py and calibration).")
    parser.add_argument("--mc-samples", type=int, default=20,
                        help="MC-dropout passes for OOF uncertainty (with --dump-oof).")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--min-samples", type=int, default=50,
                        help="Min samples per class after augmentation.")
    parser.add_argument("--min-class-size", type=int, default=5,
                        help="Drop classes with fewer samples than this per season.")
    parser.add_argument("--test-size", type=float, default=0.0,
                        help="Fraction for stratified held-out test set (e.g. 0.15).")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--feature-set", choices=["all", "raw"], default="all",
                        help="all = 10 bands + 4 indices (28 feats, default); "
                             "raw = spectral bands only (20 feats). Feature ablation.")
    parser.add_argument("--label-noise", type=float, default=0.0,
                        help="Fraction of TRAINING labels to flip (val/test "
                             "untouched). Noise ablation.")
    parser.add_argument("--label-noise-type", choices=["symmetric", "asymmetric"],
                        default="symmetric", help="Noise structure for --label-noise.")
    parser.add_argument("--noise-confusion", default=None,
                        help="Comma-separated per-class target indices for "
                             "asymmetric (confusion-structured) noise.")
    parser.add_argument("--region-adv", action="store_true",
                        help="Region-invariant (district-adversarial) training for "
                             "PhenoSSM (ms-s4): gradient reversal on a district head "
                             "to narrow the geographic coverage gap at the source.")
    parser.add_argument("--region-adv-lambda", type=float, default=1.0,
                        help="Max gradient-reversal strength (ramped over training).")
    parser.add_argument("--region-adv-weight", type=float, default=0.5,
                        help="Weight on the adversarial district-loss term.")
    parser.add_argument("--conftr", action="store_true",
                        help="Conformal-efficiency-aware training for PhenoSSM (ms-s4): "
                             "differentiable set-size penalty (Stutz et al. 2022) added "
                             "to the base loss to tighten conformal sets.")
    parser.add_argument("--conftr-weight", type=float, default=0.1,
                        help="Weight on the ConfTr set-size penalty.")
    parser.add_argument("--conftr-alpha", type=float, default=0.1,
                        help="Target miscoverage for the in-training conformal threshold.")
    parser.add_argument("--conftr-temp", type=float, default=0.1,
                        help="Temperature for the soft set-membership sigmoid.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Model:  {args.model.upper()}")

    seasons = ["rabi", "kharif"] if args.season == "both" else [args.season]

    all_results = {}
    for season in seasons:
        # Pixel-level models use processed_pixel dir, others use processed dir
        if args.model == "spatial-ssm":
            pixel_dir = Path(str(PROCESSED_PIXEL_DIR))
            x_path = pixel_dir / f"patches_{season}.npy"
            m_path = pixel_dir / f"meta_{season}.csv"
        else:
            x_path = data_dir / f"X_{season}.npy"
            m_path = data_dir / f"meta_{season}.csv"

        if not x_path.exists():
            print(f"\n  WARNING: {x_path} not found — skipping {season}.")
            continue
        if not m_path.exists():
            sys.exit(f"ERROR: {m_path} not found. Run preprocessing first.")

        metrics_df = train_season(season, args, data_dir, device)
        all_results[season] = metrics_df

    if len(all_results) > 1:
        print(f"\n{'#' * 70}")
        print(f"  OVERALL SUMMARY (ALL SEASONS) — {args.model.upper()}")
        print(f"{'#' * 70}")
        for season, mdf in all_results.items():
            cv_only = mdf[mdf["fold"] != "TEST"]   # exclude held-out test row
            print(f"\n  {season.upper()}:")
            for col in ["OA", "Weighted_F1", "Macro_F1", "Kappa"]:
                if col in cv_only:
                    print(f"    Mean {col:<12s}: {cv_only[col].mean():.4f} "
                          f"+/- {cv_only[col].std():.4f}")

    print(f"\nStage 4 complete.\n")


if __name__ == "__main__":
    main()
