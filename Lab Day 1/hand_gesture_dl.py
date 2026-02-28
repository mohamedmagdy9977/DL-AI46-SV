import os
import argparse
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


SEED       = 42
BATCH_SIZE = 256
EPOCHS     = 60
LR         = 3e-3
PATIENCE   = 10   # Early stopping patience

PALETTE = [
    '#00d4ff','#e94560','#05c46b','#ffd460','#533483',
    '#f4a261','#2d6a4f','#457b9d','#a8dadc','#e76f51',
    '#d62828','#1d3557','#2a9d8f','#e9c46a','#264653',
    '#f77f00','#fcbf49','#e63946',
]

# Dark plotting theme matching the original notebook
plt.rcParams.update({
    'figure.facecolor' : '#0f0f1a',
    'axes.facecolor'   : '#1a1a2e',
    'axes.edgecolor'   : '#444',
    'axes.labelcolor'  : '#ccc',
    'xtick.color'      : '#aaa',
    'ytick.color'      : '#aaa',
    'text.color'       : '#eee',
    'grid.color'       : '#333',
    'grid.linestyle'   : '--',
    'grid.alpha'       : 0.5,
})


# ══════════════════════════════════════════════════════════════════════════════
# 1. REPRODUCIBILITY & DEVICE
# ══════════════════════════════════════════════════════════════════════════════

def setup(seed: int = SEED):
    """
    Fix all random seeds and select compute device.

    Setting seeds ensures weight initialization and data shuffling produce
    the same results on every run — essential for fair comparisons.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'🖥️  Using device: {device}')
    return device


# ══════════════════════════════════════════════════════════════════════════════
# 2. DATA LOADING & EDA
# ══════════════════════════════════════════════════════════════════════════════

def load_data(csv_path: str = 'hand_landmarks_data.csv') -> pd.DataFrame:
    """
    Load MediaPipe landmark CSV.

    Dataset: 21 hand landmarks × (x, y, z) = 63 features + 1 label column.
    """
    df = pd.read_csv(csv_path)
    print(f'Shape  : {df.shape}')
    print(f'Labels : {sorted(df["label"].unique().tolist())}')
    print(f'Nulls  : {df.isnull().sum().sum()}')
    return df


def plot_class_distribution(df: pd.DataFrame):
    """
    Visualise sample counts per gesture class.

    Understanding class balance is critical before training.
    Imbalanced classes → the network can 'cheat' by predicting the majority
    class. We handle this later with class weights in the loss function.
    """
    fig, ax = plt.subplots(figsize=(14, 4))
    counts = df['label'].value_counts().sort_index()
    bars   = ax.bar(counts.index, counts.values, color=PALETTE[:len(counts)])
    ax.bar_label(bars, fmt='%d', padding=3, color='#eee', fontsize=9)
    ax.set_title('Class Distribution — Gesture Samples', color='#00d4ff', fontsize=14)
    ax.set_xlabel('Gesture')
    ax.set_ylabel('Count')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.show()
    print(f'Min class: {counts.min()}  |  Max class: {counts.max()}')
    print(f'Imbalance ratio: {counts.max()/counts.min():.1f}x')


# ══════════════════════════════════════════════════════════════════════════════
# 3. PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def preprocess(df: pd.DataFrame, seed: int = SEED):
    """
    Encode labels, split into train/val/test, and apply StandardScaler.

    Why three splits?
        • train (70%) — model learns weights via gradient descent
        • val   (15%) — monitor overfitting & trigger early stopping
                        (NEVER used for final reporting — that leaks info)
        • test  (15%) — single unbiased evaluation at the very end

    Why StandardScaler is critical in DL:
        Without it, features with large magnitudes (pixel coords ~200–400)
        dominate gradients during backpropagation, causing training
        instability (vanishing / exploding gradients).

    Why we fit scaler ONLY on train:
        Fitting on val/test would leak their statistics into training,
        giving an unfairly optimistic evaluation.
    """
    X     = df.drop(columns=['label']).values.astype(np.float32)
    y_raw = df['label'].values

    # LabelEncoder: maps strings → integers ('call'→0, 'dislike'→1, ...)
    # CrossEntropyLoss requires integer class indices, NOT one-hot vectors.
    le          = LabelEncoder()
    y           = le.fit_transform(y_raw).astype(np.int64)
    num_classes = len(le.classes_)
    print(f'Classes ({num_classes}): {list(le.classes_)}')

    # Stratified splits — each split mirrors the full class distribution
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=seed, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=seed, stratify=y_temp)

    print(f'Train: {X_train.shape[0]:,}  |  Val: {X_val.shape[0]:,}  |  Test: {X_test.shape[0]:,}')

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val   = scaler.transform(X_val).astype(np.float32)
    X_test  = scaler.transform(X_test).astype(np.float32)

    print(f'\n✅ Preprocessing complete.')
    print(f'Feature mean ≈ {X_train.mean():.4f}  (should be ~0)')
    print(f'Feature std  ≈ {X_train.std():.4f}  (should be ~1)')

    return X_train, X_val, X_test, y_train, y_val, y_test, le, scaler, num_classes


def make_loaders(X_train, X_val, X_test, y_train, y_val, y_test,
                 batch_size: int = BATCH_SIZE):
    """
    Wrap NumPy arrays as PyTorch TensorDatasets and return DataLoaders.

    PyTorch Tensors support automatic differentiation (autograd) —
    the core mechanism that makes backpropagation possible.

    Batch size trade-off:
        • Small (e.g. 32)  → noisy gradients, implicit regularization
        • Large (e.g. 1024) → stable gradients, faster epochs, sharp minima risk
        • 256 is a sweet spot for tabular data of this scale
    """
    def _ds(X, y):
        return TensorDataset(torch.from_numpy(X), torch.from_numpy(y))

    train_loader = DataLoader(_ds(X_train, y_train), batch_size=batch_size,
                              shuffle=True, drop_last=True)
    val_loader   = DataLoader(_ds(X_val,   y_val),   batch_size=batch_size,
                              shuffle=False)
    test_loader  = DataLoader(_ds(X_test,  y_test),  batch_size=batch_size,
                              shuffle=False)

    print(f'Train batches: {len(train_loader)}  |  Val batches: {len(val_loader)}')
    return train_loader, val_loader, test_loader


# ══════════════════════════════════════════════════════════════════════════════
# 4. MODEL ARCHITECTURE
# ══════════════════════════════════════════════════════════════════════════════

class GestureNet(nn.Module):
    """
    Multi-Layer Perceptron for 18-class hand gesture classification.

    Architecture (bottleneck funnel):
        63 → [256] → [128] → [64] → 18

    Each hidden block follows the modern recipe:
        Linear → BatchNorm → ReLU → Dropout

    Layer explanations:
        Linear    : y = Wx + b  — the actual 'learning' via weights W
        BatchNorm : Normalises activations within each mini-batch.
                    Reduces internal covariate shift, stabilises training,
                    and allows higher learning rates.
        ReLU      : f(x) = max(0, x) — non-linearity that enables the
                    Universal Approximation theorem to apply.
                    Without it, stacked Linears collapse to one matrix.
        Dropout   : Randomly zeros p% of neurons each forward pass.
                    Forces redundant representations → prevents overfitting.

    Output: raw logits (NOT softmax).
        CrossEntropyLoss internally applies LogSoftmax + NLLLoss,
        which is numerically more stable than Softmax then CrossEntropy.
    """

    def __init__(self, input_dim: int, num_classes: int, dropout_p: float = 0.3):
        super().__init__()

        # Block 1: expand feature space — room to discover landmark combinations
        self.block1 = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_p),
        )

        # Block 2: compress — form abstract gesture features
        self.block2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_p),
        )

        # Block 3: compact gesture embedding space
        self.block3 = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_p * 0.5),   # less dropout near output
        )

        # Classifier head: project embedding → class logits
        self.head = nn.Linear(64, num_classes)

        self._init_weights()

    def _init_weights(self):
        """
        He (Kaiming) initialisation: scale weights by sqrt(2 / fan_in).
        Designed for ReLU activations to maintain activation variance
        across layers — prevents vanishing/exploding gradients at init.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: data flows sequentially through each block.

        Theory: this is the 'inference' direction.
        The backward pass (backpropagation) flows gradients in reverse,
        applying the chain rule of calculus to update every weight.
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.head(x)


def build_model(input_dim: int, num_classes: int, device: torch.device):
    """Instantiate GestureNet and move it to the target device."""
    model = GestureNet(input_dim=input_dim, num_classes=num_classes, dropout_p=0.3)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable    = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total parameters : {total_params:,}')
    print(f'Trainable params : {trainable:,}')
    print(model)
    return model


# ══════════════════════════════════════════════════════════════════════════════
# 5. LOSS, OPTIMIZER & SCHEDULER
# ══════════════════════════════════════════════════════════════════════════════

def build_optimizer(model, y_train, num_classes, device,
                    lr: float = LR, epochs: int = EPOCHS):
    """
    Set up weighted CrossEntropyLoss, AdamW optimizer, and cosine LR schedule.

    Class weights:
        w_c = total_samples / (num_classes × count_c)
        Rare classes get higher weight → network pays them more attention.

    AdamW optimizer:
        Gradient descent + momentum + per-parameter adaptive LR + weight decay.
        weight_decay=1e-4 is L2 regularisation — equivalent to Ridge in classical ML.

    CosineAnnealingLR:
        Smoothly decays LR from `lr` to `eta_min` over T_max epochs.
        Like a ball rolling into a valley with ever-smaller steps — helps
        the model settle into a sharp, generalising minimum.
    """
    class_counts  = np.bincount(y_train)
    class_weights = torch.tensor(
        len(y_train) / (num_classes * class_counts.astype(np.float32)),
        dtype=torch.float32
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    print(f'Loss     : CrossEntropyLoss (with class weights)')
    print(f'Optimizer: AdamW  (lr={lr}, weight_decay=1e-4)')
    print(f'Scheduler: CosineAnnealingLR  (T_max={epochs})')
    return criterion, optimizer, scheduler


# ══════════════════════════════════════════════════════════════════════════════
# 6. TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════════

def train_epoch(model, loader, criterion, optimizer, device):
    """
    One full pass over the training set.

    The DL training cycle per batch:
        1. Forward pass  : data → network → logits
        2. Loss          : CrossEntropy(logits, targets)
        3. Backward pass : loss.backward() — chain rule → ∂L/∂w for all w
        4. Gradient clip : rescale if norm > 1.0 (prevents exploding gradients)
        5. Weight update : w ← w − lr × ∇w  (AdamW step)

    model.train() enables Dropout & BatchNorm training-time behaviour.
    """
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()                   # Clear accumulated gradients
        logits = model(X_batch)                 # Forward pass
        loss   = criterion(logits, y_batch)     # Compute loss
        loss.backward()                         # Backprop: ∂L/∂w for all w

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()                        # w ← w − lr × ∇w

        total_loss += loss.item() * y_batch.size(0)
        preds   = logits.argmax(dim=1)          # class = argmax of logits
        correct += (preds == y_batch).sum().item()
        total   += y_batch.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """
    Evaluate on val or test split.

    model.eval() is CRITICAL:
        • Disables Dropout  → all neurons active (inference mode)
        • Fixes BatchNorm statistics → uses running mean/var, not batch stats
    Forgetting model.eval() gives incorrect (pessimistic) val metrics!

    @torch.no_grad() disables gradient tracking — saves memory & speeds up.
    """
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        logits = model(X_batch)
        loss   = criterion(logits, y_batch)

        total_loss += loss.item() * y_batch.size(0)
        preds   = logits.argmax(dim=1)
        correct += (preds == y_batch).sum().item()
        total   += y_batch.size(0)

    return total_loss / total, correct / total


def train(model, train_loader, val_loader, test_loader,
          criterion, optimizer, scheduler, device,
          epochs: int = EPOCHS, patience: int = PATIENCE):
    """
    Full training loop with early stopping and best-checkpoint restoration.

    Early Stopping:
        If val loss does not improve for `patience` consecutive epochs,
        halt training and restore the best saved weights.
        This prevents overfitting without needing a separate regularisation
        hyperparameter — purely a DL training-time technique.
    """
    best_val_loss = float('inf')
    no_improve    = 0
    best_state    = None
    history       = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    print(f"\n{'Epoch':>5} | {'Train Loss':>10} | {'Train Acc':>9} | {'Val Loss':>8} | {'Val Acc':>7} | LR")
    print('-' * 65)

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        vl_loss, vl_acc = evaluate(model, val_loader,   criterion,          device)
        scheduler.step()

        history['train_loss'].append(tr_loss)
        history['val_loss'].append(vl_loss)
        history['train_acc'].append(tr_acc)
        history['val_acc'].append(vl_acc)

        current_lr = scheduler.get_last_lr()[0]
        if epoch % 5 == 0 or epoch == 1:
            print(f'{epoch:>5} | {tr_loss:>10.4f} | {tr_acc:>9.4f} | '
                  f'{vl_loss:>8.4f} | {vl_acc:>7.4f} | {current_lr:.2e}')

        # ── Early Stopping ────────────────────────────────────────────────────
        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            no_improve    = 0
            # Clone best weights — do NOT keep pointer to last epoch weights
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f'\n⏹  Early stopping at epoch {epoch} '
                      f'(no val improvement for {patience} epochs)')
                break

    # Restore the checkpoint that generalised best on the validation set
    model.load_state_dict(best_state)

    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f'\n🏆 Best Val Loss  : {best_val_loss:.4f}')
    print(f'🏆 Test Accuracy  : {test_acc:.4f} ({test_acc * 100:.2f}%)')

    return history, test_acc, test_loss


# ══════════════════════════════════════════════════════════════════════════════
# 7. VISUALISATION
# ══════════════════════════════════════════════════════════════════════════════

def plot_training_curves(history: dict, test_acc: float):
    """
    Plot loss and accuracy curves.

    Healthy training signs:
        • Train and val curves converge, val slightly behind train
    Overfitting sign:
        • Train accuracy >> Val accuracy → increase dropout / reduce model size
    Underfitting sign:
        • Both accuracies low → increase capacity or train longer
    """
    epochs_ran = range(1, len(history['train_loss']) + 1)
    fig, axes  = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(epochs_ran, history['train_loss'], color='#00d4ff', lw=2, label='Train Loss')
    ax.plot(epochs_ran, history['val_loss'],   color='#e94560', lw=2,
            label='Val Loss', linestyle='--')
    ax.set_title('Loss Curve', color='#00d4ff', fontsize=13)
    ax.set_xlabel('Epoch')  ;  ax.set_ylabel('CrossEntropy Loss')
    ax.legend()  ;  ax.grid(True)

    ax = axes[1]
    ax.plot(epochs_ran, [a * 100 for a in history['train_acc']],
            color='#05c46b', lw=2, label='Train Acc')
    ax.plot(epochs_ran, [a * 100 for a in history['val_acc']],
            color='#ffd460', lw=2, label='Val Acc', linestyle='--')
    ax.set_title('Accuracy Curve', color='#00d4ff', fontsize=13)
    ax.set_xlabel('Epoch')  ;  ax.set_ylabel('Accuracy (%)')
    ax.legend()  ;  ax.grid(True)

    fig.suptitle('GestureNet Training History', color='#a8dadc', fontsize=15)
    plt.tight_layout()
    plt.show()

    print(f'Peak Val Accuracy  : {max(history["val_acc"]) * 100:.2f}%')
    print(f'Final Test Accuracy: {test_acc * 100:.2f}%')


@torch.no_grad()
def get_predictions(model, loader, device):
    """Collect ground-truth labels and predictions over a full DataLoader."""
    model.eval()
    all_preds, all_labels = [], []
    for X_batch, y_batch in loader:
        logits = model(X_batch.to(device))
        all_preds.extend(logits.argmax(dim=1).cpu().numpy())
        all_labels.extend(y_batch.numpy())
    return np.array(all_labels), np.array(all_preds)


def plot_classification_report(y_true, y_pred, le):
    """
    Print per-class Precision, Recall, and F1-Score.

        Precision : Of all predicted as class C, how many are actually C?
        Recall    : Of all actual class C, how many did we correctly predict?
        F1-Score  : Harmonic mean of Precision and Recall — balanced metric.
    """
    print('=' * 65)
    print('     CLASSIFICATION REPORT — GestureNet (Deep Learning)')
    print('=' * 65)
    print(classification_report(y_true, y_pred, target_names=le.classes_))
    print(f'Overall Accuracy: {accuracy_score(y_true, y_pred) * 100:.2f}%')


def plot_confusion_matrix(y_true, y_pred, le):
    """
    Plot raw-count and row-normalised confusion matrices.

    Diagonal = correctly classified samples.
    Off-diagonal = misclassifications.
    Look for structurally similar gesture pairs (e.g. 'peace' vs
    'peace_inverted') — they share nearly identical landmark configurations.
    """
    cm      = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    for ax, data, title, fmt in [
        (axes[0], cm,      'Confusion Matrix (Counts)',     'd'),
        (axes[1], cm_norm, 'Confusion Matrix (Normalized)', '.2f'),
    ]:
        sns.heatmap(data, annot=True, fmt=fmt, cmap='Blues',
                    xticklabels=le.classes_, yticklabels=le.classes_,
                    linewidths=0.5, ax=ax, cbar_kws={'shrink': 0.7})
        ax.set_title(title, color='#00d4ff', fontsize=13)
        ax.set_xlabel('Predicted', color='#ccc')
        ax.set_ylabel('True',      color='#ccc')
        ax.tick_params(axis='x', rotation=30)
        ax.tick_params(axis='y', rotation=0)

    plt.suptitle('GestureNet — Test Set Evaluation', color='#a8dadc', fontsize=15)
    plt.tight_layout()
    plt.show()

    return cm


def plot_per_class_accuracy(cm, le, num_classes):
    """
    Bar chart of per-class accuracy.

    Reveals which gestures the network handles well vs struggles with.
    Confusion between structurally similar gestures is expected.
    """
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    fig, ax = plt.subplots(figsize=(14, 5))
    colors  = [PALETTE[i % len(PALETTE)] for i in range(num_classes)]
    bars    = ax.bar(le.classes_, per_class_acc * 100, color=colors, edgecolor='#333')
    ax.bar_label(bars, fmt='%.1f%%', padding=3, color='#eee', fontsize=9)
    ax.axhline(y=np.mean(per_class_acc) * 100, color='#ffd460',
               linestyle='--', lw=1.5,
               label=f'Mean = {np.mean(per_class_acc) * 100:.1f}%')
    ax.set_ylim(0, 110)
    ax.set_title('Per-Class Accuracy — GestureNet', color='#00d4ff', fontsize=14)
    ax.set_ylabel('Accuracy (%)')
    ax.set_xlabel('Gesture Class')
    ax.legend()
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.show()


def plot_comparison(test_acc: float):
    """
    Horizontal bar chart comparing GestureNet against ML1 classical baselines.

    NOTE: Update the first 5 accuracy values with your actual ML1 results.

    Key insight: For tabular data with ~25K samples and 63 features the gap
    between classical ML and DL is often small. DL's true advantage emerges
    with millions of samples, raw video frames (CNNs/LSTMs), or transfer
    learning from pre-trained gesture models.
    """
    ml_results = {
        'KNN'                 : 0.923,
        'SVM Linear'          : 0.955,
        'SVM RBF'             : 0.967,
        'Random Forest'       : 0.971,
        'Gradient Boosting'   : 0.974,
        '✨ GestureNet (MLP)' : test_acc,
    }

    fig, ax    = plt.subplots(figsize=(12, 5))
    names      = list(ml_results.keys())
    accs       = [v * 100 for v in ml_results.values()]
    bar_colors = ['#444'] * 5 + ['#00d4ff']

    bars = ax.barh(names, accs, color=bar_colors, edgecolor='#333')
    ax.bar_label(bars, fmt='%.2f%%', padding=4, color='#eee', fontsize=10)
    ax.set_xlim(85, 102)
    ax.set_title('Classical ML vs Deep Learning — Test Accuracy',
                 color='#00d4ff', fontsize=14)
    ax.set_xlabel('Test Accuracy (%)')
    ax.axvline(x=test_acc * 100, color='#00d4ff', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()
    print(f'\n📊 GestureNet Test Accuracy : {test_acc * 100:.2f}%')


# ══════════════════════════════════════════════════════════════════════════════
# 8. SINGLE-SAMPLE INFERENCE
# ══════════════════════════════════════════════════════════════════════════════

def single_inference(model, X_test, y_test, le, device, sample_idx: int = 42):
    """
    Run inference on one test sample and plot the top-5 class probabilities.

    Theory — Softmax:
        softmax(z_i) = exp(z_i) / Σ exp(z_j)
        Maps raw logits → probability distribution over all classes.
        All values ∈ [0,1] and sum to 1 → interpretable as confidence scores.

    In deployment, the 63 landmark values would come from MediaPipe in
    real-time; here we use a held-out test sample.
    """
    sample_raw = X_test[sample_idx : sample_idx + 1]
    true_label = le.classes_[y_test[sample_idx]]

    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(sample_raw).to(device))
        probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]

    pred_idx   = probs.argmax()
    pred_label = le.classes_[pred_idx]
    confidence = probs[pred_idx]

    print(f'True Label : {true_label}')
    print(f'Predicted  : {pred_label}  (confidence: {confidence * 100:.1f}%)')
    print(f'Correct?   : {"✅ YES" if true_label == pred_label else "❌ NO"}')

    top5_idx    = probs.argsort()[::-1][:5]
    top5_labels = [le.classes_[i] for i in top5_idx]
    top5_probs  = [probs[i] * 100 for i in top5_idx]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(top5_labels[::-1], top5_probs[::-1],
            color=['#00d4ff' if l == true_label else '#e94560'
                   for l in top5_labels[::-1]])
    ax.set_xlabel('Probability (%)')
    ax.set_title(f'Top-5 Class Probabilities  |  True: "{true_label}"',
                 color='#00d4ff')
    ax.set_xlim(0, 105)
    plt.tight_layout()
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# 9. ABLATION STUDY
# ══════════════════════════════════════════════════════════════════════════════

class ShallowNet(nn.Module):
    """
    Single hidden layer MLP — closer to Logistic Regression + one non-linearity.
    Used in ablation to show the value of depth.
    """
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    def forward(self, x): return self.net(x)


class DeepNoDropout(nn.Module):
    """
    Same depth as GestureNet but WITHOUT Dropout.
    Used in ablation to show the overfitting risk of skipping Dropout.
    """
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Linear(256, 128),       nn.BatchNorm1d(128), nn.ReLU(),
            nn.Linear(128, 64),        nn.BatchNorm1d(64),  nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    def forward(self, x): return self.net(x)


def run_ablation(train_loader, val_loader, input_dim, num_classes, device,
                 history, ablation_epochs: int = 20):
    """
    Ablation study: train lightweight variants for 20 epochs and compare.

    An ablation study removes/changes one component at a time to understand
    its contribution — standard practice in DL research to justify every
    architecture choice.

    Comparison:
        ShallowNet    → shows value of depth
        DeepNoDropout → shows value of Dropout
        GestureNet    → full model (best of both)
    """
    def _quick_train(model_class):
        m    = model_class(input_dim, num_classes).to(device)
        opt  = optim.AdamW(m.parameters(), lr=3e-3)
        crit = nn.CrossEntropyLoss()
        tr_accs, vl_accs = [], []
        for _ in range(ablation_epochs):
            _, tr_a = train_epoch(m, train_loader, crit, opt, device)
            _, vl_a = evaluate(m, val_loader,   crit,     device)
            tr_accs.append(tr_a)
            vl_accs.append(vl_a)
        return tr_accs, vl_accs

    print('Training ablation models (20 epochs each)...')
    sh_tr, sh_vl = _quick_train(ShallowNet)
    nd_tr, nd_vl = _quick_train(DeepNoDropout)
    print('✅ Done')

    print(f'\nShallow (1 hidden layer) val acc : {sh_vl[-1] * 100:.2f}%')
    print(f'Deep (no dropout)        val acc : {nd_vl[-1] * 100:.2f}%')
    print(f'GestureNet (full)        val acc : {max(history["val_acc"]) * 100:.2f}%')
    print('\n💡 Key Insight: Depth adds capacity; Dropout prevents overfitting.')

    # Plot
    fig, ax  = plt.subplots(figsize=(10, 5))
    epochs_a = range(1, ablation_epochs + 1)

    ax.plot(epochs_a, [v * 100 for v in sh_vl],
            color='#e94560', lw=2, label='Shallow (1 layer)')
    ax.plot(epochs_a, [v * 100 for v in nd_vl],
            color='#ffd460', lw=2, label='Deep — No Dropout')
    ax.plot(range(1, len(history['val_acc'][:ablation_epochs]) + 1),
            [v * 100 for v in history['val_acc'][:ablation_epochs]],
            color='#00d4ff', lw=2.5, label='GestureNet (full)')

    ax.set_title('Ablation Study — Val Accuracy (20 epochs)',
                 color='#00d4ff', fontsize=13)
    ax.set_xlabel('Epoch')  ;  ax.set_ylabel('Validation Accuracy (%)')
    ax.legend()  ;  ax.grid(True)
    plt.tight_layout()
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# 10. SAVE MODEL
# ══════════════════════════════════════════════════════════════════════════════

def save_model(model, le, scaler, path: str = 'models/gesture_net.pt'):
    """
    Save model weights together with the LabelEncoder and StandardScaler.

    Bundling le + scaler ensures that at inference time you can load a
    single file and reproduce the exact same preprocessing pipeline.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'model_state' : model.state_dict(),
        'le'          : le,
        'scaler'      : scaler,
    }, path)
    print(f'\n✅ Model saved → {path}')


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description='Hand Gesture Classification — Deep Learning (PyTorch MLP)')
    parser.add_argument('--data',    default='hand_landmarks_data.csv',
                        help='Path to the CSV dataset')
    parser.add_argument('--epochs',  type=int, default=EPOCHS,
                        help=f'Max training epochs (default: {EPOCHS})')
    parser.add_argument('--lr',      type=float, default=LR,
                        help=f'Learning rate (default: {LR})')
    parser.add_argument('--batch',   type=int, default=BATCH_SIZE,
                        help=f'Batch size (default: {BATCH_SIZE})')
    parser.add_argument('--patience',type=int, default=PATIENCE,
                        help=f'Early stopping patience (default: {PATIENCE})')
    parser.add_argument('--no-plots', action='store_true',
                        help='Skip all matplotlib plots')
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Setup ──────────────────────────────────────────────────────────────────
    device = setup(SEED)
    os.makedirs('models', exist_ok=True)

    # ── Data ───────────────────────────────────────────────────────────────────
    df = load_data(args.data)
    if not args.no_plots:
        plot_class_distribution(df)

    # ── Preprocessing ──────────────────────────────────────────────────────────
    (X_train, X_val, X_test,
     y_train, y_val, y_test,
     le, scaler, num_classes) = preprocess(df, seed=SEED)

    train_loader, val_loader, test_loader = make_loaders(
        X_train, X_val, X_test, y_train, y_val, y_test,
        batch_size=args.batch)

    # ── Model ──────────────────────────────────────────────────────────────────
    input_dim = X_train.shape[1]   # 63
    model     = build_model(input_dim, num_classes, device)

    # ── Optimizer ──────────────────────────────────────────────────────────────
    criterion, optimizer, scheduler = build_optimizer(
        model, y_train, num_classes, device, lr=args.lr, epochs=args.epochs)

    # ── Training ───────────────────────────────────────────────────────────────
    history, test_acc, _ = train(
        model, train_loader, val_loader, test_loader,
        criterion, optimizer, scheduler, device,
        epochs=args.epochs, patience=args.patience)

    # ── Save ───────────────────────────────────────────────────────────────────
    save_model(model, le, scaler)

    # ── Evaluation & Plots ─────────────────────────────────────────────────────
    if not args.no_plots:
        plot_training_curves(history, test_acc)

        y_true, y_pred = get_predictions(model, test_loader, device)
        plot_classification_report(y_true, y_pred, le)
        cm = plot_confusion_matrix(y_true, y_pred, le)
        plot_per_class_accuracy(cm, le, num_classes)
        plot_comparison(test_acc)

        single_inference(model, X_test, y_test, le, device, sample_idx=42)

        run_ablation(train_loader, val_loader, input_dim, num_classes,
                     device, history)
    else:
        y_true, y_pred = get_predictions(model, test_loader, device)
        plot_classification_report(y_true, y_pred, le)


if __name__ == '__main__':
    main()