import os
import sys
import time
import math

import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ─── Config ───────────────────────────────────────────────────────────────────

DATASET_PATH = r'C:\Users\Charls\Documents\PhishNET\poc_dataset_20k.csv'
MODEL_PATH   = r'models\my_phising_model'
OUTPUT_CHART = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'confusion_results.png')
BATCH_SIZE   = 32

label_map = {0: "Safe", 1: "Phishing"}

# ─── CUDA enforcement ─────────────────────────────────────────────────────────

if not torch.cuda.is_available():
    print("[ERROR] CUDA is not available on this machine.")
    print("        Verify your PyTorch installation includes CUDA support:")
    print("          pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
    print("        Also confirm your NVIDIA drivers are up to date.")
    sys.exit(1)

device = torch.device('cuda')
print(f"[INFO] GPU detected : {torch.cuda.get_device_name(0)}")
print(f"[INFO] CUDA version : {torch.version.cuda}")

# ─── Model guard ──────────────────────────────────────────────────────────────

if not os.path.isdir(MODEL_PATH):
    print(f"[ERROR] Model directory not found: {MODEL_PATH}")
    print("        Make sure 'my_phishing_model/' is present before running evaluate.py.")
    sys.exit(1)

# ─── Load model ───────────────────────────────────────────────────────────────

print(f"[INFO] Loading model from: {MODEL_PATH}")
print(f"[INFO] Device            : {device} ({torch.cuda.get_device_name(0)})")

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model     = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.to(device)
    model.eval()
except Exception as e:
    print(f"[ERROR] Failed to load model: {e}")
    sys.exit(1)

print("[INFO] Model loaded.\n")

# ─── Load dataset ─────────────────────────────────────────────────────────────

print(f"[INFO] Loading dataset: {DATASET_PATH}")
df = pd.read_csv(DATASET_PATH)
df.dropna(subset=['text'], inplace=True)
df['label'] = df['label'].astype(int)

total   = len(df)
n_phish = (df['label'] == 1).sum()
n_safe  = (df['label'] == 0).sum()

print(f"[INFO] Total samples : {total:,}")
print(f"       Phishing (1)  : {n_phish:,}")
print(f"       Safe (0)      : {n_safe:,}\n")

# ─── Inference ────────────────────────────────────────────────────────────────

true_labels = []
pred_labels = []

texts       = df['text'].astype(str).tolist()
labels_list = df['label'].tolist()
n_batches   = math.ceil(total / BATCH_SIZE)

start_time = time.time()

for i in tqdm(range(0, total, BATCH_SIZE), total=n_batches, desc="Running inference", unit="batch"):
    batch_texts  = texts[i : i + BATCH_SIZE]
    batch_labels = labels_list[i : i + BATCH_SIZE]

    inputs = tokenizer(
        batch_texts,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs       = model(**inputs)
        probabilities = F.softmax(outputs.logits, dim=1)

    predicted_classes = torch.argmax(probabilities, dim=1)

    true_labels.extend(batch_labels)
    pred_labels.extend(predicted_classes.cpu().tolist())

inference_duration = time.time() - start_time

# ─── Confusion matrix ─────────────────────────────────────────────────────────

# sklearn confusion_matrix returns [[TN, FP], [FN, TP]] for binary labels
cm  = confusion_matrix(true_labels, pred_labels, labels=[0, 1])
TN  = cm[0][0]
FP  = cm[0][1]
FN  = cm[1][0]
TP  = cm[1][1]

# ─── Metrics ──────────────────────────────────────────────────────────────────

accuracy  = (TP + TN) / total
precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
f1        = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

phishing_label = label_map[1].upper()
safe_label     = label_map[0].upper()

# ─── Console output ───────────────────────────────────────────────────────────

print("\n" + "═" * 62)
print("  CONFUSION MATRIX — PhishNET KD Evaluation")
print("═" * 62)
print(f"  {'':22} PREDICTED: {phishing_label:<12} PREDICTED: {safe_label}")
print("  " + "─" * 58)
print(f"  Actual: {phishing_label:<14}  {TP:<20,} {FN:,}")
print(f"  Actual: {safe_label:<14}  {FP:<20,} {TN:,}")
print("  " + "─" * 58)
print(f"  Total samples    : {total:,}")
print(f"  Batch size       : {BATCH_SIZE}  ({n_batches:,} batches)")
print(f"  Inference time   : {inference_duration:.2f}s  ({inference_duration/total*1000:.1f}ms/email)")
allocated = torch.cuda.memory_allocated(0) / 1024**2
reserved  = torch.cuda.memory_reserved(0)  / 1024**2
print(f"  GPU mem alloc    : {allocated:.1f} MB")
print(f"  GPU mem reserved : {reserved:.1f} MB")
print("─" * 62)
print(f"  Accuracy         : {accuracy*100:.2f}%")
print(f"  Precision        : {precision*100:.2f}%")
print(f"  Recall           : {recall*100:.2f}%")
print(f"  F1 Score         : {f1*100:.2f}%")
print("═" * 62 + "\n")

# ─── Bar chart ────────────────────────────────────────────────────────────────

categories = [
    f"True Positives\n(Actual {phishing_label}\nPredicted {phishing_label})",
    f"True Negatives\n(Actual {safe_label}\nPredicted {safe_label})",
    f"False Positives\n(Actual {safe_label}\nPredicted {phishing_label})",
    f"False Negatives\n(Actual {phishing_label}\nPredicted {safe_label})",
]
counts = [TP, TN, FP, FN]
colors = ['#2ecc71', '#3498db', '#e74c3c', '#e67e22']

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(categories, counts, color=colors, width=0.5, edgecolor='white', linewidth=1.2)

for bar, count in zip(bars, counts):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + max(counts) * 0.01,
        f'{count:,}',
        ha='center', va='bottom', fontsize=11, fontweight='bold'
    )

ax.set_title('PhishNET — Confusion Matrix Results', fontsize=14, fontweight='bold', pad=15)
ax.set_ylabel('Count', fontsize=12)
ax.set_ylim(0, max(counts) * 1.15)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):,}'))
ax.spines[['top', 'right']].set_visible(False)
ax.tick_params(axis='x', labelsize=9)

fig.tight_layout()
plt.savefig(OUTPUT_CHART, dpi=150, bbox_inches='tight')
plt.close()

print(f"[INFO] Bar chart saved to: {OUTPUT_CHART}")