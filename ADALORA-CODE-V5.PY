# Chest X-Ray Report Generation — MedGemma Multimodal Fine-tuning (AdaLoRA v3)


# =============================================================================
# Prepare GPU
# =============================================================================

"""
AUTO-SELECT GPU
===============
Memindai semua GPU yang tersedia via nvidia-smi, lalu memilih satu GPU
dengan VRAM bebas terbanyak dan utilisasi terendah.
CUDA_VISIBLE_DEVICES di-set SEBELUM torch di-import agar efektif.
"""

import os
import subprocess
import sys

# ──────────────────────────────────────────────────────────────
# KONFIGURASI: berapa GPU yang ingin dipakai (1 atau lebih)
# ──────────────────────────────────────────────────────────────
N_GPUS_TO_USE = 1   # ← ubah sesuai kebutuhan

QUERY_FIELDS = "index,name,memory.free,memory.total,utilization.gpu"
try:
    result = subprocess.run(
        ["nvidia-smi", f"--query-gpu={QUERY_FIELDS}", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    )
except FileNotFoundError:
    sys.exit("ERROR: nvidia-smi tidak ditemukan. Pastikan driver NVIDIA terinstall.")
except subprocess.CalledProcessError as e:
    sys.exit(f"ERROR: nvidia-smi gagal: {e.stderr.strip()}")

gpu_info = []
for line in result.stdout.strip().splitlines():
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 5:
        continue
    try:
        gpu_info.append({
            "index"    : int(parts[0]),
            "name"     : parts[1],
            "mem_free" : int(parts[2]),
            "mem_total": int(parts[3]),
            "util_gpu" : int(parts[4]),
        })
    except ValueError:
        continue

if not gpu_info:
    sys.exit("ERROR: Tidak ada GPU yang terdeteksi dari nvidia-smi.")

header = f"{'IDX':>3}  {'Name':<30}  {'Free MiB':>10}  {'Total MiB':>10}  {'Util%':>6}"
sep    = "=" * len(header)
print(sep)
print("  GPU INFO (nvidia-smi)")
print(sep)
print(f"  {header}")
print("-" * len(header))
for g in gpu_info:
    marker = " ◄" if g is sorted(gpu_info, key=lambda x: (-x["mem_free"], x["util_gpu"]))[0] else ""
    print(
        f"  {g['index']:>3}  {g['name']:<30}  "
        f"{g['mem_free']:>10,}  {g['mem_total']:>10,}  {g['util_gpu']:>5}%{marker}"
    )
print(sep)

sorted_gpus  = sorted(gpu_info, key=lambda g: (-g["mem_free"], g["util_gpu"]))
selected     = sorted_gpus[:N_GPUS_TO_USE]
selected_ids = [str(g["index"]) for g in selected]

# *** WAJIB di-set SEBELUM import torch ***
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(selected_ids)

print(f"\nGPU terpilih (N_GPUS_TO_USE={N_GPUS_TO_USE}):")
for rank, g in enumerate(selected):
    print(
        f"  cuda:{rank} ← GPU {g['index']}  {g['name']}"
        f"  |  VRAM bebas: {g['mem_free']:,} MiB / {g['mem_total']:,} MiB"
        f"  |  Util: {g['util_gpu']}%"
    )
print(f"\nCUDA_VISIBLE_DEVICES = \"{os.environ['CUDA_VISIBLE_DEVICES']}\"")
print("OK — lanjutkan ke cell berikutnya (import torch).")


# =============================================================================
# Login Hugging Face
# =============================================================================

import os
import getpass
import torch
from huggingface_hub import login

hf_token = getpass.getpass("Masukkan Hugging Face token (input tersembunyi): ").strip()
if not hf_token:
    raise ValueError("HF token tidak boleh kosong. Dapatkan token di https://huggingface.co/settings/tokens")

login(token=hf_token)

# Optimasi presisi untuk B200 (Blackwell — native BF16 & TF32)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32       = True

print(f"PyTorch  : {torch.__version__}")
print(f"CUDA     : {torch.version.cuda}")
print(f"Available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f"  GPU {i}: {props.name} | VRAM: {props.total_memory / 1e9:.1f} GB | SM: {props.major}.{props.minor}")


# =============================================================================
# 1. Konfigurasi Enkripsi & Path Dataset
# =============================================================================

import io
import getpass
import pandas as pd
from pathlib import Path
from cryptography.fernet import Fernet, InvalidToken

# ============================================================
# KONFIGURASI PATH  — sesuaikan dengan lokasi file Anda
# ============================================================
ENCRYPTED_CSV_PATH  = Path("SIAP_TRAINING.csv.encrypted")
ENCRYPTED_IMAGE_DIR = Path("./IMAGE/GAMBAR_TERBARU_ENKRIPSI")
OUTPUT_DIR          = "./LORA_ADAPTER_MEDGEMMA_4B_IT"

MODEL_ID = "google/medgemma-4b-it"

NUM_TRAIN_EPOCHS            = 7
PER_DEVICE_TRAIN_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 8

raw_key = getpass.getpass("Masukkan Fernet encryption key (input tersembunyi): ").strip()
if not raw_key:
    raise ValueError("Encryption key tidak boleh kosong.")

try:
    fernet = Fernet(raw_key.encode() if isinstance(raw_key, str) else raw_key)
    fernet.decrypt(fernet.encrypt(b"key_validation_test"))
    print("Encryption key valid.")
except InvalidToken:
    raise ValueError("Encryption key tidak valid — pastikan format Fernet yang benar.")
except Exception as e:
    raise ValueError(f"Error validasi key: {e}")

assert ENCRYPTED_CSV_PATH.exists(),  f"CSV terenkripsi tidak ditemukan: {ENCRYPTED_CSV_PATH}"
assert ENCRYPTED_IMAGE_DIR.exists(), f"Folder gambar tidak ditemukan: {ENCRYPTED_IMAGE_DIR}"

with open(ENCRYPTED_CSV_PATH, "rb") as f:
    encrypted_csv_bytes = f.read()

try:
    decrypted_csv_bytes = fernet.decrypt(encrypted_csv_bytes)
except InvalidToken:
    raise ValueError("Gagal dekripsi CSV — periksa encryption key dan file CSV.")

df = pd.read_csv(io.BytesIO(decrypted_csv_bytes))

# Validasi kolom wajib (termasuk exam_type untuk stratified split)
REQUIRED_COLS = ["image_file", "findings", "conclusion", "exam_type"]
missing = [c for c in REQUIRED_COLS if c not in df.columns]
assert not missing, f"Kolom CSV tidak lengkap, kurang: {missing}"

df = df.dropna(subset=REQUIRED_COLS)
df["findings"]   = df["findings"].astype(str)
df["conclusion"] = df["conclusion"].astype(str)
df["exam_type"]  = df["exam_type"].astype(str)
df = df.reset_index(drop=True)

print(f"Total data  : {len(df)} baris")
print(f"Kolom CSV   : {df.columns.tolist()}")
print(f"\nDistribusi exam_type:\n{df['exam_type'].value_counts()}")
print(f"\nContoh data :")
print(df[["image_file", "exam_type", "findings", "conclusion"]].head(3).to_string())

from transformers import AutoModelForCausalLM, AutoProcessor

n_gpu = torch.cuda.device_count()

# Sisakan ~10 GB per GPU untuk aktivasi dan optimizer states
max_memory = {i: "170GiB" for i in range(n_gpu)}
max_memory["cpu"] = "64GB"

print(f"GPU count  : {n_gpu}")
print(f"Max memory : {max_memory}")
print(f"\nLoading {MODEL_ID} dalam bfloat16...")

processor = AutoProcessor.from_pretrained(MODEL_ID, token=hf_token)

ATTN_IMPL = "sdpa"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="balanced",
    max_memory=max_memory,
    trust_remote_code=True,
    token=hf_token,
    attn_implementation=ATTN_IMPL,
)

if processor.tokenizer.pad_token is None:
    processor.tokenizer.pad_token  = processor.tokenizer.eos_token
    model.config.pad_token_id      = processor.tokenizer.eos_token_id

if hasattr(model, "hf_device_map"):
    gpus_used = set(str(v) for v in model.hf_device_map.values())
    print(f"\nModel tersebar di: {gpus_used}")
else:
    print(f"\nModel device: {next(model.parameters()).device}")

for i in range(n_gpu):
    alloc  = torch.cuda.memory_allocated(i) / 1e9
    total  = torch.cuda.get_device_properties(i).total_memory / 1e9
    print(f"  GPU {i}: {alloc:.1f} / {total:.1f} GB")

print("\nModel & Processor loaded!")


# =============================================================================
# 3. AdaLoRA Adapter
# =============================================================================

import math
from peft import AdaLoraConfig, get_peft_model

n_gpu_now       = torch.cuda.device_count()
effective_batch = PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * n_gpu_now
n_train_est     = int(len(df) * 0.8)
steps_per_epoch = math.ceil(n_train_est / effective_batch)
TOTAL_STEP      = steps_per_epoch * NUM_TRAIN_EPOCHS

# tinit  : 5% dari total_step sebagai warmup
# tfinal : 80% dari total_step — rank stabil, sisa untuk fine-tune
TINIT  = max(50,  int(TOTAL_STEP * 0.05))
TFINAL = max(200, int(TOTAL_STEP * 0.80))
DELTAT = 50   # interval rank-update, sinkron dengan eval_steps

print(f"Estimasi n_train    : {n_train_est} sampel")
print(f"Effective batch size: {effective_batch}")
print(f"Steps per epoch     : {steps_per_epoch}")
print(f"Total epochs        : {NUM_TRAIN_EPOCHS}")
print(f"TOTAL_STEP          : {TOTAL_STEP}")
print(f"tinit               : {TINIT}   (warmup sebelum pruning)")
print(f"tfinal              : {TFINAL}  (rank stabil setelah step ini)")
print(f"deltaT              : {DELTAT}  (interval rank-update)")

# ──────────────────────────────────────────────────────────────
# Aktifkan gradient checkpointing SEBELUM attach AdaLoRA
# ──────────────────────────────────────────────────────────────
model.gradient_checkpointing_enable(
    gradient_checkpointing_kwargs={"use_reentrant": False}
)
model.enable_input_require_grads()

# ──────────────────────────────────────────────────────────────
# AdaLoRA CONFIG
# ──────────────────────────────────────────────────────────────
adalora_config = AdaLoraConfig(
    # ── Rank scheduling ──────────────────────────────────────
    init_r=64,
    target_r=32,
    total_step=TOTAL_STEP,   # ← WAJIB: jumlah total training steps
    tinit=TINIT,
    tfinal=TFINAL,
    deltaT=DELTAT,
    beta1=0.85,
    beta2=0.85,

    # ── Target modules ───────────────────────────────────────
    target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)",

    # ── Modules to save (full fine-tune) ─────────────────────
    modules_to_save=["multi_modal_projector"],

    lora_alpha=64,
    lora_dropout=0.10,

    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, adalora_config)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f"\nTrainable : {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")
print()
model.print_trainable_parameters()
print()
print("Catatan: Jumlah trainable params akan BERKURANG bertahap")
print(f"         seiring AdaLoRA memangkas rank: init_r={adalora_config.init_r} → target_r={adalora_config.target_r}")


# =============================================================================
# 4. Dataset Chest X-Ray
# =============================================================================

import io
from PIL import Image
from torch.utils.data import Dataset as TorchDataset
from sklearn.model_selection import train_test_split
from cryptography.fernet import Fernet, InvalidToken
from torchvision import transforms

# ============================================================
# Template prompt — exam_type IKUT dalam output model
# ============================================================
SYSTEM_PROMPT = [
    "Anda adalah dokter radiologi. Buat laporan thorax lengkap dalam bahasa Indonesia.",
    "Tolong analisis X-ray dada ini dan tuliskan temuan radiologis beserta kesimpulannya.",
    "Sebagai spesialis radiologi, apa diagnosis Anda untuk gambar rontgen berikut?",
    "Berikan evaluasi klinis dari citra radiografi thorax ini secara terperinci.",
    "Tuliskan laporan medis dari foto rontgen dada ini, mencakup temuan dan konklusi."
]

def build_prompt(processor, exam_type: str, findings: str, conclusion: str) -> str:
    """Prompt training: model diajarkan menghasilkan exam_type + findings + conclusion."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": SYSTEM_PROMPT},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": f"Exam Type:\n{exam_type}\n\nFindings:\n{findings}\n\nConclusion:\n{conclusion}"},
            ],
        },
    ]
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


def build_inference_prompt(processor) -> str:
    """Format inferensi: hanya sisi user, tanpa jawaban model."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": SYSTEM_PROMPT},
            ],
        },
    ]
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def decrypt_image_to_pil(fernet: Fernet, img_path: Path) -> Image.Image:
    """Dekripsi file .enc di memori → PIL Image. Tidak menyentuh disk."""
    with open(img_path, "rb") as f:
        encrypted_bytes = f.read()
    try:
        decrypted_bytes = fernet.decrypt(encrypted_bytes)
    except InvalidToken as e:
        raise RuntimeError(f"Gagal dekripsi {img_path.name}: key salah atau file rusak. {e}")
    img = Image.open(io.BytesIO(decrypted_bytes)).convert("RGB")
    img = img.resize((896, 896), Image.Resampling.LANCZOS)
    return img


class ChestXRayDataset(TorchDataset):
    """
    Dataset multimodal X-ray dengan Curriculum Augmentation.

    Curriculum Augmentation: intensitas augmentasi dikurangi secara
    linear seiring epoch bertambah. Tujuannya agar:
      - Epoch awal : augmentasi kuat → paksa model belajar fitur umum
      - Epoch akhir: augmentasi ringan → model fine-tune pada fitur
                     spesifik tanpa noise yang mengganggu LR kecil
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        image_dir: Path,
        processor,
        fernet: Fernet,
        max_length: int = 2048,
        is_train: bool = False,
        total_epochs: int = 7,   # untuk menghitung skala curriculum
    ):
        self.df           = dataframe.reset_index(drop=True)
        self.image_dir    = image_dir
        self.processor    = processor
        self.fernet       = fernet
        self.max_length   = max_length
        self.is_train     = is_train
        self.total_epochs = total_epochs
        self.current_epoch = 0   # diupdate oleh EpochAugmentationCallback

        # Pembersihan data teks
        for col in ["findings", "conclusion", "exam_type"]:
            self.df[col] = (
                self.df[col].astype(str)
                    .str.replace("\n", " ")
                    .str.replace("\r", " ")
                    .str.strip()
            )

        valid_mask = self.df["image_file"].apply(lambda f: (image_dir / f).exists())
        self.df    = self.df[valid_mask].reset_index(drop=True)
        print(f"Dataset valid (is_train={is_train}): {len(self.df)} sampel")

    def set_epoch(self, epoch: int):
        """Dipanggil oleh EpochAugmentationCallback di awal setiap epoch."""
        self.current_epoch = epoch

    def _build_augmentation(self) -> transforms.Compose:
        """
        Hitung augmentasi dengan skala yang mengecil seiring epoch.

        progress: 0.0 (epoch 0) → 1.0 (epoch terakhir)
        scale   : 1.0 (penuh)  → 0.3 (30% intensitas tersisa)

        Rumus: scale = 1.0 - (0.7 × progress)
        Hasil:
          epoch 0 → scale 1.00 (rotasi ±5°, translate 3%, jitter 15%)
          epoch 3 → scale 0.65
          epoch 6 → scale 0.30 (rotasi ±1.5°, translate 0.9%, jitter 4.5%)
        """
        progress = self.current_epoch / max(self.total_epochs - 1, 1)
        scale    = 1.0 - (0.7 * progress)

        return transforms.Compose([
            transforms.RandomRotation(degrees=5 * scale),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.03 * scale, 0.03 * scale)
            ),
            transforms.ColorJitter(
                brightness=0.15 * scale,
                contrast=0.15 * scale
            ),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row      = self.df.iloc[idx]
        img_path = self.image_dir / row["image_file"]
        image    = decrypt_image_to_pil(self.fernet, img_path)

        # Augmentasi curriculum: hanya saat training, intensitas menyesuaikan epoch
        if self.is_train:
            image = self._build_augmentation()(image)

        prompt_text = build_inference_prompt(self.processor)
        full_text   = build_prompt(
            self.processor, row["exam_type"], row["findings"], row["conclusion"]
        )

        # 1. Hitung panjang prompt (instruksi + gambar), TANPA padding
        prompt_encoding  = self.processor(
            images=image,
            text=prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        actual_prompt_len = prompt_encoding["input_ids"].shape[1]

        # 2. Encode sequence penuh (prompt + response), DENGAN padding
        encoding = self.processor(
            images=image,
            text=full_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
        )

        item   = {k: v.squeeze(0) for k, v in encoding.items()}
        labels = item["input_ids"].clone()

        # Masking dinamis: loss HANYA dihitung pada token response
        real_token_indices = torch.where(item["attention_mask"] == 1)[0]
        prompt_indices     = real_token_indices[:actual_prompt_len]
        labels[prompt_indices]              = -100
        labels[item["attention_mask"] == 0] = -100

        item["labels"] = labels

        if "token_type_ids" not in item:
            item["token_type_ids"] = torch.zeros_like(item["input_ids"])

        return item


# ── STRATIFIED SPLIT (80% / 10% / 10%) ──────────────────────
train_df, temp_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df["exam_type"]
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.5, random_state=42, stratify=temp_df["exam_type"]
)

print(f"\nDistribusi Train:\n{train_df['exam_type'].value_counts()}")
print(f"\nDistribusi Val:\n{val_df['exam_type'].value_counts()}")
print(f"\nDistribusi Test:\n{test_df['exam_type'].value_counts()}\n")

train_dataset = ChestXRayDataset(
    train_df, ENCRYPTED_IMAGE_DIR, processor, fernet,
    is_train=True, total_epochs=NUM_TRAIN_EPOCHS
)
val_dataset = ChestXRayDataset(
    val_df, ENCRYPTED_IMAGE_DIR, processor, fernet,
    is_train=False, total_epochs=NUM_TRAIN_EPOCHS
)
test_dataset = ChestXRayDataset(
    test_df, ENCRYPTED_IMAGE_DIR, processor, fernet,
    is_train=False, total_epochs=NUM_TRAIN_EPOCHS
)

# Verifikasi 1 sampel end-to-end
sample = train_dataset[0]
print(f"\nSample keys   : {list(sample.keys())}")
print(f"input_ids     : {sample['input_ids'].shape}")
print(f"token_type_ids: {sample['token_type_ids'].shape}")
print(f"pixel_values  : {sample['pixel_values'].shape}")
print(f"attention_mask: {sample['attention_mask'].shape}")


# =============================================================================
# 5. Fine-Tune
# =============================================================================

import gc

gc.collect()
torch.cuda.empty_cache()

print("=== Status VRAM sebelum Training ===")
for i in range(torch.cuda.device_count()):
    total     = torch.cuda.get_device_properties(i).total_memory / 1e9
    allocated = torch.cuda.memory_allocated(i) / 1e9
    reserved  = torch.cuda.memory_reserved(i) / 1e9
    free      = total - reserved
    print(f"GPU {i} ({torch.cuda.get_device_name(i)}):")
    print(f"  Total    : {total:.1f} GB")
    print(f"  Allocated: {allocated:.1f} GB")
    print(f"  Free     : {free:.1f} GB")


def multimodal_data_collator(features):
    batch = {}
    batch["input_ids"] = torch.nn.utils.rnn.pad_sequence(
        [f["input_ids"] for f in features], batch_first=True, padding_value=0
    )
    batch["attention_mask"] = torch.nn.utils.rnn.pad_sequence(
        [f["attention_mask"] for f in features], batch_first=True, padding_value=0
    )
    batch["token_type_ids"] = torch.nn.utils.rnn.pad_sequence(
        [f["token_type_ids"] for f in features], batch_first=True, padding_value=0
    )
    batch["labels"] = torch.nn.utils.rnn.pad_sequence(
        [f["labels"] for f in features], batch_first=True, padding_value=-100
    )
    batch["pixel_values"] = torch.stack([f["pixel_values"] for f in features])
    return batch

# Verifikasi shape batch
batch = multimodal_data_collator([train_dataset[0], train_dataset[1]])
for k, v in batch.items():
    print(k, v.shape)


import os
import gc
import torch
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback, TrainerCallback

class EpochAugmentationCallback(TrainerCallback):
    """Update current_epoch di dataset setiap awal epoch."""

    def __init__(self, train_dataset):
        self.train_dataset = train_dataset

    def on_epoch_begin(self, args, state, control, **kwargs):
        epoch      = int(state.epoch) if state.epoch else 0
        progress   = epoch / max(args.num_train_epochs - 1, 1)
        scale      = 1.0 - (0.7 * progress)
        self.train_dataset.set_epoch(epoch)
        print(
            f"\n[CurriculumAug] Epoch {epoch + 1}/{int(args.num_train_epochs)} dimulai  |  "
            f"aug_scale = {scale:.2f}  "
            f"(rotasi ±{5 * scale:.1f}°, translate {3 * scale:.1f}%, jitter {15 * scale:.1f}%)"
        )


# Bebaskan VRAM sebelum training
gc.collect()
torch.cuda.empty_cache()

print("=== Status VRAM sebelum Training ===")
for i in range(torch.cuda.device_count()):
    total     = torch.cuda.get_device_properties(i).total_memory / 1e9
    allocated = torch.cuda.memory_allocated(i) / 1e9
    reserved  = torch.cuda.memory_reserved(i) / 1e9
    free      = total - reserved
    print(f"GPU {i} ({torch.cuda.get_device_name(i)}):")
    print(f"  Total    : {total:.1f} GB")
    print(f"  Allocated: {allocated:.1f} GB")
    print(f"  Free     : {free:.1f} GB")

os.environ["WANDB_DISABLED"] = "true"

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_TRAIN_EPOCHS,           # 7 epoch
    per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    learning_rate=3e-4,
    lr_scheduler_type="cosine",
    warmup_steps=TINIT,                          # sinkron dengan tinit AdaLoRA

    # ── Regularisasi ─────────────────────────────────────────
    weight_decay=0.05,                           # 0.01 → 0.05
    max_grad_norm=0.5,                           # default 1.0 → 0.5: update lebih stabil

    # ── Presisi ──────────────────────────────────────────────
    bf16=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},

    # ── Logging & Evaluasi berbasis EPOCH ────────────────────
    logging_strategy="epoch",                    # "steps" → "epoch"
    eval_strategy="epoch",                       # "steps" → "epoch"
    save_strategy="epoch",                       # "steps" → "epoch"

    # ── Checkpoint terbaik ───────────────────────────────────
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    save_total_limit=3,                          # simpan max 3 checkpoint terbaik

    # ── Misc ─────────────────────────────────────────────────
    report_to="none",
    remove_unused_columns=False,
    dataloader_num_workers=0,
    dataloader_pin_memory=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=multimodal_data_collator,
    callbacks=[
        EarlyStoppingCallback(early_stopping_patience=4),
        EpochAugmentationCallback(train_dataset),   # curriculum augmentation
    ],
)

# ── Info ringkasan sebelum training ──────────────────────────
n_gpu_actual    = torch.cuda.device_count()
effective_batch = (
    training_args.per_device_train_batch_size
    * training_args.gradient_accumulation_steps
    * n_gpu_actual
)
total_steps_actual = math.ceil(len(train_dataset) / effective_batch) * int(training_args.num_train_epochs)

print(f"\nGPU count            : {n_gpu_actual}")
print(f"Effective batch size : {effective_batch}")
print(f"Actual total steps   : ~{total_steps_actual}  (estimasi saat config: {TOTAL_STEP})")
print(f"\nAdaLoRA rank schedule:")
print(f"  init_r={adalora_config.init_r}  →  target_r={adalora_config.target_r}")
print(f"  Pruning dimulai  : step {adalora_config.tinit}   (warmup_steps = TINIT)")
print(f"  Pruning selesai  : step {adalora_config.tfinal}  (rank final/stabil)")
print(f"  Update interval  : setiap {adalora_config.deltaT} step")
print(f"\nLogging/Eval/Save    : per epoch (bukan per step)")
print(f"max_grad_norm        : {training_args.max_grad_norm}  (default 1.0 → 0.5)")
print(f"lora_dropout         : {adalora_config.lora_dropout}  (0.05 → 0.10)")
print(f"weight_decay         : {training_args.weight_decay}  (0.01 → 0.05)")
print(f"Curriculum Aug       : aktif — skala turun dari 1.0 → 0.30 selama {NUM_TRAIN_EPOCHS} epoch")
print("\nMemulai fine-tuning AdaLoRA v3...")

train_result = trainer.train()
print("\nFine-tuning selesai!")

import os
import json
import pandas as pd

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# Simpan Proses dan Hasil Training (Log & Metrik)
# =============================================================================
try:
    # 1. Simpan metrik keseluruhan dari hasil akhir
    with open(os.path.join(OUTPUT_DIR, "final_train_metrics.json"), "w") as f:
        json.dump(train_result.metrics, f, indent=4)
        
    # 2. Simpan riwayat log per epoch/step (train loss, eval loss, dll)
    log_history = trainer.state.log_history
    with open(os.path.join(OUTPUT_DIR, "trainer_log_history.json"), "w") as f:
        json.dump(log_history, f, indent=4)
        
    # 3. Ekspor ke format CSV agar mudah dibuat grafik dan dianalisa
    df_log = pd.DataFrame(log_history)
    df_log.to_csv(os.path.join(OUTPUT_DIR, "training_log_history.csv"), index=False)
    
    print(f"✅ Log dan metrik training berhasil disimpan ke dalam format JSON dan CSV di folder:\n   {OUTPUT_DIR}")
except Exception as e:
    print(f"⚠️ Terjadi kesalahan saat menyimpan log: {e}")

print("\nMenyatukan adapter (merge) dengan base model...")
model.merge_adapter()

# Simpan adapter (hanya delta weight, bukan keseluruhan model 4B)
trainer.save_model(OUTPUT_DIR)

# Simpan processor agar konfigurasi tokenizer & image processor
# identik saat inferensi
processor.save_pretrained(OUTPUT_DIR)

print(f"✅ AdaLoRA Adapter dan Processor berhasil disimpan di: {OUTPUT_DIR}")

# Tampilkan isi folder untuk konfirmasi
saved_files = sorted(os.listdir(OUTPUT_DIR))
print(f"\nFile tersimpan ({len(saved_files)} file):")
for f in saved_files:
    size_kb = os.path.getsize(os.path.join(OUTPUT_DIR, f)) / 1024
    print(f"  {f:<45} {size_kb:>8.1f} KB")