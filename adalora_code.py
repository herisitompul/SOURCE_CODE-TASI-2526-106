import os
import random
import subprocess
import sys

N_GPUS_TO_USE = 1

QUERY_FIELDS = "index,name,memory.free,memory.total,utilization.gpu"
try:
    result = subprocess.run(
        ["nvidia-smi", f"--query-gpu={QUERY_FIELDS}", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
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
        gpu_info.append(
            {
                "index": int(parts[0]),
                "name": parts[1],
                "mem_free": int(parts[2]),
                "mem_total": int(parts[3]),
                "util_gpu": int(parts[4]),
            }
        )
    except ValueError:
        continue

if not gpu_info:
    sys.exit("ERROR: Tidak ada GPU yang terdeteksi dari nvidia-smi.")

header = f"{'IDX':>3}  {'Name':<30}  {'Free MiB':>10}  {'Total MiB':>10}  {'Util%':>6}"
sep = "=" * len(header)
print(sep)
print("  GPU INFO (nvidia-smi)")
print(sep)
print(f"  {header}")
print("-" * len(header))
_sorted_for_display = sorted(gpu_info, key=lambda x: (-x["mem_free"], x["util_gpu"]))
for g in gpu_info:
    marker = " ◄" if g is _sorted_for_display[0] else ""
    print(
        f"  {g['index']:>3}  {g['name']:<30}  "
        f"{g['mem_free']:>10,}  {g['mem_total']:>10,}  {g['util_gpu']:>5}%{marker}"
    )
print(sep)

selected = _sorted_for_display[:N_GPUS_TO_USE]
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
print(f'\nCUDA_VISIBLE_DEVICES = "{os.environ["CUDA_VISIBLE_DEVICES"]}"')
print("Siap — lanjutkan ke import torch.\n")


# =============================================================================
# BAGIAN 1 — Import Utama, Login HuggingFace, Konfigurasi Path
# =============================================================================

import gc
import getpass
import io
import json
import math
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from cryptography.fernet import Fernet, InvalidToken
from huggingface_hub import login

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

print(f"PyTorch  : {torch.__version__}")
print(f"CUDA     : {torch.version.cuda}")
print(f"GPU count  : {torch.cuda.device_count()}")
for _i in range(torch.cuda.device_count()):
    _p = torch.cuda.get_device_properties(_i)
    print(
        f"  GPU {_i}: {_p.name} | VRAM: {_p.total_memory / 1e9:.1f} GB | SM: {_p.major}.{_p.minor}"
    )

# ── HuggingFace Login ─────────────────────────────────────────
hf_token = getpass.getpass("Masukkan Hugging Face token (input tersembunyi): ").strip()
if not hf_token:
    raise ValueError(
        "HF token tidak boleh kosong. Dapatkan di https://huggingface.co/settings/tokens"
    )
login(token=hf_token)
print("HuggingFace login berhasil.\n")

# ── Konfigurasi Path & Hyperparameter ────────────────────────
ENCRYPTED_CSV_PATH = Path("SIAP_TRAINING.csv.encrypted")
ENCRYPTED_IMAGE_DIR = Path("./GAMBAR_ENKRIPSI/GAMBAR_ENKRIPSI")
OUTPUT_DIR = "./RESULT_TRAIN"

MODEL_ID = "google/medgemma-4b-it"

NUM_TRAIN_EPOCHS = 20
PER_DEVICE_TRAIN_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 8
MAX_SEQ_LENGTH = 2048

SWA_START_RATIO = 0.70

NUM_EVAL_TEXT_SAMPLES = 500


# =============================================================================
# BAGIAN 2 — Dekripsi & Validasi Dataset CSV
# =============================================================================

raw_key = getpass.getpass(
    "Masukkan Fernet encryption key (input tersembunyi): "
).strip()
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

assert ENCRYPTED_CSV_PATH.exists(), (
    f"CSV terenkripsi tidak ditemukan: {ENCRYPTED_CSV_PATH}"
)
assert ENCRYPTED_IMAGE_DIR.exists(), (
    f"Folder gambar tidak ditemukan: {ENCRYPTED_IMAGE_DIR}"
)

with open(ENCRYPTED_CSV_PATH, "rb") as _f:
    encrypted_csv_bytes = _f.read()

try:
    decrypted_csv_bytes = fernet.decrypt(encrypted_csv_bytes)
except InvalidToken:
    raise ValueError("Gagal dekripsi CSV — periksa encryption key dan file CSV.")

df = pd.read_csv(io.BytesIO(decrypted_csv_bytes))

REQUIRED_COLS = ["file_gambar", "temuan", "kesimpulan", "jenis_pemeriksaan"]
missing_cols = [c for c in REQUIRED_COLS if c not in df.columns]
assert not missing_cols, f"Kolom CSV tidak lengkap, kurang: {missing_cols}"

df = df.dropna(subset=REQUIRED_COLS)
for _col in REQUIRED_COLS[1:]:
    df[_col] = df[_col].astype(str)
df = df.reset_index(drop=True)

print(f"Total data  : {len(df)} baris")
print(f"Kolom CSV   : {df.columns.tolist()}")
print(f"\nDistribusi jenis_pemeriksaan:\n{df['jenis_pemeriksaan'].value_counts()}")
print("\nContoh data:")
print(
    df[["file_gambar", "jenis_pemeriksaan", "temuan", "kesimpulan"]].head(3).to_string()
)
print()


# =============================================================================
# BAGIAN 3 — Load Model & Processor
# =============================================================================

from transformers import AutoModelForCausalLM, AutoProcessor

n_gpu = torch.cuda.device_count()
max_memory = {i: "170GiB" for i in range(n_gpu)}
max_memory["cpu"] = "64GB"

print(f"GPU count  : {n_gpu}")
print(f"Max memory : {max_memory}")
print(f"Loading {MODEL_ID} dalam bfloat16 + sdpa attention...\n")

processor = AutoProcessor.from_pretrained(MODEL_ID, token=hf_token)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="balanced",
    max_memory=max_memory,
    trust_remote_code=True,
    token=hf_token,
    attn_implementation="sdpa",
)

if processor.tokenizer.pad_token is None:
    processor.tokenizer.pad_token = processor.tokenizer.eos_token
    model.config.pad_token_id = processor.tokenizer.eos_token_id

_img_size = getattr(processor.image_processor, "size", {"height": 896, "width": 896})
if isinstance(_img_size, dict):
    IMG_H = _img_size.get("height", 896)
    IMG_W = _img_size.get("width", 896)
elif isinstance(_img_size, int):
    IMG_H = IMG_W = _img_size
else:
    IMG_H = IMG_W = 896
print(f"Ukuran gambar dari processor config: {IMG_H}×{IMG_W}")

if hasattr(model, "hf_device_map"):
    _gpus_used = set(str(v) for v in model.hf_device_map.values())
    print(f"Model tersebar di device: {_gpus_used}")
else:
    print(f"Model device: {next(model.parameters()).device}")

for _i in range(n_gpu):
    _alloc = torch.cuda.memory_allocated(_i) / 1e9
    _total = torch.cuda.get_device_properties(_i).total_memory / 1e9
    print(f"  GPU {_i}: {_alloc:.1f} / {_total:.1f} GB terpakai")

print("\nModel & Processor berhasil dimuat.\n")


# =============================================================================
# BAGIAN 4 — Dataset Split (SEBELUM AdaLoRA config)
# =============================================================================

from sklearn.model_selection import train_test_split

train_df, temp_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df["jenis_pemeriksaan"]
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.5, random_state=42, stratify=temp_df["jenis_pemeriksaan"]
)

print("Split dataset (stratified by jenis_pemeriksaan):")
print(
    f"  Train : {len(train_df)} sampel\n{train_df['jenis_pemeriksaan'].value_counts().to_string()}"
)
print(
    f"\n  Val   : {len(val_df)} sampel\n{val_df['jenis_pemeriksaan'].value_counts().to_string()}"
)
print(
    f"\n  Test  : {len(test_df)} sampel\n{test_df['jenis_pemeriksaan'].value_counts().to_string()}\n"
)


# =============================================================================
# BAGIAN 5 — Konfigurasi & Inisialisasi AdaLoRA
# =============================================================================

from peft import AdaLoraConfig, get_peft_model

n_gpu_now = torch.cuda.device_count()
effective_batch = PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * n_gpu_now

n_train_actual = len(train_df)
steps_per_epoch = math.ceil(n_train_actual / effective_batch)

TOTAL_STEP = steps_per_epoch * NUM_TRAIN_EPOCHS

TINIT = max(50, int(TOTAL_STEP * 0.05))
TFINAL = max(200, int(TOTAL_STEP * 0.70))
DELTAT = 50

print(f"n_train aktual      : {n_train_actual} sampel")
print(f"Effective batch size: {effective_batch}")
print(f"Steps per epoch     : {steps_per_epoch}")
print(f"Total epochs        : {NUM_TRAIN_EPOCHS}")
print(f"TOTAL_STEP          : {TOTAL_STEP}")
print(f"tinit               : {TINIT}   (step mulai rank pruning)")
print(f"tfinal              : {TFINAL}  (step rank berhenti berubah)")
print(f"deltaT              : {DELTAT}  (interval update rank dalam satuan step)")

model.enable_input_require_grads()

adalora_config = AdaLoraConfig(
    # ── Rank scheduling ──────────────────────────────────────────────────────
    # Konfigurasi AdaLoRA tetap di-set menggunakan init_r=64 dan target_r=32
    init_r=64,
    target_r=32,
    total_step=TOTAL_STEP,
    tinit=TINIT,
    tfinal=TFINAL,
    deltaT=DELTAT,
    beta1=0.85,
    beta2=0.85,
    # ── Target modules ───────────────────────────────────────────────────────
    target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)",
    modules_to_save=["multi_modal_projector"],
    # lora_alpha disesuaikan sama dengan target_r (32) untuk menstabilkan skala pembaruan weight
    lora_alpha=32,
    lora_dropout=0.10,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, adalora_config)

_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
_total_p = sum(p.numel() for p in model.parameters())
print(
    f"\nTrainable params : {_trainable:,} / {_total_p:,} ({100 * _trainable / _total_p:.2f}%)"
)
model.print_trainable_parameters()
print(
    f"\nCatatan: Rank akan dipangkas secara adaptif — {adalora_config.init_r} → {adalora_config.target_r}"
)
print(
    f"         lora_alpha={adalora_config.lora_alpha} = target_r — skala update tetap stabil sepanjang training\n"
)


# =============================================================================
# BAGIAN 6 — Definisi Prompt, Augmentasi Teks & Dataset
# =============================================================================

from PIL import Image
from torch.utils.data import Dataset as TorchDataset
from torchvision import transforms

SYSTEM_PROMPTS = [
    "Anda adalah dokter radiologi. Buat laporan thorax lengkap dalam bahasa Indonesia.",
    "Tolong analisis X-ray dada ini dan tuliskan temuan radiologis beserta kesimpulannya.",
    "Sebagai spesialis radiologi, apa diagnosis Anda untuk gambar rontgen berikut?",
    "Berikan evaluasi klinis dari citra radiografi thorax ini secara terperinci.",
    "Tuliskan laporan medis dari foto rontgen dada ini, mencakup temuan dan konklusi.",
    "Interpretasikan citra rontgen thorax ini dan tuliskan laporan radiologi lengkap.",
    "Lakukan pembacaan radiografi dada ini dan dokumentasikan temuan klinis yang relevan.",
]

CLINICAL_SYNONYMS = {
    "kardiomegali": ["pembesaran jantung", "kardiomegali", "cardiomegaly"],
    "efusi pleura": ["efusi pleura", "cairan pleura", "pleural effusion"],
    "infiltrat": ["infiltrat", "konsolidasi", "opasitas"],
    "hiperinflasi": ["hiperinflasi", "hiperlusen", "air trapping"],
    "pneumonia": ["pneumonia", "infeksi paru", "radang paru"],
    "atelektasis": ["atelektasis", "kolaps paru", "atelectasis"],
    "normal": ["normal", "tidak tampak kelainan", "dalam batas normal"],
    "batas jantung": ["batas jantung", "kontur jantung", "siluet jantung"],
    "diafragma": ["diafragma", "hemidiafragma", "kubah diafragma"],
    "hilus": ["hilus", "hili", "parahiler"],
}


def augment_clinical_text(text: str) -> str:
    """
    Mengacak sinonim klinis dengan pembersihan tanda baca agar pencocokan frasa 
    dua kata berjalan optimal bahkan jika diakhiri titik/koma.
    """
    words = text.split()
    result = []
    i = 0
    while i < len(words):
        matched = False
        # Cocokkan frasa dua kata
        if i + 1 < len(words):
            word1_clean = words[i].lower().rstrip(".,;:")
            word2_clean = words[i+1].lower().rstrip(".,;:")
            two_word = f"{word1_clean} {word2_clean}"
            if two_word in CLINICAL_SYNONYMS:
                synonym = random.choice(CLINICAL_SYNONYMS[two_word])
                # Tempelkan kembali tanda baca dari kata kedua asli
                punct = words[i+1][len(word2_clean):]
                result.append(synonym + punct)
                i += 2
                matched = True
        if not matched:
            one_word = words[i].lower().rstrip(".,;:")
            if one_word in CLINICAL_SYNONYMS and random.random() < 0.4:
                # 40% kemungkinan diganti sinonim
                punct = words[i][len(one_word):]
                result.append(random.choice(CLINICAL_SYNONYMS[one_word]) + punct)
            else:
                result.append(words[i])
            i += 1
    return " ".join(result)


def build_prompt(
    processor,
    jenis_pemeriksaan: str,
    temuan: str,
    kesimpulan: str,
    prompt_text: str,
) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt_text},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": (
                        f"Jenis Pemeriksaan:\n{jenis_pemeriksaan}\n\n"
                        f"Temuan:\n{temuan}\n\n"
                        f"Kesimpulan:\n{kesimpulan}"
                    ),
                },
            ],
        },
    ]
    return processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )


def build_inference_prompt(processor, prompt_text: str) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt_text},
            ],
        },
    ]
    return processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def decrypt_image_to_pil(fernet: Fernet, img_path: Path) -> Image.Image:
    with open(img_path, "rb") as _f:
        encrypted_bytes = _f.read()
    try:
        decrypted_bytes = fernet.decrypt(encrypted_bytes)
    except InvalidToken as e:
        raise RuntimeError(f"Gagal dekripsi {img_path.name}: {e}")
    img = Image.open(io.BytesIO(decrypted_bytes)).convert("RGB")
    img = img.resize((IMG_W, IMG_H), Image.Resampling.LANCZOS)
    return img


class ChestXRayDataset(TorchDataset):
    """
    Dataset multimodal X-ray teroptimasi. Menghindari triple encoding gambar
    dengan melakukan estimasi token secara dinamis menggunakan C++ fast tokenizer.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        image_dir: Path,
        processor,
        fernet: Fernet,
        max_length: int = 2048,
        is_train: bool = False,
        total_epochs: int = 7,
    ):
        self.df = dataframe.reset_index(drop=True)
        self.image_dir = image_dir
        self.processor = processor
        self.fernet = fernet
        self.max_length = max_length
        self.is_train = is_train
        self.total_epochs = total_epochs
        self.current_epoch = 0

        # Counter truncation
        self._trunc_count = 0
        self._total_seen = 0

        # Bersihkan teks dari karakter kontrol
        for _col in ["temuan", "kesimpulan", "jenis_pemeriksaan"]:
            self.df[_col] = (
                self.df[_col]
                .astype(str)
                .str.replace("\n", " ", regex=False)
                .str.replace("\r", " ", regex=False)
                .str.strip()
            )

        # Filter baris dengan gambar yang ada di disk
        _valid_mask = self.df["file_gambar"].apply(lambda f: (image_dir / f).exists())
        _n_missing = (~_valid_mask).sum()
        if _n_missing > 0:
            print(f"  ⚠️  {_n_missing} file gambar tidak ditemukan — dilewati")
        self.df = self.df[_valid_mask].reset_index(drop=True)

        # --- OPTIMASI: Estimasi token gambar secara dinamis sekali saja saat start ---
        _dummy_img = Image.new("RGB", (IMG_W, IMG_H))
        # Gunakan build_inference_prompt agar prompt berisi token <image> yang sesuai
        _dummy_prompt = build_inference_prompt(self.processor, "test")
        _dummy_enc = self.processor(images=_dummy_img, text=_dummy_prompt, return_tensors="pt")
        _dummy_text_ids = self.processor.tokenizer(_dummy_prompt)["input_ids"]
        self.num_image_tokens = _dummy_enc["input_ids"].shape[1] - len(_dummy_text_ids)

        print(f"Dataset valid (is_train={is_train}): {len(self.df)} sampel")
        print(f"Jumlah token gambar terdeteksi: {self.num_image_tokens} tokens")

    def set_epoch(self, epoch: int):
        if self._total_seen > 0:
            _pct = 100.0 * self._trunc_count / self._total_seen
            print(
                f"  [Truncation Monitor] Epoch {self.current_epoch} selesai — "
                f"{self._trunc_count}/{self._total_seen} sampel terkena truncation "
                f"({_pct:.1f}%)"
            )
        self._trunc_count = 0
        self._total_seen = 0
        self.current_epoch = epoch

    def _build_augmentation(self) -> transforms.Compose:
        _progress = self.current_epoch / max(self.total_epochs - 1, 1)
        _scale = max(0.30, 1.2 - (0.9 * _progress))

        return transforms.Compose(
            [
                transforms.RandomRotation(degrees=8.0 * _scale),
                transforms.RandomAffine(
                    degrees=0,
                    translate=(0.03 * _scale, 0.03 * _scale),
                ),
                transforms.ColorJitter(
                    brightness=0.15 * _scale,
                    contrast=0.15 * _scale,
                ),
                transforms.RandomHorizontalFlip(p=0.3),
            ]
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.image_dir / row["file_gambar"]
        image = decrypt_image_to_pil(self.fernet, img_path)

        # Augmentasi curriculum hanya saat training
        if self.is_train:
            image = self._build_augmentation()(image)

        # Pilih satu prompt
        if self.is_train:
            selected_prompt = random.choice(SYSTEM_PROMPTS)
        else:
            selected_prompt = SYSTEM_PROMPTS[0]

        # Augmentasi teks klinis
        temuan_text = row["temuan"]
        if self.is_train:
            temuan_text = augment_clinical_text(temuan_text)

        # String prompt user-turn (tanpa token asisten)
        prompt_only_text = build_inference_prompt(self.processor, selected_prompt)

        # String penuh (user-turn + asisten)
        full_text = build_prompt(
            self.processor,
            row["jenis_pemeriksaan"],
            temuan_text,
            row["kesimpulan"],
            selected_prompt,
        )

        # --- OPTIMASI: Hitung actual_prompt_len & _raw_len via fast tokenizer ---
        prompt_text_ids = self.processor.tokenizer(prompt_only_text)["input_ids"]
        actual_prompt_len = self.num_image_tokens + len(prompt_text_ids)

        full_text_ids = self.processor.tokenizer(full_text)["input_ids"]
        _raw_len = self.num_image_tokens + len(full_text_ids)

        self._total_seen += 1
        if _raw_len > self.max_length:
            self._trunc_count += 1

        # --- Encoding Tunggal: Hanya panggil processor penuh 1 kali ---
        encoding = self.processor(
            images=image,
            text=full_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
        )

        item = {k: v.squeeze(0) for k, v in encoding.items() if k in ("input_ids", "attention_mask", "pixel_values")}
        labels = item["input_ids"].clone()

        # Label masking
        real_token_indices = torch.where(item["attention_mask"] == 1)[0]
        prompt_indices = real_token_indices[:actual_prompt_len]
        labels[prompt_indices] = -100       # mask: prompt
        labels[item["attention_mask"] == 0] = -100  # mask: padding

        item["labels"] = labels

        if "token_type_ids" not in item:
            item["token_type_ids"] = torch.zeros_like(item["input_ids"])

        return item

    def report_truncation_stats(self):
        if self._total_seen == 0:
            return
        _pct = 100.0 * self._trunc_count / self._total_seen
        print(
            f"[Truncation Stats] {self._trunc_count}/{self._total_seen} "
            f"sampel ({_pct:.1f}%) terpotong (panjang asli > max_length={self.max_length}). "
            f"{'⚠️ Pertimbangkan menaikkan max_length.' if _pct > 10 else '✅ OK.'}"
        )


# =============================================================================
# BAGIAN 7 — Instansiasi Dataset & Verifikasi Sampel
# =============================================================================

train_dataset = ChestXRayDataset(
    train_df,
    ENCRYPTED_IMAGE_DIR,
    processor,
    fernet,
    max_length=MAX_SEQ_LENGTH,
    is_train=True,
    total_epochs=NUM_TRAIN_EPOCHS,
)
val_dataset = ChestXRayDataset(
    val_df,
    ENCRYPTED_IMAGE_DIR,
    processor,
    fernet,
    max_length=MAX_SEQ_LENGTH,
    is_train=False,
    total_epochs=NUM_TRAIN_EPOCHS,
)
test_dataset = ChestXRayDataset(
    test_df,
    ENCRYPTED_IMAGE_DIR,
    processor,
    fernet,
    max_length=MAX_SEQ_LENGTH,
    is_train=False,
    total_epochs=NUM_TRAIN_EPOCHS,
)

print("\n── Verifikasi sampel ──────────────────────────────────────")
_sample = train_dataset[0]
_n_valid_labels = (_sample["labels"] != -100).sum().item()
_n_total_labels = _sample["labels"].shape[0]
print(f"Kunci tensor   : {list(_sample.keys())}")
print(f"input_ids      : {_sample['input_ids'].shape}")
print(f"pixel_values   : {_sample['pixel_values'].shape}")
print(f"attention_mask : {_sample['attention_mask'].shape}")
print(
    f"labels (untuk loss) : {_n_valid_labels}/{_n_total_labels} token "
    f"({100 * _n_valid_labels / _n_total_labels:.1f}%) digunakan"
)
print("Verifikasi sampel: OK\n")


# =============================================================================
# BAGIAN 8 — Data Collator & Callback
# =============================================================================

def multimodal_data_collator(features: list) -> dict:
    batch = {}
    batch["input_ids"] = torch.nn.utils.rnn.pad_sequence(
        [f["input_ids"] for f in features],
        batch_first=True,
        padding_value=0,
    )
    batch["attention_mask"] = torch.nn.utils.rnn.pad_sequence(
        [f["attention_mask"] for f in features],
        batch_first=True,
        padding_value=0,
    )
    batch["token_type_ids"] = torch.nn.utils.rnn.pad_sequence(
        [f["token_type_ids"] for f in features],
        batch_first=True,
        padding_value=0,
    )
    batch["labels"] = torch.nn.utils.rnn.pad_sequence(
        [f["labels"] for f in features],
        batch_first=True,
        padding_value=-100,
    )
    batch["pixel_values"] = torch.stack([f["pixel_values"] for f in features])
    return batch


# Verifikasi shape batch
_batch = multimodal_data_collator([train_dataset[0], train_dataset[1]])
print("── Verifikasi batch ───────────────────────────────────────")
for _k, _v in _batch.items():
    print(f"  {_k:<18}: {_v.shape}")
print()


from transformers import (
    Trainer,
    TrainerCallback,
    TrainingArguments,
)


class EpochAugmentationCallback(TrainerCallback):
    def __init__(self, train_dataset: ChestXRayDataset):
        self.train_dataset = train_dataset

    def on_epoch_begin(self, args, state, control, **kwargs):
        _epoch = int(state.epoch) if state.epoch else 0
        _progress = _epoch / max(self.train_dataset.total_epochs - 1, 1)
        _scale = max(0.30, 1.2 - (0.9 * _progress))
        self.train_dataset.set_epoch(_epoch)
        print(
            f"\n[CurriculumAug] Epoch {_epoch + 1}/{int(args.num_train_epochs)} dimulai  |  "
            f"aug_scale={_scale:.2f}  "
            f"(rotasi ±{8.0 * _scale:.1f}°, translate {3.0 * _scale:.1f}%, "
            f"jitter {15.0 * _scale:.1f}%)"
        )


class SWACallback(TrainerCallback):
    """
    Stochastic Weight Averaging (SWA).
    Dioptimalkan dengan menyimpan data weights di CPU untuk menghemat VRAM GPU.
    """

    def __init__(self, swa_start_ratio: float = 0.70, total_epochs: int = 15):
        self.swa_start_epoch = max(1, int(total_epochs * swa_start_ratio))
        self.swa_weights = None
        self.swa_count = 0
        print(f"[SWA] Averaging dimulai dari epoch {self.swa_start_epoch}")

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        if model is None:
            return
        current_epoch = int(state.epoch) if state.epoch else 0
        if current_epoch < self.swa_start_epoch:
            return

        # Pindahkan parameter ke CPU untuk menghemat memori GPU
        trainable_params = {
            n: p.detach().cpu().float().clone()
            for n, p in model.named_parameters()
            if p.requires_grad
        }

        if self.swa_weights is None:
            self.swa_weights = trainable_params
            self.swa_count = 1
        else:
            self.swa_count += 1
            for name in self.swa_weights:
                self.swa_weights[name] = (
                    self.swa_weights[name] * (self.swa_count - 1)
                    + trainable_params[name]
                ) / self.swa_count

        print(
            f"[SWA] Epoch {current_epoch}: rata-ratakan {self.swa_count} checkpoint(s)"
        )

    def apply_swa_weights(self, model):
        if self.swa_weights is None:
            print("[SWA] Tidak ada weight yang di-average (training terlalu pendek).")
            return
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in self.swa_weights and param.requires_grad:
                    param.copy_(self.swa_weights[name].to(param.device).to(param.dtype))
        print(f"[SWA] Weights dari {self.swa_count} checkpoint berhasil diterapkan.")


class TextEvaluationCallback(TrainerCallback):
    """
    Callback Kustom untuk melakukan evaluasi teks (ROUGE-L & BERTScore) pada data
    training dan validation di akhir setiap epoch secara real-time.
    Mengevaluasi subset data representatif demi efisiensi waktu training.
    """

    def __init__(self, train_df, val_df, processor, fernet, output_dir, num_samples=50):
        self.train_df = train_df
        self.val_df = val_df
        self.processor = processor
        self.fernet = fernet
        self.output_dir = Path(output_dir)
        self.num_samples = num_samples
        self.history = []

        try:
            from rouge_score import rouge_scorer
            self.rouge_scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
            print("✅ ROUGE scorer siap digunakan untuk evaluasi tiap epoch.")
        except ImportError:
            self.rouge_scorer = None
            print("⚠️ rouge-score library tidak terinstal. Evaluasi ROUGE dilewati.")

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        if model is None or self.rouge_scorer is None:
            return

        current_epoch = int(state.epoch) if state.epoch else 0
        print(f"\n[Epoch {current_epoch}] Menjalankan Evaluasi ROUGE & BERTScore...")

        # Switch ke mode evaluasi
        model.eval()

        train_metrics = self.evaluate_subset(model, self.train_df, "train")
        val_metrics = self.evaluate_subset(model, self.val_df, "val")

        record = {
            "epoch": current_epoch,
            "train_rougeL": train_metrics["rougeL"],
            "train_bertscore": train_metrics["bertscore"],
            "val_rougeL": val_metrics["rougeL"],
            "val_bertscore": val_metrics["bertscore"],
        }
        self.history.append(record)

        print(f"  [Hasil Epoch {current_epoch}]")
        print(f"    Train -> ROUGE-L: {train_metrics['rougeL']*100:.2f}% | BERTScore: {train_metrics['bertscore']*100:.2f}%")
        print(f"    Val   -> ROUGE-L: {val_metrics['rougeL']*100:.2f}% | BERTScore: {val_metrics['bertscore']*100:.2f}%")

        # Simpan riwayat evaluasi ke JSON secara real-time
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with open(self.output_dir / "epoch_text_metrics_history.json", "w") as f:
            json.dump(self.history, f, indent=4)

        # Kembalikan model ke mode training
        model.train()

    def evaluate_subset(self, model, df, name):
        import bert_score as bs_lib

        # Ambil subset data secara acak & representatif
        subset_df = df.sample(n=min(len(df), self.num_samples), random_state=42).reset_index(drop=True)
        predictions = []
        references = []

        for idx, row in subset_df.iterrows():
            img_path = ENCRYPTED_IMAGE_DIR / row["file_gambar"]
            try:
                image = decrypt_image_to_pil(self.fernet, img_path)
            except Exception:
                continue

            prompt_text = build_inference_prompt(self.processor, SYSTEM_PROMPTS[0])
            enc = self.processor(
                images=image,
                text=prompt_text,
                return_tensors="pt",
            ).to(model.device)

            with torch.no_grad():
                out = model.generate(
                    **enc,
                    max_new_tokens=512,
                    do_sample=False,
                )

            pred = self.processor.decode(
                out[0][enc["input_ids"].shape[1]:],
                skip_special_tokens=True,
            ).strip()

            ref = (
                f"Jenis Pemeriksaan:\n{row['jenis_pemeriksaan']}\n\n"
                f"Temuan:\n{row['temuan']}\n\n"
                f"Kesimpulan:\n{row['kesimpulan']}"
            )

            predictions.append(pred)
            references.append(ref)

        if not predictions:
            return {"rougeL": 0.0, "bertscore": 0.0}

        # Hitung ROUGE-L
        rouge_scores = [self.rouge_scorer.score(ref, pred)["rougeL"].fmeasure
                        for pred, ref in zip(predictions, references)]
        avg_rouge_l = sum(rouge_scores) / len(rouge_scores)

        # Hitung BERTScore
        try:
            P, R, F1 = bs_lib.score(
                predictions,
                references,
                lang="id",
                model_type="xlm-roberta-base",
                verbose=False,
            )
            avg_bertscore = F1.mean().item()
        except Exception:
            avg_bertscore = 0.0

        return {"rougeL": avg_rouge_l, "bertscore": avg_bertscore}


# =============================================================================
# BAGIAN 9 — Konfigurasi Training & Eksekusi
# =============================================================================

gc.collect()
torch.cuda.empty_cache()

print("── Status VRAM sebelum training ──────────────────────────")
for _i in range(torch.cuda.device_count()):
    _total = torch.cuda.get_device_properties(_i).total_memory / 1e9
    _alloc = torch.cuda.memory_allocated(_i) / 1e9
    _reserved = torch.cuda.memory_reserved(_i) / 1e9
    print(
        f"  GPU {_i}: allocated={_alloc:.1f} GB | reserved={_reserved:.1f} GB | total={_total:.1f} GB"
    )
print()

os.environ["WANDB_DISABLED"] = "true"

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    # ── Epoch & Batch ─────────────────────────────────────────────────────────
    num_train_epochs=NUM_TRAIN_EPOCHS,
    per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    # ── Learning Rate ─────────────────────────────────────────────────────────
    learning_rate=1.5e-4,
    lr_scheduler_type="cosine_with_restarts",
    lr_scheduler_kwargs={"num_cycles": 2},
    warmup_steps=TINIT,
    weight_decay=0.08,
    max_grad_norm=0.5,
    label_smoothing_factor=0.05,
    bf16=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    logging_strategy="epoch",
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    save_total_limit=None, 
    report_to="none",
    remove_unused_columns=False,
    dataloader_num_workers=0,
    dataloader_pin_memory=True,
)

# Instansiasi SWACallback & TextEvaluationCallback
swa_callback = SWACallback(
    swa_start_ratio=SWA_START_RATIO,
    total_epochs=NUM_TRAIN_EPOCHS,
)

text_eval_callback = TextEvaluationCallback(
    train_df=train_df,
    val_df=val_df,
    processor=processor,
    fernet=fernet,
    output_dir=OUTPUT_DIR,
    num_samples=NUM_EVAL_TEXT_SAMPLES
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=multimodal_data_collator,
    callbacks=[
        EpochAugmentationCallback(train_dataset),
        swa_callback,
        text_eval_callback,
    ],
)

_n_gpu_actual = torch.cuda.device_count()
_eff_batch = (
    training_args.per_device_train_batch_size
    * training_args.gradient_accumulation_steps
    * _n_gpu_actual
)
_total_steps_act = math.ceil(len(train_dataset) / _eff_batch) * int(
    training_args.num_train_epochs
)

print("=" * 70)
print("  RINGKASAN KONFIGURASI TRAINING v9")
print("=" * 70)
print(f"  Model              : {MODEL_ID}")
print(f"  GPU count          : {_n_gpu_actual}")
print(
    f"  Effective batch    : {_eff_batch}  "
    f"({PER_DEVICE_TRAIN_BATCH_SIZE} × {GRADIENT_ACCUMULATION_STEPS} × {_n_gpu_actual})"
)
print(f"  n_train            : {len(train_dataset)}")
print(f"  n_val              : {len(val_dataset)}")
print(f"  n_test             : {len(test_dataset)}")
print(f"  Steps per epoch    : {steps_per_epoch}")
print(f"  Total steps (est.) : {_total_steps_act}")
print()
print("  AdaLoRA rank schedule:")
print(f"    init_r={adalora_config.init_r} → target_r={adalora_config.target_r}  (Tetap init_r=64, target_r=32)")
print(f"    lora_alpha={adalora_config.lora_alpha} = target_r")
print(f"    Pruning aktif    : step {TINIT} → {TFINAL}")
print(f"    deltaT           : {DELTAT} step")
print()
print(f"  LR                 : {training_args.learning_rate}")
print(f"  LR scheduler       : cosine_with_restarts, num_cycles=2")
print(f"  Warmup steps       : {training_args.warmup_steps}")
print(f"  max_grad_norm      : {training_args.max_grad_norm}")
print(f"  lora_dropout       : {adalora_config.lora_dropout}")
print(f"  weight_decay       : {training_args.weight_decay}")
print(f"  label_smoothing    : {training_args.label_smoothing_factor}")
print(f"  EarlyStopping      : NONAKTIF")
print(f"  Save Checkpoints   : SEMUA (save_total_limit = None)")
print(f"  SWA                : mulai epoch {swa_callback.swa_start_epoch}/{NUM_TRAIN_EPOCHS}")
print(f"  Text eval callback : Aktif di akhir setiap epoch (ROUGE + BERTScore)")
print(f"  Text augmentation  : aktif (sinonim klinis, 40% per kata)")
print(f"  Visual aug scale   : 1.2→0.30")
print("=" * 70)
print("Memulai fine-tuning AdaLoRA v9...\n")

train_result = trainer.train()
print("\nFine-tuning selesai!")

train_dataset.report_truncation_stats()


# =============================================================================
# BAGIAN 9B — Terapkan SWA Weights setelah Training
# =============================================================================

print("\n" + "=" * 70)
print("  MENERAPKAN SWA WEIGHTS")
print("=" * 70)
swa_callback.apply_swa_weights(trainer.model)

# Evaluasi val set dengan SWA weights untuk memverifikasi perbaikan
print("\nEvaluasi val set setelah SWA weights diterapkan:")
swa_val_results = trainer.evaluate(eval_dataset=val_dataset)
print(f"  Val Loss (SWA)  : {swa_val_results.get('eval_loss', float('nan')):.4f}")


# =============================================================================
# BAGIAN 10 — Simpan Log & Metrik Training
# =============================================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

try:
    with open(os.path.join(OUTPUT_DIR, "final_train_metrics.json"), "w") as _f:
        json.dump(train_result.metrics, _f, indent=4)

    _log_history = trainer.state.log_history
    with open(os.path.join(OUTPUT_DIR, "trainer_log_history_v9.json"), "w") as _f:
        json.dump(_log_history, _f, indent=4)

    pd.DataFrame(_log_history).to_csv(
        os.path.join(OUTPUT_DIR, "training_log_history_v9.csv"), index=False
    )
    print(f"✅ Log dan metrik training disimpan di: {OUTPUT_DIR}")
except Exception as _e:
    print(f"⚠️  Kesalahan saat menyimpan log: {_e}")


# =============================================================================
# BAGIAN 11 — Evaluasi Akhir Test Set dengan Metrik Teks
# =============================================================================

print("\n" + "=" * 70)
print("  EVALUASI PADA TEST SET (10% data unseen)")
print("=" * 70)

test_results = trainer.evaluate(eval_dataset=test_dataset)

print(f"\n  Test Loss       : {test_results.get('eval_loss', float('nan')):.4f}")
print(f"  Test Runtime    : {test_results.get('eval_runtime', 0):.1f} detik")
print(f"  Test Samples/s  : {test_results.get('eval_samples_per_second', 0):.2f}")

try:
    with open(os.path.join(OUTPUT_DIR, "test_results.json"), "w") as _f:
        json.dump(test_results, _f, indent=4)
    print(f"\n✅ Hasil evaluasi test set disimpan: {OUTPUT_DIR}/test_results.json")
except Exception as _e:
    print(f"⚠️  Gagal menyimpan test results: {_e}")


print("\n── Evaluasi Metrik Teks Akhir (ROUGE-L + BERTScore) ───────────────")

try:
    from rouge_score import rouge_scorer
    import bert_score as bs_lib

    _scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)

    predictions = []
    references = []

    trainer.model.eval()
    for _idx in range(min(len(test_dataset), 50)):
        _row = test_df.iloc[_idx]
        _img_path = ENCRYPTED_IMAGE_DIR / _row["file_gambar"]
        _image = decrypt_image_to_pil(fernet, _img_path)
        _prompt_text = build_inference_prompt(processor, SYSTEM_PROMPTS[0])

        _enc = processor(
            images=_image,
            text=_prompt_text,
            return_tensors="pt",
        ).to(trainer.model.device)

        with torch.no_grad():
            _out = trainer.model.generate(
                **_enc,
                max_new_tokens=512,
                do_sample=False,
            )

        _pred = processor.decode(
            _out[0][_enc["input_ids"].shape[1]:],
            skip_special_tokens=True,
        ).strip()

        _ref = (
            f"Jenis Pemeriksaan:\n{_row['jenis_pemeriksaan']}\n\n"
            f"Temuan:\n{_row['temuan']}\n\n"
            f"Kesimpulan:\n{_row['kesimpulan']}"
        )

        predictions.append(_pred)
        references.append(_ref)

    # ROUGE-L
    rouge_scores = [_scorer.score(ref, pred)["rougeL"].fmeasure
                    for pred, ref in zip(predictions, references)]
    avg_rouge_l = sum(rouge_scores) / len(rouge_scores)
    print(f"  ROUGE-L (avg)   : {avg_rouge_l:.4f}  (n={len(predictions)} sampel)")

    # BERTScore
    P, R, F1 = bs_lib.score(
        predictions,
        references,
        lang="id",
        model_type="xlm-roberta-base",
        verbose=False,
    )
    avg_bertscore_f1 = F1.mean().item()
    print(f"  BERTScore F1    : {avg_bertscore_f1:.4f}  (xlm-roberta-base)")

    # Simpan metrik teks ke JSON
    text_metrics = {
        "rouge_l": avg_rouge_l,
        "bertscore_f1": avg_bertscore_f1,
        "bertscore_precision": P.mean().item(),
        "bertscore_recall": R.mean().item(),
        "n_samples_evaluated": len(predictions),
    }
    with open(os.path.join(OUTPUT_DIR, "text_metrics.json"), "w") as _f:
        json.dump(text_metrics, _f, indent=4)
    print(f"\n✅ Metrik teks disimpan: {OUTPUT_DIR}/text_metrics.json")

    # Tampilkan beberapa contoh prediksi
    print("\n── Contoh prediksi (3 sampel pertama) ────────────────────────────")
    for _i in range(min(3, len(predictions))):
        print(f"\n[Sampel {_i+1}]")
        print(f"  Referensi : {references[_i][:150]}...")
        print(f"  Prediksi  : {predictions[_i][:150]}...")
        print(f"  ROUGE-L   : {rouge_scores[_i]:.4f}")

except Exception as _e:
    print(f"⚠️  Evaluasi metrik teks gagal: {_e}")


# =============================================================================
# BAGIAN 12 — Simpan Model Akhir
# =============================================================================

print("\n" + "=" * 70)
print("  MENYIMPAN MODEL")
print("=" * 70)

print("Menggabungkan adapter ke base model dengan merge_and_unload()...")
try:
    merged_model = trainer.model.merge_and_unload()
    merged_model.save_pretrained(
        OUTPUT_DIR,
        safe_serialization=True,
    )
    print("✅ Model ter-merge disimpan (bersih, tanpa wrapper PEFT)")
except RuntimeError as _e:
    print(f"⚠️  merge_and_unload() gagal: {_e}")
    print("   Fallback: menyimpan adapter PEFT saja")
    trainer.model.save_pretrained(OUTPUT_DIR)
    print("✅ Adapter PEFT disimpan")

processor.save_pretrained(OUTPUT_DIR)
print(f"✅ Processor disimpan di: {OUTPUT_DIR}")

# Tampilkan isi folder hasil
_saved_files = sorted(os.listdir(OUTPUT_DIR))
print(f"\nFile tersimpan ({len(_saved_files)} file):")
for _fname in _saved_files:
    _fpath = os.path.join(OUTPUT_DIR, _fname)
    _size_kb = os.path.getsize(_fpath) / 1024 if os.path.isfile(_fpath) else 0
    print(f"  {_fname:<52}  {_size_kb:>10.1f} KB")

print("\n" + "=" * 70)
print("  FINE-TUNING ADALORA v9 SELESAI")
print("=" * 70)
