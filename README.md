# Fine-Tuning Model Multimodal AI dengan Adaptive LoRA (AdaLoRA) untuk Analisis Citra Radiologi Toraks

Kode program ini mengimplementasikan *pipeline fine-tuning* multimodal *end-to-end* untuk tugas pembuatan laporan radiologi toraks otomatis menggunakan model dasar **MedGemma 4B** (`google/medgemma-4b-it`) dengan presisi `bfloat16`. Alur kerja mencakup dekripsi dataset citra dan anotasi teks klinis terenkripsi Fernet, pembagian data terstratifikasi berdasarkan jenis pemeriksaan (80% *train*, 10% *validation*, 10% *test*), penerapan augmentasi kurikulum visual adaptif beserta augmentasi sinonim klinis, serta penataan *label masking* pada *prompt* multimodal dengan panjang konteks maksimum (*max sequence length*) 2048 token. Efisiensi adaptasi parameter dicapai menggunakan metode **Adaptive LoRA (AdaLoRA)** yang memangkas alokasi *rank* secara dinamis dari `init_r = 64` menuju `target_r = 32`, dengan `lora_alpha = 32` (disamakan dengan *target rank* untuk stabilitas skala pembaruan bobot), `lora_dropout = 0.10`, interval pembaruan *rank* `deltaT = 50` langkah, serta koefisien momentum sensitivitas $\beta_1 = \beta_2 = 0.85$. Target modul AdaLoRA mencakup seluruh proyeksi atensi dan *feed-forward* pada model bahasa (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`) sekaligus melatih modul `multi_modal_projector` secara penuh (`modules_to_save`).

Proses pelatihan dikonfigurasi berjalan selama **20 epoch** dengan ukuran *batch* per perangkat sebesar **4** dan *gradient accumulation steps* sebesar **8** (menghasilkan *effective batch size* 32 untuk konfigurasi GPU tunggal). Optimalisasi model menggunakan *learning rate* puncak sebesar **$1.5 \times 10^{-4}$** dengan penjadwal *cosine with restarts* (2 siklus), *warmup steps* adaptif mengikuti ambang awal pemangkasan *rank* ($t_{\text{init}}$), *weight decay* **0.08**, *gradient clipping* (`max_grad_norm`) **0.5**, serta *label smoothing factor* **0.05** yang didukung *gradient checkpointing* untuk efisiensi memori grafis. Untuk memperkuat generalisasi pada data klinis tak terduga, sistem menerapkan teknik **Stochastic Weight Averaging (SWA)** yang mulai merata-ratakan bobot pada 70% masa pelatihan (epoch ke-14), pemantauan metrik teks generatif berkala (ROUGE-L dan BERTScore berbasis model `xlm-roberta-base`), serta peleburan bobot adapter secara langsung ke dalam model utama melalui fungsi `merge_and_unload()` di akhir pelatihan.

---

## ⚙️ Spesifikasi & Konfigurasi Hyperparameter

### 1. Konfigurasi Model Dasar
| Parameter | Nilai / Deskripsi |
| :--- | :--- |
| **Model ID** | `google/medgemma-4b-it` |
| **Precision** | `torch.bfloat16` |
| **Attention Implementation** | SDPA (`sdpa` - PyTorch Scaled Dot-Product Attention) |
| **Device Map** | `balanced` |
| **Max Sequence Length** | 2048 tokens |
| **Dimensi Resolusi Citra** | $896 \times 896$ piksel |

### 2. Konfigurasi Adaptive LoRA (AdaLoRA)
| Parameter | Nilai | Keterangan |
| :--- | :--- | :--- |
| `init_r` | `64` | Nilai *rank* awal sebelum *pruning* dinamis |
| `target_r` | `32` | Nilai *rank* akhir target setelah proses alokasi adaptif |
| `lora_alpha` | `32` | Skala amplifikasi LoRA (disamakan dengan `target_r`) |
| `lora_dropout` | `0.10` | Probabilitas *dropout* pada layer adapter |
| `target_modules` | `r".*language_model.*\.(q_proj\|k_proj\|v_proj\|o_proj\|gate_proj\|up_proj\|down_proj)"` | Lapisan *attention* dan *feed-forward* pada model bahasa |
| `modules_to_save` | `["multi_modal_projector"]` | Modul jembatan multimodal dilatih secara penuh |
| `tinit` | `max(50, int(TOTAL_STEP * 0.05))` | Langkah awal aktivasi pemangkasan *rank* |
| `tfinal` | `max(200, int(TOTAL_STEP * 0.70))` | Langkah penghentian perubahan alokasi *rank* |
| `deltaT` | `50` | Interval langkah kalkulasi pentingnya parameter (*importance update*) |
| `beta1`, `beta2` | `0.85`, `0.85` | Faktor pemulusan eksponensial (*EMA*) |

### 3. Hyperparameter Pelatihan (*Training Arguments*)
| Parameter | Nilai | Keterangan |
| :--- | :--- | :--- |
| **Num Train Epochs** | `20` | Total epoch pelatihan |
| **Per-Device Batch Size** | `4` | Ukuran *batch* per GPU |
| **Gradient Accumulation** | `8` | Akumulasi gradien (efektif *batch* = 32 per GPU) |
| **Learning Rate** | `1.5e-4` ($0.00015$) | Tingkat pembelajaran puncak |
| **LR Scheduler** | `cosine_with_restarts` | Penjadwal penurunan *learning rate* kosinus dengan 2 siklus |
| **Warmup Steps** | Mengikuti $t_{\text{init}}$ | Langkah pemanasan laju pembelajaran |
| **Weight Decay** | `0.08` | Regularisasi penalti bobot |
| **Max Grad Norm** | `0.5` | Pemotongan gradien (*gradient clipping*) |
| **Label Smoothing** | `0.05` | Regularisasi distribusi probabilitas label |
| **Gradient Checkpointing**| `True` (`use_reentrant=False`) | Penghematan konsumsi VRAM GPU |
| **SWA Start Ratio** | `0.70` (Epoch 14) | *Stochastic Weight Averaging* pada 30% epoch terakhir |

---

## 📂 Struktur Berkas Workspace

- [`adalora_code.py`](file:///g:/HPC/SOURCE_CODE-TASI-2526-106/adalora_code.py): Skrip utama implementasi *training pipeline*, konfigurasi AdaLoRA, augmentasi kurikulum, serta evaluasi SWA.
- [`evaluation_adaptive_model.ipynb`](file:///g:/HPC/SOURCE_CODE-TASI-2526-106/evaluation_adaptive_model.ipynb): *Notebook* evaluasi inferensi mendalam untuk model hasil adaptasi AdaLoRA.
- [`evaluation_baseline_model.ipynb`](file:///g:/HPC/SOURCE_CODE-TASI-2526-106/evaluation_baseline_model.ipynb): *Notebook* pengujian dan komparasi performa model *baseline* (tanpa adaptasi / *zero-shot*).
- [`PREPROCESSING-REPORT.ipynb`](file:///g:/HPC/SOURCE_CODE-TASI-2526-106/PREPROCESSING-REPORT.ipynb): *Notebook* visualisasi dan analisis data eksploratif untuk tahap pra-pemrosesan laporan klinis.

---

