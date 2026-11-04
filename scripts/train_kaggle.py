"""Fine-tune document-pretrained DiT and produce an id,label submission.

Kaggle: edit the paths below, then run the entire file. No notebook state is needed.
This script skips duplicate hashing and label-conflict analysis. It caches resized
images once, records unreadable train files, and requires every test file to decode.
"""
from pathlib import Path

# Pengaturan yang perlu disesuaikan.
DATA_DIR = Path("/kaggle/input/competitions/document-image-classification-sd234504")
TRAIN_DIR = DATA_DIR / "train/train"
TEST_DIR = DATA_DIR / "test/test"
OUTPUT_DIR = Path("/kaggle/working/dit_pretrained")
CHECKPOINT_PATH = OUTPUT_DIR / "best_model.pt"
SUBMISSION_PATH = Path("/kaggle/working/submission.csv")
MODEL_NAME = "microsoft/dit-base"  # Backbone dokumen tanpa fine-tuning RVL-CDIP.
MODEL_CONFIG = None
NORMALIZE_MEAN = [0.5, 0.5, 0.5]
NORMALIZE_STD = [0.5, 0.5, 0.5]
IMAGE_SIZE = 224
BATCH_SIZE = 16  # Turunkan menjadi 8 jika memori GPU tidak cukup.
EPOCHS = 10
HEAD_EPOCHS = 0  # Langsung fine-tuning seluruh model dengan learning rate kecil.
LEARNING_RATE = 2e-5
HEAD_LEARNING_RATE = 1e-4
PATIENCE = 3
NUM_WORKERS = 2  # Gunakan 0 jika menjalankan lewat sel dan worker bermasalah.
PREPROCESS_WORKERS = 2
SEED = 42
PREDICT_ONLY = False

import hashlib
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
try:
    from transformers import AutoImageProcessor, BeitConfig, BeitForImageClassification
except ImportError as exc:
    raise ImportError("Instal dependensi dahulu: pip install transformers==4.57.1 safetensors") from exc
from tqdm.auto import tqdm

CLASS_NAMES = [
    "advertisement", "budget", "email", "file_folder", "form", "handwritten",
    "invoice", "letter", "memo", "news_article", "presentation", "questionnaire",
    "resume", "scientific_publication", "scientific_report", "specification",
]


def seed_everything(seed):
    """Atur seed eksperimen; CUDA autotuning tetap dapat menghasilkan variasi kecil."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def seed_worker(worker_id):
    """Atur seed Python dan NumPy pada setiap worker DataLoader."""
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)

def load_tables():
    """Validasi CSV, pemetaan label, dan susun path berdasarkan folder pengguna."""
    root = Path(DATA_DIR)
    train = pd.read_csv(root / "train.csv", dtype={"id": str})
    test = pd.read_csv(root / "test.csv", dtype={"id": str})
    sample = pd.read_csv(root / "sample_submission.csv", dtype={"id": str})
    for name, df, required in [
        ("train", train, ["id", "filename", "class_name", "label"]),
        ("test", test, ["id", "filename"]),
        ("sample_submission", sample, ["id", "label"]),
    ]:
        if df.empty or not set(required).issubset(df.columns):
            raise ValueError(f"CSV {name} kosong atau kolomnya tidak lengkap.")
        if df[required].isna().any().any() or df["id"].duplicated().any():
            raise ValueError(f"Ada nilai kosong atau ID berulang pada {name}.")
    labels = pd.to_numeric(train["label"], errors="raise")
    if not labels.isin(range(16)).all():
        raise ValueError("Label train harus berupa integer 0 sampai 15.")
    train["label"] = labels.astype(int)
    mapping = train["label"].map(dict(enumerate(CLASS_NAMES)))
    if not train["class_name"].eq(mapping).all():
        raise ValueError("class_name tidak sesuai dengan referensi label.")
    if set(test["id"]) != set(sample["id"]) or list(sample.columns) != ["id", "label"]:
        raise ValueError("ID atau format sample submission tidak sesuai.")
    for split, df, folder in [("train", train, TRAIN_DIR), ("test", test, TEST_DIR)]:
        folder = Path(folder)
        if not folder.is_dir():
            raise FileNotFoundError(f"Folder {split} tidak ditemukan: {folder}")
        paths = []
        for row in df.itertuples(index=False):
            filename = str(row.filename).replace("\\", "/").split("/")[-1]
            path = folder / row.class_name / filename if split == "train" else folder / filename
            paths.append(str(path))
        df["path"] = paths
        if not any(Path(path).is_file() for path in paths):
            raise FileNotFoundError(f"Tidak ada gambar {split} ditemukan. Periksa folder {folder}.")
    return train, test, sample

def cache_one_image(path, cache_dir, size):
    """Cache halaman pertama sebagai PNG kecil; key memakai metadata file, bukan piksel."""
    try:
        source = Path(path)
        stat = source.stat()
        key = json.dumps([str(source.resolve()), stat.st_size, stat.st_mtime_ns, size, "dit-stretch-rgb-v1"])
        name = hashlib.sha256(key.encode()).hexdigest()
        target = Path(cache_dir) / f"{name}.png"
        if target.is_file():
            return str(target), "", True
        with Image.open(source) as image:
            image.seek(0)
            rgb = ImageOps.exif_transpose(image).convert("RGB")
            # Resize seluruh halaman tanpa center crop. Proporsi berubah ke persegi.
            canvas = rgb.resize((size, size), Image.Resampling.BICUBIC)
        temporary = target.with_suffix(".tmp")
        canvas.save(temporary, format="PNG", compress_level=1)
        os.replace(temporary, target)
        return str(target), "", False
    except Exception as exc:
        return "", f"{type(exc).__name__}: {exc}", False

def prepare_cache(df, split, output):
    """Siapkan gambar kecil secara paralel dan simpan laporan kegagalan pembacaan."""
    cache_dir = output / "cache" / split
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = df["path"].tolist()
    # Deduplikasi path hanya mencegah dua thread menulis file cache yang sama.
    unique_paths = list(dict.fromkeys(paths))
    def prepare(path):
        """Siapkan satu path dengan ukuran dan folder cache eksperimen ini."""
        return cache_one_image(path, cache_dir, IMAGE_SIZE)
    with ThreadPoolExecutor(max_workers=PREPROCESS_WORKERS) as pool:
        results = list(tqdm(pool.map(prepare, unique_paths), total=len(unique_paths),
                            desc=f"Cache {split} ({IMAGE_SIZE}px)"))
    lookup = dict(zip(unique_paths, results))
    result = df.copy()
    result["cached_path"] = [lookup[path][0] for path in paths]
    result["read_error"] = [lookup[path][1] for path in paths]
    bad = result.loc[result["read_error"].ne("")]
    bad.to_csv(output / f"unreadable_{split}.csv", index=False)
    print(f"{split}: gagal {len(bad)}; cache lama dipakai {sum(item[2] for item in results)}")
    if split == "test" and not bad.empty:
        raise ValueError("Ada test gagal dibaca. Periksa unreadable_test.csv; semua ID test harus diprediksi.")
    result = result.loc[result["read_error"].eq("")].reset_index(drop=True)
    if result.empty:
        raise ValueError(f"Tidak ada gambar {split} yang dapat digunakan.")
    return result

class Documents(Dataset):
    """Baca PNG cache yang telah diperkecil agar TIFF tidak didekode setiap epoch."""

    def __init__(self, df, training=False, labeled=True):
        """Simpan tabel dan susun augmentasi ringan khusus data pelatihan."""
        self.df = df.reset_index(drop=True)
        self.labeled = labeled
        steps = []
        if training:
            steps.append(transforms.RandomAffine(degrees=2, translate=(0.015, 0.015), fill=255))
        steps.extend([
            transforms.ToTensor(),
            transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
        ])
        self.transform = transforms.Compose(steps)

    def __len__(self):
        """Kembalikan jumlah gambar dalam tabel."""
        return len(self.df)

    def __getitem__(self, index):
        """Baca satu gambar cache dan kembalikan tensor beserta label atau indeks."""
        row = self.df.iloc[index]
        with Image.open(row["cached_path"]) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, int(row["label"]) if self.labeled else index

def make_loader(df, device, training=False, labeled=True):
    """Buat pemuat data paralel dengan seed dan pinned memory ketika memakai CUDA."""
    return DataLoader(
        Documents(df, training, labeled), batch_size=BATCH_SIZE, shuffle=training,
        num_workers=NUM_WORKERS, persistent_workers=NUM_WORKERS > 0,
        pin_memory=device.type == "cuda", worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(SEED),
    )


class DitClassifier(nn.Module):
    """Bungkus DiT dengan kepala klasifikasi dan antarmuka pemisahan parameter."""

    def __init__(self, pretrained=True):
        """Muat backbone dokumen atau bangun arsitektur checkpoint tanpa mengunduh bobot."""
        super().__init__()
        if pretrained:
            self.network = BeitForImageClassification.from_pretrained(
                MODEL_NAME, num_labels=len(CLASS_NAMES),
                id2label=dict(enumerate(CLASS_NAMES)),
                label2id={name: i for i, name in enumerate(CLASS_NAMES)},
                ignore_mismatched_sizes=True,
            )
        else:
            if MODEL_CONFIG is None:
                raise ValueError("Konfigurasi arsitektur checkpoint belum tersedia.")
            self.network = BeitForImageClassification(BeitConfig.from_dict(MODEL_CONFIG))

    @property
    def features(self):
        """Kembalikan backbone BEiT yang digunakan oleh DiT."""
        return self.network.beit

    @property
    def classifier(self):
        """Kembalikan kepala klasifikasi yang dilatih untuk kategori tugas."""
        return self.network.classifier

    def forward(self, images):
        """Hasilkan logits 16 kelas dari tensor gambar yang sudah dinormalisasi."""
        return self.network(pixel_values=images).logits

def build_model(pretrained=True):
    """Buat model DiT; kegagalan unduhan tidak diganti dengan bobot acak."""
    return DitClassifier(pretrained=pretrained)

def autocast_context(device):
    """Gunakan FP16 autocast hanya pada perangkat CUDA."""
    return torch.autocast("cuda", dtype=torch.float16) if device.type == "cuda" else nullcontext()

def predict_probabilities(model, data_loader, device, labeled=True):
    """Hitung probabilitas dan loss validation tanpa menyimpan gradient."""
    model.eval()
    probabilities, targets, total_loss = [], [], 0.0
    with torch.inference_mode():
        for images, labels in tqdm(data_loader, desc="Validation" if labeled else "Prediksi test", leave=False):
            images = images.to(device, non_blocking=True)
            with autocast_context(device):
                logits = model(images)
            if labeled:
                target = labels.to(device, non_blocking=True)
                total_loss += nn.functional.cross_entropy(logits.float(), target, reduction="sum").item()
                targets.extend(labels.tolist())
            probabilities.append(logits.float().softmax(1).cpu().numpy())
    probs = np.concatenate(probabilities)
    return probs, np.asarray(targets), total_loss / len(targets) if labeled else None

def make_submission(test_ids, probabilities, sample, path):
    """Cocokkan ID dengan sample submission dan simpan hanya kolom id,label."""
    if probabilities.shape != (len(test_ids), 16) or not np.isfinite(probabilities).all():
        raise ValueError("Probabilitas test memiliki bentuk tidak sesuai atau nilai tidak valid.")
    predictions = pd.DataFrame({"id": list(test_ids), "label": probabilities.argmax(1)})
    submission = sample[["id"]].merge(predictions, on="id", how="left", sort=False, validate="one_to_one")
    if submission["label"].isna().any() or not submission["id"].equals(sample["id"]):
        raise ValueError("ID submission tidak lengkap atau urutannya berubah.")
    submission["label"] = submission["label"].astype(int)
    if not submission["label"].isin(range(16)).all():
        raise ValueError("Prediksi label berada di luar rentang 0-15.")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(path, index=False)
    return submission


def main():
    """Jalankan persiapan cache, fine-tuning, evaluasi, dan pembuatan submission."""
    global MODEL_CONFIG, NORMALIZE_MEAN, NORMALIZE_STD
    if IMAGE_SIZE != 224:
        raise ValueError("Versi ini memakai input native DiT 224 piksel. Jangan ubah ukuran tanpa adaptasi posisi.")
    if min(EPOCHS, BATCH_SIZE, IMAGE_SIZE, PATIENCE, PREPROCESS_WORKERS) < 1 or NUM_WORKERS < 0:
        raise ValueError("Pengaturan epoch, batch, ukuran, patience, dan worker tidak valid.")
    if not 0 <= HEAD_EPOCHS < EPOCHS or min(LEARNING_RATE, HEAD_LEARNING_RATE) <= 0:
        raise ValueError("Gunakan 0 <= HEAD_EPOCHS < EPOCHS dan learning rate positif.")
    output = Path(OUTPUT_DIR)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(CHECKPOINT_PATH)
    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
        torch.backends.cudnn.benchmark = True
    else:
        print("GPU tidak tersedia; training pada CPU dapat memerlukan waktu lama.")
    train, test, sample = load_tables()
    if PREDICT_ONLY:
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        MODEL_CONFIG = saved["hf_config"]
        NORMALIZE_MEAN = saved["normalize_mean"]
        NORMALIZE_STD = saved["normalize_std"]
    else:
        processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
        NORMALIZE_MEAN = list(processor.image_mean)
        NORMALIZE_STD = list(processor.image_std)
        processor.save_pretrained(output / "processor")
    print("Pretrained:", MODEL_NAME, "; ukuran:", IMAGE_SIZE)
    started = time.perf_counter()
    history = []
    if not PREDICT_ONLY:
        train = prepare_cache(train, "train", output)
        if set(train["label"]) != set(range(16)) or train["label"].value_counts().min() < 2:
            raise ValueError("Setiap kelas membutuhkan minimal dua gambar train yang dapat dibaca.")
        train_part, val_part = train_test_split(train, test_size=0.2, stratify=train["label"], random_state=SEED)
        split = train.copy()
        split["split"] = "train"
        split.loc[val_part.index, "split"] = "validation"
        split.to_csv(output / "split.csv", index=False)
        print(f"Train: {len(train_part):,}; validation: {len(val_part):,}. Tanpa pemeriksaan duplikat.")
        train_loader = make_loader(train_part, device, training=True)
        val_loader = make_loader(val_part, device)
        model = build_model(pretrained=True).to(device)
        optimizer = torch.optim.AdamW([
            {"params": model.features.parameters(), "lr": LEARNING_RATE},
            {"params": model.classifier.parameters(), "lr": HEAD_LEARNING_RATE},
        ], weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
        criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
        best_accuracy, stale = -1.0, 0
        print(f"Persiapan train: {(time.perf_counter() - started) / 60:.1f} menit")
        for epoch in range(1, EPOCHS + 1):
            tick = time.perf_counter()
            freeze = epoch <= HEAD_EPOCHS
            for parameter in model.features.parameters():
                parameter.requires_grad_(not freeze)
            model.train()
            if freeze:
                model.features.eval()
            total, seen = 0.0, 0
            for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}"):
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with autocast_context(device):
                    loss = criterion(model(images), labels)
                if not torch.isfinite(loss):
                    raise RuntimeError("Loss tidak valid; periksa data atau learning rate.")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                total += loss.item() * len(labels)
                seen += len(labels)
            probs, targets, val_loss = predict_probabilities(model, val_loader, device)
            predictions = probs.argmax(1)
            accuracy = accuracy_score(targets, predictions)
            macro_f1 = f1_score(targets, predictions, labels=list(range(16)), average="macro", zero_division=0)
            seconds = time.perf_counter() - tick
            history.append(dict(epoch=epoch, head_only=freeze, train_loss=total / seen,
                                val_loss=val_loss, accuracy=accuracy, macro_f1=macro_f1, epoch_seconds=seconds))
            pd.DataFrame(history).to_csv(output / "history.csv", index=False)
            print(f"Epoch {epoch}: {seconds / 60:.1f} menit; accuracy={accuracy:.4f}; macro-F1={macro_f1:.4f}")
            if accuracy > best_accuracy:
                best_accuracy, stale = accuracy, 0
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"model_state": model.state_dict(), "image_size": IMAGE_SIZE,
                            "classes": CLASS_NAMES, "epoch": epoch, "val_accuracy": accuracy,
                            "hf_config": model.network.config.to_dict(), "model_name": MODEL_NAME,
                            "normalize_mean": NORMALIZE_MEAN, "normalize_std": NORMALIZE_STD}, checkpoint_path)
            else:
                stale += 1
            scheduler.step()
            if not freeze and stale >= PATIENCE:
                print("Early stopping berdasarkan validation accuracy.")
                break
        del optimizer, scaler, train_loader
    else:
        model = build_model(pretrained=False).to(device)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint["classes"] != CLASS_NAMES or checkpoint["image_size"] != IMAGE_SIZE:
        raise ValueError("Ukuran gambar atau pemetaan kelas berbeda dari checkpoint.")
    model.load_state_dict(checkpoint["model_state"])
    if not PREDICT_ONLY:
        probs, targets, val_loss = predict_probabilities(model, val_loader, device)
        predictions = probs.argmax(1)
        report = classification_report(targets, predictions, labels=list(range(16)), target_names=CLASS_NAMES,
                                       output_dict=True, zero_division=0)
        pd.DataFrame(report).transpose().to_csv(output / "classification_report.csv")
        with open(output / "metrics.json", "w", encoding="utf-8") as file:
            json.dump({"best_epoch": checkpoint["epoch"], "accuracy": accuracy_score(targets, predictions),
                       "macro_f1": f1_score(targets, predictions, average="macro", zero_division=0),
                       "val_loss": val_loss}, file, indent=2)
        pd.DataFrame(confusion_matrix(targets, predictions, labels=list(range(16))),
                     index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(output / "confusion_matrix.csv")
        details = val_part[["id", "class_name", "label"]].copy()
        details["predicted_label"] = predictions
        details["confidence"] = probs.max(1)
        details.to_csv(output / "validation_predictions.csv", index=False)
        np.save(output / "validation_probabilities.npy", probs)
        print(f"Model terbaik: epoch {checkpoint['epoch']}; accuracy {checkpoint['val_accuracy']:.4f}")

    test = prepare_cache(test, "test", output)
    test_loader = make_loader(test, device, labeled=False)
    probs, _, _ = predict_probabilities(model, test_loader, device, labeled=False)
    submission = make_submission(test["id"], probs, sample, SUBMISSION_PATH)
    submission.to_csv(output / "submission.csv", index=False)
    probability_table = pd.DataFrame(probs, columns=[f"prob_{i}" for i in range(16)])
    probability_table.insert(0, "id", test["id"].to_numpy())
    probability_table.to_csv(output / "test_probabilities.csv", index=False)
    config_keys = ["DATA_DIR", "TRAIN_DIR", "TEST_DIR", "OUTPUT_DIR", "CHECKPOINT_PATH", "SUBMISSION_PATH",
                   "IMAGE_SIZE", "BATCH_SIZE", "EPOCHS", "HEAD_EPOCHS", "LEARNING_RATE", "HEAD_LEARNING_RATE",
                   "PATIENCE", "NUM_WORKERS", "PREPROCESS_WORKERS", "SEED", "PREDICT_ONLY", "MODEL_NAME",
                   "NORMALIZE_MEAN", "NORMALIZE_STD"]
    with open(output / "run_config.json", "w", encoding="utf-8") as file:
        json.dump({key: str(globals()[key]) if isinstance(globals()[key], Path) else globals()[key]
                   for key in config_keys}, file, indent=2)
    print("\n" + submission.head(10).to_csv(index=False))
    print(f"Selesai: {len(submission):,} prediksi; file: {SUBMISSION_PATH}")


if __name__ == "__main__":
    main()
