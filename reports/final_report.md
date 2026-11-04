# Laporan model final DiT

## Tujuan dan data

Tugas ini mengklasifikasikan gambar dokumen menjadi 16 kategori menggunakan neural network PyTorch. Dataset memiliki 32.003 gambar train dan 7.994 gambar test. Pemetaan label tersedia pada `configs/classes.json`.

## Hasil eksplorasi data

Notebook EDA berasal dari run Kaggle pada `eda-textimage.ipynb`. Pemeriksaan CSV menunjukkan kolom, ID, label, dan sample submission sesuai. Tidak ditemukan nilai kosong atau ID dan filename yang berulang. Isi gambar duplikat belum diperiksa.

Jumlah sampel per kategori berkisar dari 1.948 pada `questionnaire` sampai 2.056 pada `scientific_publication`. Rasio terbesar terhadap terkecil sekitar 1,055. Distribusi kelas relatif serupa, sehingga pembagian data mempertahankan proporsi kelas.

Metadata 32.002 file train berhasil dibaca dan satu file gagal, yaitu `train_08795`. Seluruh 7.994 metadata test berhasil dibaca. Pada file yang berhasil diperiksa, tinggi gambar adalah 1.000 piksel, mode warna grayscale, dan jumlah halaman TIFF satu. Resolusi terbanyak pada train dan test adalah 754 x 1.000 piksel. Lebar train berkisar 606 sampai 2.542 piksel dan test 606 sampai 1.473 piksel.

EDA memakai `READ_PIXELS=False`. Keterbacaan tersebut berlaku untuk metadata dan belum menjamin seluruh piksel bebas kerusakan. Grafik EDA di folder figures berasal dari hasil ekspor notebook yang dikirim.

## Metode

Model yang dipilih adalah `microsoft/dit-base`, yang menjalani pretraining tanpa label pada gambar dokumen IIT-CDIP. Model ditambah kepala klasifikasi 16 kelas, lalu seluruh model dilatih kembali menggunakan data tugas. Kode memakai backbone awal, bukan checkpoint yang telah di-fine-tune pada label RVL-CDIP. [Keterangan model](https://huggingface.co/microsoft/dit-base).

Pipeline membaca halaman pertama TIFF, menyesuaikan orientasi EXIF, mengubahnya menjadi RGB, lalu me-resize seluruh halaman ke 224 x 224 tanpa crop. Proporsi halaman berubah menjadi persegi. Normalisasi mengikuti processor DiT. Gambar kecil disimpan ke cache agar TIFF tidak perlu didekode ulang setiap epoch. Augmentasi train memakai rotasi maksimal 2 derajat dan translasi kecil; validation dan test tidak memakai augmentasi acak.

Konfigurasi pada notebook terbaru menggunakan batch size 16, maksimal 10 epoch, AdamW, learning rate backbone 0,00002, serta learning rate kepala klasifikasi 0,0001. Cross-entropy memakai label smoothing 0,05 dan learning rate dijadwalkan dengan cosine annealing. Mixed precision aktif pada CUDA dan gradient clipping memakai nilai 1,0. Early stopping memakai patience 3 berdasarkan validation accuracy. Parameter yang terbaca dari kode dicatat pada `settings_from_notebook.json`.

Log menunjukkan satu file train dikeluarkan karena gagal dibaca. Dari 32.002 gambar tersisa, sebanyak 25.601 digunakan untuk train dan 6.401 untuk validation. Pembagian dilakukan secara stratifikasi dengan seed 42. Versi ini tidak melakukan pemeriksaan duplikasi piksel atau konflik label.

## Hasil run terbaru

Notebook training terbaru yang sudah menjalankan seluruh sel training dan pemeriksaan hasil. GPU yang tercetak adalah Tesla T4. Persiapan train tercetak sekitar 5,8 menit. Waktu training dan validation per epoch tercetak sekitar 5,2 menit. Angka waktu tersebut dibulatkan oleh program.

Validation accuracy naik dari 0,8288 pada epoch 1 menjadi 0,8924 pada epoch 9. Pada epoch 10 nilainya turun menjadi 0,8920. Program memilih checkpoint epoch 9 untuk memprediksi test.

Metrik lengkap model terbaik yang tercetak pada sel pemeriksaan hasil adalah:

| Metrik | Nilai |
| --- | --- |
| Epoch terbaik | 9 |
| Validation accuracy | 0,8923605686611467 |
| Validation macro-F1 | 0,891977735841067 |
| Validation loss | 0,4297408316004483 |
| Jumlah prediksi test | 7.994 |

Nilai lengkap tersebut disimpan pada `metrics.json` berdasarkan dictionary yang tercetak di notebook. Log per epoch pada `training_history_from_log.csv` tetap mengikuti pembulatan empat desimal yang dicetak program. File itu tidak memuat loss pada seluruh epoch.

Program menyelesaikan 7.994 prediksi dan menampilkan contoh submission dengan kolom `id,label`. Public score untuk submission run terbaru belum dilaporkan. Public score 0,89711 yang pernah disampaikan berasal dari run sebelumnya, sehingga tidak dipakai sebagai skor submission baru. Private score juga belum tersedia.

## Interpretasi

Validation accuracy sekitar 89,24% berarti sekitar 89 dari 100 sampel validation diprediksi dengan benar. Accuracy dan macro-F1 memiliki nilai agregat yang serupa. Hal ini belum menunjukkan bahwa semua kategori memiliki performa yang sama.

Penurunan accuracy dari epoch 9 ke epoch 10 menunjukkan bahwa memakai epoch terakhir tidak selalu menghasilkan skor validation tertinggi. Penyimpanan checkpoint berdasarkan validation accuracy membuat prediksi test menggunakan epoch 9. Penurunan kecil pada satu epoch belum cukup untuk menentukan penyebabnya.

Validation loss checkpoint terbaik adalah sekitar 0,42974. Notebook tidak mencetak seluruh train loss dan validation loss per epoch, sehingga kurva loss belum dibuat dari log yang tersedia.

Isi `classification_report.csv`, `confusion_matrix.csv`, dan `validation_predictions.csv` belum disertakan sebagai berkas terpisah. Kategori yang paling sering salah belum dapat ditentukan. Notebook evaluasi disediakan untuk membaca berkas tersebut dan membuat grafik saat hasil run ditambahkan.

## Reproduksi dan keterbatasan

Notebook yang sudah dirapikan telah dijalankan ulang oleh pemilik eksperimen di Kaggle. Repository memakai notebook terbaru tersebut dan mempertahankan kode serta outputnya. Penjelasan markdown diperbarui mengikuti hasil run baru. Function dan class pada script mandiri memiliki isi yang sama dengan kode notebook terbaru.

Sel instalasi memasang Transformers 4.57.1 dan safetensors. Output instalasi menampilkan perbedaan kebutuhan versi beberapa library lain pada lingkungan Kaggle, tetapi proses DiT tetap selesai sampai prediksi. Versi lengkap seluruh library belum dicatat. Python pada metadata notebook terbaru adalah 3.13.16. Seed tetap tidak menjamin hasil identik pada semua lingkungan GPU.

Pemeriksaan duplikat belum dilakukan. Gambar identik yang tersebar antar-split dapat membuat validation accuracy terlalu tinggi. Pretraining memakai IIT-CDIP; kemungkinan dokumen kompetisi berasal dari korpus yang sama belum diperiksa. Penggunaan halaman pertama dan resize ke 224 piksel juga dapat menghilangkan informasi pada halaman lain atau detail kecil.

Checkpoint, submission lengkap, loss per epoch, dan tabel evaluasi per kelas masih perlu diambil dari folder Output run Kaggle. Berkas tersebut tidak dapat direkonstruksi sepenuhnya dari contoh prediksi dan log yang tercetak.
