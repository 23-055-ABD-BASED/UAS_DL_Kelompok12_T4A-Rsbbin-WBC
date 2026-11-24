# UAS_DL_Kelompok12_T4A-Rsbbin-WBC

# Segmentasi Nukleus dan Sitoplasma Sel Darah Putih pada Raabin-WBC (T4-A)

Tugas besar pengganti UAS, mata kuliah Deep Learning for Image – Computer Vision Lanjutan.
Tim: [ISI NAMA TIM] — [ISI NAMA ANGGOTA DAN NIM]

> **Prototipe riset, bukan alat diagnosis.** Model dan hasil di repositori ini tidak boleh dipakai untuk keputusan klinis.

## Ringkasan

Dua arsitektur segmentasi semantik berbobot pralatih ImageNet, **U-Net** dan **DeepLabV3+** (encoder ResNet-34), dibandingkan pada subset segmentasi Raabin-WBC (nukleus, sitoplasma, latar).

Audit data menunjukkan subset ini tidak memuat label kamera atau mikroskop yang dapat diverifikasi, sehingga skema *leave-one-device-out* murni tidak dapat dilakukan. Sebagai gantinya kami menguji empat skenario:

| Skenario | Latih → uji | Tujuan |
|---|---|---|
| Apusan tak terlihat (`grouped`) | Split per apusan | Skenario utama, bebas kebocoran antar-apusan |
| Split bocor vs bersih (`random` vs `grouped`) | Split per sel | Mengukur inflasi skor akibat kebocoran |
| Antar-sesi (`cross_session`) | Batch 95-8 → apusan 95-8 lain dan batch 95-5/95-7/95-9 | Pergeseran antar-sesi (bukan antar-perangkat) |
| Eksternal | Raabin-WBC → Zheng et al. | [BELUM DIKERJAKAN — isi ringkasan hasil bila sudah ada] |

## Struktur repositori

```
code/        data.py, runlog.py, metrics.py, models.py, train.py, analysis.py
notebooks/   01_data_audit_raabin_wbc, 02_train_unet_deeplab, 03_eval_analysis, 04_error_gallery_gradcam
splits/      daftar berkas tiap subset (seed 42), semua skema
logs/        log_<anggota>.csv dan LOG_GABUNGAN.csv (semua run, termasuk yang dihentikan dini)
results/     tabel dan gambar hasil (CSV dan PNG)
docs/        MODEL_CARD.md
```

## Data

| | |
|---|---|
| Sumber | Raabin-WBC, subset dengan *ground truth* nukleus dan sitoplasma (1.145 sel). Kouzehkanan dkk., *Scientific Reports* 12:1123 (2022) |
| Dipakai | 927 sel dari 54 apusan: limfosit 242, neutrofil 242, monosit 242, eosinofil 201 |
| Dikeluarkan | 218 sel basofil, karena masker basofil hanya bernilai {0, 255} (seluruh sel tanpa pemisahan nukleus), sedangkan jenis sel lain {0, 100, 255}. Keputusan diambil dari audit data sebelum pelatihan |
| Kode masker | 0 = latar, 100 = nukleus, 255 = sitoplasma (diverifikasi visual) |
| Ukuran citra | 575 × 575 piksel, satu sel utama per citra |
| Eksternal | Zheng dkk., *Micron* 107 (2018), Mendeley Data, doi: 10.17632/w7cvnmn4c5.1 |

**Dataset tidak disertakan di repositori.** Unduh manual lewat browser dari raabindata.com (bagian *Nucleus and Cytoplasm Ground Truths*, berkas `GroundTruths_bmpformat.rar`), karena server Raabin tidak dapat dijangkau dari Colab. Kami juga mencoba LISC (Rezatofighi & Soltanian-Zadeh, 2011) tetapi tautan unduhannya tidak berfungsi, sehingga diganti Zheng atas persetujuan dosen.

### Pembagian data (seed 42, sel dari satu apusan tidak dipisah)

| Skema | Latih / validasi / uji | Kebocoran latih→uji |
|---|---|---|
| `grouped` | 728 / 91 / 108 | 0 apusan, 0 citra asal |
| `random` | 728 / 91 / 108 | 37 apusan dan 5 citra asal bersama |
| `cross_session` | 509 / 147 / uji dalam-sesi 69 / uji lintas-sesi 202 | 0 apusan, 0 citra asal |

*Perceptual hash* tidak menemukan duplikat pada skema mana pun (jarak minimum 12), termasuk pada `random` yang jelas bocor per apusan, karena tiap potongan berpusat pada sel yang berbeda. Pemeriksaan per apusan dan per citra asal adalah yang informatif.

## Cara mereproduksi

1. Unduh dataset (lihat atas), unggah `GroundTruths_bmpformat.rar` ke Google Drive pada folder `PROYEK UAS DEEP LEARNING/`, dan taruh folder `code/` dan `splits/` dari repositori ini di folder yang sama. Kode memakai jalur `/content/drive/MyDrive/PROYEK UAS DEEP LEARNING`.
2. Colab dengan GPU T4. Pasang paket dari `requirements.txt` dan `apt-get install unrar`.
3. Jalankan notebook berurutan:
   - `01` audit data, split, pemeriksaan kebocoran, EDA (hasil sudah ada di `splits/` dan `results/`)
   - `02` pelatihan: baseline, ablasi, perbandingan utama 3 seed (`run_experiment` mencatat tiap run ke `logs/`)
   - `03` tabel hasil, per jenis sel, efisiensi, uji berpasangan, tabel ablasi
   - `04` galeri kesalahan dan Grad-CAM
4. Bobot final: [ISI TAUTAN UNDUH BOBOT — unet_final_seed{42,43,44}.pt dan deeplabv3plus_final_seed{42,43,44}.pt]. Bobot tidak disimpan di repositori karena ukurannya.

Angka utama di laporan dapat ditelusuri ke `logs/LOG_GABUNGAN.csv`. Pelatihan memakai `cudnn.benchmark = True`, sehingga angka dapat berbeda tipis antar-run walau seed sama.

## Konfigurasi pelatihan

U-Net dan DeepLabV3+ (`segmentation_models_pytorch` 0.6.0), encoder ResNet-34 ImageNet. Masukan 256 × 256. Loss CE + Dice (Dice pada nukleus dan sitoplasma). AdamW, lr 1e-3, weight decay 1e-4, jadwal cosine, batch 16, presisi campuran, maksimal 20 epoch, *early stopping* patience 5 pada konfigurasi final (patience 20, yaitu tanpa early stopping, pada ablasi yang diulang). Augmentasi: flip horizontal/vertikal, rotasi 90°, *color jitter*. Epoch terbaik dipilih dari rerata Dice nukleus dan sitoplasma pada **validasi**. Set uji hanya dipakai pada run final.

## Hasil utama (set uji `grouped`, 108 sel, rerata ± SD antar 3 seed)

| Metrik | U-Net | DeepLabV3+ |
|---|---|---|
| Dice sitoplasma | 0,9089 ± 0,0030 | 0,9088 ± 0,0010 |
| Dice nukleus | 0,9696 ± 0,0008 | 0,9699 ± 0,0002 |
| mIoU (3 kelas) | 0,9225 ± 0,0016 | 0,9226 ± 0,0006 |
| HD95 sitoplasma (piksel) | 9,80 ± 1,81 | 8,68 ± 0,27 |
| HD95 nukleus (piksel) | 5,38 ± 1,37 | 5,64 ± 0,56 |
| Parameter (juta) | 24,44 | 22,44 |
| Latensi GPU T4 / CPU, batch 1, 256 px (ms) | 7,37 / 241,27 | 6,74 / 213,53 |

Tidak ditemukan perbedaan yang berarti antara kedua arsitektur (selisih Dice sitoplasma +0,0001, CI 95% bootstrap [−0,0034, +0,0044]).

Dice per jenis sel (rerata 3 seed): sitoplasma limfosit paling sulit (±0,85), jenis lain 0,92–0,94; Dice nukleus 0,96–0,98. Jumlah sel uji: eosinofil 13, limfosit 31, monosit 33, neutrofil 31.

### Ablasi (U-Net, validasi, 1 seed = 42)

| Faktor | Nilai | Dice sitoplasma | Dice nukleus | Catatan |
|---|---|---|---|---|
| Baseline | CE+Dice, warna aktif, 256 px, pralatih | 0,8963 | 0,9724 | |
| Loss | CE (patience 20) | 0,8953 | 0,9719 | |
| Loss | Dice | 0,8948 | 0,9716 | dihentikan dini; [ISI HASIL dice_p20] |
| Loss | Focal+Dice | 0,8963 | 0,9710 | |
| Augmentasi warna | nonaktif (patience 20) | 0,8963 | 0,9738 | |
| Resolusi | 384 | 0,8976 | 0,9725 | |
| Encoder | inisialisasi acak | 0,8790 | 0,9710 | |
| Skema split | `random` | 0,9183 | 0,9699 | validasi bocor; bukan perbandingan di set yang sama |

Run CE dan warna nonaktif yang pertama berhenti dini sebelum jadwal laju belajar selesai sehingga tidak sebanding; keduanya diulang tanpa early stopping dan tetap tercatat di log. Pada satu seed, selisih antar-konfigurasi (≤ 0,002) jauh di bawah SD Dice per citra, sehingga tidak ada bukti perbedaan, kecuali inisialisasi acak yang lebih rendah.

### Skenario antar-sesi dan eksternal

[ISI dari kolom test_insession_* dan test_* di LOG_GABUNGAN.csv, dan hasil Zheng bila sudah ada.]

## Keterbatasan

- Subset berisi satu sel utama per citra (potongan 575 × 575). Kinerja pada apusan lapang pandang penuh tidak diuji.
- Basofil dikeluarkan; model tidak berlaku untuk basofil. Asal perangkat akuisisi citra basofil tidak dapat diverifikasi.
- Label perangkat per citra tidak tersedia; skenario antar-perangkat yang murni tidak dapat dilakukan.
- Dua apusan menyumbang ±27% data (139 dan 110 sel). Eosinofil hanya 13 sel di set uji.
- Tiga seed saja. Beberapa seed pada run final dihentikan lebih awal oleh early stopping (lihat `LOG_GABUNGAN.csv`), sehingga lama pelatihan tidak seragam. Konfigurasi tidak diubah setelah set uji dipakai.
- Kategori kesalahan pada notebook 04 dihitung dari aturan (jenis kesalahan piksel terbanyak), bukan label manusia. Grad-CAM adalah implementasi sendiri (Seg-Grad-CAM pada keluaran terakhir encoder).

## Penggunaan AI generatif

[SALIN paragraf pernyataan penggunaan AI dari laporan, setelah dicocokkan dengan penggunaan sebenarnya seluruh anggota.]

## Sitasi dan lisensi data

Gunakan dataset sesuai lisensi masing-masing dan sitasi makalah sumber. Dataset tidak didistribusikan ulang di repositori ini. [ISI LISENSI KODE bila diminta dosen.]
