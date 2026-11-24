# Model Card — Segmentasi Nukleus dan Sitoplasma Sel Darah Putih (T4-A)

> **Prototipe riset, bukan alat diagnosis.** Tidak boleh dipakai untuk keputusan klinis atau penghitungan diferensial yang memengaruhi pasien.

**Model:** U-Net dan DeepLabV3+, encoder ResNet-34 berbobot pralatih ImageNet, 3 kelas (latar, sitoplasma, nukleus). Masukan RGB 256 × 256. Tim: [ISI NAMA TIM]. Versi: bobot final seed 42, 43, 44 untuk tiap arsitektur.

## Data latih

Subset segmentasi **Raabin-WBC** (Kouzehkanan dkk., _Sci. Rep._ 12:1123, 2022): potongan 575 × 575 berpusat pada satu sel, dengan masker nukleus dan sitoplasma. Dipakai 927 sel dari 54 apusan (limfosit 242, neutrofil 242, monosit 242, eosinofil 201). **Basofil (218 sel) dikeluarkan** karena format masker berbeda. Latih/validasi/uji = 728 / 91 / 108 sel, dibagi per apusan (seed 42), tanpa apusan yang sama di dua bagian. Menurut makalah sumber, data berasal dari darah tepi normal.

## Kinerja (set uji 108 sel, rerata ± SD antar 3 seed)

| Metrik                                    | U-Net                | DeepLabV3+           |
| ----------------------------------------- | -------------------- | -------------------- |
| Dice sitoplasma                           | 0,909 ± 0,003        | 0,909 ± 0,001        |
| Dice nukleus                              | 0,970 ± 0,001        | 0,970 ± 0,000        |
| HD95 sitoplasma / nukleus (piksel)        | 9,8 / 5,4            | 8,7 / 5,6            |
| Parameter; latensi T4 / CPU (ms, batch 1) | 24,4 juta; 7,4 / 241 | 22,4 juta; 6,7 / 214 |

| Dice per jenis sel (rerata 2 model) | Sitoplasma | Nukleus | n uji |
| ----------------------------------- | ---------- | ------- | ----- |
| Eosinofil                           | 0,934      | 0,959   | 13    |
| Limfosit                            | 0,851      | 0,977   | 31    |
| Monosit                             | 0,924      | 0,977   | 33    |
| Neutrofil                           | 0,940      | 0,961   | 31    |

Tidak ada perbedaan berarti antara kedua arsitektur. Antar-sesi dan eksternal: [ISI dari LOG_GABUNGAN.csv dan hasil Zheng bila sudah ada].

## Keterbatasan

- Sitoplasma limfosit paling sulit disegmentasi (Dice ±0,85, variasi antar-sel besar).
- Hanya diuji pada potongan satu sel; tidak diuji pada apusan lapang pandang penuh.
- Satu sumber data (satu laboratorium); label perangkat tidak tersedia sehingga pergeseran antar-perangkat tidak terukur. Pergeseran antar-sesi dan eksternal [ISI ringkasan].
- Dua apusan menyumbang sekitar 27% data; eosinofil hanya 13 sel di set uji; tiga seed.
- Anotasi mengandung noise kecil (lubang dan bintik pada masker nukleus).

## Kondisi di mana model tidak boleh dipakai

- Diagnosis, skrining, atau penghitungan diferensial untuk pasien.
- **Basofil**: tidak dilatih dan tidak divalidasi.
- Apusan dengan pewarnaan, kamera, mikroskop, atau perbesaran yang berbeda dari Raabin-WBC tanpa validasi ulang; sampel patologis (mis. sel blas leukemia) dan sel di luar empat jenis yang dilatih.
- Citra lapang pandang penuh, sel bertumpuk, atau lebih dari satu sel utama per citra.
- Menghitung rasio nukleus/sitoplasma tanpa pengecekan manual, karena kesalahan sistematis pada limfosit akan memengaruhinya.

## Etika dan data

Dataset dipakai sesuai lisensinya dan tidak didistribusikan ulang. Penggunaan AI generatif dinyatakan di laporan.
