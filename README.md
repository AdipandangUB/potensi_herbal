# 🌿 WebGIS SDM — Tanaman Herbal TNBTS

Aplikasi Streamlit untuk memprediksi potensi tumbuh (habitat suitability)
tanaman herbal di kawasan Taman Nasional Bromo Tengger Semeru menggunakan
metode **Species Distribution Modelling (SDM)**.

## Struktur Folder
```
sdm_app/
├── app.py                 # aplikasi utama (jalankan file ini)
├── requirements.txt
├── README.md
└── data/                  # letakkan 8 file raster reklasifikasi di sini
    ├── Tinggi-Reclass.tif
    ├── Hujan-reclass.tif
    ├── LST-Reclass.tif
    ├── Lereng-Reclass.tif
    ├── Moisture-reclass.tif
    ├── NDVI-Reclass.tif
    ├── Pemukiman-Reclass.tif
    └── Soil-Reclass.tif
```

## Instalasi & Menjalankan Lokal
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy ke Streamlit Community Cloud
1. Buat repository GitHub berisi seluruh folder ini (termasuk folder `data/`
   dengan 8 file `.tif`, atau biarkan kosong dan gunakan fitur unggah manual
   di sidebar aplikasi).
2. Masuk ke https://streamlit.io/cloud → **New app** → hubungkan repo →
   pilih `app.py` sebagai entry point → **Deploy**.

## Dua Metode SDM yang Tersedia
1. **Fuzzy Environmental Envelope (MCE)** — cocok bila Anda ingin hasil
   instan berdasarkan rentang toleransi ekologis tiap tanaman (ketinggian,
   curah hujan, suhu) yang sudah terdatabase, dikombinasikan dengan bobot
   AHP untuk layer lain (kelerengan, kelembapan, NDVI, jarak pemukiman,
   jenis tanah).
2. **Statistik / Machine Learning (presence-background)** — cocok bila Anda
   punya/ingin memasukkan titik lokasi kemunculan aktual tanaman (GPS),
   menghasilkan peta probabilitas berbasis Random Forest / Regresi
   Logistik (mirip prinsip MaxEnt).

## Kalibrasi Penting
Buka `app.py`, cari dictionary `CLASS_MIDPOINT` (±baris 105). Nilai di
situ adalah **asumsi** nilai tengah tiap kelas hasil reklasifikasi Anda
(mis. kelas 1 Ketinggian = 350 mdpl). **Sesuaikan dengan breakpoint
reklasifikasi asli** yang Anda gunakan di ArcGIS/QGIS agar hasil SDM
akurat secara kuantitatif.

Database ekologi 22 spesies (`SPECIES_DB`) diringkas dari dokumen
"Tanaman Herbal di Kawasan TNBTS..." — silakan tambah/lengkapi entri
lain sesuai kebutuhan riset.
