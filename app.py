# -*- coding: utf-8 -*-
"""
================================================================================
 WebGIS Interaktif — Species Distribution Modelling (SDM)
 Potensi Tumbuh Tanaman Herbal di Kawasan Taman Nasional Bromo Tengger Semeru
================================================================================
Dibuat untuk: Dr. Adipandang Yudono — Dept. PWK, Universitas Brawijaya

DESKRIPSI
---------
Aplikasi ini memprediksi peluang/kesesuaian tumbuh (habitat suitability) tiap
jenis tanaman herbal di kawasan TNBTS menggunakan 8 layer raster hasil
reklasifikasi (Curah Hujan, LST, Kelerengan, Kelembapan Tanah, NDVI, Jarak
Pemukiman, Jenis Tanah, dan Ketinggian) yang sudah Anda siapkan.

METODE SPECIES DISTRIBUTION MODELLING (SDM) YANG DIIMPLEMENTASIKAN
--------------------------------------------------------------------
1) MODE ENVIRONMENTAL ENVELOPE / FUZZY-MCE (mirip prinsip algoritma BIOCLIM)
   - Untuk layer yang punya data toleransi kuantitatif per spesies
     (Ketinggian, Curah Hujan, Suhu/LST), nilai kelas raster dipetakan ke
     nilai representatif dunia-nyata (lihat CLASS_MIDPOINT), lalu dihitung
     derajat keanggotaan fuzzy (trapezoidal) terhadap rentang toleransi
     ekologis spesies (dari database Syarat Hidup pada dokumen sumber).
   - Untuk layer yang tidak punya data spesifik-spesies (Kelerengan,
     Kelembapan, NDVI, Jarak Pemukiman, Jenis Tanah), digunakan skor arah
     preferensi umum (naik/turun/tengah) yang bisa diatur pengguna, sesuai
     prinsip Multi-Criteria Evaluation (MCE) / AHP yang lazim pada kajian
     kesesuaian lahan.
   - Seluruh skor digabung sebagai rata-rata terbobot (weighted overlay)
     menjadi Indeks Kesesuaian (Suitability Index/SI) 0-1 (0-100%).

2) MODE STATISTIK / MACHINE LEARNING (presence-background SDM)
   - Menyerupai pendekatan MaxEnt/berbasis kehadiran (presence-only SDM).
   - Pengguna mengunggah/menandai titik kemunculan spesies (presence),
     lalu sistem menghasilkan titik latar (pseudo-absence/background)
     secara acak pada kawasan kajian.
   - 8 layer raster dipakai sebagai variabel prediktor lingkungan untuk
     melatih model klasifikasi (Random Forest atau Regresi Logistik),
     kemudian diprediksi probabilitas kesesuaian pada seluruh piksel
     kawasan -> peta probabilitas sebaran potensial.

CATATAN PENTING TENTANG BREAKPOINT KELAS RASTER
------------------------------------------------
Nilai pada kamus CLASS_MIDPOINT (baris ~90) adalah ASUMSI nilai tengah
tiap kelas hasil reklasifikasi (mis. kelas 1 Ketinggian = 350 mdpl, dst).
KARENA breakpoint asli reklasifikasi Anda di ArcGIS/QGIS tidak tercantum
pada metadata raster, SILAKAN SESUAIKAN nilai-nilai ini dengan breakpoint
reklasifikasi yang sebenarnya Anda gunakan, agar hasil SDM akurat.

CARA MENJALANKAN
-----------------
1) pip install -r requirements.txt
2) Letakkan 8 file raster (*.tif) pada folder ./data (nama file harus
   mengandung kata kunci: Hujan, LST, Lereng, Moisture, NDVI, Pemukiman,
   Soil, Tinggi) — atau unggah langsung lewat sidebar aplikasi.
3) streamlit run app.py
4) Untuk deploy: push folder ini + data/*.tif ke GitHub, lalu deploy di
   Streamlit Community Cloud (streamlit.io/cloud).
================================================================================
"""

import base64
import io
import os
import tempfile

import folium
import numpy as np
import pandas as pd
import rasterio
import streamlit as st
from folium.plugins import Draw, MousePosition
from rasterio.transform import array_bounds
from rasterio.warp import Resampling, calculate_default_transform, reproject
from streamlit_folium import st_folium

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False

import matplotlib.cm as cm
import matplotlib.colors as mcolors
from PIL import Image


# ==============================================================================
# 1. KONFIGURASI DATA & DOMAIN KNOWLEDGE
# ==============================================================================

st.set_page_config(
    page_title="SDM Tanaman Herbal TNBTS",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# Nama file raster yang diharapkan (keyword pencocokan nama file, tidak case-sensitive)
RASTER_KEYWORDS = {
    "Tinggi": "tinggi",
    "Hujan": "hujan",
    "LST": "lst",
    "Lereng": "lereng",
    "Moisture": "moisture",
    "NDVI": "ndvi",
    "Pemukiman": "pemukiman",
    "Soil": "soil",
}

LAYER_LABELS = {
    "Tinggi": "Ketinggian (mdpl)",
    "Hujan": "Curah Hujan (mm/th)",
    "LST": "Suhu Permukaan / LST (°C)",
    "Lereng": "Kelerengan (%)",
    "Moisture": "Kelembapan Tanah",
    "NDVI": "Kerapatan Vegetasi (NDVI)",
    "Pemukiman": "Jarak dari Pemukiman",
    "Soil": "Jenis / Kesuburan Tanah",
}

# --- Nilai representatif (midpoint) tiap kelas hasil reklasifikasi. ---------
# !! SESUAIKAN dengan breakpoint reklasifikasi asli Anda (ArcGIS/QGIS) !!
CLASS_MIDPOINT = {
    "Tinggi": {1: 350, 2: 1100, 3: 2000, 4: 3000},                 # mdpl
    "Hujan":  {1: 1000, 2: 2100, 3: 3200},                          # mm/tahun
    "LST":    {1: 34, 2: 30, 3: 26, 4: 22, 5: 17},                  # °C (kelas1=terpanas)
}

# Arah preferensi default untuk layer tanpa data toleransi spesifik-spesies.
# "naik"  = kelas makin tinggi makin sesuai
# "turun" = kelas makin rendah makin sesuai
# "tengah"= kelas menengah paling sesuai
DEFAULT_DIRECTION = {
    "Lereng": "turun",       # lahan landai lebih sesuai utk tumbuh & panen
    "Moisture": "naik",      # kelembapan lebih tinggi umumnya lebih sesuai
    "NDVI": "naik",          # tutupan vegetasi rapat -> humus/naungan baik
    "Pemukiman": "naik",     # makin jauh dari pemukiman -> habitat lebih alami
    "Soil": "naik",          # asumsi kelas lebih tinggi = kesuburan lebih baik
}

# Bobot default AHP (silakan ubah lewat slider di sidebar). Total tak harus 100,
# akan dinormalisasi otomatis.
DEFAULT_WEIGHTS = {
    "Tinggi": 20, "Hujan": 18, "LST": 15, "Lereng": 10,
    "Moisture": 10, "NDVI": 10, "Pemukiman": 7, "Soil": 10,
}


# --- Database spesies tanaman herbal (diringkas dari dokumen sumber) --------
# elev = (min, max) mdpl | hujan = (min, max) mm/th | suhu = (min, max) °C
# ph = (min, max) — disimpan untuk info, belum ada layer pH langsung
SPECIES_DB = [
    {"nama": "Ajeran / Ketul", "latin": "Bidens pilosa", "elev": (0, 3600), "hujan": (500, 3600), "suhu": (15, 45), "ph": (5.0, 6.5),
     "fungsi": "Obat luka, antiradang, antibakteri, antidiare, penurun demam"},
    {"nama": "Adas", "latin": "Foeniculum vulgare", "elev": (1600, 2400), "hujan": (0, 2500), "suhu": (15, 20), "ph": (5.3, 7.8),
     "fungsi": "Pencernaan, batuk, kesehatan jantung, gejala menopause"},
    {"nama": "Alang-alang", "latin": "Imperata cylindrica", "elev": (0, 2000), "hujan": (500, 3500), "suhu": (20, 40), "ph": (4.0, 7.5),
     "fungsi": "Batu ginjal, infeksi ginjal, hepatitis, kencing tidak lancar"},
    {"nama": "Andong", "latin": "Cordyline fruticosa", "elev": (0, 1900), "hujan": (300, 2500), "suhu": (15, 35), "ph": (5.5, 6.5),
     "fungsi": "Hentikan pendarahan, obat luka, diare, gangguan haid"},
    {"nama": "Awar-awar", "latin": "Ficus septica", "elev": (0, 1800), "hujan": (300, 2500), "suhu": (28, 30), "ph": (6.0, 7.0),
     "fungsi": "Antiradang, bisul, antibakteri, penurun demam"},
    {"nama": "Bakung", "latin": "Crinum asiaticum", "elev": (0, 700), "hujan": (2000, 3000), "suhu": (20, 34), "ph": (6.5, 7.5),
     "fungsi": "Bengkak, keseleo, nyeri sendi & rematik, sakit kepala"},
    {"nama": "Klandingan", "latin": "Leucas lavandulifolia", "elev": (0, 1500), "hujan": (1000, 2000), "suhu": (20, 35), "ph": (6.0, 7.0),
     "fungsi": "Insomnia, sakit kepala, antiinflamasi, antimikroba"},
    {"nama": "Jarak Hitam", "latin": "Euphorbiaceae sp.", "elev": (0, 1700), "hujan": (300, 1200), "suhu": (20, 30), "ph": (5.0, 6.5),
     "fungsi": "Radang telinga, demam, sembelit, penyakit kulit"},
    {"nama": "Jarak", "latin": "Jatropha curcas", "elev": (300, 800), "hujan": (300, 1500), "suhu": (20, 30), "ph": (5.0, 6.5),
     "fungsi": "Sakit gigi, sariawan, penyembuhan luka, nyeri sendi"},
    {"nama": "Buah Delima", "latin": "Punica granatum", "elev": (0, 1000), "hujan": (800, 1200), "suhu": (25, 30), "ph": (6.5, 7.5),
     "fungsi": "Kesehatan jantung, antikanker, antiradang, pencernaan"},
    {"nama": "Labu Siam Hitam", "latin": "Sicyos edulis", "elev": (0, 1000), "hujan": (800, 1200), "suhu": (25, 30), "ph": (6.5, 7.5),
     "fungsi": "Tekanan darah, gula darah, kesehatan janin, ASI"},
    {"nama": "Pepaya Gunung / Carica", "latin": "Vasconcellea pubescens", "elev": (1500, 3000), "hujan": (800, 1200), "suhu": (10, 20), "ph": (5.5, 6.5),
     "fungsi": "Cacingan, pencernaan, antioksidan, osteoporosis"},
    {"nama": "Bit Merah", "latin": "Beta vulgaris", "elev": (1000, 1200), "hujan": (500, 550), "suhu": (15, 25), "ph": (6.0, 7.0),
     "fungsi": "Tekanan darah, antikanker, anemia, kerja saraf & otot"},
    {"nama": "Daun Otot", "latin": "Stellaria saxatilis", "elev": (0, 2200), "hujan": (300, 1200), "suhu": (20, 30), "ph": (5.0, 6.5),
     "fungsi": "Pegal linu, nyeri otot, peradangan"},
    {"nama": "Ciplukan", "latin": "Physalis minima", "elev": (700, 2300), "hujan": (1500, 2300), "suhu": (18, 35), "ph": (4.5, 8.2),
     "fungsi": "Demam, masuk angin, batuk-pilek, antiradang"},
    {"nama": "Pegagan / Calingan", "latin": "Centella asiatica", "elev": (0, 2500), "hujan": (1500, 2500), "suhu": (20, 30), "ph": (5.5, 6.5),
     "fungsi": "Penyembuhan luka, kesehatan kulit, antiinflamasi"},
    {"nama": "Paku Rane", "latin": "Selaginella sp.", "elev": (1500, 2356), "hujan": (1500, 3000), "suhu": (15, 28), "ph": (3.9, 7.0),
     "fungsi": "Jantung, stroke, obat luka, antikanker, antimikroba"},
    {"nama": "Parijoto", "latin": "Medinilla speciosa", "elev": (500, 2300), "hujan": (1500, 2300), "suhu": (18, 25), "ph": (5.5, 6.5),
     "fungsi": "Kesuburan, kesehatan ibu hamil, sariawan, kolesterol"},
    {"nama": "Sawi Ireng", "latin": "Brassica juncea", "elev": (5, 1200), "hujan": (1000, 1500), "suhu": (10, 25), "ph": (6.0, 6.8),
     "fungsi": "Pencernaan, kolesterol, jantung, antibakteri"},
    {"nama": "Jamur Lingzhi", "latin": "Ganoderma lucidum", "elev": (400, 600), "hujan": (2000, 2500), "suhu": (25, 30), "ph": (5.5, 6.5),
     "fungsi": "Insomnia, hipertensi, imunitas, antitumor"},
    {"nama": "Tebu Ireng", "latin": "Saccharum officinarum", "elev": (0, 1000), "hujan": (200, 2500), "suhu": (24, 30), "ph": (6.0, 6.5),
     "fungsi": "Gula darah, pencernaan, antibakteri, antikanker"},
    {"nama": "Tempuyung / Ketiuw", "latin": "Sonchus arvensis", "elev": (50, 1600), "hujan": (800, 3000), "suhu": (25, 32), "ph": (5.5, 7.0),
     "fungsi": "Batu ginjal, asam urat, tekanan darah, radang"},
]
SPECIES_NAMES = [s["nama"] for s in SPECIES_DB]


# ==============================================================================
# 2. UTILITAS RASTER
# ==============================================================================

def find_raster_path(layer_key: str, uploaded_files: dict) -> str:
    """Cari path raster: prioritas file yang diunggah pengguna, lalu folder data/."""
    if uploaded_files.get(layer_key) is not None:
        return uploaded_files[layer_key]
    if os.path.isdir(DATA_DIR):
        kw = RASTER_KEYWORDS[layer_key]
        for fn in os.listdir(DATA_DIR):
            if kw in fn.lower() and fn.lower().endswith((".tif", ".tiff")):
                return os.path.join(DATA_DIR, fn)
    return None


@st.cache_data(show_spinner=False)
def get_reference_grid(ref_path: str, downsample: int):
    """Definisikan grid referensi (transform, crs, width, height) hasil downsample."""
    with rasterio.open(ref_path) as src:
        transform = src.transform
        crs = src.crs
        width = max(1, src.width // downsample)
        height = max(1, src.height // downsample)
        new_transform = transform * transform.scale(src.width / width, src.height / height)
    return new_transform, crs.to_wkt(), width, height


@st.cache_data(show_spinner=False)
def load_raster_aligned(path: str, ref_transform, ref_crs_wkt: str, width: int, height: int):
    """Baca & reproject satu raster kategori ke grid referensi (nearest neighbor)."""
    ref_crs = rasterio.crs.CRS.from_wkt(ref_crs_wkt)
    with rasterio.open(path) as src:
        dst = np.full((height, width), np.nan, dtype=np.float32)
        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=ref_transform,
            dst_crs=ref_crs,
            src_nodata=src.nodata,
            dst_nodata=np.nan,
            resampling=Resampling.nearest,
        )
    return dst


@st.cache_data(show_spinner=False)
def load_all_rasters(paths_tuple, downsample):
    """paths_tuple: tuple of (layer_key, path) sorted. Mengembalikan dict layer->array + grid info."""
    paths = dict(paths_tuple)
    # pilih raster referensi = layer dengan ukuran piksel paling umum (Hujan / Lereng dll)
    ref_key = "Hujan" if "Hujan" in paths else list(paths.keys())[0]
    ref_transform, ref_crs_wkt, width, height = get_reference_grid(paths[ref_key], downsample)
    arrays = {}
    for key, path in paths.items():
        arrays[key] = load_raster_aligned(path, ref_transform, ref_crs_wkt, width, height)
    return arrays, ref_transform, ref_crs_wkt, width, height


def reproject_array_to_wgs84(arr, transform, crs_wkt):
    """Reproject array 2D (grid referensi UTM) ke EPSG:4326 untuk ditampilkan di folium."""
    src_crs = rasterio.crs.CRS.from_wkt(crs_wkt)
    dst_crs = "EPSG:4326"
    h, w = arr.shape
    left, bottom, right, top = array_bounds(h, w, transform)
    dst_transform, dst_w, dst_h = calculate_default_transform(
        src_crs, dst_crs, w, h, left, bottom, right, top
    )
    dst = np.full((dst_h, dst_w), np.nan, dtype=np.float32)
    reproject(
        source=arr,
        destination=dst,
        src_transform=transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        src_nodata=np.nan,
        dst_nodata=np.nan,
        resampling=Resampling.bilinear,
    )
    bounds = array_bounds(dst_h, dst_w, dst_transform)  # left, bottom, right, top
    return dst, bounds, dst_transform


def get_colormap(cmap_name):
    """Ambil colormap matplotlib secara kompatibel lintas versi.
    matplotlib >= 3.9 menghapus cm.get_cmap(); gunakan matplotlib.colormaps[...]."""
    try:
        return cm.colormaps[cmap_name]  # matplotlib >= 3.6
    except (AttributeError, KeyError, TypeError):
        try:
            return cm.get_cmap(cmap_name)  # matplotlib lama
        except AttributeError:
            import matplotlib.pyplot as plt
            return plt.get_cmap(cmap_name)


def array_to_png_overlay(arr, cmap_name="RdYlGn", vmin=0.0, vmax=1.0):
    """Ubah array float (0-1 atau lainnya) jadi PNG RGBA base64 (nan = transparan)."""
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    cmap = get_colormap(cmap_name)
    rgba = cmap(norm(np.nan_to_num(arr, nan=vmin)))
    rgba[..., 3] = np.where(np.isnan(arr), 0.0, 0.85)  # alpha
    rgba_uint8 = (rgba * 255).astype(np.uint8)
    img = Image.fromarray(rgba_uint8, mode="RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


# ==============================================================================
# 3. FUNGSI SDM — MODE 1: FUZZY ENVIRONMENTAL ENVELOPE / MCE
# ==============================================================================

def species_fuzzy_score(class_arr, class_midpoint_map, species_min, species_max, buffer_frac=0.3):
    """Skor keanggotaan fuzzy trapezoidal: 1 di dalam rentang toleransi spesies,
    menurun linear di zona penyangga di luar rentang, 0 di luar zona penyangga."""
    val_arr = np.full(class_arr.shape, np.nan, dtype=np.float32)
    for c, v in class_midpoint_map.items():
        val_arr[class_arr == c] = v

    width = max(species_max - species_min, 1e-6)
    buffer = max(width * buffer_frac, 1e-6)
    lower_edge, upper_edge = species_min - buffer, species_max + buffer

    score = np.zeros_like(val_arr, dtype=np.float32)
    inside = (val_arr >= species_min) & (val_arr <= species_max)
    score[inside] = 1.0
    left = (val_arr >= lower_edge) & (val_arr < species_min)
    score[left] = (val_arr[left] - lower_edge) / buffer
    right = (val_arr > species_max) & (val_arr <= upper_edge)
    score[right] = (upper_edge - val_arr[right]) / buffer
    score[np.isnan(val_arr)] = np.nan
    return np.clip(score, 0, 1)


def generic_direction_score(class_arr, direction):
    """Skor 0-1 berdasar arah preferensi umum terhadap nilai kelas raster."""
    valid = ~np.isnan(class_arr)
    if valid.sum() == 0:
        return np.full_like(class_arr, np.nan)
    cmin, cmax = np.nanmin(class_arr), np.nanmax(class_arr)
    rng = max(cmax - cmin, 1e-6)
    norm = (class_arr - cmin) / rng
    if direction == "naik":
        score = norm
    elif direction == "turun":
        score = 1 - norm
    else:  # "tengah"
        score = 1 - np.abs(norm - 0.5) * 2
    return np.clip(score, 0, 1)


def compute_suitability_fuzzy(rasters: dict, species: dict, weights: dict, directions: dict):
    """Hitung Indeks Kesesuaian (SI) 0-1 dengan weighted overlay fuzzy-MCE."""
    scores = {}
    scores["Tinggi"] = species_fuzzy_score(rasters["Tinggi"], CLASS_MIDPOINT["Tinggi"], *species["elev"])
    scores["Hujan"] = species_fuzzy_score(rasters["Hujan"], CLASS_MIDPOINT["Hujan"], *species["hujan"])
    scores["LST"] = species_fuzzy_score(rasters["LST"], CLASS_MIDPOINT["LST"], *species["suhu"])
    for layer in ["Lereng", "Moisture", "NDVI", "Pemukiman", "Soil"]:
        scores[layer] = generic_direction_score(rasters[layer], directions[layer])

    total_w = sum(weights.values()) or 1.0
    shape = scores["Tinggi"].shape
    si = np.zeros(shape, dtype=np.float32)
    wsum = np.zeros(shape, dtype=np.float32)
    for layer, sc in scores.items():
        w = weights[layer] / total_w
        valid = ~np.isnan(sc)
        si[valid] += sc[valid] * w
        wsum[valid] += w
    si = np.where(wsum > 0, si / np.where(wsum == 0, 1, wsum), np.nan)
    return si, scores


# ==============================================================================
# 4. FUNGSI SDM — MODE 2: STATISTIK / MACHINE LEARNING (presence-background)
# ==============================================================================

def sample_rasters_at_points(rasters: dict, rows, cols):
    """Ambil nilai kelas 8 layer pada indeks baris/kolom tertentu -> matrix (n, 8)."""
    layer_order = list(rasters.keys())
    n = len(rows)
    X = np.zeros((n, len(layer_order)), dtype=np.float32)
    for j, layer in enumerate(layer_order):
        X[:, j] = rasters[layer][rows, cols]
    return X, layer_order


def latlon_to_rowcol(lat, lon, transform, crs_wkt):
    """Konversi lat/lon (WGS84) -> baris/kolom pada grid referensi (UTM)."""
    dst_crs = rasterio.crs.CRS.from_wkt(crs_wkt)
    from pyproj import Transformer
    transformer = Transformer.from_crs("EPSG:4326", dst_crs, always_xy=True)
    x, y = transformer.transform(lon, lat)
    col, row = ~transform * (x, y)
    return int(row), int(col)


def fit_statistical_sdm(rasters, presence_rowcol, model_type="Random Forest", n_background=1000, seed=42):
    """Latih model presence-background dan kembalikan peta probabilitas + info model."""
    rng = np.random.default_rng(seed)
    ref_layer = next(iter(rasters.values()))
    h, w = ref_layer.shape
    valid_mask = ~np.isnan(ref_layer)
    valid_rows, valid_cols = np.where(valid_mask)

    # presence
    pres_rows = np.array([r for r, c in presence_rowcol if 0 <= r < h and 0 <= c < w])
    pres_cols = np.array([c for r, c in presence_rowcol if 0 <= r < h and 0 <= c < w])
    if len(pres_rows) < 3:
        raise ValueError("Minimal 3 titik kemunculan (presence) valid diperlukan untuk melatih model.")

    # background / pseudo-absence acak dari kawasan valid
    idx = rng.choice(len(valid_rows), size=min(n_background, len(valid_rows)), replace=False)
    bg_rows, bg_cols = valid_rows[idx], valid_cols[idx]

    X_pres, layer_order = sample_rasters_at_points(rasters, pres_rows, pres_cols)
    X_bg, _ = sample_rasters_at_points(rasters, bg_rows, bg_cols)
    X = np.vstack([X_pres, X_bg])
    y = np.concatenate([np.ones(len(X_pres)), np.zeros(len(X_bg))])

    # buang baris dengan nan
    ok = ~np.isnan(X).any(axis=1)
    X, y = X[ok], y[ok]

    if model_type == "Random Forest":
        clf = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=seed, class_weight="balanced")
    else:
        clf = LogisticRegression(max_iter=1000, class_weight="balanced")

    if len(np.unique(y)) < 2:
        raise ValueError("Data presence/background tidak cukup bervariasi untuk melatih model.")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=seed, stratify=y)
    clf.fit(X_train, y_train)
    try:
        auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
    except Exception:
        auc = np.nan

    # prediksi seluruh piksel valid
    Xall, _ = sample_rasters_at_points(rasters, valid_rows, valid_cols)
    ok_all = ~np.isnan(Xall).any(axis=1)
    proba = np.full(len(valid_rows), np.nan, dtype=np.float32)
    proba[ok_all] = clf.predict_proba(Xall[ok_all])[:, 1]

    prob_map = np.full((h, w), np.nan, dtype=np.float32)
    prob_map[valid_rows, valid_cols] = proba

    importances = None
    if model_type == "Random Forest":
        importances = dict(zip(layer_order, clf.feature_importances_))
    else:
        importances = dict(zip(layer_order, np.abs(clf.coef_[0])))

    return prob_map, auc, importances, len(X_pres), len(X_bg)


# ==============================================================================
# 5. ANTARMUKA STREAMLIT
# ==============================================================================

st.title("🌿 WebGIS SDM — Potensi Tumbuh Tanaman Herbal TNBTS")
st.caption(
    "Species Distribution Modelling berbasis 8 layer raster reklasifikasi "
    "(Ketinggian, Curah Hujan, LST, Kelerengan, Kelembapan, NDVI, Jarak "
    "Pemukiman, Jenis Tanah) — Taman Nasional Bromo Tengger Semeru."
)

# ---- Sidebar: sumber data raster -------------------------------------------
st.sidebar.header("📁 1. Data Raster")
use_upload = st.sidebar.checkbox(
    "Unggah manual file raster (.tif)", value=not os.path.isdir(DATA_DIR)
)

uploaded_paths = {}
if use_upload:
    tmp_dir = tempfile.mkdtemp()
    for key in RASTER_KEYWORDS:
        f = st.sidebar.file_uploader(f"{key} ({LAYER_LABELS[key]})", type=["tif", "tiff"], key=f"up_{key}")
        if f is not None:
            p = os.path.join(tmp_dir, f.name)
            with open(p, "wb") as out:
                out.write(f.getbuffer())
            uploaded_paths[key] = p
else:
    st.sidebar.info(f"Membaca otomatis dari folder: `{DATA_DIR}`")

raster_paths = {}
missing = []
for key in RASTER_KEYWORDS:
    p = find_raster_path(key, uploaded_paths)
    if p is None:
        missing.append(key)
    else:
        raster_paths[key] = p

if missing:
    st.warning(
        "⚠️ Layer raster berikut belum tersedia: **" + ", ".join(missing) +
        "**. Silakan unggah file atau letakkan pada folder `data/` "
        "dengan nama file mengandung kata kunci: " +
        ", ".join(RASTER_KEYWORDS[k] for k in missing) + "."
    )
    st.stop()

downsample = st.sidebar.slider(
    "🔧 Faktor downsample (performa vs detail)", min_value=2, max_value=15, value=6,
    help="Nilai lebih besar = proses lebih cepat & ringan, resolusi tampilan lebih kasar."
)

with st.spinner("Memuat & menyelaraskan raster..."):
    paths_tuple = tuple(sorted(raster_paths.items()))
    rasters, ref_transform, ref_crs_wkt, grid_w, grid_h = load_all_rasters(paths_tuple, downsample)

st.sidebar.success(f"✅ 8 layer raster dimuat — grid {grid_w} x {grid_h} px")

# ---- Sidebar: pilih mode SDM ------------------------------------------------
st.sidebar.header("🧬 2. Metode SDM")
mode = st.sidebar.radio(
    "Pilih metode pemodelan:",
    ["Fuzzy Environmental Envelope (MCE)", "Statistik / Machine Learning (presence-background)"],
)

tab1, tab2, tab3 = st.tabs(["🗺️ Peta Kesesuaian", "📊 Data & Layer", "ℹ️ Tentang Metode"])


# ==============================================================================
# TAB 2 — Data & Layer (ditaruh duluan agar bisa dirujuk)
# ==============================================================================
with tab2:
    st.subheader("Database Ekologi Tanaman Herbal")
    df_species = pd.DataFrame([
        {
            "Tanaman": s["nama"], "Nama Latin": s["latin"],
            "Elevasi (mdpl)": f'{s["elev"][0]}–{s["elev"][1]}',
            "Curah Hujan (mm/th)": f'{s["hujan"][0]}–{s["hujan"][1]}',
            "Suhu (°C)": f'{s["suhu"][0]}–{s["suhu"][1]}',
            "pH Tanah": f'{s["ph"][0]}–{s["ph"][1]}',
            "Khasiat Utama": s["fungsi"],
        } for s in SPECIES_DB
    ])
    st.dataframe(df_species, use_container_width=True, height=380)

    st.subheader("Statistik Kelas per Layer Raster (grid termuat)")
    stat_rows = []
    for key, arr in rasters.items():
        vals, counts = np.unique(arr[~np.isnan(arr)], return_counts=True)
        stat_rows.append({
            "Layer": key, "Deskripsi": LAYER_LABELS[key],
            "Kelas tersedia": ", ".join(str(int(v)) for v in vals),
            "Jumlah piksel valid": int(counts.sum()),
        })
    st.dataframe(pd.DataFrame(stat_rows), use_container_width=True)


# ==============================================================================
# TAB 1 — Peta Kesesuaian
# ==============================================================================
with tab1:
    if mode == "Fuzzy Environmental Envelope (MCE)":
        colA, colB = st.columns([1, 2])
        with colA:
            species_name = st.selectbox("🌱 Pilih Tanaman Herbal", SPECIES_NAMES)
            species = next(s for s in SPECIES_DB if s["nama"] == species_name)
            st.markdown(f"**Nama Latin:** *{species['latin']}*")
            st.markdown(f"**Elevasi optimal:** {species['elev'][0]}–{species['elev'][1]} mdpl")
            st.markdown(f"**Curah hujan optimal:** {species['hujan'][0]}–{species['hujan'][1]} mm/th")
            st.markdown(f"**Suhu optimal:** {species['suhu'][0]}–{species['suhu'][1]} °C")
            st.markdown(f"**Khasiat:** {species['fungsi']}")

            st.markdown("---")
            st.markdown("**⚖️ Bobot AHP tiap layer (%)**")
            weights = {}
            for key in rasters:
                weights[key] = st.slider(LAYER_LABELS[key], 0, 100, DEFAULT_WEIGHTS[key], key=f"w_{key}")

            st.markdown("**↕️ Arah preferensi (layer non-spesifik-spesies)**")
            directions = {}
            for key in ["Lereng", "Moisture", "NDVI", "Pemukiman", "Soil"]:
                directions[key] = st.selectbox(
                    LAYER_LABELS[key], ["naik", "turun", "tengah"],
                    index=["naik", "turun", "tengah"].index(DEFAULT_DIRECTION[key]),
                    key=f"dir_{key}",
                )
            run = st.button("▶️ Jalankan Pemodelan", type="primary", use_container_width=True)

        with colB:
            if run or "si_result" in st.session_state:
                if run:
                    with st.spinner("Menghitung indeks kesesuaian..."):
                        si, scores = compute_suitability_fuzzy(rasters, species, weights, directions)
                        st.session_state["si_result"] = si
                        st.session_state["si_species"] = species_name

                si = st.session_state["si_result"]
                si_wgs, bounds, dst_transform = reproject_array_to_wgs84(si, ref_transform, ref_crs_wkt)
                left, bottom, right, top = bounds
                center = [(bottom + top) / 2, (left + right) / 2]

                m = folium.Map(location=center, zoom_start=11, tiles="OpenStreetMap")
                folium.TileLayer(
                    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                    attr="Esri", name="Citra Satelit", overlay=False,
                ).add_to(m)

                overlay_url = array_to_png_overlay(si_wgs, cmap_name="RdYlGn", vmin=0, vmax=1)
                folium.raster_layers.ImageOverlay(
                    image=overlay_url,
                    bounds=[[bottom, left], [top, right]],
                    opacity=0.75,
                    name=f"Kesesuaian: {st.session_state.get('si_species','')}",
                ).add_to(m)
                folium.LayerControl().add_to(m)
                MousePosition().add_to(m)

                st_folium(m, use_container_width=True, height=560)

                valid_si = si[~np.isnan(si)]
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Rerata Kesesuaian", f"{np.nanmean(si)*100:.1f}%")
                c2.metric("Maks Kesesuaian", f"{np.nanmax(si)*100:.1f}%")
                c3.metric("Area Sangat Sesuai (>75%)", f"{(valid_si>0.75).mean()*100:.1f}%")
                c4.metric("Area Tidak Sesuai (<25%)", f"{(valid_si<0.25).mean()*100:.1f}%")

                st.markdown("**Legenda:** 🟩 Sangat Sesuai — 🟨 Cukup Sesuai — 🟥 Kurang Sesuai")

                # Unduh GeoTIFF hasil (grid referensi UTM asli, resolusi sesuai downsample)
                buf = io.BytesIO()
                out_profile = {
                    "driver": "GTiff", "height": si.shape[0], "width": si.shape[1],
                    "count": 1, "dtype": "float32",
                    "crs": rasterio.crs.CRS.from_wkt(ref_crs_wkt),
                    "transform": ref_transform, "nodata": np.nan,
                }
                with rasterio.io.MemoryFile() as memfile:
                    with memfile.open(**out_profile) as dst:
                        dst.write(si.astype("float32"), 1)
                    buf.write(memfile.read())
                st.download_button(
                    "⬇️ Unduh Peta Kesesuaian (GeoTIFF)", data=buf.getvalue(),
                    file_name=f"kesesuaian_{species_name.replace(' ', '_')}.tif",
                    mime="image/tiff",
                )
            else:
                st.info("Atur bobot & arah preferensi di panel kiri, lalu klik **Jalankan Pemodelan**.")

    else:  # ---------------- MODE STATISTIK / ML ----------------
        if not SKLEARN_OK:
            st.error("scikit-learn belum terpasang. Tambahkan `scikit-learn` pada requirements.txt.")
            st.stop()

        colA, colB = st.columns([1, 2])
        with colA:
            st.markdown("**📍 Titik Kemunculan Spesies (Presence)**")
            st.caption(
                "Klik pada peta di sebelah kanan untuk menambah titik, atau unggah CSV "
                "dengan kolom `lat` dan `lon`."
            )
            up_csv = st.file_uploader("Unggah CSV titik kemunculan", type=["csv"])
            model_type = st.selectbox("Algoritma", ["Random Forest", "Regresi Logistik"])
            n_background = st.slider("Jumlah titik latar (background/pseudo-absence)", 200, 5000, 1000, step=100)

            if "presence_points" not in st.session_state:
                st.session_state["presence_points"] = []
            if up_csv is not None:
                df_up = pd.read_csv(up_csv)
                if {"lat", "lon"}.issubset(df_up.columns):
                    st.session_state["presence_points"] = list(zip(df_up["lat"], df_up["lon"]))
                    st.success(f"{len(df_up)} titik dimuat dari CSV.")
                else:
                    st.error("CSV harus memiliki kolom `lat` dan `lon`.")

            if st.button("🗑️ Hapus semua titik"):
                st.session_state["presence_points"] = []

            st.write(f"Jumlah titik saat ini: **{len(st.session_state['presence_points'])}**")
            run_ml = st.button("▶️ Latih Model & Prediksi", type="primary", use_container_width=True)

        with colB:
            left, bottom, right, top = array_bounds(grid_h, grid_w, ref_transform)
            # bounds ini masih UTM; konversi cepat pusatnya ke wgs84 lewat reproject helper
            dummy = np.zeros((2, 2), dtype=np.float32)
            _, wgs_bounds, _ = reproject_array_to_wgs84(dummy, ref_transform, ref_crs_wkt)
            wl, wb, wr, wt = wgs_bounds
            center = [(wb + wt) / 2, (wl + wr) / 2]

            m2 = folium.Map(location=center, zoom_start=11, tiles="OpenStreetMap")
            for lat, lon in st.session_state["presence_points"]:
                folium.CircleMarker([lat, lon], radius=5, color="red", fill=True, fill_color="red").add_to(m2)
            Draw(export=False, draw_options={"polyline": False, "polygon": False, "rectangle": False, "circle": False, "marker": True, "circlemarker": False}).add_to(m2)

            map_state = st_folium(m2, use_container_width=True, height=460, key="ml_map")
            if map_state and map_state.get("last_clicked"):
                latlon = (map_state["last_clicked"]["lat"], map_state["last_clicked"]["lng"])
                if latlon not in st.session_state["presence_points"]:
                    st.session_state["presence_points"].append(latlon)
                    st.rerun()

            if run_ml:
                pts = st.session_state["presence_points"]
                if len(pts) < 3:
                    st.error("Tambahkan minimal 3 titik kemunculan terlebih dahulu.")
                else:
                    with st.spinner("Melatih model & memprediksi seluruh kawasan..."):
                        rowcols = [latlon_to_rowcol(lat, lon, ref_transform, ref_crs_wkt) for lat, lon in pts]
                        try:
                            prob_map, auc, importances, n_pres, n_bg = fit_statistical_sdm(
                                rasters, rowcols, model_type=model_type, n_background=n_background
                            )
                            st.session_state["ml_result"] = (prob_map, auc, importances, n_pres, n_bg)
                        except ValueError as e:
                            st.error(str(e))

            if "ml_result" in st.session_state:
                prob_map, auc, importances, n_pres, n_bg = st.session_state["ml_result"]
                prob_wgs, bounds, _ = reproject_array_to_wgs84(prob_map, ref_transform, ref_crs_wkt)
                bl, bb, br, bt = bounds
                m3 = folium.Map(location=[(bb + bt) / 2, (bl + br) / 2], zoom_start=11)
                overlay_url = array_to_png_overlay(prob_wgs, cmap_name="viridis", vmin=0, vmax=1)
                folium.raster_layers.ImageOverlay(
                    image=overlay_url, bounds=[[bb, bl], [bt, br]], opacity=0.75,
                    name="Probabilitas Sebaran",
                ).add_to(m3)
                folium.LayerControl().add_to(m3)
                st_folium(m3, use_container_width=True, height=460, key="ml_result_map")

                c1, c2 = st.columns(2)
                c1.metric("AUC (uji)", f"{auc:.3f}" if not np.isnan(auc) else "N/A")
                c2.metric("Titik presence / background", f"{n_pres} / {n_bg}")

                st.markdown("**Kepentingan Variabel (Feature Importance)**")
                imp_df = pd.DataFrame(
                    {"Layer": list(importances.keys()), "Kepentingan": list(importances.values())}
                ).sort_values("Kepentingan", ascending=False)
                st.bar_chart(imp_df.set_index("Layer"))


# ==============================================================================
# TAB 3 — Tentang Metode
# ==============================================================================
with tab3:
    st.markdown(__doc__)
