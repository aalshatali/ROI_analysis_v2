"""
CT vs PCCT ROI Comparator
=========================
Compares HU values between a reference CT scan and a PCCT scan
for a single patient, single ROI, with voxel-wise analysis.

Dependencies:
    pip install streamlit pydicom numpy pandas scipy scikit-image SimpleITK matplotlib
"""

import atexit
import logging
import os
import shutil
import tempfile
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import pydicom
import SimpleITK as sitk
import streamlit as st
from scipy.ndimage import binary_erosion
from skimage.draw import polygon

# =====================================================
# LOGGING
# =====================================================
logging.basicConfig(level=logging.WARNING)
log = logging.getLogger("ct_pcct")

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="CT · PCCT Comparator",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =====================================================
# CUSTOM CSS  — clinical/precision aesthetic
# =====================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Sora:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Sora', sans-serif;
}

/* ── Base text: always light on dark ── */
.stApp {
    background-color: #0d1117;
    color: #e6edf3;
}

/* All plain paragraph / label text */
p, span, div, label, li,
.stMarkdown, .stText,
[data-testid="stMarkdownContainer"] p {
    color: #e6edf3 !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #161b22;
    border-right: 1px solid #30363d;
}
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] div {
    color: #e6edf3 !important;
}

/* Headings */
h1 {
    font-family: 'Sora', sans-serif;
    font-weight: 700;
    font-size: 1.8rem;
    letter-spacing: -0.03em;
    color: #e6edf3 !important;
}
h2, h3, h4 {
    font-family: 'Sora', sans-serif;
    font-weight: 600;
    color: #c9d1d9 !important;
    letter-spacing: -0.02em;
}

/* Streamlit native widgets — inputs, selects, number inputs */
.stSelectbox label, .stMultiSelect label,
.stNumberInput label, .stRadio label,
.stFileUploader label, .stSlider label {
    color: #c9d1d9 !important;
    font-size: 0.85rem;
}
.stRadio div[role="radiogroup"] label {
    color: #e6edf3 !important;
}

/* Expander header text */
.streamlit-expanderHeader {
    color: #e6edf3 !important;
    background-color: #161b22 !important;
}
.streamlit-expanderContent {
    background-color: #0d1117 !important;
    border: 1px solid #30363d;
}

/* st.code / st.text blocks */
.stCodeBlock pre, code {
    background-color: #161b22 !important;
    color: #e6edf3 !important;
    border: 1px solid #30363d;
}

/* Dataframe / table text */
[data-testid="stDataFrame"] {
    color: #e6edf3;
}

/* Alert / info / warning / error boxes */
.stAlert {
    color: #e6edf3 !important;
}
div[data-testid="stAlert"] p {
    color: #e6edf3 !important;
}

/* Upload area */
[data-testid="stFileUploader"] {
    background-color: #161b22;
    border: 1px dashed #30363d;
    border-radius: 6px;
}
[data-testid="stFileUploader"] span,
[data-testid="stFileUploader"] p {
    color: #c9d1d9 !important;
}

/* ── Custom components ── */
.badge-ct {
    display: inline-block;
    background: #0d4f8c;
    color: #79c0ff !important;
    border: 1px solid #1f6feb;
    border-radius: 4px;
    padding: 2px 10px;
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
    font-weight: 500;
    letter-spacing: 0.08em;
    margin-bottom: 8px;
}

.badge-pcct {
    display: inline-block;
    background: #3d1f63;
    color: #d2a8ff !important;
    border: 1px solid #8957e5;
    border-radius: 4px;
    padding: 2px 10px;
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
    font-weight: 500;
    letter-spacing: 0.08em;
    margin-bottom: 8px;
}

.metric-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 16px 20px;
    margin: 6px 0;
}

.metric-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    color: #8b949e !important;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 4px;
}

.metric-value {
    font-family: 'DM Mono', monospace;
    font-size: 1.4rem;
    font-weight: 500;
    color: #e6edf3 !important;
}

.section-rule {
    border: none;
    border-top: 1px solid #30363d;
    margin: 2rem 0 1.5rem 0;
}

.info-box {
    background: #0d2137;
    border-left: 3px solid #1f6feb;
    border-radius: 0 6px 6px 0;
    padding: 12px 16px;
    font-size: 0.85rem;
    color: #c9d1d9 !important;
    margin: 12px 0;
}

.stratum-chip {
    display: inline-block;
    border-radius: 20px;
    padding: 2px 12px;
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    margin: 2px;
}

/* ── Selectbox / dropdown — light background needs dark text ── */
div[data-baseweb="select"] span,
div[data-baseweb="select"] div,
div[data-baseweb="select"] input,
div[data-baseweb="select"] > div > div {
    color: #0d1117 !important;
    background-color: #f0f2f6 !important;
}
ul[data-baseweb="menu"] li,
ul[data-baseweb="menu"] span,
[role="listbox"] li,
[role="option"] {
    color: #0d1117 !important;
    background-color: #ffffff !important;
}
[role="option"]:hover,
[role="option"][aria-selected="true"] {
    background-color: #dde1e7 !important;
    color: #0d1117 !important;
}
/* Number / text inputs */
input[type="number"], input[type="text"] {
    color: #0d1117 !important;
    background-color: #f0f2f6 !important;
}
/* Radio button labels (already light enough, but ensure contrast) */
.stRadio > div label p {
    color: #e6edf3 !important;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# TEMP DIR
# =====================================================
if "temp_dir" not in st.session_state:
    td = tempfile.mkdtemp(prefix="ct_pcct_")
    st.session_state["temp_dir"] = td
    atexit.register(shutil.rmtree, td, ignore_errors=True)

TEMP_DIR: str = st.session_state["temp_dir"]

# =====================================================
# HEADER
# =====================================================
st.markdown("# CT · PCCT ROI Comparator")
st.markdown(
    '<div class="info-box">Single-patient, single-ROI voxel-wise HU comparison between '
    'a reference <b>CT</b> scan and a <b>PCCT</b> scan. '
    'Registration is assumed to be complete prior to upload.</div>',
    unsafe_allow_html=True,
)

# =====================================================
# SIDEBAR — configuration
# =====================================================
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    st.markdown("---")

    erosion_voxels = st.number_input(
        "Mask erosion (voxels)",
        min_value=0,
        max_value=10,
        value=2,
        step=1,
        help="Erodes the ROI mask inward by N voxels to exclude boundary artefacts from misregistration.",
    )

    st.markdown("---")
    st.markdown("### 🎚️ Tissue HU Thresholds")
    st.markdown(
        '<div class="info-box">Define the HU boundaries separating tissue strata. '
        'Calibrate per study.</div>',
        unsafe_allow_html=True,
    )

    air_upper = st.number_input("Air  |  upper bound (HU)", value=-900, step=10)
    fat_upper = st.number_input("Fat  |  upper bound (HU)", value=-100, step=10)
    soft_upper = st.number_input("Soft tissue  |  upper bound (HU)", value=200, step=10)
    # Bone = everything above soft_upper

    strata_bounds = [
        ("Air",         None,        air_upper),
        ("Fat",         air_upper,   fat_upper),
        ("Soft Tissue", fat_upper,   soft_upper),
        ("Bone/Dense",  soft_upper,  None),
    ]

    st.markdown("---")
    st.markdown("### 📊 Bland–Altman")
    ba_level = st.radio(
        "Aggregation level",
        ["Per slice (mean per slice)", "Per voxel (subsampled)"],
        index=0,
    )
    if "subsampled" in ba_level:
        ba_max_pts = st.number_input("Max voxels to plot", value=5000, step=500)
    else:
        ba_max_pts = None

    st.markdown("---")
    st.markdown(
        '<div class="info-box" style="font-size:0.75rem;">CT is treated as the '
        'reference standard.<br>Difference = CT − PCCT.<br>'
        'Positive values → CT reads higher.</div>',
        unsafe_allow_html=True,
    )

# =====================================================
# UPLOADERS
# =====================================================
col_a, col_b = st.columns(2)

with col_a:
    st.markdown('<span class="badge-ct">CT — REFERENCE</span>', unsafe_allow_html=True)
    zip_ct = st.file_uploader("Upload CT ZIP", type=["zip"], key="zip_ct")

with col_b:
    st.markdown('<span class="badge-pcct">PCCT — CHALLENGER</span>', unsafe_allow_html=True)
    zip_pcct = st.file_uploader("Upload PCCT ZIP", type=["zip"], key="zip_pcct")

# =====================================================
# HELPER FUNCTIONS
# =====================================================

@st.cache_data(show_spinner=False)
def extract_zip(zip_bytes: bytes, label: str) -> str:
    """Extract a ZIP to a labelled subfolder of TEMP_DIR. Returns extract path."""
    extract_dir = os.path.join(TEMP_DIR, label)
    os.makedirs(extract_dir, exist_ok=True)
    zip_path = os.path.join(TEMP_DIR, f"{label}.zip")
    with open(zip_path, "wb") as f:
        f.write(zip_bytes)
    with zipfile.ZipFile(zip_path, "r") as zr:
        zr.extractall(extract_dir)
    return extract_dir


@st.cache_data(show_spinner=False)
def scan_dicom_dir(extract_dir: str):
    """
    Walk a directory and return:
        ct_files  : list of CT DICOM file paths (all series merged, assumed single series)
        rt_file   : path to the RTSTRUCT file
        roi_names : list of ROI names in the RTSTRUCT
    """
    ct_files = []
    rt_file = None

    for f in Path(extract_dir).rglob("*"):
        if not f.is_file():
            continue
        try:
            ds = pydicom.dcmread(str(f), stop_before_pixels=True)
            mod = getattr(ds, "Modality", None)
            if mod == "CT":
                ct_files.append(str(f))
            elif mod == "RTSTRUCT":
                rt_file = str(f)
        except Exception:
            pass

    roi_names = []
    if rt_file:
        rt_ds = pydicom.dcmread(rt_file)
        roi_names = [r.ROIName for r in rt_ds.StructureSetROISequence]

    return ct_files, rt_file, roi_names


def _decode_pixel_array(ds) -> np.ndarray:
    """
    Decode pixel data from a pydicom Dataset, handling both uncompressed
    and compressed transfer syntaxes (JPEG, JPEG-LS, JPEG2000).

    Compressed DICOM requires optional codec plugins:
      pip install pylibjpeg pylibjpeg-libjpeg pylibjpeg-openjpeg gdcm

    If all decoders fail the slice is skipped (caller handles missing slices).
    """
    # 1. Standard path — works for uncompressed and most JPEG variants
    #    when the right plugin is installed.
    try:
        return ds.pixel_array.astype(np.float32)
    except Exception:
        pass

    # 2. Force numpy handler (uncompressed fallback via raw buffer)
    try:
        import pydicom.pixel_data_handlers.numpy_handler as np_handler
        if np_handler.supports_transfer_syntax(ds.file_meta.TransferSyntaxUID):
            ds.decompress()
            return ds.pixel_array.astype(np.float32)
    except Exception:
        pass

    # 3. gdcm handler
    try:
        import pydicom.pixel_data_handlers.gdcm_handler as gdcm_handler
        if gdcm_handler.supports_transfer_syntax(ds.file_meta.TransferSyntaxUID):
            gdcm_handler.needs_to_convert_these_transfer_syntaxes(ds.file_meta.TransferSyntaxUID)
            ds.decompress()
            return ds.pixel_array.astype(np.float32)
    except Exception:
        pass

    # 4. pylibjpeg handler
    try:
        import pydicom.pixel_data_handlers.pylibjpeg_handler as pljpeg
        if pljpeg.supports_transfer_syntax(ds.file_meta.TransferSyntaxUID):
            return pljpeg.get_pixeldata(ds).astype(np.float32).reshape(
                ds.Rows, ds.Columns
            )
    except Exception:
        pass

    raise ValueError(
        f"Cannot decode pixel data for SOP {getattr(ds, 'SOPInstanceUID', '?')}. "
        "Install codec plugins: pip install pylibjpeg pylibjpeg-libjpeg "
        "pylibjpeg-openjpeg  or  pip install python-gdcm"
    )


@st.cache_data(show_spinner=False)
def load_ct_volume(files: tuple[str, ...]):
    """
    Load sorted CT series → HU volume.
    Returns: volume (rows, cols, slices), slices, z_positions, spacing, origin, direction
    """
    raw_slices = [pydicom.dcmread(f) for f in files]
    raw_slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))

    imgs, zs, slices = [], [], []
    decode_errors = 0
    for s in raw_slices:
        try:
            img = _decode_pixel_array(s)
        except Exception as exc:
            decode_errors += 1
            log.warning("Skipping slice %s: %s", getattr(s, "SOPInstanceUID", "?"), exc)
            continue
        slope = float(getattr(s, "RescaleSlope", 1.0))
        intercept = float(getattr(s, "RescaleIntercept", 0.0))
        imgs.append(img * slope + intercept)
        zs.append(float(s.ImagePositionPatient[2]))
        slices.append(s)

    if decode_errors:
        log.warning("%d slice(s) could not be decoded and were skipped.", decode_errors)

    if not imgs:
        raise ValueError(
            "No slices could be decoded. The DICOM files may use a compressed "
            "transfer syntax that requires additional codec plugins.\n"
            "Install them with:\n"
            "  pip install pylibjpeg pylibjpeg-libjpeg pylibjpeg-openjpeg\n"
            "or:\n"
            "  pip install python-gdcm"
        )

    volume = np.stack(imgs, axis=-1)
    z_positions = np.array(zs)

    ps = slices[0].PixelSpacing
    row_sp = float(ps[0])
    col_sp = float(ps[1])
    slice_sp = abs(z_positions[1] - z_positions[0]) if len(z_positions) > 1 else 1.0
    spacing = (row_sp, col_sp, slice_sp)
    origin = np.array([float(v) for v in slices[0].ImagePositionPatient])

    # ImageOrientationPatient: [row_cosine(3), col_cosine(3)]
    iop = [float(v) for v in slices[0].ImageOrientationPatient]
    row_cosine   = np.array(iop[:3])
    col_cosine   = np.array(iop[3:])
    slice_cosine = np.cross(row_cosine, col_cosine)
    # SimpleITK direction: column-major (row_cos | col_cos | slice_cos)
    direction = tuple(row_cosine) + tuple(col_cosine) + tuple(slice_cosine)

    return volume, slices, z_positions, spacing, origin, direction


def build_roi_mask(rt_path: str, roi_name: str, volume_shape, z_positions, spacing, origin,
                   z_tol: float = 3.0) -> np.ndarray:
    """Convert RTSTRUCT contours → boolean voxel mask."""
    rt = pydicom.dcmread(rt_path)
    roi_map = {r.ROIName: r.ROINumber for r in rt.StructureSetROISequence}
    roi_num = roi_map.get(roi_name)
    mask = np.zeros(volume_shape, dtype=bool)

    if roi_num is None:
        return mask

    roi_contours = None
    for rc in rt.ROIContourSequence:
        if rc.ReferencedROINumber == roi_num:
            roi_contours = rc
            break

    if roi_contours is None or not hasattr(roi_contours, "ContourSequence"):
        return mask

    row_sp, col_sp, _ = spacing
    x0, y0 = origin[0], origin[1]

    for contour in roi_contours.ContourSequence:
        pts = np.array(contour.ContourData).reshape(-1, 3)
        z = pts[0, 2]
        dists = np.abs(z_positions - z)
        si = int(np.argmin(dists))
        if dists[si] > z_tol:
            continue
        rows = (pts[:, 1] - y0) / row_sp
        cols = (pts[:, 0] - x0) / col_sp
        rr, cc = polygon(rows, cols, shape=volume_shape[:2])
        mask[rr, cc, si] = True

    return mask


def erode_mask(mask: np.ndarray, n: int) -> np.ndarray:
    """
    Erode a 3D boolean mask slice-by-slice (2D erosion per slice).

    Why 2D and not 3D:
      - CT slice thickness (e.g. 1–3 mm) is typically much larger than
        in-plane pixel spacing (e.g. 0.5–1 mm), so a 3D cube structuring
        element aggressively removes voxels through-plane and can wipe
        out thin ROIs entirely.
      - 2D erosion per slice is standard practice in radiomics for this
        reason and correctly reflects the in-plane boundary uncertainty.
    """
    if n == 0:
        return mask
    struct_2d = np.ones((2 * n + 1, 2 * n + 1), dtype=bool)
    eroded = np.zeros_like(mask)
    for sl in range(mask.shape[2]):
        if mask[:, :, sl].any():
            eroded[:, :, sl] = binary_erosion(
                mask[:, :, sl], structure=struct_2d
            )
    return eroded


def volume_to_sitk(
    volume: np.ndarray,
    spacing: tuple,
    origin: np.ndarray,
    direction: tuple | None = None,
) -> sitk.Image:
    """
    Convert numpy HU volume (rows, cols, slices) -> SimpleITK image.

    SimpleITK axis order is (x=cols, y=rows, z=slices), so the array is
    transposed to (slices, rows, cols) before wrapping.
    Spacing  -> (col_sp, row_sp, slice_sp)
    Direction cosines from DICOM ImageOrientationPatient MUST be set;
    without them SimpleITK assumes an identity direction matrix and two
    physically-overlapping grids can appear non-overlapping to the resampler.
    """
    img = sitk.GetImageFromArray(np.transpose(volume, (2, 0, 1)).astype(np.float32))
    img.SetSpacing((float(spacing[1]), float(spacing[0]), float(spacing[2])))
    img.SetOrigin(tuple(float(v) for v in origin))
    if direction is not None:
        img.SetDirection(direction)
    return img


def resample_to_reference(moving: sitk.Image, fixed: sitk.Image) -> sitk.Image:
    """Resample moving image onto fixed image grid (identity transform — registration pre-done)."""
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(-1024.0)
    return resampler.Execute(moving)


def sitk_to_numpy(img: sitk.Image) -> np.ndarray:
    """Convert SimpleITK image back to numpy (rows, cols, slices)."""
    arr = sitk.GetArrayFromImage(img)   # shape: (slices, rows, cols)
    return np.transpose(arr, (1, 2, 0))


def assign_stratum(hu_values: np.ndarray, bounds) -> np.ndarray:
    """Return stratum index array for each voxel HU value."""
    out = np.full(hu_values.shape, len(bounds) - 1, dtype=np.int8)
    for i, (_, lo, hi) in enumerate(bounds):
        cond = np.ones(hu_values.shape, dtype=bool)
        if lo is not None:
            cond &= hu_values > lo
        if hi is not None:
            cond &= hu_values <= hi
        out[cond] = i
    return out


def compute_stats(vals: np.ndarray, label: str) -> dict:
    return {
        "Scanner": label,
        "N voxels": len(vals),
        "Mean HU": round(float(np.mean(vals)), 2),
        "STD HU": round(float(np.std(vals)), 2),
        "Min HU": round(float(np.min(vals)), 2),
        "Max HU": round(float(np.max(vals)), 2),
        "Median HU": round(float(np.median(vals)), 2),
    }


# =====================================================
# MATPLOTLIB STYLE
# =====================================================
PLOT_BG = "#0d1117"
PLOT_FG = "#e6edf3"
PLOT_GRID = "#21262d"
CT_COLOR = "#1f6feb"
PCCT_COLOR = "#8957e5"
DIFF_COLOR = "#f0883e"

def apply_dark_style(ax, fig):
    fig.patch.set_facecolor(PLOT_BG)
    ax.set_facecolor("#161b22")
    ax.tick_params(colors=PLOT_FG, labelsize=8)
    ax.xaxis.label.set_color(PLOT_FG)
    ax.yaxis.label.set_color(PLOT_FG)
    ax.title.set_color(PLOT_FG)
    for spine in ax.spines.values():
        spine.set_edgecolor(PLOT_GRID)
    ax.grid(color=PLOT_GRID, linewidth=0.5, linestyle="--", alpha=0.7)


# =====================================================
# MAIN ANALYSIS — triggered when both ZIPs are uploaded
# =====================================================
if zip_ct and zip_pcct:

    st.markdown('<hr class="section-rule">', unsafe_allow_html=True)

    # --- Extract ZIPs ---
    with st.spinner("Extracting ZIPs…"):
        dir_ct = extract_zip(zip_ct.getbuffer().tobytes(), "ct")
        dir_pcct = extract_zip(zip_pcct.getbuffer().tobytes(), "pcct")

    # --- Scan DICOM files ---
    with st.spinner("Scanning DICOM files…"):
        ct_files, ct_rt, ct_rois = scan_dicom_dir(dir_ct)
        pcct_files, pcct_rt, pcct_rois = scan_dicom_dir(dir_pcct)

    if not ct_files:
        st.error("❌ No CT DICOM files found in the CT ZIP.")
        st.stop()
    if not pcct_files:
        st.error("❌ No CT DICOM files found in the PCCT ZIP.")
        st.stop()
    if not ct_rt:
        st.error("❌ No RTSTRUCT found in the CT ZIP.")
        st.stop()
    if not pcct_rt:
        st.error("❌ No RTSTRUCT found in the PCCT ZIP.")
        st.stop()

    # --- ROI selection ---
    union_rois = sorted(set(ct_rois) | set(pcct_rois))
    only_ct = set(ct_rois) - set(pcct_rois)
    only_pcct = set(pcct_rois) - set(ct_rois)

    st.markdown("### 🎯 ROI Selection")

    if only_ct:
        st.warning(f"ROIs only in CT: {', '.join(sorted(only_ct))}")
    if only_pcct:
        st.warning(f"ROIs only in PCCT: {', '.join(sorted(only_pcct))}")

    common_rois = sorted(set(ct_rois) & set(pcct_rois))
    if not common_rois:
        st.error("❌ No ROIs in common between CT and PCCT RTSTRUCTs.")
        st.stop()

    selected_roi = st.selectbox(
        "Select ROI for analysis",
        common_rois,
        help="Only ROIs present in both RTSTRUCTs are listed.",
    )

    # --- Run button ---
    st.markdown('<hr class="section-rule">', unsafe_allow_html=True)

    if st.button("▶ Run Analysis", type="primary"):

        # ── Load CT volume ──
        with st.spinner("Loading CT volume…"):
            ct_vol, ct_slices, ct_z, ct_spacing, ct_origin, ct_direction = load_ct_volume(
                tuple(sorted(ct_files))
            )

        # ── Load PCCT volume ──
        with st.spinner("Loading PCCT volume…"):
            pcct_vol, pcct_slices, pcct_z, pcct_spacing, pcct_origin, pcct_direction = load_ct_volume(
                tuple(sorted(pcct_files))
            )

        # ── Align PCCT volume to CT grid (index-space) ──
        # Background: registration guarantees anatomy is aligned, but DICOM
        # ImagePositionPatient Z origins can differ by hundreds of mm when the
        # registration transform was NOT written back into the CT headers.
        # Physical-space resampling therefore fails.  We instead align the two
        # volumes slice-by-rank: we identify which PCCT slices spatially
        # correspond to which CT slices by matching their ranked contour Z
        # positions, then resample PCCT in-plane only onto the CT pixel grid.
        with st.spinner("Aligning PCCT onto CT grid (index-space)…"):

            # 1. Build masks in each scanner's own native pixel space
            mask_ct   = build_roi_mask(ct_rt,   selected_roi, ct_vol.shape,
                                       ct_z,   ct_spacing,   ct_origin)
            mask_pcct = build_roi_mask(pcct_rt, selected_roi, pcct_vol.shape,
                                       pcct_z, pcct_spacing, pcct_origin)

            # 2. Find which slice indices have ROI voxels in each mask
            ct_roi_slices   = sorted(np.unique(np.where(mask_ct)[2]))
            pcct_roi_slices = sorted(np.unique(np.where(mask_pcct)[2]))

            n_ct_mask_raw   = int(mask_ct.sum())
            n_pcct_mask_raw = int(mask_pcct.sum())

            # 3. Match PCCT ROI slices to CT ROI slices by rank order.
            #    Registration guarantees the Nth ROI slice in PCCT corresponds
            #    to the Nth ROI slice in CT, regardless of Z coordinates.
            n_common = min(len(ct_roi_slices), len(pcct_roi_slices))
            slice_pairs = list(zip(ct_roi_slices[:n_common],
                                   pcct_roi_slices[:n_common]))

            # 4. For each pair, resample the PCCT slice in-plane onto CT grid
            #    using SimpleITK (handles different pixel spacings correctly).
            ct_rows,   ct_cols   = ct_vol.shape[:2]
            pcct_rows, pcct_cols = pcct_vol.shape[:2]
            needs_inplane_resample = (
                ct_rows != pcct_rows or ct_cols != pcct_cols or
                abs(ct_spacing[0] - pcct_spacing[0]) > 0.001 or
                abs(ct_spacing[1] - pcct_spacing[1]) > 0.001
            )

            # Allocate resampled PCCT volume on CT grid
            pcct_vol_r = np.full(ct_vol.shape, -1024.0, dtype=np.float32)
            mask_pcct_r = np.zeros(ct_vol.shape, dtype=bool)

            for ct_sl, pcct_sl in slice_pairs:
                if needs_inplane_resample:
                    # Wrap single PCCT slice as a 2D SimpleITK image
                    pcct_slice_sitk = sitk.GetImageFromArray(
                        pcct_vol[:, :, pcct_sl].astype(np.float32)
                    )
                    pcct_slice_sitk.SetSpacing(
                        (float(pcct_spacing[1]), float(pcct_spacing[0]))
                    )
                    pcct_slice_sitk.SetOrigin(
                        (float(pcct_origin[0]), float(pcct_origin[1]))
                    )
                    # Reference CT slice grid
                    ct_ref_sitk = sitk.GetImageFromArray(
                        ct_vol[:, :, ct_sl].astype(np.float32)
                    )
                    ct_ref_sitk.SetSpacing(
                        (float(ct_spacing[1]), float(ct_spacing[0]))
                    )
                    ct_ref_sitk.SetOrigin(
                        (float(ct_origin[0]), float(ct_origin[1]))
                    )
                    resampler_2d = sitk.ResampleImageFilter()
                    resampler_2d.SetReferenceImage(ct_ref_sitk)
                    resampler_2d.SetInterpolator(sitk.sitkLinear)
                    resampler_2d.SetTransform(sitk.Transform(2, sitk.sitkIdentity))
                    resampler_2d.SetDefaultPixelValue(-1024.0)
                    pcct_resampled_2d = sitk.GetArrayFromImage(
                        resampler_2d.Execute(pcct_slice_sitk)
                    )
                    pcct_vol_r[:, :, ct_sl] = pcct_resampled_2d

                    # Resample PCCT mask slice
                    pcct_mask_sitk_2d = sitk.GetImageFromArray(
                        mask_pcct[:, :, pcct_sl].astype(np.float32)
                    )
                    pcct_mask_sitk_2d.SetSpacing(
                        (float(pcct_spacing[1]), float(pcct_spacing[0]))
                    )
                    pcct_mask_sitk_2d.SetOrigin(
                        (float(pcct_origin[0]), float(pcct_origin[1]))
                    )
                    resampler_nn2d = sitk.ResampleImageFilter()
                    resampler_nn2d.SetReferenceImage(ct_ref_sitk)
                    resampler_nn2d.SetInterpolator(sitk.sitkNearestNeighbor)
                    resampler_nn2d.SetTransform(sitk.Transform(2, sitk.sitkIdentity))
                    resampler_nn2d.SetDefaultPixelValue(0)
                    mask_pcct_r[:, :, ct_sl] = sitk.GetArrayFromImage(
                        resampler_nn2d.Execute(pcct_mask_sitk_2d)
                    ).astype(bool)
                else:
                    # Same grid — just copy
                    pcct_vol_r[:, :, ct_sl]  = pcct_vol[:, :, pcct_sl]
                    mask_pcct_r[:, :, ct_sl] = mask_pcct[:, :, pcct_sl]

            # 5. Diagnostics
            n_ct_mask   = int(mask_ct.sum())
            n_pcct_mask = int(mask_pcct_r.sum())
            combined_mask_pre = mask_ct & mask_pcct_r
            n_intersection = int(combined_mask_pre.sum())

            with st.expander("🔍 Mask diagnostics (expand to debug)", expanded=False):
                st.markdown(
                    f"| Step | Voxels |\n|---|---|\n"
                    f"| CT mask (native grid) | `{n_ct_mask:,}` |\n"
                    f"| PCCT mask (aligned to CT grid) | `{n_pcct_mask:,}` |\n"
                    f"| Intersection (CT ∩ PCCT) | `{n_intersection:,}` |\n"
                    f"| After {erosion_voxels}-voxel 2D erosion | computed below |"
                )
                st.markdown("**CT grid**")
                st.code(
                    f"origin  : {[round(v,2) for v in ct_origin]}\n"
                    f"spacing : {[round(v,4) for v in ct_spacing]}\n"
                    f"shape   : {ct_vol.shape}\n"
                    f"roi slices ({len(ct_roi_slices)}): {ct_roi_slices}"
                )
                st.markdown("**PCCT grid (native)**")
                st.code(
                    f"origin  : {[round(v,2) for v in pcct_origin]}\n"
                    f"spacing : {[round(v,4) for v in pcct_spacing]}\n"
                    f"shape   : {pcct_vol.shape}\n"
                    f"roi slices ({len(pcct_roi_slices)}): {pcct_roi_slices}"
                )
                st.markdown("**Slice pairing (CT slice → PCCT slice)**")
                st.code("\n".join(f"CT sl {c:3d}  ←→  PCCT sl {p:3d}" for c, p in slice_pairs))
                st.markdown(
                    f"In-plane resample needed: `{needs_inplane_resample}` "
                    f"(CT spacing {ct_spacing[:2]}, PCCT spacing {pcct_spacing[:2]})"
                )
                if n_ct_mask == 0:
                    st.error("CT mask is empty — check RTSTRUCT and ROI name.")
                if n_pcct_mask_raw == 0:
                    st.error("PCCT mask is empty in its native grid — check RTSTRUCT.")
                if len(slice_pairs) == 0:
                    st.error(
                        "No slice pairs found. Neither mask has any ROI voxels. "
                        "Check that the selected ROI has contours in both RTSTRUCTs."
                    )

            if n_intersection == 0:
                st.error(
                    "❌ CT and PCCT masks have zero overlap after index-space alignment.\n"
                    "Expand 'Mask diagnostics' above to inspect slice pairing."
                )
                st.stop()

            # Combined mask: intersection of both, then erode
            if erosion_voxels > 0:
                combined_mask = erode_mask(combined_mask_pre, erosion_voxels)
            else:
                combined_mask = combined_mask_pre

            n_voxels = int(combined_mask.sum())

        if n_voxels == 0:
            st.error(
                f"❌ Mask is empty after {erosion_voxels}-voxel erosion "
                f"(intersection had {n_intersection:,} voxels before erosion). "
                "Try reducing the erosion value in the sidebar — the ROI may be "
                "too thin in-plane for the current setting."
            )
            st.stop()

        # ── Extract HU values ──
        ct_vals = ct_vol[combined_mask].astype(np.float32)
        pcct_vals = pcct_vol_r[combined_mask].astype(np.float32)
        diff_vals = ct_vals - pcct_vals  # CT − PCCT

        # ── Assign tissue strata using CT as reference ──
        strata_labels = assign_stratum(ct_vals, strata_bounds)

        # ── Slice indices for each voxel ──
        vox_z_idx = np.where(combined_mask)[2]

        # ==============================================
        # SECTION 1 — CT RAW RESULTS
        # ==============================================
        st.markdown("## 🔵 Section 1 — CT Raw Results")
        st.markdown('<span class="badge-ct">CT — REFERENCE</span>', unsafe_allow_html=True)

        overall_ct = compute_stats(ct_vals, "CT")
        c1, c2, c3, c4, c5 = st.columns(5)
        for col, (k, v) in zip(
            [c1, c2, c3, c4, c5],
            [(k, overall_ct[k]) for k in ["Mean HU", "STD HU", "Min HU", "Max HU", "Median HU"]],
        ):
            col.markdown(
                f'<div class="metric-card"><div class="metric-label">{k}</div>'
                f'<div class="metric-value" style="color:#79c0ff;">{v}</div></div>',
                unsafe_allow_html=True,
            )

        st.markdown(f"**Total voxels in ROI:** `{n_voxels:,}` "
                    f"(after {erosion_voxels}-voxel erosion)")

        # Per-stratum CT stats
        st.markdown("#### Per tissue stratum")
        ct_stratum_rows = []
        for si, (sname, lo, hi) in enumerate(strata_bounds):
            sv = ct_vals[strata_labels == si]
            if len(sv) == 0:
                continue
            row = compute_stats(sv, "CT")
            row["Stratum"] = sname
            row["HU range"] = f"{'−∞' if lo is None else lo} → {'∞' if hi is None else hi}"
            ct_stratum_rows.append(row)

        df_ct_strata = pd.DataFrame(ct_stratum_rows)[
            ["Stratum", "HU range", "N voxels", "Mean HU", "STD HU", "Min HU", "Max HU", "Median HU"]
        ]
        st.dataframe(df_ct_strata, use_container_width=True)

        # HU histogram CT
        fig_ct, ax_ct = plt.subplots(figsize=(8, 3))
        ax_ct.hist(ct_vals, bins=120, color=CT_COLOR, alpha=0.85, edgecolor="none")
        ax_ct.set_xlabel("HU")
        ax_ct.set_ylabel("Voxel count")
        ax_ct.set_title(f"CT HU distribution — {selected_roi}")
        apply_dark_style(ax_ct, fig_ct)
        fig_ct.tight_layout()
        st.pyplot(fig_ct)
        plt.close(fig_ct)

        # ==============================================
        # SECTION 2 — PCCT RAW RESULTS
        # ==============================================
        st.markdown('<hr class="section-rule">', unsafe_allow_html=True)
        st.markdown("## 🟣 Section 2 — PCCT Raw Results")
        st.markdown('<span class="badge-pcct">PCCT — CHALLENGER</span>', unsafe_allow_html=True)

        overall_pcct = compute_stats(pcct_vals, "PCCT")
        p1, p2, p3, p4, p5 = st.columns(5)
        for col, (k, v) in zip(
            [p1, p2, p3, p4, p5],
            [(k, overall_pcct[k]) for k in ["Mean HU", "STD HU", "Min HU", "Max HU", "Median HU"]],
        ):
            col.markdown(
                f'<div class="metric-card"><div class="metric-label">{k}</div>'
                f'<div class="metric-value" style="color:#d2a8ff;">{v}</div></div>',
                unsafe_allow_html=True,
            )

        # Per-stratum PCCT stats
        st.markdown("#### Per tissue stratum")
        pcct_stratum_rows = []
        for si, (sname, lo, hi) in enumerate(strata_bounds):
            sv = pcct_vals[strata_labels == si]
            if len(sv) == 0:
                continue
            row = compute_stats(sv, "PCCT")
            row["Stratum"] = sname
            row["HU range"] = f"{'−∞' if lo is None else lo} → {'∞' if hi is None else hi}"
            pcct_stratum_rows.append(row)

        df_pcct_strata = pd.DataFrame(pcct_stratum_rows)[
            ["Stratum", "HU range", "N voxels", "Mean HU", "STD HU", "Min HU", "Max HU", "Median HU"]
        ]
        st.dataframe(df_pcct_strata, use_container_width=True)

        # HU histogram PCCT (overlaid with CT)
        fig_pcct, ax_pcct = plt.subplots(figsize=(8, 3))
        ax_pcct.hist(ct_vals, bins=120, color=CT_COLOR, alpha=0.5, label="CT", edgecolor="none")
        ax_pcct.hist(pcct_vals, bins=120, color=PCCT_COLOR, alpha=0.5, label="PCCT", edgecolor="none")
        ax_pcct.set_xlabel("HU")
        ax_pcct.set_ylabel("Voxel count")
        ax_pcct.set_title(f"HU distribution overlay — {selected_roi}")
        ax_pcct.legend(facecolor="#161b22", edgecolor=PLOT_GRID, labelcolor=PLOT_FG)
        apply_dark_style(ax_pcct, fig_pcct)
        fig_pcct.tight_layout()
        st.pyplot(fig_pcct)
        plt.close(fig_pcct)

        # ==============================================
        # SECTION 3 — COMPARISON
        # ==============================================
        st.markdown('<hr class="section-rule">', unsafe_allow_html=True)
        st.markdown("## ⚖️ Section 3 — CT vs PCCT Comparison")

        # ── 3a. Global difference stats ──
        st.markdown("### 3a. Global Difference Stats (CT − PCCT)")

        mean_diff = float(np.mean(diff_vals))
        std_diff = float(np.std(diff_vals))
        loa_upper = mean_diff + 1.96 * std_diff
        loa_lower = mean_diff - 1.96 * std_diff

        d1, d2, d3, d4 = st.columns(4)
        for col, label, val, color in [
            (d1, "Mean diff (HU)", f"{mean_diff:+.2f}", DIFF_COLOR),
            (d2, "STD of diff", f"{std_diff:.2f}", PLOT_FG),
            (d3, "LoA upper (+1.96σ)", f"{loa_upper:+.2f}", "#3fb950"),
            (d4, "LoA lower (−1.96σ)", f"{loa_lower:+.2f}", "#f85149"),
        ]:
            col.markdown(
                f'<div class="metric-card"><div class="metric-label">{label}</div>'
                f'<div class="metric-value" style="color:{color};">{val}</div></div>',
                unsafe_allow_html=True,
            )

        # ── 3b. Per-stratum difference ──
        st.markdown("### 3b. Per-Stratum Difference (CT − PCCT)")

        comp_rows = []
        for si, (sname, lo, hi) in enumerate(strata_bounds):
            sv_ct = ct_vals[strata_labels == si]
            sv_pcct = pcct_vals[strata_labels == si]
            sv_diff = sv_ct - sv_pcct
            if len(sv_diff) == 0:
                continue
            comp_rows.append({
                "Stratum": sname,
                "N voxels": len(sv_diff),
                "CT Mean HU": round(float(np.mean(sv_ct)), 2),
                "PCCT Mean HU": round(float(np.mean(sv_pcct)), 2),
                "Mean diff": round(float(np.mean(sv_diff)), 2),
                "STD diff": round(float(np.std(sv_diff)), 2),
                "LoA upper": round(float(np.mean(sv_diff) + 1.96 * np.std(sv_diff)), 2),
                "LoA lower": round(float(np.mean(sv_diff) - 1.96 * np.std(sv_diff)), 2),
            })

        df_comp = pd.DataFrame(comp_rows)
        st.dataframe(df_comp, use_container_width=True)

        # ── 3c. Bland–Altman plot ──
        st.markdown("### 3c. Bland–Altman Plot")

        if "Per slice" in ba_level:
            # Aggregate per slice
            unique_slices = np.unique(vox_z_idx)
            ba_means, ba_diffs = [], []
            for si in unique_slices:
                m = vox_z_idx == si
                if m.sum() == 0:
                    continue
                ba_means.append(float(np.mean((ct_vals[m] + pcct_vals[m]) / 2)))
                ba_diffs.append(float(np.mean(ct_vals[m] - pcct_vals[m])))
            ba_means = np.array(ba_means)
            ba_diffs = np.array(ba_diffs)
            ba_label = "slice-level mean"
        else:
            # Subsample voxels
            n_pts = min(ba_max_pts, len(ct_vals))
            idx = np.random.choice(len(ct_vals), n_pts, replace=False)
            ba_means = (ct_vals[idx] + pcct_vals[idx]) / 2
            ba_diffs = ct_vals[idx] - pcct_vals[idx]
            ba_label = f"voxel-level (n={n_pts:,})"

        ba_mean_diff = float(np.mean(ba_diffs))
        ba_std_diff = float(np.std(ba_diffs))
        ba_loa_up = ba_mean_diff + 1.96 * ba_std_diff
        ba_loa_lo = ba_mean_diff - 1.96 * ba_std_diff

        fig_ba, ax_ba = plt.subplots(figsize=(8, 4))
        ax_ba.scatter(ba_means, ba_diffs, color=DIFF_COLOR, alpha=0.6, s=18, edgecolors="none",
                      label=ba_label)
        ax_ba.axhline(ba_mean_diff, color="#e3b341", linewidth=1.5,
                      linestyle="-", label=f"Mean diff: {ba_mean_diff:+.2f} HU")
        ax_ba.axhline(ba_loa_up, color="#3fb950", linewidth=1.2, linestyle="--",
                      label=f"+1.96σ: {ba_loa_up:+.2f} HU")
        ax_ba.axhline(ba_loa_lo, color="#f85149", linewidth=1.2, linestyle="--",
                      label=f"−1.96σ: {ba_loa_lo:+.2f} HU")
        ax_ba.axhline(0, color=PLOT_FG, linewidth=0.5, linestyle=":", alpha=0.4)
        ax_ba.set_xlabel("Mean of CT & PCCT (HU)")
        ax_ba.set_ylabel("CT − PCCT (HU)")
        ax_ba.set_title(f"Bland–Altman — {selected_roi}  [{ba_label}]")
        ax_ba.legend(facecolor="#161b22", edgecolor=PLOT_GRID, labelcolor=PLOT_FG, fontsize=8)
        apply_dark_style(ax_ba, fig_ba)
        fig_ba.tight_layout()
        st.pyplot(fig_ba)
        plt.close(fig_ba)

        # ── 3d. Difference heatmap per slice ──
        st.markdown("### 3d. Voxel-wise Difference Heatmap (CT − PCCT)")
        st.markdown(
            '<div class="info-box">Each slice shows only voxels inside the eroded ROI mask. '
            'Color encodes CT − PCCT in HU. Grey = outside mask.</div>',
            unsafe_allow_html=True,
        )

        # Find slices that have ROI voxels
        active_slices = sorted(np.unique(np.where(combined_mask)[2]))

        # Build difference volume
        diff_vol = np.full(ct_vol.shape, np.nan, dtype=np.float32)
        diff_vol[combined_mask] = diff_vals

        # Symmetric color scale
        vmax = float(np.nanpercentile(np.abs(diff_vals), 98))
        vmin = -vmax

        n_cols = 4
        n_rows_grid = int(np.ceil(len(active_slices) / n_cols))

        fig_hm, axes = plt.subplots(
            n_rows_grid, n_cols,
            figsize=(n_cols * 3, n_rows_grid * 3),
        )
        fig_hm.patch.set_facecolor(PLOT_BG)
        axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

        cmap = plt.cm.RdBu_r

        for ax_idx, ax in enumerate(axes_flat):
            ax.set_facecolor(PLOT_BG)
            ax.axis("off")
            if ax_idx >= len(active_slices):
                continue
            sl = active_slices[ax_idx]
            # Background: CT in grayscale, windowed to soft tissue
            bg = ct_vol[:, :, sl]
            ax.imshow(
                bg,
                cmap="gray",
                vmin=-200, vmax=400,
                origin="upper",
                interpolation="nearest",
            )
            # Overlay: difference, masked
            diff_slice = diff_vol[:, :, sl].copy()
            rgba = cmap((diff_slice - vmin) / (vmax - vmin + 1e-9))
            rgba[..., 3] = np.where(combined_mask[:, :, sl], 0.75, 0.0)
            ax.imshow(rgba, origin="upper", interpolation="nearest")
            ax.set_title(f"z={sl}", color=PLOT_FG, fontsize=7, pad=2)

        # Colorbar
        sm = plt.cm.ScalarMappable(
            cmap=cmap, norm=mcolors.Normalize(vmin=vmin, vmax=vmax)
        )
        sm.set_array([])
        cbar = fig_hm.colorbar(sm, ax=axes_flat, fraction=0.015, pad=0.02)
        cbar.set_label("CT − PCCT (HU)", color=PLOT_FG, fontsize=9)
        cbar.ax.tick_params(colors=PLOT_FG, labelsize=7)

        fig_hm.suptitle(
            f"Difference heatmap — {selected_roi}  (erosion: {erosion_voxels} vx)",
            color=PLOT_FG, fontsize=11, y=1.005,
        )
        st.pyplot(fig_hm)
        plt.close(fig_hm)

        # ── 3e. Per-stratum Bland–Altman ──
        st.markdown("### 3e. Per-Stratum Bland–Altman")

        n_strata_active = len(df_comp)
        if n_strata_active > 0:
            fig_sba, axes_sba = plt.subplots(
                1, n_strata_active,
                figsize=(4.5 * n_strata_active, 4),
                squeeze=False,
            )
            fig_sba.patch.set_facecolor(PLOT_BG)

            for si_plot, (si, (sname, lo, hi)) in enumerate(
                [(si, b) for si, b in enumerate(strata_bounds)
                 if (ct_vals[strata_labels == si]).size > 0]
            ):
                ax_s = axes_sba[0, si_plot]
                sv_ct = ct_vals[strata_labels == si]
                sv_pcct = pcct_vals[strata_labels == si]
                sv_z = vox_z_idx[strata_labels == si]

                if "Per slice" in ba_level:
                    s_means, s_diffs = [], []
                    for sli in np.unique(sv_z):
                        m = sv_z == sli
                        s_means.append(float(np.mean((sv_ct[m] + sv_pcct[m]) / 2)))
                        s_diffs.append(float(np.mean(sv_ct[m] - sv_pcct[m])))
                    s_means = np.array(s_means)
                    s_diffs = np.array(s_diffs)
                else:
                    n_s = min(ba_max_pts, len(sv_ct))
                    idx_s = np.random.choice(len(sv_ct), n_s, replace=False)
                    s_means = (sv_ct[idx_s] + sv_pcct[idx_s]) / 2
                    s_diffs = sv_ct[idx_s] - sv_pcct[idx_s]

                s_md = float(np.mean(s_diffs))
                s_sd = float(np.std(s_diffs))

                ax_s.scatter(s_means, s_diffs, color=DIFF_COLOR, alpha=0.6, s=14, edgecolors="none")
                ax_s.axhline(s_md, color="#e3b341", linewidth=1.4, label=f"Mean: {s_md:+.2f}")
                ax_s.axhline(s_md + 1.96 * s_sd, color="#3fb950", linewidth=1.1,
                             linestyle="--", label=f"+1.96σ: {s_md + 1.96 * s_sd:+.2f}")
                ax_s.axhline(s_md - 1.96 * s_sd, color="#f85149", linewidth=1.1,
                             linestyle="--", label=f"−1.96σ: {s_md - 1.96 * s_sd:+.2f}")
                ax_s.axhline(0, color=PLOT_FG, linewidth=0.4, linestyle=":", alpha=0.4)
                ax_s.set_title(sname, fontsize=9)
                ax_s.set_xlabel("Mean HU", fontsize=8)
                ax_s.set_ylabel("CT − PCCT (HU)", fontsize=8)
                ax_s.legend(facecolor="#161b22", edgecolor=PLOT_GRID,
                            labelcolor=PLOT_FG, fontsize=7)
                apply_dark_style(ax_s, fig_sba)

            fig_sba.tight_layout()
            st.pyplot(fig_sba)
            plt.close(fig_sba)

        # ==============================================
        # DOWNLOADS
        # ==============================================
        st.markdown('<hr class="section-rule">', unsafe_allow_html=True)
        st.markdown("### ⬇️ Downloads")

        # Per-voxel CSV
        rows_x, rows_y, rows_z = np.where(combined_mask)
        df_voxels = pd.DataFrame({
            "voxel_row": rows_x,
            "voxel_col": rows_y,
            "slice_idx": rows_z,
            "CT_HU": ct_vals,
            "PCCT_HU": pcct_vals,
            "Diff_CT_minus_PCCT": diff_vals,
            "Stratum": [strata_bounds[i][0] for i in strata_labels],
        })

        # Summary CSV
        df_summary_ct = df_ct_strata.copy()
        df_summary_ct.insert(0, "Scanner", "CT")
        df_summary_pcct = df_pcct_strata.copy()
        df_summary_pcct.insert(0, "Scanner", "PCCT")
        df_summary = pd.concat([df_summary_ct, df_summary_pcct], ignore_index=True)

        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            st.download_button(
                "⬇ Per-voxel CSV (CT, PCCT, diff, stratum)",
                df_voxels.to_csv(index=False),
                f"voxels_{selected_roi}.csv",
                "text/csv",
            )
        with col_dl2:
            st.download_button(
                "⬇ Summary stats CSV (CT + PCCT per stratum)",
                df_summary.to_csv(index=False),
                f"summary_{selected_roi}.csv",
                "text/csv",
            )

        st.success("✅ Analysis complete.")

elif zip_ct and not zip_pcct:
    st.info("Upload the PCCT ZIP to proceed.")
elif zip_pcct and not zip_ct:
    st.info("Upload the CT ZIP to proceed.")
else:
    st.markdown(
        '<div class="info-box">Upload both ZIPs above to begin. '
        'CT is treated as the reference standard.</div>',
        unsafe_allow_html=True,
    )
