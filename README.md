# ROI_analysis_v2
CT · PCCT ROI Comparator
========================

A Streamlit application for voxel-wise HU comparison between a reference CT scan
and a Photon-Counting CT (PCCT) scan for a single patient and a single ROI.
Designed for scanner harmonisation and technology superiority studies in the
head and neck region.

Registration between the two scans is assumed to be complete prior to upload.
The app resamples the PCCT volume onto the CT grid internally.


REQUIREMENTS
------------
Python 3.10 or higher is recommended.

Install all dependencies with:

    pip install -r requirements.txt


HOW TO RUN
----------
    streamlit run ct_pcct_comparator.py

The app will open in your default browser at http://localhost:8501


INPUT FORMAT
------------
Two ZIP files are required, one per scanner:

  - CT ZIP   : contains all CT DICOM slices (.dcm) and the corresponding
               RTSTRUCT file for the patient, in any folder structure.

  - PCCT ZIP : same structure for the PCCT scan of the same patient.

The app will automatically detect CT and RTSTRUCT modalities by reading
the DICOM Modality tag. No specific folder naming convention is required.

Important: both scans must be co-registered before upload. The app applies
an identity transform when resampling PCCT onto the CT grid — it does NOT
perform registration itself.


CONFIGURATION (sidebar)
-----------------------
Mask erosion (voxels)
    Erodes the ROI mask inward by N voxels before analysis to exclude boundary
    artefacts caused by residual misregistration at tissue/air interfaces.
    Default: 2 voxels. Set to 0 to disable.

Tissue HU thresholds
    Define the upper HU boundary for each tissue stratum:
      - Air         : below the air upper bound      (default −900 HU)
      - Fat         : air upper → fat upper           (default −100 HU)
      - Soft Tissue : fat upper → soft tissue upper   (default  +200 HU)
      - Bone/Dense  : above soft tissue upper
    These thresholds are calibrated from the CT (reference) values and
    should be adjusted per study and anatomy.

Bland–Altman aggregation level
    Per slice (recommended): one point per slice = mean HU of all ROI
    voxels in that slice. Clean and interpretable.
    Per voxel (subsampled): one point per voxel, subsampled to a
    user-defined maximum for performance.


OUTPUT SECTIONS
---------------
Section 1 — CT Raw Results
    Global metrics (mean, STD, min, max, median HU) and per-stratum
    breakdown for the CT scan. Includes HU histogram.

Section 2 — PCCT Raw Results
    Same structure as Section 1 for the PCCT scan. HU histogram is
    overlaid with the CT distribution for direct visual comparison.

Section 3 — CT vs PCCT Comparison
    3a. Global difference statistics (CT − PCCT):
        mean difference, STD of difference, limits of agreement (±1.96σ).

    3b. Per-stratum difference table:
        mean and STD of CT − PCCT broken down by tissue type, with
        limits of agreement per stratum.

    3c. Bland–Altman plot (global):
        x = mean of CT and PCCT HU per point
        y = CT − PCCT
        Lines shown for mean difference and ±1.96σ limits of agreement.
        Positive values indicate CT reads higher than PCCT.

    3d. Voxel-wise difference heatmap:
        One panel per active slice. CT image shown as greyscale background
        (windowed −200 to +400 HU). Colour overlay (RdBu_r) encodes
        CT − PCCT in HU, restricted to the eroded ROI mask.

    3e. Per-stratum Bland–Altman:
        Separate Bland–Altman panel for each active tissue stratum.

Downloads
    Per-voxel CSV  : voxel coordinates, CT_HU, PCCT_HU, CT−PCCT, stratum.
    Summary CSV    : per-stratum stats for CT and PCCT side by side.


NOTES
-----
- Difference convention: CT − PCCT. Positive = CT reads higher.
- CT is treated as the reference standard throughout.
- The combined ROI mask is the intersection of the CT and PCCT masks,
  both brought onto the CT grid, then eroded.
- Tissue strata are assigned using CT HU values as the reference.
- The app uses @st.cache_data on volume loading so re-runs within the
  same session do not reload DICOM files from disk.


DEPENDENCIES
------------
streamlit       — web application framework
pydicom         — DICOM file reading
numpy           — array operations
pandas          — tabular data and CSV export
scipy           — binary erosion (ndimage)
scikit-image    — polygon rasterisation for contour-to-mask
SimpleITK       — image resampling onto reference grid
matplotlib      — all plots and heatmaps
