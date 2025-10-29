# Heartbeat Time-Series Classification (Notebook)

This notebook walks through a practical pipeline for classifying heartbeat audio:
1) **Visualize raw signals** (normal vs abnormal)
2) **Denoise & simplify** with rectification and rolling averages (signal envelope)
3) **Engineer features**:
   - Global stats (mean, std, max of envelopes)
   - **Tempo** features from `librosa`
   - **Spectrogram** features: spectral centroid & bandwidth
4) **Train & evaluate** a **LinearSVC** classifier

> The emphasis is on *feature engineering* for time series, using audio as the example domain.

---

## Contents (by steps)

- **Data loading**  
  - CSVs: `invariance_normal.csv`, `invariance_abnormal.csv`, `audio.csv`  
  - TXT (converted to CSV in-notebook): `normal_full.txt`, `abnormal_full.txt`, etc.  
  - HDF5 bundle: `audio_munged.hdf5` with keys:
    - `h5io/key_data` → (time-indexed audio columns)
    - `h5io/key_meta` → labels and metadata
    - `h5io/key_sfreq` → sampling frequency (e.g., 2205 Hz)

- **Visualization**  
  - Multi-panel raw waveform plots  
  - Envelope plots after rectification + rolling mean  
  - Spectrograms (`librosa.stft` + `amplitude_to_db`) with overlays

- **Feature engineering**  
  - Envelope stats: mean / std / max per recording  
  - Tempo estimates & summary stats  
  - Spectral centroid & bandwidth (sequence and summary)

- **Model**  
  - `sklearn.svm.LinearSVC` with simple train/test split and accuracy

---

## Quickstart

```bash
# 1) Create and activate a virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# 2) Install dependencies
pip install -r requirements.txt

#3) Git Clone
git clone https://github.com/Joe-Naz01/time_series.git
cd time_series

# 3) Launch Jupyter
jupyter lab   # or: jupyter notebook
