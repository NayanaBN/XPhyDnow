## X-PhyDNow: Explainable Physics-Enhanced Deep Learning for Nowcasting

**X-PhyDNow** is a physics-informed deep learning framework designed to predict short-term rainfall intensity (0-3 hours) using satellite imagery. Unlike standard "black-box" AI models, X-PhyDNow integrates thermodynamic laws (CAPE, TCWV) directly into the neural network, ensuring predictions are scientifically valid, interpretable, and robust to sensor failures.

---

##  Key Features

* ** Physics-Informed Dual-Encoder:** Fuses high-resolution satellite vision (Himawari-9) with atmospheric physics (ERA5 Reanalysis) to prevent "hallucinations" in dry conditions.
* ** Autonomous Physics Fallback:** Detects satellite sensor failures (black images/noise) and automatically switches to a "Physics-Only" mode to maintain situational awareness.
* ** Explainable AI (XAI):** Features a real-time **Physical Consistency Score (PCS)** gauge and **Branch-Wise Contribution** charts, showing forecasters *why* a storm is predicted.
* ** Lightweight Deployment:** Optimized via quantization to run on standard **CPUs (Intel i5)** with <3s latency, making it accessible for low-resource meteorological centers.
* ** Flood Risk Alerts:** Automatically flags regions with rainfall intensity >50mm/hr for potential flash floods.

---

##  System Architecture

The model uses a **Dual-Encoder ConvLSTM** architecture:

1. **Vision Encoder:** Extracts spatiotemporal cloud textures from Infrared (IR) Satellite data.
2. **Physics Encoder:** Extracts thermodynamic states from atmospheric variables (CAPE, Moisture).
3. **PCS-Gated Fusion:** A custom attention layer that dynamically weights the vision stream based on physical validity.

*(Replace with actual architecture diagram from Chapter 5)*

---

##  Installation

### Prerequisites

* Python 3.9+
* Git

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/X-PhyDNow.git
cd X-PhyDNow

```

### 2. Create a Virtual Environment

```bash
# Windows
python -m venv env
env\Scripts\activate

# Mac/Linux
python3 -m venv env
source env/bin/activate

```

### 3. Install Dependencies

```bash
pip install -r requirements.txt

```

---

##  Dataset Structure

The system expects data in the following directory structure. (Note: Raw data is not included in the repo due to size).

```
data/
├── raw/
│   ├── himawari/       # Raw Satellite NetCDF files
│   └── era5/           # Raw Physics (CAPE/TCWV) NetCDF files
├── processed/
│   ├── train/          # Aligned .npy tensors (2023-2024)
│   └── test/           # Aligned .npy tensors (2025)
└── weights/
    └── best_model_csi_0.72.pth  # Trained Model Checkpoint

```

---

##  Usage

### 1. Run the Dashboard

To launch the interactive forecaster interface:

```bash
streamlit run app.py

```

The dashboard will open in your browser at `http://localhost:8501`.

### 2. Run Inference (CLI)

To generate a forecast for a specific timestamp via command line:

```bash
python inference.py --date "2025-11-04" --time "14:30" --mode "dual_encoder"

```

---

##  Performance & Results

Tested on the **Indian Monsoon Dataset (2025)**:

| Metric | Score | Target | Description |
| --- | --- | --- | --- |
| **CSI** | **0.72** | ≥ 0.70 | Critical Success Index (Rain Detection) |
| **FAR** | **0.14** | ≤ 0.20 | False Alarm Rate (Hallucination Check) |
| **SSIM** | **0.86** | ≥ 0.80 | Structural Similarity (Image Quality) |
| **Latency** | **2.8s** | ≤ 3.0s | CPU Inference Time per Step |

---

##  Screenshots




<img width="1164" height="716" alt="Screenshot 2026-01-18 200756" src="https://github.com/user-attachments/assets/3aa5c022-2c55-4cc1-a1f5-8ab878282827" />




<img width="1920" height="1080" alt="Screenshot 2026-01-27 123317" src="https://github.com/user-attachments/assets/55835b08-f0f8-4fe7-853b-0482d36412e2" />




<img width="1025" height="736" alt="Screenshot 2026-01-18 192616" src="https://github.com/user-attachments/assets/829c0da5-aaee-487e-9fa5-04e07d39a2f6" />

---

##  Future Scope

* Integration of **Doppler Weather Radar (DWR)** data for microphysical precision.
* Extension to **Lightning Prediction** using ice-flux proxies.
* Deployment on **Edge Devices (NVIDIA Jetson)** for UAV-based monitoring.

---


## Contributors

* **Your Name** Nayana B N
* **Teammate Name** Tushar L G 
* **Guide Name** Ms Meena Kumari K S

---

##  Acknowledgements

* **JAXA / SSEC** for Himawari-9 Satellite Data.
* **ECMWF** for ERA5 Reanalysis Data.
* **Google Colab** for GPU computational resources.
