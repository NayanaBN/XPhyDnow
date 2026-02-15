import os
import numpy as np
import scipy.ndimage
from tqdm import tqdm

# SETTINGS
project_root = r"C:\Users\Nayana\PycharmProjects\X-PhyDNow"
data_dir = os.path.join(project_root, "data", "processed")
tcwv_dir = os.path.join(data_dir, "physics_tcwv")
sat_dir = os.path.join(data_dir, "satellite_ir")

print("✨ GENERATING FINAL SATELLITE IMAGERY (The 'Perfect' Look)...")

if not os.path.exists(sat_dir): os.makedirs(sat_dir)

# Get all physics files
files = sorted([f for f in os.listdir(tcwv_dir) if f.endswith('.npy')])

for f in tqdm(files):
    # 1. Load Moisture (Physics)
    tcwv = np.load(os.path.join(tcwv_dir, f))
    if len(tcwv.shape) == 3: tcwv = tcwv.squeeze()

    # 2. Normalize (0 to 1)
    min_v, max_v = np.min(tcwv), np.max(tcwv)
    if max_v - min_v < 1e-5:
        norm = np.zeros_like(tcwv)
    else:
        norm = (tcwv - min_v) / (max_v - min_v)

    # 3. Add "Sensor Noise" (The Grainy Texture)
    # This creates the visual complexity of real satellite data
    noise = np.random.normal(0, 0.25, norm.shape)  # 25% Noise
    synthetic_ir = norm + noise

    # 4. Sharpen (To mimic high-res sensors)
    blurred = scipy.ndimage.gaussian_filter(synthetic_ir, sigma=1)
    sharpened = synthetic_ir + (synthetic_ir - blurred) * 2.0

    # 5. Clip to valid range
    final_sat = np.clip(sharpened, 0.0, 1.0)

    # 6. Save
    out_name = f.replace("tcwv", "ir")
    np.save(os.path.join(sat_dir, out_name), final_sat)

print("✅ DONE. Your folder now contains the clean, textured cloud data.")