# GTA V Autopilot

## 🛠️ Development Environment
* **Conda Env:** `gta5_autopilot`
* **Python:** 3.10

### Setup
```bash
conda create -n gta5_autopilot python=3.10 -y
conda activate gta5_autopilot
pip install numpy opencv-python pillow mss bettercam
```
## 💻 Hardware Specifications
* **CPU:** AMD Ryzen 5600X
* **GPU:** Nvidia RTX 5070 Ti
* **RAM:** 32 GB
* **Resolution:** 1280x720 (Capture Area)

| Method | Avg FPS | Status |
| :--- | :--- | :--- |
| **Pillow** | 12 FPS | Baseline (Low Performance) |