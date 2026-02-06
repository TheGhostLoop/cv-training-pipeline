# CV Training Pipeline Generator 🚀

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A **production-ready, configurable deep learning pipeline** for image classification that automatically analyzes datasets, recommends architectures, and generates **Google Colab-ready training scripts** using transfer learning.

Built with **engineering rigor** — focusing on correctness, usability, and real-world robustness over toy benchmarks.

---

## 🎯 Problem Statement

Building image classifiers shouldn't break because of:
- ❌ Inconsistent dataset structures (single-folder vs pre-split)
- ❌ Manual boilerplate for every project
- ❌ Fragile training scripts that fail on edge cases
- ❌ No clear separation between training and inference

**This pipeline solves all of that.**

---

## ✨ What Makes This Different

Unlike typical ML projects that focus only on accuracy:

- 🔍 **Intelligent Dataset Analysis** — Automatically detects structure, validates ImageFolder format, analyzes class distribution
- 🧠 **Smart Recommendations** — Suggests optimal model, training approach, batch size, and epochs based on dataset characteristics
- 🎨 **Interactive UI** — Streamlit-powered interface for dataset upload, visualization, and configuration
- 📄 **Code Generation** — Produces clean, executable `train.py` and `inference.py` scripts
- ☁️ **Colab-First Design** — Generated scripts work out-of-the-box on Google Colab (GPU-friendly)
- 🛡️ **Robust Error Handling** — Handles messy real-world datasets gracefully

---

## 🏗️ Architecture Overview
```
┌─────────────────┐
│  Dataset Input  │ ← Upload ZIP or URL
└────────┬────────┘
         ↓
┌─────────────────┐
│ Structure Check │ ← Detect single-folder vs pre-split
└────────┬────────┘
         ↓
┌─────────────────┐
│    Analysis     │ ← Class distribution, recommendations
└────────┬────────┘
         ↓
┌─────────────────┐
│ Configuration   │ ← Override model/training settings
└────────┬────────┘
         ↓
┌─────────────────┐
│ Script Gen      │ ← Generate train.py + inference.py
└────────┬────────┘
         ↓
┌─────────────────┐
│  Colab Exec     │ ← Train on GPU, save model
└─────────────────┘
```

---

## 🚀 Features

### 📂 Dataset Handling
- ✅ Automatic ImageFolder structure validation
- ✅ Support for **single-folder** and **pre-split** datasets
- ✅ Smart detection of `train/val/test` folder aliases
- ✅ Class distribution visualization with recommendations
- ✅ Upload via ZIP file or direct URL

### 🧠 Model & Training
- ✅ Transfer learning with **ResNet18** and **MobileNetV2**
- ✅ Two training modes:
  - **Final Head Only** (fast, small datasets)
  - **Layer4 + Final Head** (better accuracy, larger datasets)
- ✅ Adaptive learning rate with `ReduceLROnPlateau`
- ✅ Early stopping at target accuracy
- ✅ Automatic class mapping (`class_mapping.json`)

### 📊 Analysis & Insights
- ✅ Class distribution bar charts
- ✅ Dataset statistics (total images, classes, balance)
- ✅ Training curves (accuracy over epochs)
- ✅ Automatic hyperparameter recommendations

### 🔧 Generated Scripts
- ✅ **`train.py`** — Fully configured training script
- ✅ **`inference.py`** — Interactive prediction script
- ✅ Clean, well-commented, production-ready code
- ✅ Works seamlessly in Google Colab

---

## 📦 Installation

### Local Setup
```bash
# Clone the repository
git clone https://github.com/theghostloop/cv-training-pipeline.git
cd cv-training-pipeline

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Requirements
```txt
streamlit>=1.28.0
torch>=2.0.0
torchvision>=0.15.0
pandas>=2.0.0
matplotlib>=3.7.0
Pillow>=10.0.0
requests>=2.31.0
```

---

## 🎮 Quick Start

### 1️⃣ Launch Streamlit App
```bash
streamlit run app.py
```

### 2️⃣ Upload Your Dataset

**Option A:** Upload ZIP file
- Drag & drop a ZIP containing your dataset

**Option B:** Provide URL
- Enter direct download link to dataset ZIP

### 3️⃣ Review Analysis

The app will:
- Detect dataset structure
- Show class distribution
- Recommend model & training settings

### 4️⃣ Configure (Optional)

Override recommendations:
- Model: ResNet18 vs MobileNetV2
- Training mode: Final Head vs Layer4 + Head
- Batch size: 16-64
- Epochs: 1-50

### 5️⃣ Download Scripts

Click **"Generate Scripts"** to download:
- `train.py` — Training script
- `inference.py` — Prediction script

### 6️⃣ Train in Colab
```python
# Upload train.py to Colab
!python train.py

# Follow prompts to load dataset
# Training begins automatically
```

### 7️⃣ Run Inference
```python
# After training completes
!python inference.py

# Predict on new images
```

---

## 📁 Supported Dataset Formats

### Single-Folder Structure
```
dataset/
├── class_1/
│   ├── img1.jpg
│   ├── img2.jpg
├── class_2/
│   ├── img3.jpg
└── class_3/
    ├── img4.jpg
```

### Pre-Split Structure
```
dataset/
├── train/
│   ├── class_1/
│   ├── class_2/
├── val/
│   ├── class_1/
│   ├── class_2/
└── test/  (optional)
```

**Supported aliases:**
- Train: `train`, `training`
- Val: `val`, `valid`, `validation`
- Test: `test`, `testing`

---

## 🧪 Example Workflow
```python
# 1. User uploads cat/dog dataset (1000 images, 2 classes)
# 2. App detects: single-folder structure
# 3. Recommendations:
#    - Model: MobileNetV2 (lighter for binary classification)
#    - Mode: Final Head Only
#    - Batch: 32
#    - Epochs: 15
# 4. User downloads train.py
# 5. Runs in Colab → achieves 95% accuracy
# 6. Saves best_model.pth
# 7. Uses inference.py for predictions
```

---

## 📊 Output Files

After training:
```
outputs/
├── best_model.pth           # Trained model weights
├── class_mapping.json       # Class index mapping
└── training_curves.png      # Accuracy/loss plots
```

---

## 🔬 Technical Deep Dive

### Dataset Structure Detection
```python
def find_dataset_root(base_dir):
    """
    Intelligently detects:
    1. Pre-split (train/val/test folders)
    2. Single-folder (class subfolders)
    Handles nested ZIPs and edge cases
    """
```

### Smart Recommendations
```python
def get_recommended_details(df):
    """
    Based on:
    - Total images
    - Number of classes
    - Class balance
    
    Returns:
    - Model choice
    - Training approach
    - Batch size
    - Epoch count
    """
```

### Transfer Learning Strategy

**ResNet18:**
- Final Head Only: Freeze all, train `fc` layer
- Layer4 + Head: Unfreeze `layer4` + `fc`

**MobileNetV2:**
- Final Head Only: Train `classifier` only
- Deep Finetune: Unfreeze last 6 feature layers

---

## ⚙️ Configuration Options

| Parameter | Options | Description |
|-----------|---------|-------------|
| **Model** | ResNet18, MobileNetV2 | Architecture choice |
| **Training Mode** | Final Head Only, Layer4 + Head | Freezing strategy |
| **Batch Size** | 16, 24, 32, 40, 48, 56, 64 | Training batch size |
| **Epochs** | 1-50 | Maximum training epochs |

---

## ⚠️ Known Limitations (By Design)

This is a **learning-focused, prototyping pipeline**, not a production ML system:

- ❌ No hyperparameter search (Grid/Random/Bayesian)
- ❌ No ensemble methods
- ❌ No data augmentation beyond basic transforms
- ❌ No distributed training
- ❌ No model versioning (MLflow/Weights & Biases)
- ❌ No deployment (FastAPI/TensorFlow Serving)

**Why?** These omissions keep the codebase clean, understandable, and focused on **core engineering principles**.

---

## 🧠 Design Philosophy

1. **Simplicity > Complexity** — No unnecessary abstractions
2. **Robustness > Assumptions** — Handle messy real-world data
3. **Reproducibility > One-offs** — Generated scripts are version-controllable
4. **Education > Black Boxes** — Code is readable and well-commented

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| **Backend** | Python 3.8+ |
| **ML Framework** | PyTorch 2.0+ |
| **Computer Vision** | Torchvision |
| **UI** | Streamlit |
| **Data Viz** | Matplotlib, Pandas |
| **Image Processing** | Pillow |

---

## 📈 Future Enhancements

- [ ] Support for multi-label classification
- [ ] Custom augmentation policies
- [ ] Integration with Hugging Face Hub
- [ ] Export to ONNX/TorchScript
- [ ] Gradio interface alternative
- [ ] Automatic dataset balancing

---

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repo
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📜 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- PyTorch team for excellent documentation
- Streamlit for making ML UIs accessible
- ImageNet for pretrained weights

---

## 📧 Contact

**Your Name**  
📧 Email: princeverma2005@gmail.com  
🐙 GitHub: [@TheGhostLoop](https://github.com/TheGhostLoop)  
💼 LinkedIn: Prince Verma(https://www.linkedin.com/in/prince-verma-80a94b374?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app)

---

## ⚖️ Ethical Note

This project is intended for **educational and research purposes**. It is **not designed** for:
- Medical diagnosis
- Biometric identification
- Surveillance systems
- Safety-critical applications

Use responsibly and ensure compliance with relevant data protection regulations.

---

**⭐ If this project helped you, please star the repo!**

---

## 📝 Additional Files to Add

### `.gitignore`
```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/

# Dataset folders
datasets/
outputs/

# Streamlit
.streamlit/

# OS
.DS_Store
Thumbs.db
```

### `requirements.txt`
```
streamlit==1.28.0
torch==2.0.1
torchvision==0.15.2
pandas==2.0.3
matplotlib==3.7.2
Pillow==10.0.0
requests==2.31.0
```