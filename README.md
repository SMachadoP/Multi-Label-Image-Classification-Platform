# Multi-Label Classification System: A Deep Learning Platform for Continuous Object Recognition
**Python** · **TensorFlow/Keras** · **FastAPI** · **MLflow** · **License**

This documentation describes the architecture and capabilities of the **Multi-Label Classification System**, a production-ready platform designed to identify multiple objects within images using state-of-the-art Transfer Learning techniques and Continuous Learning paradigms.

**Project Status:** Active Development - Production Ready  
**Focus:** Multi-Object Recognition, Transfer Learning, Continuous Model Retraining, MLflow Experiment Tracking.

---

## 👨‍💻 Engineering Profile
**Multi-Label Classification Team** | Computer Vision & Deep Learning

Demonstrated expertise:

🏗️ **Modular Architecture:** Clean separation between data preparation, training, and inference pipelines.  
🤖 **Transfer Learning:** Implementation of ResNet50, EfficientNetB0, and MobileNetV2 for efficient feature extraction.  
🧠 **Continuous Learning:** Incremental retraining system with user feedback integration.  
📊 **Experiment Tracking:** MLflow-based model versioning and metrics visualization.  
📡 **REST API:** FastAPI server with real-time prediction and retraining endpoints.  
🎯 **Production Ready:** Automated model loading, timestamp-based versioning, and zero-downtime updates.

---

## 🎯 Problem & Solution

### The Challenge
Traditional image classification systems are limited to single-label predictions and require complete retraining cycles when new data becomes available. Real-world applications demand:

- **Multi-object recognition** in complex scenes  
- **Adaptive learning** from user corrections  
- **Rapid deployment** of retrained models  
- **Transparent experiment tracking** for reproducibility  

### The Solution: Multi-Label Classification System
An intelligent computer vision platform that:

✅ **Identifies multiple objects simultaneously** using sigmoid activation (person, chair, dog, sofa).  
✅ **Compares 3 state-of-the-art architectures** to select the optimal model for production.  
✅ **Retrains incrementally** with user-provided corrections, saving models with automatic timestamps.  
✅ **Tracks all experiments** via MLflow, enabling model comparison and rollback.  
✅ **Serves predictions** through a REST API with interactive web interface.  

---

## ✨ Key Features

🚀 **Transfer Learning Pipeline:** Leverages pre-trained ImageNet weights for rapid convergence.  
🧠 **Multi-Architecture Comparison:** ResNet50, EfficientNetB0, MobileNetV2 trained and evaluated side-by-side.  
🔄 **Continuous Retraining:** API endpoint accepts new images + labels, retrains model, and auto-deploys.  
📈 **MLflow Integration:** Automatic experiment logging, metric tracking, and model versioning.  
📡 **Production API:** FastAPI server with `/predict` and `/retrain` endpoints.  
🌐 **Interactive Web UI:** Drag-and-drop interface for testing predictions and providing feedback.  
🎯 **Robust Preprocessing:** Automatic image resizing, normalization, and augmentation.

---

## 🏗️ System Architecture

### High-Level Data Flow
The system implements a three-stage pipeline: Data Preparation → Model Training → Inference & Retraining.

```
┌──────────────────────────────────────────────────────────┐
│  CLIENT (Web UI + REST API Calls)                        │
└────────────────────┬─────────────────────────────────────┘
                     │ HTTPS / JSON + Multipart Form Data
┌────────────────────▼─────────────────────────────────────┐
│  API GATEWAY (FastAPI Server)                             │
│  ┌────────────────────────────────────────────────────┐  │
│  │ 1. Model Loading (load_best_model)                 │  │
│  │    - MLflow Tracking URI Configuration            │  │
│  │    - Latest Model Retrieval (by timestamp)         │  │
│  └──────────────────────┬─────────────────────────────┘  │
│                         │ Keras Model Object             │
│  ┌──────────────────────▼─────────────────────────────┐  │
│  │ 2. Inference Engine (/predict endpoint)            │  │
│  │    - Image Preprocessing (224x224, normalize)      │  │
│  │    - Multi-label Prediction (sigmoid threshold)    │  │
│  │    - Confidence Scores for all classes             │  │
│  └──────────────────────┬─────────────────────────────┘  │
│                         │ Predictions + Probabilities    │
│  ┌──────────────────────▼─────────────────────────────┐  │
│  │ 3. Retraining Pipeline (/retrain endpoint)         │  │
│  │    - User Feedback Integration                     │  │
│  │    - Data Augmentation (10x replication)           │  │
│  │    - Incremental Fine-tuning (30 epochs)           │  │
│  │    - MLflow Model Persistence (timestamped runs)   │  │
│  └────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────┘
          ▲                                      │
          │                                      ▼
┌─────────┴─────────┐               ┌────────────────────┐
│  MLflow Tracking  │               │  Processed Data    │
│  (mlflow_data/)   │               │  (processed_data/) │
│  - Experiments    │               │  - X_train.npy     │
│  - Models         │               │  - y_train.npy     │
│  - Metrics        │               │  - X_val.npy       │
└───────────────────┘               │  - X_test.npy      │
                                    └────────────────────┘
```

---

## 📁 Project Structure

```
Multi-Label_Classification_proyecto final/
├── mlflow_data/                              # 📊 MLflow Tracking Server
│   ├── mlflow_env/                          # Isolated Python environment
│   ├── mlflow.db                            # SQLite metadata database
│   └── mlruns/                              # Model artifacts and runs
│
└── Multi-Label_Classification/               # 🧠 Main Project
    ├── notebooks/                           # 📓 3-Stage Pipeline
    │   ├── 01_preparacion_datos.ipynb       # Data Ingestion & Preprocessing
    │   ├── 02_entrenamiento_modelos.ipynb   # Multi-Architecture Training
    │   └── 03_prediccion_reentrenamiento.ipynb  # Inference & Retraining
    │
    ├── api/                                 # 🌐 REST API
    │   ├── main.py                          # FastAPI server (inline functions)
    │   └── requirements.txt                 # Dependencies
    │
    ├── web/                                 # 💻 Frontend
    │   ├── index.html
    │   ├── app.js
    │   └── styles.css
    │
    ├── processed_data/                      # 📦 NumPy arrays (generated)
    ├── pascal_2007/                         # 🗂️ Dataset (auto-downloaded)
    ├── model_config.npy                     # ⚙️ Configuration
    └── requirements.txt                     # 📋 Python dependencies
```

---

## 🛠️ Technology Stack

### Backend Infrastructure
| Component | Technology | Version | Purpose |
|-----------|-----------|---------|---------|
| Framework | TensorFlow + Keras | 2.15+ | Deep learning engine |
| API | FastAPI | 0.115+ | High-performance REST server |
| ASGI Server | Uvicorn | 0.32+ | Production web server |
| Tracking | MLflow | 2.18+ | Model versioning & metrics |

### Machine Learning
| Component | Technology | Purpose |
|-----------|-----------|---------|
| Architectures | ResNet50, EfficientNetB0, MobileNetV2 | Transfer Learning |
| Loss | Binary Crossentropy | Multi-label objective |
| Optimizer | Adam | Adaptive learning rate |
| Metrics | F1-Score, Hamming Loss, Precision/Recall | Evaluation |
| Image Processing | PIL (Pillow) | Loading & transformation |

### Development
| Component | Technology | Purpose |
|-----------|-----------|---------|
| Notebooks | Jupyter | Interactive development |
| Dataset | Pascal VOC 2007 | Benchmark (9,963 images) |
| Version Control | Git | Source management |

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8 - 3.11
- 5 GB storage (dataset + models)
- 8 GB RAM (16 GB recommended)
- GPU optional (CUDA 11.8+ for TensorFlow GPU)

### 1. Main Environment Setup

```bash
cd "c:\Users\salej\Desktop\Multi-Label_Classification_proyecto final\Multi-Label_Classification"
python -m venv venv312

# Windows
venv312\Scripts\activate

# Linux/Mac
source venv312/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. MLflow Environment

```bash
cd ..\mlflow_data
.\mlflow_env\Scripts\Activate.ps1  # Windows

# Start MLflow UI
mlflow ui --backend-store-uri ./
# Access: http://localhost:5000
```

---

## 📝 Usage Examples

### Workflow 1: Complete Training (First Time)

**Step 1: Data Preparation** (~10 min)
```bash
jupyter notebook notebooks/01_preparacion_datos.ipynb
```
- Downloads Pascal 2007 dataset
- Filters images with target classes
- Saves to `processed_data/`

**Step 2: Model Training** (~30-90 min)
```bash
jupyter notebook notebooks/02_entrenamiento_modelos.ipynb
```
- Trains 3 architectures
- Saves models to `mlflow_data/`

**Step 3: View Results**
```bash
cd ..\mlflow_data
mlflow ui --backend-store-uri ./
```

---

### Workflow 2: Production API (Existing Models)

**Start Server:**
```bash
cd "c:\Users\salej\Desktop\Multi-Label_Classification_proyecto final\Multi-Label_Classification"
.\venv312\Scripts\Activate.ps1
python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

**Access UI:** `http://localhost:8000`

---

### API Examples

#### Health Check
```bash
curl http://localhost:8000/health
```

#### Prediction
```bash
curl -X POST "http://localhost:8000/predict" -F "files=@image.jpg"
```

**Response:**
```json
[{
  "filename": "image.jpg",
  "labels": ["person", "dog"],
  "probabilities": {
    "person": 0.94,
    "chair": 0.12,
    "dog": 0.89,
    "sofa": 0.08
  }
}]
```

#### Retraining
```bash
curl -X POST "http://localhost:8000/retrain" -F "files=@corrected_image.jpg" -F 'labels=[["person", "chair"]]'
```

**What happens:**
1. Images preprocessed (224×224, normalized)
2. Labels converted to binary vectors
3. Data replicated 10× for augmentation
4. Model fine-tuned for 30 epochs
5. Saved to MLflow: `Retrained_20260209_151045`

---

## 🎨 Design Patterns

| Pattern | Implementation | Purpose |
|---------|---------------|---------|
| **Pipeline** | 3 Sequential Notebooks | Data → Training → Inference separation |
| **Strategy** | Multiple architectures (ResNet50, EfficientNetB0, MobileNetV2) | Algorithm interchangeability |
| **Repository** | MLflow Tracking | Centralized model storage |
| **Facade** | API endpoints | Simplified interface to complex ML logic |

---

## 🔧 Troubleshooting

### Error: "No se encontraron experimentos en mlflow_data"
**Fix:** Run Notebook 2 to train initial models.

### Error: "Modelo no cargado" (API)
**Fix:** Ensure `mlflow_data/` contains trained models. Check MLflow UI at `http://localhost:5000`.

### Error: "Module 'tensorflow' has no attribute 'keras'"
**Fix:** Install correct TensorFlow version:
```bash
pip install tensorflow>=2.15.0
```

---

## 📊 Model Performance

### Benchmark Results (Pascal VOC 2007)

| Architecture | F1-Score | Parameters | Inference Time |
|--------------|----------|------------|----------------|
| **MobileNetV2** ⭐ | 0.87 | 3.5M | 15 ms |
| EfficientNetB0 | 0.89 | 5.0M | 22 ms |
| ResNet50 | 0.86 | 23.5M | 48 ms |

**Winner:** MobileNetV2 (best speed/accuracy trade-off)

---

## 📄 License

Copyright © 2026 Multi-Label Classification Team. All Rights Reserved.

This project is for educational and research purposes. Unauthorized commercial use is prohibited.

---

## 📧 Contact

For questions or collaboration:
- **Author:** Sebastián Machado
- **Email:** salejomac1210@gmail.com
- **LinkedIn:** www.linkedin.com/in/sebastian-machado-eng
- **Date:** February 2026
