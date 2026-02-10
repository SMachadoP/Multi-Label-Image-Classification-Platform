# Multi-Label Classification System: A Deep Learning Platform for Continuous Object Recognition

**Python** · **TensorFlow/Keras** · **FastAPI** · **MLflow** · **Transfer Learning**

**Author:** Sebastián Machado and Sebastian Verdugo 
**Date:** February 2026  

---

## 1. ABSTRACT / RESUMEN

**Problem:** Traditional image classification systems are limited to single-label predictions and require complete retraining cycles when new data becomes available. Real-world computer vision applications demand the ability to identify multiple objects simultaneously in complex scenes, adapt to user corrections through incremental learning, and maintain transparent experiment tracking for reproducibility and model comparison.

**Proposal:** This paper presents a Multi-Label Classification System that leverages Transfer Learning with three state-of-the-art pre-trained architectures (ResNet50, EfficientNetB0, and MobileNetV2) to identify multiple objects (person, chair, dog, sofa) in images. The system implements a continuous learning paradigm where users can correct predictions through an interactive web interface, triggering incremental model retraining with automatic versioning via MLflow. A production-ready REST API built with FastAPI enables real-time predictions and seamless model updates with zero downtime.

**Dataset:** The system uses **Pascal VOC 2007**, a benchmark dataset containing 9,963 images with multi-label annotations across 20 object categories. For this implementation, images are filtered to include only the four target classes: person, chair, dog, and sofa. Images are preprocessed to 224×224 pixels and split into training (70%), validation (15%), and testing (15%) sets.

**Quality Measures:** Model performance is evaluated using standard multi-label classification metrics: **F1-Score** (harmonic mean of precision and recall), **Precision** (proportion of true positives among predicted positives), **Recall** (proportion of true positives among all actual positives), **Accuracy** (overall classification correctness), and **Hamming Loss** (fraction of incorrect labels). The best-performing model achieved an F1-Score of 0.82, Precision of 0.84, Recall of 0.83, and Accuracy of 0.85 on the test set.

**Future Work:** Future enhancements include: (1) expanding the target classes to cover all 20 Pascal VOC categories, (2) implementing active learning strategies to prioritize uncertain predictions for retraining, (3) exploring ensemble methods combining the three architectures, (4) deploying the system with GPU acceleration for faster inference, and (5) integrating explainability techniques (GradCAM) to visualize model decision-making.

---

## 2. PROPOSED METHOD / MÉTODO PROPUESTO

### 2.1 System Architecture

The Multi-Label Classification System follows a three-phase pipeline:

**Phase 1 - Data Preparation:**
1. **Dataset selection:** Download and extract Pascal VOC 2007 dataset
2. **Class filtering:** Select images containing target classes (person, chair, dog, sofa)
3. **Image preprocessing:** Resize images to 224×224 pixels, normalize pixel values to [0,1]
4. **Label encoding:** Convert multi-label annotations to binary vectors (4 classes)
5. **Data splitting:** Divide dataset into train (70%), validation (15%), test (15%)
6. **Data augmentation:** Apply random rotations, flips, and brightness adjustments during training

**Phase 2 - Model Training & Comparison:**
1. **Transfer Learning:** Load pre-trained models (ResNet50, EfficientNetB0, MobileNetV2) with ImageNet weights
2. **Architecture customization:** 
   - Freeze base layers
   - Add Global Average Pooling
   - Add Dense layer (128 units, ReLU activation)
   - Add Dropout (0.2)
   - Add output layer (4 units, sigmoid activation)
3. **Training:** Compile with Adam optimizer (lr=0.0001), binary crossentropy loss, train for 10 epochs
4. **Evaluation:** Compare models on validation set using F1-Score, Precision, Recall, Accuracy
5. **Model selection:** Select best-performing architecture for production deployment

**Phase 3 - Deployment & Continuous Learning:**
1. **API deployment:** FastAPI server loads best model, exposes `/predict` and `/retrain` endpoints
2. **Real-time inference:** Process uploaded images, return predictions with confidence scores
3. **User feedback:** Interactive web UI allows label corrections
4. **Incremental retraining:** 
   - Collect corrected images and labels
   - Apply data augmentation (10× replication)
   - Fine-tune model for 30 epochs with low learning rate
   - Save retrained model with timestamp to MLflow
   - Automatically load new model for future predictions

### 2.2 System Diagram

```
┌──────────────────────────────────────────────────────────┐
│  CLIENT (Web UI + REST API Calls)                        │
└────────────────────┬─────────────────────────────────────┘
                     │ HTTPS / JSON + Multipart Form Data
┌────────────────────▼─────────────────────────────────────┐
│  API GATEWAY (FastAPI Server)                            │
│  ┌────────────────────────────────────────────────────┐  │
│  │ 1. Model Loading (load_best_model)                 │  │
│  │    - MLflow Tracking URI Configuration             │  │
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

### 2.3 Method Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Image Size | 224×224×3 | Input dimensions for all models |
| Batch Size | 16 | Training batch size |
| Initial Epochs | 10 | Epochs for initial training |
| Retraining Epochs | 30 | Epochs for incremental retraining |
| Learning Rate | 0.0001 | Adam optimizer learning rate |
| Dropout Rate | 0.2 | Dropout regularization |
| Augmentation | 10× | Replication factor for retraining data |
| Classes | 4 | person, chair, dog, sofa |
| Loss Function | Binary Crossentropy | Multi-label classification loss |
| Activation (output) | Sigmoid | Multi-label output activation |

---

## 3. EXPERIMENTAL DESIGN / DISEÑO DE EXPERIMENTOS

### 3.1 Dataset Characteristics

| Dataset | Number of Images | Dimensions | Target Classes |
|---------|------------------|------------|----------------|
| Pascal VOC 2007 (total) | 9,963 | Variable (resized to 224×224) | 20 categories |
| Filtered Dataset | ~2,500 | 224×224×3 | person, chair, dog, sofa |
| Training Set | 1,750 (70%) | 224×224×3 | Multi-label annotations |
| Validation Set | 375 (15%) | 224×224×3 | Multi-label annotations |
| Test Set | 375 (15%) | 224×224×3 | Multi-label annotations |

**Class Distribution (filtered dataset):**
- **person:** ~60% of images
- **chair:** ~35% of images
- **dog:** ~25% of images
- **sofa:** ~20% of images

### 3.2 Model Optimization Parameters

| Architecture | Base Parameters | Trainable Parameters | Total Parameters |
|-------------|-----------------|---------------------|------------------|
| **ResNet50** | 23,587,712 | 525,316 | 24,113,028 |
| **EfficientNetB0** | 4,049,564 | 525,316 | 4,574,880 |
| **MobileNetV2** | 2,257,984 | 525,316 | 2,783,300 |

**Training Configuration:**
- **Optimizer:** Adam (Adaptive Moment Estimation)
- **Learning Rate:** 0.0001
- **Loss Function:** Binary Crossentropy
- **Metrics:** Accuracy, Precision, Recall, F1-Score
- **Batch Size:** 16
- **Initial Epochs:** 10
- **Retraining Epochs:** 30
- **Validation Split:** 15% of training data
- **Early Stopping:** Monitoring validation F1-Score (patience=3)
- **Data Augmentation:** Random rotation (±15°), horizontal flip, brightness adjustment (±0.2)

---

## 4. RESULTS AND DISCUSSION / RESULTADOS Y DISCUSIÓN

### 4.1 Model Comparison

Following the proposed three-phase method, three architectures were trained and evaluated on the Pascal VOC 2007 filtered dataset:

| Model | Accuracy | Precision | Recall | F1-Score | Hamming Loss | Training Time |
|-------|----------|-----------|--------|----------|--------------|---------------|
| **MobileNetV2_FineTuned** | **0.85** | **0.84** | **0.83** | **0.82** | **0.09** | ~30 min |
| EfficientNetB0 | 0.84 | 0.83 | 0.82 | 0.81 | 0.10 | ~45 min |
| ResNet50 | 0.83 | 0.82 | 0.81 | 0.80 | 0.11 | ~60 min |
| MobileNetV2 (base) | 0.80 | 0.79 | 0.78 | 0.77 | 0.13 | ~20 min |

### 4.2 Key Findings

1. **Best Performance:** **MobileNetV2 with fine-tuning** achieved the highest F1-Score (0.82) and overall accuracy (0.85), demonstrating that incremental retraining significantly improves model performance (+0.05 F1-Score compared to base MobileNetV2).

2. **Efficiency:** MobileNetV2 has the smallest model size (2.78M parameters) and fastest inference time (~15ms per image), making it ideal for production deployment.

3. **Trade-offs:** ResNet50 provides strong feature extraction capabilities but requires 8× more parameters and 3× longer training time compared to MobileNetV2, with only marginal performance differences.

4. **Multi-label Performance:** All models achieved low Hamming Loss (<0.13), indicating accurate prediction of individual labels. The sigmoid activation function effectively handles multi-label scenarios where images contain multiple objects.

### 4.3 Continuous Learning Results

After implementing the retraining pipeline with user-corrected predictions:

- **Retraining Time:** ~2 minutes for 3 corrected images (30 augmented samples)
- **F1-Score Improvement:** +0.03 on corrected sample set
- **Model Versioning:** Automatic timestamp-based versioning via MLflow
- **Zero Downtime:** New model loaded without service interruption
- **Total Retraining Cycles Tested:** 5 cycles, consistent performance improvement

### 4.4 Example Predictions

Sample predictions from the deployed MobileNetV2_FineTuned model:

| Image | Predicted Labels | Confidence Scores | Ground Truth | Correctness |
|-------|-----------------|-------------------|--------------|-------------|
| img_001.jpg | person, dog | person: 0.94, dog: 0.89 | person, dog | ✓ Correct |
| img_002.jpg | chair, sofa | chair: 0.76, sofa: 0.82 | chair, sofa | ✓ Correct |
| img_003.jpg | person, chair | person: 0.88, chair: 0.71 | person, chair, dog | ✗ Missed dog |

---

## 5. CONCLUSIONS / CONCLUSIONES

This work successfully developed and deployed a Multi-Label Classification System capable of identifying multiple objects in images with high accuracy (F1-Score=0.82, Accuracy=0.85) using Transfer Learning approaches. The principal contributions and conclusions are:

1. **Multi-Architecture Evaluation:** Comparative analysis of three state-of-the-art architectures (ResNet50, EfficientNetB0, MobileNetV2) determined that **MobileNetV2 with fine-tuning** provides the optimal balance between accuracy, model size, and inference speed for real-time multi-label classification tasks.

2. **Continuous Learning Success:** The implemented incremental retraining pipeline demonstrates that models can effectively adapt to user corrections through the interactive web interface. Data augmentation (10× replication) combined with fine-tuning (30 epochs, low learning rate) achieves consistent performance improvements (+0.03 F1-Score per retraining cycle) without catastrophic forgetting.

3. **Production-Ready Architecture:** The complete end-to-end platform integrating FastAPI (REST API), MLflow (experiment tracking), and an interactive web frontend enables seamless deployment, real-time predictions, automatic model versioning, and zero-downtime updates. The system successfully handles multiple concurrent requests while maintaining fast inference times (<20ms per image).

4. **Transfer Learning Effectiveness:** Pre-trained ImageNet weights combined with architecture customization (GAP, Dense layers, Dropout) enable rapid convergence (10 epochs) and strong generalization to Pascal VOC 2007 classes. This approach reduces training time by ~90% compared to training from scratch while achieving competitive performance.

5. **Multi-Label Classification Challenges:** Analysis reveals that class imbalance (person: 60%, sofa: 20%) affects model performance, with minority classes showing lower recall. The sigmoid activation function with binary crossentropy loss effectively handles multi-label scenarios, achieving low Hamming Loss (<0.09 for best model).

**Future Work Considerations:**

- **Scale Expansion:** Extend to all 20 Pascal VOC categories for comprehensive object recognition
- **Active Learning:** Implement uncertainty-based sample selection to prioritize retraining on challenging images
- **Ensemble Methods:** Combine predictions from multiple architectures to improve robustness
- **Hardware Acceleration:** Deploy on GPU infrastructure for sub-10ms inference times
- **Explainability:** Integrate GradCAM visualization to provide model decision transparency
- **Class Balancing:** Apply weighted loss functions or oversampling to address class imbalance

---

## 6. REFERENCES / REFERENCIAS

[1] M. Everingham, L. Van Gool, C. K. I. Williams, J. Winn, and A. Zisserman. "The Pascal Visual Object Classes (VOC) Challenge." *International Journal of Computer Vision*, vol. 88, no. 2, pp. 303–338, 2010. DOI: 10.1007/s11263-009-0275-4

[2] K. He, X. Zhang, S. Ren, and J. Sun. "Deep Residual Learning for Image Recognition." *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2016, pp. 770-778. DOI: 10.1109/CVPR.2016.90

[3] M. Sandler, A. Howard, M. Zhu, A. Zhmoginov, and L.-C. Chen. "MobileNetV2: Inverted Residuals and Linear Bottlenecks." *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2018, pp. 4510-4520.

[4] M. Tan and Q. V. Le. "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks." *Proceedings of the 36th International Conference on Machine Learning (ICML)*, 2019, pp. 6105-6114.

[5] D. P. Kingma and J. Ba. "Adam: A Method for Stochastic Optimization." *Proceedings of the International Conference on Learning Representations (ICLR)*, 2015.

[6] M. Chen, A. Goel, M. Sennesh, et al. "MLflow: A Platform for Machine Learning Lifecycle Management." *Proceedings of the 4th International Workshop on Data Management for End-to-End Machine Learning*, 2020.

[7] Pascal VOC 2007 Dataset. "The PASCAL Visual Object Classes Challenge 2007 (VOC2007)." Retrieved from: http://host.robots.ox.ac.uk/pascal/VOC/voc2007/

---

## 7. GETTING STARTED / CÓMO EJECUTAR

### Prerequisites
- Python 3.8+
- 5 GB storage (dataset + models)
- 8 GB RAM minimum

### Quick Start

**1. Install dependencies:**
```bash
cd Multi-Label_Classification
pip install -r requirements.txt
```

**2. Download pre-trained models:**
```bash
# Run the multilabel_classification_fixed.ipynb notebook
# It downloads the complete MLflow directory with 5 models
jupyter notebook notebooks/multilabel_classification_fixed.ipynb
```

**3. Start the API:**
```bash
python -m uvicorn api.main:app --reload --port 8000
```

**4. Access the web interface:**
```
http://localhost:8000
```

### Training from Scratch (Optional)

If you want to train your own models, run the notebooks in sequence:

```bash
# 1. Data preparation (~10 min)
jupyter notebook notebooks/01_preparacion_datos.ipynb

# 2. Model training (~30-90 min)
jupyter notebook notebooks/02_entrenamiento_modelos.ipynb

# 3. Inference and retraining
jupyter notebook notebooks/03_prediccion_reentrenamiento.ipynb
```

---
**Contact**
**Email:** salejomac1210@gmail.com | **LinkedIn:** [sebastian-machado-eng](https://www.linkedin.com/in/sebastian-machado-eng)
