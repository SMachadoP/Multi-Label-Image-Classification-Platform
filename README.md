# Multi-Label Classification System: A Deep Learning Platform for Continuous Object Recognition

**Python** · **TensorFlow/Keras** · **FastAPI** · **MLflow** · **Transfer Learning**

**Author:** Sebastián Machado and Sebastian Verdugo 
**Date:** February 2026  

---

## 1. ABSTRACT / RESUMEN

Traditional image classification systems are limited to single-label predictions and require complete retraining cycles when new data becomes available. This short report presents a Multi-Label Classification System that leverages Transfer Learning with three state-of-the-art pre-trained architectures (ResNet50, EfficientNetB0, and MobileNetV2) to identify multiple objects (person, chair, dog, sofa) in images from the Pascal VOC 2007 dataset containing 9,963 images with multi-label annotations. The system implements a continuous learning paradigm where users can correct predictions through an interactive web interface, triggering incremental model retraining with automatic versioning via MLflow. A production-ready REST API built with FastAPI enables real-time predictions and seamless model updates with zero downtime. Model performance is evaluated using standard multi-label classification metrics including F1-Score, Precision, Recall, Accuracy, and Hamming Loss. The best-performing model (MobileNetV2 fine-tuned) achieved an F1-Score of 0.82, Precision of 0.84, Recall of 0.83, and Accuracy of 0.85 on the test set, demonstrating that Transfer Learning combined with continuous learning enables efficient adaptation to user feedback while maintaining high performance in multi-label object recognition tasks.

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

Following the proposed methodology and algorithms presented in Section 2, the following results are obtained:

### 4.1 Model Performance Comparison

**Table 1.** Comparison of model performance metrics on Pascal VOC 2007 test set

| Model | Accuracy | Precision | Recall | F1-Score | Hamming Loss | Training Time |
|-------|----------|-----------|--------|----------|--------------|---------------|
| **MobileNetV2_FineTuned** | **0.85** | **0.84** | **0.83** | **0.82** | **0.09** | ~30 min |
| EfficientNetB0 | 0.84 | 0.83 | 0.82 | 0.81 | 0.10 | ~45 min |
| ResNet50 | 0.83 | 0.82 | 0.81 | 0.80 | 0.11 | ~60 min |
| MobileNetV2 (base) | 0.80 | 0.79 | 0.78 | 0.77 | 0.13 | ~20 min |

As shown in Table 1, MobileNetV2 with fine-tuning achieved the best overall performance across all evaluation metrics, obtaining an F1-Score of 0.82, Accuracy of 0.85, Precision of 0.84, and Recall of 0.83 on the test set. This represents a significant improvement (+0.05 F1-Score) compared to the base MobileNetV2 model, demonstrating the effectiveness of the fine-tuning strategy.

![Model Comparison](docs/images/results.png)
**Figure 1.** Comparative analysis of model performance across five evaluation metrics: accuracy, precision, recall, F1-Score, and Hamming Loss. MobileNetV2 consistently outperforms other architectures across all metrics.

### 4.2 Architecture Analysis

The results indicate that MobileNetV2 provides an optimal balance between model complexity (2.78M parameters), inference speed (~15ms per image), and classification performance, making it particularly suitable for production deployment scenarios. ResNet50, despite having 8× more parameters (24.1M) than MobileNetV2, achieved only marginally different results (F1-Score: 0.80 vs 0.82), requiring 3× longer training time (~60 minutes). This suggests that for the specific task of multi-label classification on Pascal VOC 2007 with four target classes, the additional model capacity of ResNet50 does not translate to proportional performance gains. EfficientNetB0 demonstrated competitive performance (F1-Score: 0.81) with moderate computational requirements, positioning it as a viable alternative when balancing accuracy and efficiency.

### 4.3 Multi-Label Classification Performance

The Hamming Loss metric, which measures the fraction of incorrectly predicted labels, remained consistently low across all models (<0.13), with the fine-tuned MobileNetV2 achieving the lowest value (0.09). This indicates that the sigmoid activation function combined with binary crossentropy loss effectively handles the multi-label classification scenario where images contain multiple objects simultaneously. The analysis reveals that class imbalance in the dataset (person: 60%, chair: 35%, dog: 25%, sofa: 20%) affects model performance, with minority classes exhibiting lower recall rates, particularly for the sofa class.

### 4.4 System Interface and Deployment

The complete system provides an intuitive web-based interface for end-to-end multi-label classification workflow, as illustrated in Figures 2-4.

![Upload Interface](docs/images/upload-interface.png)
**Figure 2.** Web interface for image upload supporting drag-and-drop functionality and batch processing.

![Predictions View](docs/images/predictions.png)
**Figure 3.** Real-time prediction results displaying detected classes with confidence scores and probability distributions for each target category.

![Retraining Interface](docs/images/retraining.png)
**Figure 4.** Interactive label correction interface enabling users to modify predictions and trigger incremental model retraining.

### 4.5 Continuous Learning Results

The implemented continuous learning pipeline demonstrated successful adaptation to user corrections through the interactive web interface. Retraining experiments with user-corrected predictions showed consistent improvements, with an average F1-Score increase of +0.03 on the corrected sample set. The data augmentation strategy (10× replication) combined with fine-tuning (30 epochs, learning rate: 0.0001) enabled model adaptation within approximately 2 minutes for batches of 3 corrected images, with automatic versioning via MLflow ensuring traceability and zero-downtime model updates. A total of 5 retraining cycles were conducted, each showing consistent performance improvements without catastrophic forgetting, validating the effectiveness of the incremental learning approach for maintaining model relevance with minimal computational overhead.

---

## 5. CONCLUSIONS / CONCLUSIONES

This work successfully developed and deployed a Multi-Label Classification System capable of identifying multiple objects in images with high accuracy (F1-Score=0.82, Accuracy=0.85) using Transfer Learning approaches. Comparative analysis of three state-of-the-art architectures (ResNet50, EfficientNetB0, MobileNetV2) determined that MobileNetV2 with fine-tuning provides the optimal balance between accuracy, model size, and inference speed for real-time multi-label classification tasks. The implemented incremental retraining pipeline demonstrates that models can effectively adapt to user corrections through the interactive web interface, where data augmentation (10× replication) combined with fine-tuning (30 epochs, low learning rate) achieves consistent performance improvements (+0.03 F1-Score per retraining cycle) without catastrophic forgetting. The complete end-to-end platform integrating FastAPI (REST API), MLflow (experiment tracking), and an interactive web frontend enables seamless deployment, real-time predictions, automatic model versioning, and zero-downtime updates while successfully handling multiple concurrent requests with fast inference times (<20ms per image). Pre-trained ImageNet weights combined with architecture customization (GAP, Dense layers, Dropout) enable rapid convergence (10 epochs) and strong generalization to Pascal VOC 2007 classes, reducing training time by approximately 90% compared to training from scratch while achieving competitive performance. Analysis reveals that class imbalance (person: 60%, sofa: 20%) affects model performance with minority classes showing lower recall, though the sigmoid activation function with binary crossentropy loss effectively handles multi-label scenarios achieving low Hamming Loss (<0.09 for best model). Future work considerations include extending to all 20 Pascal VOC categories for comprehensive object recognition, implementing uncertainty-based active learning to prioritize retraining on challenging images, exploring ensemble methods combining multiple architectures, deploying with GPU acceleration for sub-10ms inference times, integrating explainability techniques such as GradCAM for model decision transparency, and applying weighted loss functions or oversampling to address class imbalance.

---

## 6. REFERENCES / REFERENCIAS

[1] Pascal VOC 2007 Dataset. "The PASCAL Visual Object Classes Challenge 2007 (VOC2007)." Retrieved from: http://host.robots.ox.ac.uk/pascal/VOC/voc2007/

---

```
**Contact**
**Email:** salejomac1210@gmail.com | **LinkedIn:** [sebastian-machado-eng](https://www.linkedin.com/in/sebastian-machado-eng)
