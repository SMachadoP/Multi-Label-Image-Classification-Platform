"""
FastAPI Server - Multi-Label Classification
Endpoints: /predict, /retrain, /model-info

Este API contiene las mismas funciones del Notebook 3 para predicción y reentrenamiento.
"""

import os
import io
import json
from pathlib import Path
from typing import List, Dict
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import numpy as np
from PIL import Image
import tensorflow as tf
import mlflow
import mlflow.keras

# Directorios
BASE_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BASE_DIR.parent
MLFLOW_DATA_DIR = PROJECT_ROOT / "mlflow_data"
WEB_DIR = BASE_DIR / "web"

# Configuración
IMG_SIZE = 224
TARGET_CLASSES = ['person', 'chair', 'dog', 'sofa']
NUM_CLASSES = 4

# Variables globales
model = None


# ============================================================================
# FUNCIONES (mismas del Notebook 3)
# ============================================================================

def load_best_model():
    """
    Carga el último modelo (más reciente) desde mlflow_data.
    Busca directamente en las carpetas de modelos registrados.
    
    Returns:
        Modelo de Keras cargado o None si no hay modelo disponible
    """
    try:
        print(f"Buscando modelos en: {MLFLOW_DATA_DIR}")
        
        # Buscar modelos directamente en la carpeta de artifacts
        experiment_dir = MLFLOW_DATA_DIR / "155436956194197961"
        models_dir = experiment_dir / "models"
        
        if not models_dir.exists():
            print("⚠️ No se encontró directorio de modelos")
            return None
        
        # Mapeo de model_id a run_id y nombre
        model_mapping = {
            "m-c6564c9babab4b539faa729e7970bd3b": ("222666ee5fbe4985a1eff7358f450931", "ResNet50"),
            "m-99854b79979347eeb3caee8f1b635470": ("270ee779a46e4c5b871356c9230785f2", "MobileNetV2"),
            "m-4c60b9c893914cb5943ac56ab1a1f4c0": ("be67f6f0dac648be8a2e37a0ca7fc967", "EfficientNetB0"),
            "m-df89104e077249f699dd661c6103763b": ("6ea310a9d4fe4578a16698db6d3d51b7", "MobileNetV2_FineTuned"),
            "m-e15625d0deab4b9a8daea4933d25d183": ("54be9661823b471c9ecdaaf554c46916", "Retrained"),
        }
        
        # Ordenar por prioridad: Retrained > FineTuned > otros
        priority_order = [
            "m-e15625d0deab4b9a8daea4933d25d183",  # Retrained
            "m-df89104e077249f699dd661c6103763b",  # MobileNetV2_FineTuned
            "m-99854b79979347eeb3caee8f1b635470",  # MobileNetV2
            "m-4c60b9c893914cb5943ac56ab1a1f4c0",  # EfficientNetB0
            "m-c6564c9babab4b539faa729e7970bd3b",  # ResNet50
        ]
        
        # Intentar cargar modelos en orden de prioridad
        for model_id in priority_order:
            if model_id not in model_mapping:
                continue
                
            run_id, model_name = model_mapping[model_id]
            model_path = models_dir / model_id / "artifacts" / "data" / "model.keras"
            
            print(f"\nIntentando cargar: {model_name}")
            print(f"  - Ruta: {model_path}")
            
            if model_path.exists():
                try:
                    model = tf.keras.models.load_model(str(model_path))
                    print(f"✅ Modelo cargado exitosamente: {model_name}")
                    print(f"  - Parámetros: {model.count_params():,}")
                    return model
                except Exception as e:
                    print(f"  ⚠️ Error cargando: {str(e)[:100]}...")
                    continue
            else:
                print(f"  ⚠️ Archivo no encontrado")
        
        print("\n⚠️ No se pudo cargar ningún modelo")
        return None
            
    except Exception as e:
        print(f"❌ Error buscando modelos: {e}")
        return None


def preprocess_image(image_source, target_size=None):
    """
    Preprocesa una imagen para predicción.
    
    Args:
        image_source: Path (str/Path), bytes, o PIL Image
        target_size: Tupla (height, width) para redimensionar
    
    Returns:
        Tupla (array numpy normalizado, imagen PIL)
    """
    if target_size is None:
        target_size = (IMG_SIZE, IMG_SIZE)
    
    # Cargar imagen según el tipo de entrada
    if isinstance(image_source, bytes):
        image = Image.open(io.BytesIO(image_source))
    elif isinstance(image_source, (str, Path)):
        image = Image.open(image_source)
    elif isinstance(image_source, Image.Image):
        image = image_source
    else:
        raise ValueError(f"Tipo de imagen no soportado: {type(image_source)}")
    
    # Convertir a RGB y redimensionar
    image = image.convert('RGB').resize(target_size)
    
    # Convertir a array y normalizar
    image_array = np.array(image, dtype=np.float32) / 255.0
    
    return image_array, image


def predict_image(model, image_array, threshold=0.5):
    """
    Predice las clases de una imagen.
    
    Args:
        model: Modelo de Keras
        image_array: Array numpy de la imagen normalizada
        threshold: Umbral de confianza para clasificación
    
    Returns:
        dict con 'labels' (lista de clases) y 'probs' (dict de probabilidades)
    """
    # Asegurar que la imagen tenga la dimensión de batch
    if len(image_array.shape) == 3:
        image_array = np.expand_dims(image_array, axis=0)
    
    # Predecir
    predictions = model.predict(image_array, verbose=0)[0]
    
    # Extraer etiquetas con probabilidad > threshold
    labels = [TARGET_CLASSES[i] for i, p in enumerate(predictions) if p >= threshold]
    if not labels:
        labels = ['ninguna']
    
    # Crear diccionario de probabilidades
    probs = {TARGET_CLASSES[i]: float(predictions[i]) for i in range(NUM_CLASSES)}
    
    return {
        'labels': labels,
        'probs': probs
    }


def retrain_model(model, new_images, new_labels, epochs=30, batch_size=4, learning_rate=0.0001):
    """
    Reentrena el modelo con nuevas imágenes.
    
    Args:
        model: Modelo de Keras a reentrenar
        new_images: Array numpy de imágenes (N, H, W, 3), normalizado
        new_labels: Array numpy de etiquetas (N, num_classes)
        epochs: Número de epochs
        batch_size: Tamaño del batch
        learning_rate: Tasa de aprendizaje
    
    Returns:
        Tupla (modelo reentrenado, historial)
    """
    # Verificar tipos de datos
    if new_images.dtype != np.float32:
        new_images = new_images.astype(np.float32)
    if new_labels.dtype != np.float32:
        new_labels = new_labels.astype(np.float32)
    
    # Replicar datos para tener más muestras
    X_rep = np.tile(new_images, (10, 1, 1, 1))
    y_rep = np.tile(new_labels, (10, 1))
    
    print(f"Reentrenando con {len(X_rep)} muestras generadas...")
    
    # Compilar y entrenar
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    history = model.fit(
        X_rep, y_rep,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )
    
    print("Reentrenamiento completado")
    return model, history


def save_model(model, run_name=None):
    """
    Guarda el modelo reentrenado en mlflow_data con fecha.
    
    Args:
        model: Modelo de Keras
        run_name: Nombre opcional para el run (si None, usa fecha actual)
    """
    try:
        # Configurar MLflow para usar servidor local
        tracking_uri = "http://127.0.0.1:5000"
        mlflow.set_tracking_uri(tracking_uri)
        
        # Buscar o crear experimento
        experiments = mlflow.search_experiments()
        if experiments:
            exp_name = experiments[0].name
            mlflow.set_experiment(exp_name)
        else:
            mlflow.set_experiment("MultiLabel_Pascal2007")
        
        # Nombre del run con fecha
        if run_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"Retrained_{timestamp}"
        
        print(f"\n{'='*60}")
        print(f"   GUARDANDO MODELO REENTRENADO")
        print(f"{'='*60}")
        print(f"Ubicación: {MLFLOW_DATA_DIR}")
        print(f"Run: {run_name}")
        
        # Guardar en MLflow
        with mlflow.start_run(run_name=run_name):
            # Registrar parámetros
            mlflow.log_param("model_type", "Retrained")
            mlflow.log_param("timestamp", datetime.now().isoformat())
            mlflow.log_param("img_size", IMG_SIZE)
            mlflow.log_param("classes", TARGET_CLASSES)
            
            # Guardar modelo
            mlflow.keras.log_model(
                model,
                artifact_path="model",
                registered_model_name="RetrainedModel"
            )
            
            # Tags
            mlflow.set_tag("retrained", "True")
            mlflow.set_tag("source", "api_retraining")
            
        print(f"✅ Modelo guardado en mlflow_data")
        print(f"   Run: {run_name}")
        print(f"{'='*60}\n")
        
    except Exception as e:
        print(f"❌ Error guardando modelo: {e}")
        raise


def create_label_vector(labels_list):
    """
    Convierte una lista de etiquetas a vector binario.
    
    Args:
        labels_list: Lista de strings con nombres de clases
    
    Returns:
        Array numpy de shape (num_classes,) con valores 0.0 o 1.0
    """
    vector = np.array([1.0 if cls in labels_list else 0.0 for cls in TARGET_CLASSES], dtype=np.float32)
    return vector


# ============================================================================
# CONFIGURACIÓN FASTAPI
# ============================================================================

def load_model_api():
    """Carga el modelo para el API."""
    global model
    model = load_best_model()
    return model


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle: carga modelo al iniciar."""
    print("🚀 Iniciando API...")
    loaded_model = load_model_api()
    if loaded_model is not None:
        print("✅ Modelo cargado correctamente")
    else:
        print("⚠️ API iniciada sin modelo - Usa /retrain para crear uno")
    yield
    print("👋 Cerrando API...")


# App FastAPI
app = FastAPI(
    title="Multi-Label Classification API",
    description="API para clasificación multi-label de imágenes (person, chair, dog, sofa)",
    version="2.0.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Modelos Pydantic
class PredictResponse(BaseModel):
    filename: str
    labels: List[str]
    probabilities: Dict[str, float]


class ModelInfo(BaseModel):
    model_name: str
    classes: List[str]
    image_size: int
    num_classes: int


class RetrainResponse(BaseModel):
    status: str
    message: str
    final_loss: float
    predictions: List[Dict]


# Endpoints
@app.get("/")
async def root():
    """Servir página web."""
    index_path = WEB_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"message": "Multi-Label Classification API", "docs": "/docs"}


@app.get("/health")
async def health():
    """Health check."""
    return {"status": "ok", "model_loaded": model is not None}


@app.get("/model-info", response_model=ModelInfo)
async def get_model_info():
    """Obtener información del modelo."""
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado")
    
    return ModelInfo(
        model_name=model.name if hasattr(model, 'name') else "Unknown",
        classes=TARGET_CLASSES,
        image_size=IMG_SIZE,
        num_classes=NUM_CLASSES
    )


@app.post("/predict", response_model=List[PredictResponse])
async def predict(files: List[UploadFile] = File(...)):
    """
    Predecir etiquetas de imágenes.
    
    Endpoint para clasificación multi-label de imágenes.
    Utiliza la función predict_image de utils.py.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No se enviaron imágenes")
    
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado")
    
    results = []
    for file in files:
        try:
            # Leer contenido del archivo
            content = await file.read()
            
            # Preprocesar imagen usando utils.py
            img_array, _ = preprocess_image(content)
            
            # Predecir usando utils.py
            prediction = predict_image(model, img_array)
            
            results.append(PredictResponse(
                filename=file.filename,
                labels=prediction['labels'],
                probabilities=prediction['probs']
            ))
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Error procesando {file.filename}: {str(e)}"
            )
    
    return results


@app.post("/retrain", response_model=RetrainResponse)
async def retrain(
    files: List[UploadFile] = File(...),
    labels: str = Form(...)
):
    """
    Reentrenar modelo con nuevas imágenes y etiquetas.
    
    Utiliza la función retrain_model de utils.py.
    
    Args:
        files: Lista de archivos de imagen
        labels: JSON string con lista de listas de etiquetas
               Ejemplo: '[["person", "dog"], ["chair"]]'
    
    Returns:
        Información sobre el reentrenamiento y nuevas predicciones
    """
    global model
    
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado")
    
    # Parsear etiquetas
    try:
        labels_list = json.loads(labels)
    except Exception as e:
        raise HTTPException(
            status_code=400, 
            detail=f"Formato de etiquetas inválido: {str(e)}"
        )
    
    if len(files) != len(labels_list):
        raise HTTPException(
            status_code=400, 
            detail=f"Número de imágenes ({len(files)}) no coincide con etiquetas ({len(labels_list)})"
        )
    
    # Procesar imágenes
    X_new = []
    y_new = []
    
    try:
        for file, img_labels in zip(files, labels_list):
            # Leer y preprocesar imagen
            content = await file.read()
            img_array, _ = preprocess_image(content)
            X_new.append(img_array)
            
            # Convertir etiquetas a vector usando utils.py
            vector = create_label_vector(img_labels)
            y_new.append(vector)
        
        X_new = np.array(X_new, dtype=np.float32)
        y_new = np.array(y_new, dtype=np.float32)
        
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Error procesando datos: {str(e)}"
        )
    
    # Reentrenar usando utils.py
    try:
        print(f"\n{'='*60}")
        print(f"   REENTRENANDO MODELO")
        print(f"{'='*60}")
        print(f"Imágenes: {len(X_new)}")
        print(f"Etiquetas: {labels_list}")
        
        model, history = retrain_model(
            model, 
            X_new, 
            y_new, 
            epochs=30, 
            batch_size=4, 
            learning_rate=0.0001
        )
        
        # Guardar modelo usando utils.py
        save_model(model, run_name=None)  # Se genera nombre con fecha automáticamente
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error durante reentrenamiento: {str(e)}"
        )
    
    # Predecir nuevamente para mostrar resultados
    new_predictions = []
    for i, img_array in enumerate(X_new):
        pred = predict_image(model, img_array)
        new_predictions.append({
            'filename': files[i].filename,
            'correct_labels': labels_list[i],
            'predicted_labels': pred['labels'],
            'probabilities': pred['probs']
        })
    
    print(f"\n✅ Reentrenamiento completado")
    print(f"{'='*60}\n")
    
    return RetrainResponse(
        status="success",
        message=f"Modelo reentrenado con {len(X_new)} imágenes",
        final_loss=float(history.history['loss'][-1]),
        predictions=new_predictions
    )


# Servir archivos estáticos
print(f"Web directory: {WEB_DIR}")

if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")
    print(f"✅ Static files mounted from {WEB_DIR}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

