// Multi-Label Classification App

const API_URL = window.location.origin;
const CLASSES = ['person', 'chair', 'dog', 'sofa'];

// State
let uploadedFiles = [];
let predictions = [];

// DOM Elements
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const imagesSection = document.getElementById('imagesSection');
const imagesGrid = document.getElementById('imagesGrid');
const predictionsSection = document.getElementById('predictionsSection');
const predictionsGrid = document.getElementById('predictionsGrid');
const retrainSection = document.getElementById('retrainSection');
const retrainGrid = document.getElementById('retrainGrid');
const resultsSection = document.getElementById('resultsSection');
const resultsGrid = document.getElementById('resultsGrid');
const loading = document.getElementById('loading');
const loadingText = document.getElementById('loadingText');

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    setupDropZone();
    setupButtons();
    loadModelInfo();
});

// Setup drop zone
function setupDropZone() {
    dropZone.addEventListener('click', () => fileInput.click());

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('dragover');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        handleFiles(e.dataTransfer.files);
    });

    fileInput.addEventListener('change', (e) => {
        handleFiles(e.target.files);
    });
}

// Setup buttons
function setupButtons() {
    document.getElementById('predictBtn').addEventListener('click', predict);
    document.getElementById('retrainBtn').addEventListener('click', showRetrainSection);
    document.getElementById('confirmRetrainBtn').addEventListener('click', retrain);
    document.getElementById('newPredictionBtn').addEventListener('click', resetApp);
}

// Handle file selection
function handleFiles(files) {
    uploadedFiles = Array.from(files).filter(f => f.type.startsWith('image/'));

    if (uploadedFiles.length === 0) {
        alert('Por favor selecciona imágenes válidas');
        return;
    }

    displayImages();
}

// Display uploaded images
function displayImages() {
    imagesGrid.innerHTML = '';

    uploadedFiles.forEach((file, i) => {
        const reader = new FileReader();
        reader.onload = (e) => {
            const card = document.createElement('div');
            card.className = 'image-card';
            card.innerHTML = `
                <img src="${e.target.result}" alt="${file.name}">
                <div class="info">
                    <div class="filename">${file.name}</div>
                </div>
            `;
            imagesGrid.appendChild(card);
        };
        reader.readAsDataURL(file);
    });

    imagesSection.style.display = 'block';
    predictionsSection.style.display = 'none';
    retrainSection.style.display = 'none';
    resultsSection.style.display = 'none';
}

// Predict
async function predict() {
    showLoading('Analizando imágenes...');

    const formData = new FormData();
    uploadedFiles.forEach(file => formData.append('files', file));

    try {
        const response = await fetch(`${API_URL}/predict`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) throw new Error('Error en predicción');

        predictions = await response.json();
        displayPredictions();
    } catch (error) {
        alert('Error: ' + error.message);
    } finally {
        hideLoading();
    }
}

// Display predictions
function displayPredictions() {
    predictionsGrid.innerHTML = '';

    // Create placeholder cards first to maintain order
    const cards = predictions.map((pred, i) => {
        const card = document.createElement('div');
        card.className = 'prediction-card';
        card.dataset.index = i;
        predictionsGrid.appendChild(card);
        return card;
    });

    // Then load images into the correct cards
    predictions.forEach((pred, i) => {
        const reader = new FileReader();
        reader.onload = (e) => {
            const labelsHtml = CLASSES.map(c => {
                const isActive = pred.labels.includes(c);
                return `<span class="label-tag ${isActive ? '' : 'inactive'}">${c}</span>`;
            }).join('');

            const probsHtml = Object.entries(pred.probabilities).map(([cls, prob]) => `
                <div class="prob-bar">
                    <div class="bar-label">
                        <span>${cls}</span>
                        <span>${Math.round(prob * 100)}%</span>
                    </div>
                    <div class="bar-bg">
                        <div class="bar-fill" style="width: ${prob * 100}%"></div>
                    </div>
                </div>
            `).join('');

            cards[i].innerHTML = `
                <img src="${e.target.result}" alt="${pred.filename}">
                <div class="content">
                    <div class="labels">${labelsHtml}</div>
                    ${probsHtml}
                </div>
            `;
        };
        reader.readAsDataURL(uploadedFiles[i]);
    });

    predictionsSection.style.display = 'block';
}

// Show retrain section
function showRetrainSection() {
    retrainGrid.innerHTML = '';

    // Create placeholder cards first to maintain order
    const cards = predictions.map((pred, i) => {
        const card = document.createElement('div');
        card.className = 'retrain-card';
        card.dataset.index = i;
        retrainGrid.appendChild(card);
        return card;
    });

    // Then load images into the correct cards
    predictions.forEach((pred, i) => {
        const reader = new FileReader();
        reader.onload = (e) => {
            const checkboxesHtml = CLASSES.map(c => {
                const checked = pred.labels.includes(c) ? 'checked' : '';
                return `
                    <label class="checkbox-item">
                        <input type="checkbox" name="labels-${i}" value="${c}" ${checked}>
                        ${c}
                    </label>
                `;
            }).join('');

            cards[i].innerHTML = `
                <img src="${e.target.result}" alt="Imagen ${i + 1}">
                <div class="checkboxes" data-index="${i}">
                    ${checkboxesHtml}
                </div>
            `;
        };
        reader.readAsDataURL(uploadedFiles[i]);
    });

    retrainSection.style.display = 'block';
}

// Retrain
async function retrain() {
    showLoading('Reentrenando modelo (30 epochs)...');

    // Collect labels
    const labels = [];
    document.querySelectorAll('.retrain-card .checkboxes').forEach((container, i) => {
        const checked = Array.from(container.querySelectorAll('input:checked'))
            .map(input => input.value);
        labels.push(checked.length > 0 ? checked : ['ninguna']);
    });

    // Prepare form data
    const formData = new FormData();
    uploadedFiles.forEach(file => formData.append('files', file));
    formData.append('labels', JSON.stringify(labels));

    try {
        const response = await fetch(`${API_URL}/retrain`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) throw new Error('Error en reentrenamiento');

        const result = await response.json();
        displayResults(result, labels);
    } catch (error) {
        alert('Error: ' + error.message);
    } finally {
        hideLoading();
    }
}

// Display results
function displayResults(result, correctLabels) {
    resultsGrid.innerHTML = '';

    // Create placeholder cards first to maintain order
    const cards = result.predictions.map((pred, i) => {
        const card = document.createElement('div');
        card.className = 'result-card';
        card.dataset.index = i;
        resultsGrid.appendChild(card);
        return card;
    });

    // Then load images into the correct cards
    result.predictions.forEach((pred, i) => {
        const reader = new FileReader();
        reader.onload = (e) => {
            const oldLabels = predictions[i]?.labels?.join(', ') || 'N/A';
            const newLabels = pred.predicted_labels.join(', ');
            const correct = pred.correct_labels.join(', ');

            cards[i].innerHTML = `
                <img src="${e.target.result}" alt="${pred.filename}">
                <div class="content">
                    <div class="comparison">
                        <div><strong>Correcta:</strong> ${correct}</div>
                        <div>Antes: ${oldLabels}</div>
                        <div class="arrow">→ Después: ${newLabels}</div>
                    </div>
                </div>
            `;
        };
        reader.readAsDataURL(uploadedFiles[i]);
    });

    retrainSection.style.display = 'none';
    resultsSection.style.display = 'block';
}

// Reset app
function resetApp() {
    uploadedFiles = [];
    predictions = [];
    fileInput.value = '';
    imagesSection.style.display = 'none';
    predictionsSection.style.display = 'none';
    retrainSection.style.display = 'none';
    resultsSection.style.display = 'none';
}

// Load model info
async function loadModelInfo() {
    try {
        const response = await fetch(`${API_URL}/model-info`);
        if (response.ok) {
            const info = await response.json();
            document.getElementById('modelName').textContent = info.model_name;
            document.getElementById('modelF1').textContent =
                info.metrics.f1_score ? info.metrics.f1_score.toFixed(4) : '-';
            document.getElementById('modelPrecision').textContent =
                info.metrics.precision ? info.metrics.precision.toFixed(4) : '-';
            document.getElementById('modelRecall').textContent =
                info.metrics.recall ? info.metrics.recall.toFixed(4) : '-';
            document.getElementById('modelAccuracy').textContent =
                info.metrics.accuracy ? info.metrics.accuracy.toFixed(4) : '-';
        }
    } catch (error) {
        console.log('API no disponible');
    }
}

// Loading functions
function showLoading(text = 'Procesando...') {
    loadingText.textContent = text;
    loading.style.display = 'flex';
}

function hideLoading() {
    loading.style.display = 'none';
}
