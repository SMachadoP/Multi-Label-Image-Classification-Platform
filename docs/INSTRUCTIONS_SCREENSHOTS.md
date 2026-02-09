# 📸 Guía para Tomar Capturas de Pantalla del Frontend

## Preparación
✅ Servidor corriendo en: http://127.0.0.1:8000
✅ Navegador abierto
✅ Carpeta creada: `docs/images/`

---

## 📷 Capturas a Tomar

### 1. `upload-interface.png` - Pantalla Inicial
**Qué capturar:**
- Pantalla completa del navegador mostrando la interfaz inicial
- El área de "Clasificación Multi-Label" en el header
- El drag & drop zone con el ícono 📷
- Las etiquetas: person, chair, dog, sofa
- El footer con las métricas del modelo

**Cómo:**
1. Asegúrate de que no haya imágenes cargadas (pantalla inicial limpia)
2. Presiona `Windows + Shift + S` para tomar captura
3. Selecciona el área de la ventana del navegador
4. Guarda como: `docs/images/upload-interface.png`

---

### 2. `predictions.png` - Resultados de Predicciones
**Qué capturar:**
- Sección "Predicciones" con al menos 3-5 imágenes
- Las etiquetas detectadas (badges azules)
- Las barras de probabilidad de cada clase
- El botón "🔄 Reentrenar"

**Cómo:**
1. Sube 3-5 imágenes de perros, personas, sillas o sofás
2. Haz clic en "🔮 Predecir"
3. Espera a que aparezcan los resultados
4. Captura toda la sección de predicciones
5. Guarda como: `docs/images/predictions.png`

**Nota:** Puedes usar imágenes de ejemplo de internet (Google Images):
- Busca: "person with dog"
- Busca: "person sitting on chair"
- Busca: "sofa living room"

---

### 3. `retraining.png` - Interfaz de Reentrenamiento
**Qué capturar:**
- Sección "Corregir Etiquetas"
- Las imágenes con los checkboxes de cada clase
- Al menos 3 imágenes donde estés marcando/corrigiendo etiquetas
- El botón "✅ Confirmar y Reentrenar"

**Cómo:**
1. Desde la pantalla de predicciones, haz clic en "🔄 Reentrenar"
2. Marca o desmarca algunos checkboxes para corregir
3. Captura toda la sección de corrección
4. Guarda como: `docs/images/retraining.png`

---

### 4. `results.png` - Resultados del Reentrenamiento
**Qué capturar:**
- Sección "Resultados del Reentrenamiento"
- Las comparaciones: "Correcta", "Antes", "Después"
- Al menos 3 imágenes mostrando el cambio
- El botón "🔮 Nueva Predicción"

**Cómo:**
1. Después de corregir las etiquetas, haz clic en "✅ Confirmar y Reentrenar"
2. Espera a que termine el reentrenamiento (puede tomar 1-2 minutos)
3. Captura toda la sección de resultados
4. Guarda como: `docs/images/results.png`

---

## 💡 Consejos para Buenas Capturas

1. **Resolución:** Usa pantalla completa o máximo tamaño de ventana
2. **Zoom:** Asegúrate de que el zoom del navegador esté al 100%
3. **Contenido:** Incluye suficiente contexto pero enfócate en lo importante
4. **Calidad:** Formato PNG para mejor calidad
5. **Tamaño:** No te preocupes por el tamaño, GitHub los optimizará

---

## 📤 Subir las Capturas

Una vez tengas las 4 imágenes guardadas en `docs/images/`:

```bash
cd C:\Users\salej\Desktop\Multi-Label_Classification_proyectofinal\Multi-Label_Classification
git add docs/
git commit -m "Add frontend screenshots for documentation"
git push
```

---

## ✅ Checklist

- [ ] upload-interface.png
- [ ] predictions.png
- [ ] retraining.png
- [ ] results.png
- [ ] Todas guardadas en `docs/images/`
- [ ] Subidas al repositorio

---

**Nota:** Si necesitas reiniciar el proceso, simplemente recarga la página en el navegador (F5).
