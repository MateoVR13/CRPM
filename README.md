# Predicción de Recurrencia del Cáncer Tiroideo

Este proyecto explora el uso de modelos de aprendizaje automático para predecir la recurrencia del cáncer tiroideo. Utilizamos datos históricos de pacientes y varias técnicas de modelado para analizar patrones y mejorar la precisión predictiva.

## Modelos Utilizados

Los siguientes algoritmos de clasificación se emplearon para construir modelos predictivos:

- **XGBoost**
- **Logistic Regression**
- **Random Forest**
- **AdaBoost**

## Flujo del Análisis

1. **Exploración de Datos**:
   - Se cargaron y analizaron los datos desde un archivo CSV (`thyroid_dyff.csv`).
   - Visualizaciones iniciales para entender la distribución de las características.

2. **Preprocesamiento**:
   - División de los datos en conjuntos de entrenamiento y prueba (80%-20%).
   - Selección de características relevantes.

3. **Entrenamiento y Evaluación**:
   - Entrenamiento de cada modelo con `scikit-learn` y `xgboost`.
   - Evaluación basada en métricas como precisión (accuracy), AUC-ROC y matrices de confusión.
   - Generación de curvas ROC para cada modelo.

## Resultados

Cada modelo fue evaluado en un conjunto de prueba independiente. Los resultados incluyen:

- **Precisión**: Varió entre los modelos, destacando XGBoost por su desempeño constante.
- **AUC-ROC**: Métrica utilizada para evaluar la capacidad discriminativa de los modelos.
- **Matriz de Confusión**: Proporcionó información sobre las tasas de falsos positivos y falsos negativos.

## Visualizaciones

Las visualizaciones incluyen:

- **Matriz de Confusión**: Para cada modelo.
- **Curva ROC**: Comparación gráfica del desempeño de los modelos.

## Requisitos

- Python 3.x
- Bibliotecas:
  - `pandas`, `numpy`, `matplotlib`, `seaborn`, `plotly`
  - `scikit-learn`, `xgboost`

## Ejecución

1. Clonar el repositorio.
2. Instalar dependencias usando `pip install -r requirements.txt`.
3. Ejecutar el archivo Jupyter Notebook para reproducir los resultados.

## Contribuciones

Las contribuciones al proyecto son bienvenidas. Por favor, abre un issue o envía un pull request para discutir cambios o mejoras.

---

*Este proyecto fue realizado para estudiar la viabilidad de modelos predictivos en el ámbito médico.*