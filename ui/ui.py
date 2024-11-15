import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Configuración de la página
st.set_page_config(
    page_title="Predictor de Recurrencia",
    layout="wide"
)

st.title("Predictor de Recurrencia")
st.write("Esta aplicación predice la probabilidad de recurrencia usando diferentes modelos de Machine Learning.")

# Diccionarios para codificación
GENDER_MAP = {'M': 0, 'F': 1}
BINARY_MAP = {'No': 0, 'Yes': 1}
NORMAL_MAP = {'Normal': 0, 'Abnormal': 1}
PATHOLOGY_MAP = {'Type1': 0, 'Type2': 1, 'Type3': 2}
FOCALITY_MAP = {'Unifocal': 0, 'Multifocal': 1}
RISK_MAP = {'Low': 0, 'Medium': 1, 'High': 2}
T_MAP = {'T1': 0, 'T2': 1, 'T3': 2, 'T4': 3}
N_MAP = {'N0': 0, 'N1': 1, 'N2': 2, 'N3': 3}
M_MAP = {'M0': 0, 'M1': 1}
STAGE_MAP = {'I': 0, 'II': 1, 'III': 2, 'IV': 3}
RESPONSE_MAP = {'None': 0, 'Partial': 1, 'Complete': 2}

# Cargar los modelos y el scaler
@st.cache_resource
def load_models():
    models = {
        "Random Forest": joblib.load("Random Forest_model.pkl"),
        "Ada Boost": joblib.load("Ada Boost_model.pkl"),
        "Logistic Regression": joblib.load("Logistic Regression_model.pkl"),
        "XGBoost": joblib.load("XGBoost_model.pkl")
    }
    scaler = joblib.load("scaler.pkl")
    return models, scaler

# Function to get prediction and probability
def get_prediction_and_probability(model, X):
    prediction = model.predict(X)[0]
    
    # Check if model has predict_proba method
    if hasattr(model, 'predict_proba'):
        probability = model.predict_proba(X)[0][1]
    else:
        # For SVM without probability estimation, use decision_function
        if hasattr(model, 'decision_function'):
            decision = model.decision_function(X)[0]
            # Convert decision function to probability-like score using sigmoid function
            probability = 1 / (1 + np.exp(-decision))
        else:
            # If neither method is available, use binary prediction as probability
            probability = float(prediction)
    
    return prediction, probability

try:
    models, scaler = load_models()
    st.success("✅ Modelos cargados exitosamente")
except Exception as e:
    st.error(f"Error al cargar los modelos: {str(e)}")
    st.stop()

# Crear el formulario de entrada
with st.form("prediction_form"):
    st.write("### Ingrese los datos del paciente")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        form_age = st.number_input("Age", min_value=0, max_value=120, value=50)
        form_gender = st.selectbox("Gender", list(GENDER_MAP.keys()))
        form_smoking = st.selectbox("Smoking", list(BINARY_MAP.keys()))
        form_hx_smoking = st.selectbox("Hx Smoking", list(BINARY_MAP.keys()))
        form_hx_radiotherapy = st.selectbox("Hx Radiotherapy", list(BINARY_MAP.keys()))
        
    with col2:
        form_adenopathy = st.selectbox("Adenopathy", list(BINARY_MAP.keys()))
        form_focality = st.selectbox("Focality", list(FOCALITY_MAP.keys()))
        form_risk = st.selectbox("Risk", list(RISK_MAP.keys()))
        form_stage = st.selectbox("Stage", list(STAGE_MAP.keys()))
        
    with col3:
        form_t_stage = st.selectbox("T", list(T_MAP.keys()))
        form_n_stage = st.selectbox("N", list(N_MAP.keys()))
        form_m_stage = st.selectbox("M", list(M_MAP.keys()))
        form_response = st.selectbox("Response", list(RESPONSE_MAP.keys()))
    
    submitted = st.form_submit_button("Predecir")

    # Procesar el formulario cuando se envía
    if submitted:
        # Preparar los datos para la predicción
        input_data = {
            'Age': form_age,
            'Gender': GENDER_MAP[form_gender],
            'Smoking': BINARY_MAP[form_smoking],
            'Hx Smoking': BINARY_MAP[form_hx_smoking],
            'Hx Radiothreapy': BINARY_MAP[form_hx_radiotherapy],
            'Adenopathy': BINARY_MAP[form_adenopathy],
            'Focality': FOCALITY_MAP[form_focality],
            'Risk': RISK_MAP[form_risk],
            'T': T_MAP[form_t_stage],
            'N': N_MAP[form_n_stage],
            'M': M_MAP[form_m_stage],
            'Stage': STAGE_MAP[form_stage],
            'Response': RESPONSE_MAP[form_response]
        }
        
        # Convertir a DataFrame
        input_df = pd.DataFrame([input_data])
        
        try:
            # Escalar los datos
            input_scaled = scaler.transform(input_df)
            
            # Hacer predicciones con todos los modelos
            st.write("### Resultados de la predicción")
            
            col1, col2 = st.columns(2)
            
            probabilities = []
            with col1:
                st.write("#### Predicciones por modelo:")
                for name, model in models.items():
                    prediction, probability = get_prediction_and_probability(model, input_scaled)
                    probabilities.append(probability)
                    
                    st.write(f"**{name}:**")
                    st.write(f"- Predicción: {'Recurrencia' if prediction == 1 else 'No recurrencia'}")
                    st.write(f"- Probabilidad de recurrencia: {probability:.2%}")
                    st.write("---")
            
            with col2:
                # Calcular el promedio de las probabilidades
                avg_probability = np.mean(probabilities)
                
                st.write("#### Consenso de los modelos:")
                st.write(f"Probabilidad promedio de recurrencia: {avg_probability:.2%}")
                                
                # Visualizar el nivel de riesgo
                risk_level = "Alto" if avg_probability > 0.66 else "Medio" if avg_probability > 0.33 else "Bajo"
                risk_color = "red" if risk_level == "Alto" else "yellow" if risk_level == "Medio" else "green"

                st.markdown(f"""
                <div style='background-color: {risk_color}; padding: 10px; border-radius: 5px;'>
                    <h4 style='color: black; margin: 0;'>Nivel de riesgo: {risk_level}</h4>
                </div>
                """, unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"Error al procesar los datos: {str(e)}")
            st.write("DataFrame de entrada:")
            st.write(input_df)

# Agregar información adicional
st.sidebar.write("### Información del modelo")
st.sidebar.write("""
Esta aplicación utiliza varios modelos de machine learning para predecir la probabilidad de recurrencia basándose en diferentes características del paciente y del tumor.

Los modelos utilizados son:
- Random Forest
- Support Vector Machine
- Ada Boost
- Gradient Boosting
- Logistic Regression
- XGBoost

La predicción final se basa en el consenso de todos estos modelos.
""")

st.sidebar.write("### Explicación de Variables")
st.sidebar.write("""
**Variables del Paciente:**
- Age: Edad del paciente en años
- Gender: Sexo del paciente (M/F)
- Smoking: Indica si el paciente fuma actualmente
- Hx Smoking: Historia previa de tabaquismo
- Hx Radiotherapy: Historia previa de radioterapia

**Variables Clínicas:**
- Thyroid Function: Estado de la función tiroidea (Normal/Anormal)
- Physical Examination: Resultado del examen físico (Normal/Anormal)
- Adenopathy: Presencia de adenopatías (Sí/No)

**Características del Tumor:**
- Pathology: Tipo patológico del tumor (Type1/Type2/Type3)
- Focality: Distribución del tumor (Unifocal/Multifocal)
- Risk: Nivel de riesgo evaluado (Low/Medium/High)

**Estadificación:**
- T: Tamaño del tumor primario (T1-T4)
- N: Afectación de ganglios linfáticos (N0-N3)
- M: Presencia de metástasis (M0/M1)
- Stage: Fase general de la enfermedad (I-IV)

**Respuesta al Tratamiento:**
- Response: Nivel de respuesta al tratamiento (None/Partial/Complete)
""")

st.sidebar.write("### Interpretación")
st.sidebar.write("""
- Probabilidad < 33%: Riesgo Bajo
- Probabilidad 33-66%: Riesgo Medio
- Probabilidad > 66%: Riesgo Alto

Nota: Esta herramienta es solo para fines informativos y no debe utilizarse como único criterio para la toma de decisiones médicas.
""")