import tkinter as tk
from tkinter import ttk
import joblib
import pandas as pd
import numpy as np

class CancerPredictorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Predictor de Recurrencia")
        
        # Diccionarios para codificación
        self.GENDER_MAP = {'M': 0, 'F': 1}
        self.BINARY_MAP = {'No': 0, 'Yes': 1}
        self.NORMAL_MAP = {'Normal': 0, 'Abnormal': 1}
        self.PATHOLOGY_MAP = {'Type1': 0, 'Type2': 1, 'Type3': 2}
        self.FOCALITY_MAP = {'Unifocal': 0, 'Multifocal': 1}
        self.RISK_MAP = {'Low': 0, 'Medium': 1, 'High': 2}
        self.T_MAP = {'T1': 0, 'T2': 1, 'T3': 2, 'T4': 3}
        self.N_MAP = {'N0': 0, 'N1': 1, 'N2': 2, 'N3': 3}
        self.M_MAP = {'M0': 0, 'M1': 1}
        self.STAGE_MAP = {'I': 0, 'II': 1, 'III': 2, 'IV': 3}
        self.RESPONSE_MAP = {'None': 0, 'Partial': 1, 'Complete': 2}
        
        # Cargar modelos
        try:
            self.models = {
                "Random Forest": joblib.load("Random Forest_model.pkl"),
                "Ada Boost": joblib.load("Ada Boost_model.pkl"),
                "Logistic Regression": joblib.load("Logistic Regression_model.pkl"),
                "XGBoost": joblib.load("XGBoost_model.pkl")
            }
            self.scaler = joblib.load("scaler.pkl")
        except Exception as e:
            tk.messagebox.showerror("Error", f"Error al cargar los modelos: {str(e)}")
            root.destroy()
            return

        self.create_widgets()

    def create_widgets(self):
        # Frame principal
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Variables para almacenar los valores
        self.age_var = tk.StringVar(value="50")
        self.gender_var = tk.StringVar(value="M")
        self.smoking_var = tk.StringVar(value="No")
        self.hx_smoking_var = tk.StringVar(value="No")
        self.hx_radiotherapy_var = tk.StringVar(value="No")
        self.adenopathy_var = tk.StringVar(value="No")
        self.focality_var = tk.StringVar(value="Unifocal")
        self.risk_var = tk.StringVar(value="Low")
        self.t_stage_var = tk.StringVar(value="T1")
        self.n_stage_var = tk.StringVar(value="N0")
        self.m_stage_var = tk.StringVar(value="M0")
        self.stage_var = tk.StringVar(value="I")
        self.response_var = tk.StringVar(value="None")

        # Crear campos del formulario
        row = 0
        # Columna 1
        ttk.Label(main_frame, text="Age:").grid(row=row, column=0, sticky=tk.W)
        ttk.Entry(main_frame, textvariable=self.age_var).grid(row=row, column=1)
        row += 1

        self.create_combobox(main_frame, "Gender:", self.gender_var, self.GENDER_MAP.keys(), row)
        row += 1
        self.create_combobox(main_frame, "Smoking:", self.smoking_var, self.BINARY_MAP.keys(), row)
        row += 1
        self.create_combobox(main_frame, "Hx Smoking:", self.hx_smoking_var, self.BINARY_MAP.keys(), row)
        row += 1
        self.create_combobox(main_frame, "Hx Radiotherapy:", self.hx_radiotherapy_var, self.BINARY_MAP.keys(), row)
        row += 1

        # Columna 2

        self.create_combobox(main_frame, "Adenopathy:", self.adenopathy_var, self.BINARY_MAP.keys(), row)
        row += 1
        self.create_combobox(main_frame, "Focality:", self.focality_var, self.FOCALITY_MAP.keys(), row)
        row += 1

        # Columna 3
        self.create_combobox(main_frame, "Risk:", self.risk_var, self.RISK_MAP.keys(), row)
        row += 1
        self.create_combobox(main_frame, "T:", self.t_stage_var, self.T_MAP.keys(), row)
        row += 1
        self.create_combobox(main_frame, "N:", self.n_stage_var, self.N_MAP.keys(), row)
        row += 1
        self.create_combobox(main_frame, "M:", self.m_stage_var, self.M_MAP.keys(), row)
        row += 1
        self.create_combobox(main_frame, "Stage:", self.stage_var, self.STAGE_MAP.keys(), row)
        row += 1
        self.create_combobox(main_frame, "Response:", self.response_var, self.RESPONSE_MAP.keys(), row)
        row += 1

        # Botón de predicción
        ttk.Button(main_frame, text="Predecir", command=self.predict).grid(row=row, column=0, columnspan=2, pady=10)

        # Área de resultados
        self.result_text = tk.Text(main_frame, height=10, width=50)
        self.result_text.grid(row=row+10, column=0, columnspan=2, pady=10)

    def create_combobox(self, parent, label, variable, values, row):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky=tk.W)
        ttk.Combobox(parent, textvariable=variable, values=list(values), state="readonly").grid(row=row, column=1)

    def get_prediction_and_probability(self, model, X):
        prediction = model.predict(X)[0]
        
        if hasattr(model, 'predict_proba'):
            probability = model.predict_proba(X)[0][1]
        else:
            if hasattr(model, 'decision_function'):
                decision = model.decision_function(X)[0]
                probability = 1 / (1 + np.exp(-decision))
            else:
                probability = float(prediction)
        
        return prediction, probability

    def predict(self):
        try:
            input_data = {
                'Age': float(self.age_var.get()),
                'Gender': self.GENDER_MAP[self.gender_var.get()],
                'Smoking': self.BINARY_MAP[self.smoking_var.get()],
                'Hx Smoking': self.BINARY_MAP[self.hx_smoking_var.get()],
                'Hx Radiothreapy': self.BINARY_MAP[self.hx_radiotherapy_var.get()],
                'Adenopathy': self.BINARY_MAP[self.adenopathy_var.get()],
                'Focality': self.FOCALITY_MAP[self.focality_var.get()],
                'Risk': self.RISK_MAP[self.risk_var.get()],
                'T': self.T_MAP[self.t_stage_var.get()],
                'N': self.N_MAP[self.n_stage_var.get()],
                'M': self.M_MAP[self.m_stage_var.get()],
                'Stage': self.STAGE_MAP[self.stage_var.get()],
                'Response': self.RESPONSE_MAP[self.response_var.get()]
            }

            input_df = pd.DataFrame([input_data])
            input_scaled = self.scaler.transform(input_df)

            self.result_text.delete(1.0, tk.END)
            probabilities = []

            for name, model in self.models.items():
                prediction, probability = self.get_prediction_and_probability(model, input_scaled)
                probabilities.append(probability)
                
                self.result_text.insert(tk.END, f"{name}:\n")
                self.result_text.insert(tk.END, f"Predicción: {'Recurrencia' if prediction == 1 else 'No recurrencia'}\n")
                self.result_text.insert(tk.END, f"Probabilidad: {probability:.2%}\n\n")

            avg_probability = np.mean(probabilities)
            risk_level = "Alto" if avg_probability > 0.66 else "Medio" if avg_probability > 0.33 else "Bajo"
            
            self.result_text.insert(tk.END, f"\nPromedio de probabilidad: {avg_probability:.2%}\n")
            self.result_text.insert(tk.END, f"Nivel de riesgo: {risk_level}\n")

        except Exception as e:
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, f"Error al procesar los datos: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = CancerPredictorApp(root)
    root.mainloop()