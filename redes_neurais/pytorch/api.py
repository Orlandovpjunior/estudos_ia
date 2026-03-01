from typing import List, Dict, Any

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# 2. Constrói os caminhos usando essa base
MODEL_PATH = str(BASE_DIR / 'best_model.pth')
PREPROCESSOR_PATH = str(BASE_DIR / 'preprocessor.pkl')
MAXMINSCALER_PATH = str(BASE_DIR / 'maxminscaler.pkl')

app = FastAPI(title="Preço de veículos - API Inferência")

class Payload(BaseModel):
    records : List[Dict[str, Any]]

# Criar uma arquitetura de rede neural com Pytorch
class NeuralNetwork(nn.Module):
    def __init__(self, input_size,hidden_layer_sizes=[128,64,32,16],output_size=1, dropout_rate = 0.2):
        super(NeuralNetwork,self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_layer_sizes[0])
        self.bn1 = nn.BatchNorm1d(hidden_layer_sizes[0])
        # self.dropout1 = nn.Dropout(dropout_rate)
        self.layer2 = nn.Linear(hidden_layer_sizes[0], hidden_layer_sizes[1])
        self.bn2 = nn.BatchNorm1d(hidden_layer_sizes[1])
        # self.dropout2 = nn.Dropout(dropout_rate)
        self.layer3 = nn.Linear(hidden_layer_sizes[1], hidden_layer_sizes[2])
        self.bn3 = nn.BatchNorm1d(hidden_layer_sizes[2])
        # self.dropout3 = nn.Dropout(dropout_rate)
        self.layer4 = nn.Linear(hidden_layer_sizes[2], hidden_layer_sizes[3])
        self.bn4 = nn.BatchNorm1d(hidden_layer_sizes[3])
        # self.dropout4 = nn.Dropout(dropout_rate)
        self.output_layer = nn.Linear(hidden_layer_sizes[3], output_size)
        self.relu = nn.ReLU()

    def forward(self,X):
        X = self.relu(self.bn1(self.layer1(X)))
        # X = self.dropout1(X)
        X = self.relu(self.bn2(self.layer2(X)))
        # X = self.dropout2(X)
        X = self.relu(self.bn3(self.layer3(X)))
        # X = self.dropout3(X)
        X = self.relu(self.bn4(self.layer4(X)))
        # X = self.dropout4(X)
        X = self.output_layer(X)
        return X
    
def carregar_preprocessors():
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    maxminscaler = joblib.load(MAXMINSCALER_PATH)

    return preprocessor, maxminscaler

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(request: Payload):

    preprocessor , maxminscaler = carregar_preprocessors()

    records = request.records

    df_veiculos = pd.DataFrame(records)

    X_proc = preprocessor.transform(df_veiculos)

    # Carregar modelo treinado X_proc[1] qtde de colunas
    input_size = X_proc.shape[1]
    modelo = NeuralNetwork(input_size=input_size, hidden_layer_sizes=[64,32,16,8], output_size=1)
    modelo.load_state_dict(torch.load(MODEL_PATH))

    X_tensor = torch.tensor(X_proc, dtype=torch.float32)
    modelo.eval()
    with torch.no_grad():
        outputs = modelo(X_tensor)
    
    print(outputs)

    # Inverter a escala dos resultados
    preds = maxminscaler.inverse_transform(outputs).reshape(-1).tolist()

    return {"predictions": preds}



