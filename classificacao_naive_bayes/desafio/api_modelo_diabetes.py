from flask import Flask, request,jsonify
from pydantic import BaseModel, ValidationError
import joblib
import pandas as pd

app = Flask(__name__)

class request_body(BaseModel):
    glicemia: int
    pressao_arterial: int

try:
    modelo_diabetes = joblib.load('./modelo_diabetes.pkl')
except FileNotFoundError:
    print("Erro: Arquivo 'modelo_diabetes.pkl' não encontrado. Certifique-se de que ele está no mesmo diretório.")
    modelo_diabetes = None

@app.route("/predict", methods=['POST'])
def predict():
    if modelo_diabetes is None:
        return jsonify({"Erro": "Modelo de predição não está disponível"}), 500
    json_data = request.get_json()
    if not json_data:
        return jsonify({"erro": "Corpo da requisição vazio ou em formato inválido."}), 400
    try:
        body = request_body.model_validate(json_data)
    except ValidationError as e:
        return jsonify({"erro": "Dados de entrada inválidos", "detalhes": e.errors()}), 400
    
    predict_df = pd.DataFrame(body.model_dump(),index=[0])

    features_necessarias = [
        'pressao_arterial'
    ]

    predict_df = predict_df[features_necessarias]

    y_pred = modelo_diabetes.predict(predict_df)
    resultado = int(y_pred[0])

    return jsonify({'diabetes': resultado})

if __name__ == '__main__':
    app.run(port=5000, debug=True)