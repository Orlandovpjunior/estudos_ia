from flask import Flask, request, jsonify
from pydantic import BaseModel, ValidationError
import joblib
import pandas as pd

# Inicializa a aplicação Flask
app = Flask(__name__)

# Define a estrutura do corpo da requisição usando Pydantic
class request_body(BaseModel):
    Genero_Masculino: int
    Idade: int
    Historico_Familiar_Sobrepeso:int
    Consumo_Alta_Caloria_Com_Frequencia: int
    Consumo_Vegetais_Com_Frequencia: int
    Refeicoes_Dia:int
    Consumo_Alimentos_entre_Refeicoes: int
    Fumante: int
    Consumo_Agua: int
    Monitora_Calorias_Ingeridas: int
    Nivel_Atividade_Fisica: int
    Nivel_Uso_Tela: int
    Consumo_Alcool: int
    Transporte_Automovel: int
    Transporte_Bicicleta: int
    Transporte_Motocicleta: int
    Transporte_Publico: int
    Transporte_Caminhada: int

# Carrega o modelo de machine learning treinado
try:
    modelo_obesidade = joblib.load('./modelo_obesidade.pkl')
except FileNotFoundError:
    print("Erro: Arquivo 'modelo_obesidade.pkl' não encontrado. Certifique-se de que ele está no mesmo diretório.")
    modelo_obesidade = None

# Define a rota para predição, aceitando apenas o método POST
@app.route("/predict", methods=['POST'])
def predict():
    # Verifica se o modelo foi carregado corretamente
    if modelo_obesidade is None:
        return jsonify({"erro": "Modelo de predição não está disponível."}), 500

    # 1. Obter o corpo da requisição JSON
    json_data = request.get_json()
    if not json_data:
        return jsonify({"erro": "Corpo da requisição vazio ou em formato inválido."}), 400

    try:
        # 2. Validar os dados recebidos usando o modelo Pydantic
        body = request_body.model_validate(json_data)
    except ValidationError as e:
        # Retorna um erro 400 (Bad Request) se os dados forem inválidos
        return jsonify({"erro": "Dados de entrada inválidos", "detalhes": e.errors()}), 400

    # 3. Transformar os dados validados em um DataFrame
    # Usamos model_dump() para converter o objeto Pydantic em um dicionário
    predict_df = pd.DataFrame(body.model_dump(), index=[0])

    # 4. Criar a feature 'Faixa_Etaria'
    bins = [10,20,30,40,50,60,70]
    bins_ordinal = [0,1,2,3,4,5]
    predict_df['Faixa_Etaria'] = pd.cut(x=predict_df['Idade'], bins=bins, labels=bins_ordinal, include_lowest=True)

    # 5. Selecionar apenas as features que o modelo espera
    features_necessarias = [
        'Historico_Familiar_Sobrepeso',
        'Consumo_Alta_Caloria_Com_Frequencia',
        'Consumo_Alimentos_entre_Refeicoes',
        'Monitora_Calorias_Ingeridas',
        'Nivel_Atividade_Fisica',
        'Nivel_Uso_Tela',
        'Transporte_Caminhada',
        'Faixa_Etaria'
    ]
    predict_df = predict_df[features_necessarias]

    # 6. Realizar a predição
    y_pred = modelo_obesidade.predict(predict_df)
    
    # Extrai o primeiro valor da predição e converte para um tipo nativo do Python
    resultado_predicao = int(y_pred[0])

    # 7. Retornar o resultado da predição em formato JSON
    return jsonify({'obesidade': resultado_predicao})

if __name__ == '__main__':
    app.run(port=5000, debug=True)
