import streamlit as st
import json
import requests

st.title("Modelo de Predição de Receita de vendas")
st.write("Quantos meses o profissional tem de experiência?")
tempo_de_experiencia = st.slider("Meses", min_value=1,max_value=119, value=60,step=1)
st.write("Qual o numero de vendas do profissional na empresa?")
numero_de_vendas = st.slider("Nível", min_value=10,max_value=100, value=20,step=1)
st.write("Qual o fator sazonal?")
fator_sazonal = st.slider("Nível", min_value=1,max_value=10, value=5,step=1)

input_features = {
    'tempo_de_experiencia': tempo_de_experiencia,
    'numero_de_vendas': numero_de_vendas,
    'fator_sazonal': fator_sazonal
}

if st.button("Estimar Receita"):
    res = requests.post("http://127.0.0.1:8000/predict",data=json.dumps(input_features))
    retorno_json = json.loads(res.text)
    receita_em_reais = round(retorno_json['receita_em_reais'], 2)
    st.subheader(f'O receita estimada é de R$ {receita_em_reais}')

# streamlit run app_streamlit_vendas.py 