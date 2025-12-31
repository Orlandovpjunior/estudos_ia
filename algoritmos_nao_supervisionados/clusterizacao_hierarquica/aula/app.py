import streamlit as st
import pandas as pd

# 1. Configurar layout largo para melhor visualização
st.set_page_config(page_title="Recomendador de Laptops", layout="wide")

@st.cache_data
def carregar_dados():
    # Certifique-se de que o caminho do arquivo está correto
    return pd.read_csv('./dataset/clusterizacao_laptops.csv')

df = carregar_dados()

st.sidebar.header("Filtros")

# Selecionar o modelo
model_selected = st.sidebar.selectbox('Selecionar Modelo', df['model'].unique())

# Filtrar o cluster do modelo escolhido
# Usamos .values[0] para pegar o valor do cluster de forma segura
cluster_id = df[df['model'] == model_selected]['cluster'].values[0]

# Filtrar todos os laptops que pertencem ao mesmo cluster
df_recomendados = df[df['cluster'] == cluster_id]

st.title("Sistema de Recomendação de Laptops")
st.write(f"Exibindo modelos similares ao **{model_selected}** (Cluster {cluster_id})")

# 2. Usar dataframe em vez de table para ter barras de rolagem
st.dataframe(df_recomendados)