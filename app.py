"""
Checkpoint 5 - previsão do preço de apartamentos em São Paulo
Grupo: Enzo Augusto (RM562249), Gustavo Neres (RM561785),
       Rafaell Santiago (RM563486), Sebastian Iriarte (RM563619)

Executar com: streamlit run app.py
"""
import joblib
import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Preço de apartamentos em SP",
    layout="wide",
)

colunas_numericas = ["area_m2", "quartos", "banheiros", "suites", "vagas", "condominio",
                     "elevador", "mobiliado", "piscina", "novo", "latitude", "longitude"]
colunas_x = colunas_numericas + ["distrito"]


@st.cache_data
def carregar_modelo():
    return joblib.load("modelo_final.joblib")


@st.cache_data
def carregar_csv(nome):
    return pd.read_csv(nome)


# mesma função usada no notebook
def limpar_imovel(df):
    df = df.copy()
    df["condominio"] = df["condominio"].replace(0, np.nan)
    fora_sp = ~((df["latitude"] >= -24) & (df["latitude"] <= -23.3) &
                (df["longitude"] >= -47) & (df["longitude"] <= -46.3))
    df.loc[fora_sp, "latitude"] = np.nan
    df.loc[fora_sp, "longitude"] = np.nan
    return df


def reais(valor):
    # formato brasileiro: R$ 1.234.567,89
    texto = f"{valor:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return "R$ " + texto


modelo = carregar_modelo()
base = carregar_csv("base_tratada.csv")
metricas = carregar_csv("metricas_teste.csv")
paridade = carregar_csv("paridade.csv")

st.title("Quanto vale um apartamento em São Paulo?")
st.write(
    "Estimativa do preço de venda anunciado de apartamentos em São Paulo. "
    "Dados: São Paulo Real Estate - Sale/Rent - April 2019 (Kaggle). "
    f"Modelo: {metricas['Modelo'][0]}."
)

col1, col2, col3, col4 = st.columns(4)
col1.metric("RMSLE no teste", f"{metricas['RMSLE'][0]:.3f}".replace(".", ","))
col2.metric("RMSE no teste", reais(metricas["RMSE"][0]))
col3.metric("MAE no teste", reais(metricas["MAE"][0]))
col4.metric("R² no teste", f"{metricas['R2'][0]:.3f}".replace(".", ","))


# =========================================================
# DADOS DO APARTAMENTO
# =========================================================

st.sidebar.header("Dados do apartamento")

distritos = sorted(base["distrito"].unique())
distrito = st.sidebar.selectbox("Distrito", distritos, index=distritos.index("Moema"))
area = st.sidebar.number_input("Área (m²)", min_value=20, max_value=1000, value=70, step=1)
quartos = st.sidebar.number_input("Quartos", min_value=1, max_value=10, value=2, step=1)
suites = st.sidebar.number_input("Suítes", min_value=0, max_value=10, value=1, step=1)
banheiros = st.sidebar.number_input("Banheiros", min_value=1, max_value=10, value=2, step=1)
vagas = st.sidebar.number_input("Vagas", min_value=0, max_value=10, value=1, step=1)
condominio = st.sidebar.number_input("Condomínio (R$/mês, 0 se não souber)", min_value=0, max_value=20000, value=0, step=50)
elevador = st.sidebar.selectbox("Elevador", ["Sim", "Não"])
piscina = st.sidebar.selectbox("Piscina", ["Não", "Sim"])
mobiliado = st.sidebar.selectbox("Mobiliado", ["Não", "Sim"])
novo = st.sidebar.selectbox("Imóvel novo", ["Não", "Sim"])

# localização: começa no centro típico do distrito (mediana das coordenadas da base)
centro = base.groupby("distrito")[["latitude", "longitude"]].median()
latitude = st.sidebar.number_input("Latitude", min_value=-24.0, max_value=-23.3,
                                   value=float(centro.loc[distrito, "latitude"]), format="%.6f")
longitude = st.sidebar.number_input("Longitude", min_value=-47.0, max_value=-46.3,
                                    value=float(centro.loc[distrito, "longitude"]), format="%.6f")

entrada = pd.DataFrame({
    "area_m2": [area],
    "quartos": [quartos],
    "banheiros": [banheiros],
    "suites": [suites],
    "vagas": [vagas],
    "condominio": [float(condominio)],
    "elevador": [1 if elevador == "Sim" else 0],
    "mobiliado": [1 if mobiliado == "Sim" else 0],
    "piscina": [1 if piscina == "Sim" else 0],
    "novo": [1 if novo == "Sim" else 0],
    "latitude": [latitude],
    "longitude": [longitude],
    "distrito": [distrito],
})


# =========================================================
# PREVISÃO
# =========================================================

# o modelo prevê o log do preço; a conta inversa volta para reais
previsao = np.exp(modelo.predict(limpar_imovel(entrada))[0]) - 1

st.subheader("Preço estimado")
st.metric("Previsão do modelo", reais(previsao))
erro_tipico = 100 * (np.exp(metricas["RMSLE"][0]) - 1)
st.write(
    f"Em dados novos, a previsão costuma ficar uns {erro_tipico:.0f}% acima ou abaixo do preço anunciado "
    f"(RMSLE no teste) e o erro médio é de {reais(metricas['MAE'][0])} (MAE no teste)."
)

if area < base["area_m2"].min() or area > base["area_m2"].max():
    st.info("A área digitada está fora da faixa vista no treino; a previsão pode ser pouco confiável.")
if suites > quartos:
    st.info("O número de suítes é maior que o de quartos.")


# =========================================================
# TESTE DE PARIDADE
# =========================================================

st.subheader("Teste de paridade")
st.write(
    "Anúncio do conjunto de teste usado no notebook. O app aplica a mesma limpeza e o mesmo "
    "modelo salvo; as duas previsões precisam ser iguais. Digitando esses valores no formulário "
    "ao lado (condomínio vazio = 0), a previsão acima também fica igual."
)
st.dataframe(paridade[colunas_x], hide_index=True)

previsao_paridade = np.exp(modelo.predict(limpar_imovel(paridade[colunas_x]))[0]) - 1
col1, col2 = st.columns(2)
col1.metric("Previsão no notebook", reais(paridade["previsao_notebook"][0]))
col2.metric("Previsão no app", reais(previsao_paridade))

with st.expander("Ver amostra da base tratada"):
    st.dataframe(base.sample(10, random_state=1))
    st.dataframe(base.describe())
