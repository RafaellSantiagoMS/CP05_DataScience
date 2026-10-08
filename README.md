# Checkpoint 5 - Preço de apartamentos em São Paulo

Data Science & Statistical Computing - FIAP 2026

Grupo: Enzo Augusto (RM562249), Gustavo Neres (RM561785), Rafaell Santiago (RM563486), Sebastian Iriarte (RM563619)

Continuação do CP04. Usamos a mesma base e o mesmo problema (estimar o preço de venda de apartamentos em São Paulo), agora comparando Random Forest, XGBoost e LightGBM com validação cruzada, Grid Search e Optuna.

## Dados

São Paulo Real Estate - Sale/Rent - April 2019 (Kaggle): https://www.kaggle.com/datasets/argonalyst/sao-paulo-real-estate-sale-rent-april-2019. O notebook lê o CSV direto por URL, então não é preciso baixar nada. Depois do tratamento ficaram 6.240 anúncios de venda.

## Resultado

Os modelos aprendem o log do preço (`log(1 + preço)`), porque o erro de preço é proporcional: errar R$ 100 mil é muito num imóvel de R$ 300 mil e pouco num de R$ 3 milhões. A métrica principal é o RMSLE, o erro na escala do log, que mede o erro em proporção ao preço. Toda previsão volta para reais com `exp(previsão) - 1`.

Os três modelos usaram o mesmo pipeline e os mesmos 5 folds de validação cruzada. O teste (20% da base) só foi usado no final.

| Configuração | RMSLE na validação cruzada | Desvio entre folds | Gap treino x validação |
|---|---|---|---|
| XGBoost (Optuna) | 0,1828 | 0,0068 | 121% |
| XGBoost (Grid Search) | 0,1828 | 0,0056 | 126% |
| LightGBM (Grid Search) | 0,1832 | 0,0080 | 76% |
| LightGBM (Optuna) | 0,1834 | 0,0077 | 99% |

Essas quatro configurações ficaram a menos de 1% do melhor erro, então foram consideradas empatadas. O modelo final é o LightGBM do Grid Search, que tem o menor gap entre treino e validação. No teste: RMSLE de 0,178 (previsões costumam ficar uns 20% acima ou abaixo do preço), RMSE de R$ 184,2 mil, MAE de R$ 85,5 mil e R² de 0,94. A tabela completa com as nove configurações está no Exercício 6 do notebook.

## Arquivos

- `Checkpoint05_RF_XGBoost_LightGBM.ipynb`: notebook com os exercícios 1 a 7
- `app.py`: aplicação Streamlit
- `requirements.txt`: dependências
- `modelo_final.joblib`: pipeline final (preparo dos dados + modelo, que prevê o log do preço), gerado pelo notebook
- `base_tratada.csv`: base depois do tratamento, gerada pelo notebook
- `metricas_teste.csv`: métricas do modelo final no teste, geradas pelo notebook
- `paridade.csv`: anúncio do teste de paridade e a previsão do notebook

## Como rodar

```
pip install -r requirements.txt
streamlit run app.py
```

Para gerar o modelo de novo, execute o notebook do início ao fim; ele recria os arquivos `.joblib` e `.csv` na mesma pasta. O arquivo do modelo precisa das mesmas versões de scikit-learn, xgboost e lightgbm usadas para gerá-lo (as que estão no `requirements.txt`). Se o notebook for rodado em outro ambiente, ajuste as versões do `requirements.txt` para as desse ambiente.

## Teste de paridade

O app mostra o anúncio de `paridade.csv` (35 m² em Artur Alvim) e a previsão do notebook ao lado da previsão do app. As duas são R$ 177.598,32. O mesmo valor aparece ao digitar esse anúncio no formulário.

## Limitações

Os preços são de anúncios de 2019, e não de vendas efetivas. O erro típico fica em torno de 20% do valor do imóvel e, em reais, é maior nos imóveis caros, então o modelo serve como referência de preço, e não como avaliação de um imóvel específico.

## Links

- GitHub: https://github.com/RafaellSantiagoMS/DataScience_CP05
- Aplicação Streamlit: https://datasciencecp05-tnjy9hyxbr6y7vqhhibo4p.streamlit.app/
