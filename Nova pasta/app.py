
import streamlit as st    # interface grafica 
import pandas as pd       # tratamento de dados
from sklearn.linear_model import LinearRegression # o tipo de treinamento do modelo

# NOTAS DE ESTUDOS 

import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

st.header('ANALISE DE NOTAS - PREVENDO')

estudos = pd.DataFrame({
'notas':[1,2,4,6,8,10],
'horas':[2,4,5,7,9,10]
})

#st.scatter_chart(estudos, x = 'horas', y= 'notas')
modelo_escola = LinearRegression() 
modelo_escola.fit(estudos[['horas']], estudos['notas'])

h_estudo = st.slider('horas de estudos', 0,12,5)
nota_final = modelo_escola.predict([[h_estudo]])
print(nota_final)

st.metric(f'sua nota seria' ,f'{min(nota_final[0], 10.0):.1f}')



# st.header('PREVISÃO DE VENDAS')
# dados_vendas = pd.DataFrame({

#    'investimentos':[100,200,300,550,750,800],
#    'faturamento':[1200,2500,3700,3900,5500,6900]

# })

# st.write(dados_vendas)

# # treinar os dados 
# X = dados_vendas[['investimentos']]
# y = dados_vendas['faturamento']

# model = LinearRegression().fit(X,y) # treina o modelo com os dados

# investimento =  st.number_input('Digite o investimento', value = 150)

# if investimento:
#    if st.button('Analisar:'):
    
#         previsao = model.predict([[investimento]])[0] #previsão
#         st.write(f'Faturamento -  previsto R${previsao:.2f} **')# resultado





# Analisar a previsão de vendas do mes de setembro

import streamlit as st    # interface grafica 
import pandas as pd       # tratamento de dados
from sklearn.linear_model import LinearRegression # o tipo de treinamento do modelo


dados =  pd.read_csv('vendas.csv')
df  =  pd.DataFrame(dados)

print(df)




