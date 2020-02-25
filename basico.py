#source: https://medium.com/open-machine-learning-course/open-machine-learning-course-topic-1-exploratory-data-analysis-with-pandas-de57880f1a68
import pandas as pd
import numpy as np

df = pd.read_csv('./data/telecom_churn.csv')

print(df.head()) #abre as 5 primeiras linhas
print(df.shape) #3333 linhas e 20 colunas
print(df.columns) #os nomes das colunas
print(df.info()) #1 é bool, 8 são float, 8 sao inteiros e 3 sao objetos
#astype muda o tipo da coluna: df['Churn'] = df['Churn'].astype('int64'), mudando de false para 0 e true para 1
df['Churn'] = df['Churn'].astype('int64')
print(df.describe()) #faz o calculo da media nas colunas e calcula o basico das outras
print(df["Churn"].mean()) #pode-se pegar o valor por coluna, por exemplo
print(df.describe(include=['object', 'bool']))
print(df['Churn'].value_counts()) #conta eles
print(df['Churn'].value_counts(normalize=True)) #proporção
#descendente: df.sort_values(by=['Churn', 'Total day charge'], ascending=[True, False]).head()

#pegar a media de onde os churn são = 1, onde se constrói temporariamente um novo df só com essas informações
print(df[df['Churn'] == 1].mean())
#ou só um valor dessa df
print(df[df['Churn'] == 1]['Total day minutes'].mean())
print(df[(df['Churn'] == 0) & (df['International plan'] == 'No')]['Total intl minutes'].max())
#loc acha por nome e iloc acha por index
print(df.loc[0:5, 'State':'Area code'])
print(df.iloc[0:5, 0:3])
#aplicar uma funcao em tudo
