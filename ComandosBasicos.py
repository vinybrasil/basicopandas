#source: https://medium.com/open-machine-learning-course/open-machine-learning-course-topic-1-exploratory-data-analysis-with-pandas-de57880f1a68
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
print(df.iloc[0:5, 0:3]) #o max não é pego?
#aplicar uma funcao em tudo
print(df.apply(np.max))
#selecionar os estados cujo nome começa com W:
print(df[df['State'].apply(lambda state: state[0] == 'W')].head())
#trocar valores usando um dicionario{old value: new value}:
d = { 'No' : False,
      'Yes': True}
df['International plan'] = df['International plan'].map(d)
print(df.head())
#da pra usar o map ou o replace
df = df.replace({'Voice mail plan': d})
print(df.head())
#grouping data: df.groupby(by=grouping_columns)[columns_to_show].function()
colunas_para_mostrar = ['Total day minutes', 'Total eve minutes', 'Total night minutes']
print(df.groupby(['Churn'])[colunas_para_mostrar].describe(percentiles=[]))
#da pra passar uma lista de funções com o agg:
print(df.groupby(['Churn'])[colunas_para_mostrar].agg([np.mean, np.std, np.min, np.max]))
#crosstab cria uma tabela de contingenciamento
print(pd.crosstab(df['Churn'], df['International plan']))
print(pd.crosstab(df['Churn'], df['Voice mail plan'], normalize=True))
#pivot tables do excel com os parametros: values, index e aggfunc(sum, mean, maximum, etc)
print(df.pivot_table(['Total day calls', 'Total eve calls', 'Total night calls'], ['Area code'], aggfunc='mean'))
#adicinando novas colunas: (nesse caso, uma nova que calcula o total de chamadas)
total_calls = df['Total day calls'] + df['Total eve calls'] + df['Total night calls'] + df['Total intl calls']
df.insert(loc=len(df.columns), column='Total calls', value=total_calls)
print(df.head())
#deletar: use 1 para a coluna e 0 para a linha. inplace=True mexe 
#no df originarl e inplace=false faz um df novo
#dropando colunas
df.drop(['Total calls'], axis=1, inplace=True) #da pra df.drop(['Total charge', 'Total calls'])
#dropando linhas, o index fica zuado
print(df.drop([1,2]).head())

#plotando com o seaborn
print(pd.crosstab(df['Churn'], df['International plan'], margins=True)) #o margins=True poe a coluna do all
plt.rcParams['figure.figsize'] = (8, 6)
sns.countplot(x='International plan', hue='Churn', data=df)
print(pd.crosstab(df['Churn'], df['Customer service calls'], margins=True))
sns.countplot(x='Customer service calls', hue='Churn', data=df)
#plt.show()

#adicionando um binario para Customer service calls > 3
df["Many_services_calls"] = (df['Customer service calls'] > 3).astype('int')
print(pd.crosstab(df['Many_services_calls'], df['Churn'], margins=True))
sns.countplot(x='Many_services_calls', hue='Churn', data=df)
#plt.plot()
#para saber se vai ser churn dado os many services calls
print(pd.crosstab(df['Many_services_calls'] & df['International plan'], df['Churn']))
print(pd.crosstab(df['Many_services_calls'] & df['International plan'], df['Churn'], normalize=True))
#With the help of a simple forecast that can be expressed by the following formula: 
# “(Customer Service calls > 3) & (International plan = True) => Churn = 1, else Churn = 0”,
# we can expect a guessing rate of 85.8%, which is just above 85.5%. 
# Subsequently, we’ll talk about decision trees and figure out how to find such rules 
# automatically based only on the input data;We got these two baselines without applying machine learning,
# and they’ll serve as the starting point for our subsequent models.