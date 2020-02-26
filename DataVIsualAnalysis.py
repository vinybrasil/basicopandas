#source: https://mlcourse.ai/articles/topic2-visual-data-analysis-in-python/
#Aula 2
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("darkgrid") #darkgrid, whitegrid, white, dark e ticks
df = pd.read_csv('./data/telecom_churn.csv')
print(df.head())

#=======================Visualizacao univariada====================================

features = ['Total day minutes', 'Total intl calls'] #classe no eixo x e quantidade no y
df[features].hist(figsize=(10, 4))
#kernel density plots:
df[features].plot(kind='density', subplots=True, layout=(1, 2), sharex=False, figsize=(10, 4))
#plt.show()
#o sns.displot() plota o histograma e o kernel density estimate(KDE)
sns.distplot(df['Total intl calls'])
# cuidado ao colocar multiple plots, um deles pode sobrepor o outro(só se for o msm conjunto de dados)
# O box plot: (The box by itself illustrates the interquartile spread of the distribution; its length is 
# determined by the 25th(Q1) and 75th(Q3) percentiles. 
# The vertical line inside the box marks the median (50%) of the distribution.The whiskers are the lines
# extending from the box. They represent the entire scatter of data points, 
# specifically the points that fall within the interval (Q1−1.5⋅IQR,Q3+1.5⋅IQR), 
# where IQR=Q3−Q1 is the interquartile range.)
sns.boxplot(x='Total intl calls', data=df)
#plt.show()
#the violin plot:
_, axes = plt.subplots(1, 2, sharey=True, figsize=(6,4))
sns.boxplot(data=df['Total intl calls'], ax=axes[0])
sns.violinplot(data=df['Total intl calls'], ax=axes[1])
#plt.show()
print(df.describe())
print(df[features].describe())
#frequency table
print(df['Churn'].value_counts()) #Only a small part of the clients canceled their subscription to the telecom service
#bar plot:
_, axes = plt.subplots(nrows=1, ncols=2, figsize=(12,4))
sns.countplot(x='Churn', data=df, ax=axes[0])
sns.countplot(x='Customer service calls', data=df, ax=axes[1])
plt.show()
#===============Visualizacao multivaridada============================================

#Matriz de correlação:

#primeiro tirar as variaveis nao numericas
numericas = list(set(df.columns) - set(['State','International Plan', 'Voice mail plan', 
                                        'Area code', 'Churn', 'Customer service calls' ]))
corr_matriz = df[numericas].corr()
sns.heatmap(corr_matriz)
#tirando as variaveis dependentes:
numericas = list(set(numericas) - set(['Total day charge', 'Total eve charge', 'Total night charge', 'Total intl charge']))
plt.show()
plt.scatter(df['Total day minutes'], df['Total night minutes'])
#juntar dois plots:
sns.set_style("darkgrid")
sns.jointplot(x='Total day minutes', y='Total night minutes', data=df, kind='scatter') 
plt.show()
sns.jointplot('Total day minutes', 'Total night minutes', data=df, kind='kde', color='g')
# %config InlineBackend.figure_format = 'png'
# %config InlineBackend.figure_format = 'retina'
# para arrumar no notebook
#sns.pairplot(df[numericas]) # é mto ruim e pesado

#quantitativa e categorica
sns.lmplot('Total day minutes', 'Total night minutes', data=df, hue='Churn', fit_reg=False)
plt.show()