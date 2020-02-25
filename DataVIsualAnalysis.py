#source: https://mlcourse.ai/articles/topic2-visual-data-analysis-in-python/
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('./data/telecom_churn.csv')
print(df.head())
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
