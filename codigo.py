#referencias: https://medium.com/open-machine-learning-course/open-machine-learning-course-topic-9-time-series-analysis-in-python-a270cb05e0b3
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns #plotador2

from dateutil.relativedelta import relativedelta
from scipy.optimize import minimize

import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt
import statsmodels.api as smf
import scipy.stats as scs

from itertools import product
import warnings
warnings.filterwarnings('ignore')

ads = pd.read_csv("./data/ads.csv", index_col=['Time'], parse_dates=['Time'])
currency = pd.read_csv('./data/currency.csv', index_col=['Time'], parse_dates=['Time'])

plt.figure(figsize=(15,7))
plt.plot(ads.Ads) #a coluna Ads do dataframe ads    
plt.title("Ads watched (hourly data)")
plt.grid(True)
plt.show()

plt.figure(figsize=(15,7))
plt.plot(currency.GEMS_GEMS_SPENT)
plt.title("In-game currency spent(daily data)")
plt.grid(True)
plt.show()



