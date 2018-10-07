import matplotlib.pyplot as plt
from sklearn import linear_model
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import eli5
from eli5.sklearn import PermutationImportance

train = pd.read_csv('/Users/walkerrowe/Documents/keras/votes.csv',usecols=['Trump','White','Black','Hispanic','age65plus','Edu_highschool','SEX255214'])
 
xx = train[['White','Black','Hispanic','age65plus','Edu_highschool','SEX255214']].values
yy = train[['Trump']].values
reg = linear_model.LinearRegression()
model=reg.fit(xx,yy)
print('Coefficients: \n', reg.coef_)
perm = PermutationImportance(reg, random_state=1).fit(xx, yy)
eli5.show_weights(perm)
