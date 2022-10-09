# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 08:45:00 2022
@author: sahakyz
"""

# First install data that comes with the textbook  
# Type in the right lower window $pip install wooldridge

#%%
import wooldridge as woo

#%%
# After that we need our standard packages

import pandas as pd # data import and manipulation, pandas stands for Python Data Analysis Library 
import numpy as np # math operations
import statsmodels.formula.api as smf #regressions
import seaborn as sns # library for plotting /easy version
import matplotlib.pyplot as plt # library for plotting
from scipy import stats #library for statistical functions

#%%
# Load Data wage2.csv file

wage2df = woo.dataWoo('wage2')

#%%

#%%
'Drop missing observations'
wage2df.dropna(inplace=True) #Drop NAN observations
wage2df.reset_index(drop=True, inplace=True) #Reset index


#%%

# Examine the data'
print(wage2df[:5]) #show me the first 5 observations

#%%
print(wage2df.head()) #show me the first (5) observations

print(wage2df.tail()) #show me the last (5) observations


#%%
#Simple plot
plt.plot(wage2df.educ, wage2df.wage, 'ro')
plt.xlabel("Education")
plt.ylabel("Wage Rate")
plt.title("Hourly Wage Rate vs Years of Schooling")
plt.show()


#%%
'Create a new variable and assign to your DataFrame'

wage2df['lnwage']=np.log(wage2df.wage) # log(wage)

#%%
wage2df['educ2']=np.square(wage2df.educ) # educ**2


#%%
# Plot Data with regression line going through it
sns.regplot(x='educ', y='wage', data=wage2df).set_title('Return on Education')


#%%
#Run regression 
reg = smf.ols('wage ~ educ', data=wage2df, missing='drop').fit()
print(reg.summary2())

#%%
#Get fitted (predicted) values of y and print them  
'wagehat =b0+b1*educ'

print(reg.fittedvalues)

#%%
'Get Fitted values/predicted values and assign them to yhat variable in your dataframe'
wage2df['wagehat']=reg.fittedvalues

#%%
'Get residuals and assign to your datatframe'
wage2df['uhat']=reg.resid


#%%
# Residual Plot
plt.scatter(wage2df.wagehat, wage2df.uhat)
plt.show()


#%%
#Or use sns package
#Residual Plot
sns.residplot(x='educ', y='wage', \
            data=wage2df).set_title('Residuals vs Years of Schooling')


#%%
#Creating new dataframe where wage >900 and save as a new datafrae 

wage2df900=wage2df[wage2df['wage']>900]

#%%
#Get the number of observations in this dataframe
print(wage2df900.shape[0])

#%%
# Report your results in for hoemwork like this
#Question 2 part a 
print('The number of observation that have \
      wage rate above 900 is', wage2df900.shape[0])


#%%
#You can also use count() function
print(wage2df900.count())

#%%
#Report as this
print(wage2df900.count()[1])

#%%
#Describe wage column in wage2 dataframe
print(wage2df['wage'].describe())

#%%

print(wage2df['wage'].describe()[2])

#%%
print('The standard deviation of wage rate in our suample is', wage2df['wage'].describe()[2])



    
    