# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 10:30:18 2021
F-Distribution
@author: sahakyz
"""
# %%

import numpy as np
# the package that will help us find critical values for F-distribution
from scipy.stats import f
# the package that will help us find critical values for normal distribution
from scipy.stats import norm
# the package that will help us find critical values for t-distribution
from scipy.stats import t
# the package that will help us find critical values for chi2-distribution
from scipy.stats import chi2


# %%
n1 = 11
n2 = 13
df1 = n1-1  # degrees of freedom from sample 1
df2 = n2-1
var1 = 6.7
var2 = 2.68

# %%
# Critical Value for F distribution (Right tail)
cv_f1 = f.isf(0.05, df1, df2)

# %%
print(cv_f1)

# %%
cv_f2 = f.isf(0.95, 10, 12)

# %%
# Critical Value for F distribution (Left tail)

print(cv_f2)  # critical value
# %%
print('Hello World')

print('Critical Value for F distribution (Right tail)', cv_f2)

print('Critical Value for F distribution (Left tail)', cv_f1)

# %%
# Test Statistics
ts = var1/var2  # test statistic

print('My calculated t.s. is equal to \n', ts)
# %%
print('This is what my t.s.', ts)


# %%
# P-value

'p-value'

pvalue = 2*f.sf(ts, df1, df2)

# %%

print('This is my p-value', pvalue)

# %%

print('This is my p-value', np.round(pvalue, 4))

# %%
# critical values for building 95%  Confidence Intervale, here alpha  is 0.05
# Since CI is always a two-tailed, we need to take half of our alpha
# Note the probabilities: 0.025 and 0.975


cvci_f1 = f.isf(0.025, 10, 12)

print(cvci_f1)
# %%
# Give references not the exact values whenever possible
cvci_f1 = f.isf(0.025, df1, df2)


# %%
print(round(cvci_f1, 4))

# %%
cvci_f2 = f.isf(0.975, 10, 12)

# %%
cvci_f2 = f.isf(0.975, df1, df2)
print(cvci_f2)

# %%
# Z distribution
alpha = 0.05
Z_cv = norm.isf(alpha/2)  # two-tailed test
print(Z_cv)

# %%
# %
Z_cv1 = norm.isf(alpha)  # one-tailed test
print(Z_cv1)

# %%
# t distribution
alpha = 0.05
n = 100
t_cv = t.isf(alpha/2, n-1)  # two-tailed test
print(t_cv)

# %%
# %
t_cv1 = t.isf(alpha, n-1)  # one-tailed test
print(t_cv1)
