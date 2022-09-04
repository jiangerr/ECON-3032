# %%
import numpy as np
from scipy.stats import f

alpha = .05
s1 = 16
s2 = 12
F = pow(s1/s2, 2)
# print(F)
lowBound = f.isf(alpha, 19, 19)
print(lowBound)
p = f.sf(F, 19, 19)
print(p)

# %%
alpha = .1
lowerCV = F/f.isf(alpha/2, 19, 19)
upperCV = F/f.isf(1-alpha/2, 19, 19)
print('CI: [{}, {}]'.format(lowerCV, upperCV))
# %%
alpha = .1
s1 = 2.8
s2 = 3.3
F = s1/s2
print(F)
upperBound = f.isf(.9, 149, 299)
print(upperBound)
# %%
alpha = .05
df1 = 149
df2 = 299
lowerCV = F/f.isf(alpha/2, df1, df2)
upperCV = F/f.isf(1-alpha/2, df1, df2)
print('CI: [{}, {}]'.format(lowerCV, upperCV))
# %%
