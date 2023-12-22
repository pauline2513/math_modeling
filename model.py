import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

def mnk(y, x1, x2, x3):
    y_matrix = np.array(y).transpose()
    ct = np.array([[1] * len(y), x1, x2, x3 ])
    c = ct.transpose()
    ct_c = np.dot(ct, c)
    ct_c_inv = np.linalg.inv(ct_c)
    coeff = np.dot(ct_c_inv, ct)
    coeff = np.dot(coeff, y_matrix)
    return coeff

def model_y(coeff, median_age, smoking_rate, pollution):
    result = []
    n = len(median_age)
    for i in range (n):
        result.append(coeff[0] + coeff[1]*median_age[i] + coeff[2]*smoking_rate[i] + coeff[3]*pollution[i])
    return result

f = open("rate_of_cases.txt")
cases = []
for line in f:
    cases.append(float(line[:-1:]))
# print(cases)
n = len(cases)
f = open("smoking_rate.txt")
smoking_rate = []
for line in f:
    smoking_rate.append(float(line[:-1:]))
# print(smoking_rate)

f = open("median_age.txt")
median_age = []
for line in f:
    median_age.append(float(line[:-1:]))
# print(median_age)

f = open("pollution.txt")
pollution = []
for line in f:
    pollution.append(float(line[:-1:]))
# print(pollution)

data = {'X1': median_age,
        'X2': smoking_rate,
        'X3': pollution,
        'Y': cases}

df = pd.DataFrame(data)

print(df.corr())


coeff = mnk(cases, median_age, smoking_rate, pollution)

model_cases = model_y(coeff, median_age, smoking_rate, pollution)

# анализ полученной модели
y_avg = sum(cases)/n
x1_avg = sum(median_age)/n
x2_avg = sum(smoking_rate)/n
x3_avg = sum(pollution)/n

# эластичность 
e1 = coeff[1] * (x1_avg/y_avg)
e2 = coeff[2] * (x2_avg/y_avg)
e3 = coeff[3] * (x3_avg/y_avg)

# дисперсионный анализ
m = 3
df_y = 1/(n-1)
df_fact = 1/m
df_e = 1/(n - m - 1)


sy = 0
se = 0
sfact = 0
for i in range(n):
    sy += (cases[i] - y_avg)**2
    sfact += (model_cases[i] - y_avg)**2
    se += (cases[i] - model_cases[i])**2

s2y = sy*df_y
s2fact = df_fact*sfact
s2e = df_e*se

print(s2y, s2fact, s2e)

f_criteria = s2fact/s2e
print(f_criteria)

X = sm.add_constant(df[['X1', 'X2', 'X3']])

model = sm.OLS(df['Y'], X).fit()
print(model.summary())


# исследование на мультиколлинеарность
r_x = df.corr()
new_data = {'X1': median_age,
            'Y': cases}

df_1 = pd.DataFrame(new_data)
X = sm.add_constant(df[['X1']])

model = sm.OLS(df['Y'], X).fit()
print(model.summary())
new_y_model = []
for x in median_age:
    new_y_model.append(214.5263-1.2478*x)

m = 1
df_y = 1/(n-1)
df_fact = 1/m
df_e = 1/(n - m - 1)

sy = 0
se = 0
sfact = 0
for i in range(n):
    sy += (cases[i] - y_avg)**2
    sfact += (new_y_model[i] - y_avg)**2
    se += (cases[i] - new_y_model[i])**2

print(se, sy, sfact)
s2y = sy*df_y
s2fact = df_fact*sfact
s2e = df_e*se

print(s2y, s2fact, s2e)


r_x = df.corr()
new_data = {'X1': median_age,
            'X2': smoking_rate,
            'Y': cases}

df_1 = pd.DataFrame(new_data)
X = sm.add_constant(df[['X1', 'X2']])
print(df_1.corr())
model = sm.OLS(df['Y'], X).fit()
print(model.summary())
final_model_cases = []
for i in range(n):
    final_model_cases.append(189.3626-1.0900*median_age[i] +0.5714*smoking_rate[i])

m = 2
df_y = 1/(n-1)
df_fact = 1/m
df_e = 1/(n - m - 1)

sy = 0
se = 0
sfact = 0
for i in range(n):
    sy += (cases[i] - y_avg)**2
    sfact += (final_model_cases[i] - y_avg)**2
    se += (cases[i] - final_model_cases[i])**2

s2y = sy*df_y
s2fact = df_fact*sfact
s2e = df_e*se

