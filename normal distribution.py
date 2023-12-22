import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

f = open("rate_of_cases.txt")
data = []
for line in f:
    data.append(float(line[:-1:]))

min_rate = min(data)
max_rate = max(data)

data_np = np.array(data)
n = len(data_np)

#дискретный ряд распределения
discr_distribution_series = Counter(data_np)
print(discr_distribution_series)
# размах
r = max_rate - min_rate
# дискретное распределение вероятностей
discr_random_values = dict()
for x in discr_distribution_series:
    discr_random_values[x] = discr_distribution_series[x]/n
#мат ожидание
print(discr_random_values)
expected_m_x = 0
for x in discr_random_values:
    expected_m_x += x * discr_random_values[x]
print("мат ожидание", expected_m_x)
#центральный момент 3 порядка
central_moment_3 = 0
#центральный момент 4 порядка
central_moment_4 = 0
#дисперсия
dispersion = 0
for x in discr_random_values:
    dispersion += ((x - expected_m_x)**2) * discr_random_values[x]
    central_moment_3 += ((x - expected_m_x)**3) * discr_random_values[x]
    central_moment_4 += ((x - expected_m_x)**4) * discr_random_values[x]

#среднее квадратическое отклонение
sigma = math.sqrt(dispersion)
#асимметрия
asymmetry = central_moment_3/(sigma ** 3)
#эксцесс
excess = central_moment_4/(sigma ** 4) - 3

print("асимметрия:", asymmetry, "эксцесс:", excess)
print("Сигма", sigma )

#проверка правила 3 сигм
left = expected_m_x - 3 * sigma
right = expected_m_x + 3 * sigma
sum_in_3_sigma = 0
for i in range(n):
    if (data[i] >= left) and (data[i] <= right):
        sum_in_3_sigma += 1
print("Процент значений в пределах 3 сигм", sum_in_3_sigma/n*100)

left = expected_m_x - 2 * sigma
right = expected_m_x + 2 * sigma
sum_in_2_sigma = 0
for i in range(n):
    if (data[i] >= left) and (data[i] <= right):
        sum_in_2_sigma += 1
print("Процент значений в пределах 2 сигм", sum_in_2_sigma/n*100)

left = expected_m_x - sigma
right = expected_m_x + sigma
sum_in_1_sigma = 0
for i in range(n):
    if (data[i] >= left) and (data[i] <= right):
        sum_in_1_sigma += 1
print("Процент значений в пределах 1 сигмы", sum_in_1_sigma/n*100)


#количество интервалов по формуле Стерджесса
m = math.ceil((1 + 3.222*math.log10(r)))
#длина интервалов
h = math.ceil((r/m))

interval_dist_series = {}
start_age = math.floor(min_rate)
while start_age < math.ceil(max_rate):
    index = ((start_age + start_age + h)/2)
    interval_dist_series[index] = 0
    for x in discr_distribution_series:
        if x >= start_age and x < start_age + h:
            interval_dist_series[index] += discr_distribution_series[x]
    start_age += h
print("интервальный ряд распределения", interval_dist_series)
#построение гистограммы
plt.bar(interval_dist_series.keys(), interval_dist_series.values(), width=h)
plt.title("Гистограмма частот")
plt.xlabel("Число новых случаев на 100 тыс. населения")
plt.ylabel("Частота")
plt.show()

