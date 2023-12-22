import numpy as np

f = open("rate_of_cases.txt")
data = []
for line in f:
    data.append(float(line[:-1:]))

data_np = np.array(data)
discr_min_age = np.min(data_np)
discr_max_age = np.max(data_np)
discr_mean = np.mean(data_np)
print("максимальное значение:", discr_max_age)
print("минимальное значение:", discr_min_age)
print("Cредние значение:", discr_mean)
discr_standart_deviation = np.std(data_np)
print("среднее квадратичное отклонение:", discr_standart_deviation)
discr_dispersion = np.var(data_np)
print("дисперсия:", discr_dispersion)
discr_variation_coeff = (discr_standart_deviation / discr_mean) * 100
print("коэффициент вариации:", discr_variation_coeff)
