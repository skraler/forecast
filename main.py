import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from pmdarima import auto_arima
import warnings

warnings.simplefilter(action = 'ignore', category = Warning)

def addfuller_test(data):
    result = adfuller(data['Всего'])
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))
    if result[1] < 0.05:
        return True
    '''else:
        data['Всего'] = np.log(data['Всего'])
        addfuller_test(data)'''

def parametr_search(data):
    parameter_search = auto_arima(data, start_p = 1, start_q = 1, max_p = 3, max_q = 3, m = 12, start_P = 0, seasonal = True,
                         d = None, D = 1, trace = True, error_action ='ignore', suppress_warnings = True,  stepwise = True)
    # Сохраняем оптимальные параметры в переменную
    order = parameter_search.order 
    seasonal_order = parameter_search.seasonal_order 
    return order, seasonal_order

def wape(actual, predicted):
    actual = actual.reset_index()
    predicted = predicted.reset_index()
    # Вычисляем ошибку WAPE
    wape=[]
    for index in actual.index:
        abs_diff =np.abs(int(actual.loc[index, 'Всего'])- int(predicted.loc[index, 'predicted_mean']))
        wape_loc = abs_diff/(int(actual.loc[index, 'Всего']))*100
        acc = 100 - wape_loc
        if acc <0:
            acc = 0
        wape.append(acc)
    return wape

# Загружаем данные из таблицы Excel
df = pd.read_excel('D:/project/month.xlsx', index_col='Месяц', parse_dates=True)
# Преобразуем данные в формат временного ряда

# Выводим DataFrame
df = df.groupby(['Месяц', 'SKU'])['Всего'].sum().reset_index()
df.set_index('Месяц', inplace=True)
print(df)
#sku = ['ГР0002-0004-0006','ГР0050-0050-0054', 'ГР0002-0004-0008']
result = []
count_error = 0
for sku in df['SKU'].unique():
    try:
        result_df=df[df['SKU'] == sku]
        result_df.index = result_df.index.strftime('%Y-%d')
        # обучающая выборка будет включать данные до декабря 2022 года включительно
        train = result_df[:'2022-12']
        # тестовая выборка начнется с января 2023 (по сути, один год)
        test = result_df['2023-01': '2023-12']

        train.drop('SKU', axis=1, inplace=True)
        test.drop('SKU', axis=1, inplace=True)


        order, seasonal_order = parametr_search(train)
        # обучаем модель с соответствующими параметрами
        # создадим объект этой модели
        model = SARIMAX(train, order=order, seasonal_order=seasonal_order)
                            
        # применим метод fit
        result_forecast = model.fit()

        # Делаем прогноз на тестовой выборке
        forecast = result_forecast.forecast(steps=3)
        forecast.index = forecast.index.strftime('%Y-%m')
        forecast = forecast.round()
        test_for_wape = test[test.index.isin(forecast.index)]
        wape_iter = wape(test_for_wape, forecast)
        data = pd.DataFrame({'Код товара': sku,
                        'прогноз': forecast, 
                        'тестовые данные': test_for_wape['Всего'].to_list(), 
                        'точность': wape_iter}).rename(index={'predicted_mean': 'Дата'})
        result.append(data)
    except Exception as e:
        count_error += 1

result_table = pd.concat(result)
print(count_error)

result_table.to_excel('output.xlsx', index=True)
'''# Создаем график
plt.figure(figsize=(12, 6))

# Рисуем график тренировочной выборки
plt.plot(train, label='Тренировочная выборка')

# Рисуем график тестовой выборки
plt.plot(test, label='Тестовая выборка')
plt.plot(forecast, label='Прогноз')
plt.gca().xaxis.set_major_locator(plt.MultipleLocator(4))
# Устанавливаем метки для осей
plt.xlabel('Месяц')
plt.ylabel('Всего')

# Устанавливаем легенду
plt.legend()
# Отображаем график
plt.show()'''
