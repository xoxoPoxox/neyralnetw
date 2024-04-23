import numpy as np
def f(x):
    return 2 / (1 + np.exp(-x)) -1

def df(x):
    return 0.5 * (1 - x)*(1+x)

# Установим начальные значения весов
W1 = np.array([[0.4, 0.6], [0.5, 0.63], [0.3, 0.65]])
W2 = np.array([[0.1, 0.3, 0.63], [0.2, 0.25, 0.64]])
W3 = np.array([0.2, 0.3])

def go_forward(inp):
    sum1 = np.dot(W1, inp)
    out1 = np.array([f(x) for x in sum1])

    sum2 = np.dot(W2, out1)
    out2 = np.array([f(x) for x in sum2])
    sum3 = np.dot(W3, out2)
    out = np.append(out1, out2, axis=0)

    #print(out)
    y = f(sum3)
    return (y, out)


def train(epoch):
    global W1, W2, W3
    lmd = 0.01 # шаг обучения
    N = 100000 # число итераций при обучении
    count = len(epoch)
    for k in range(N):
        x = epoch[np.random.randint(0, count)] # случайный выбор входного сигнала из обучающей выборки
        y, out =  go_forward(x[0:2]) # прямой проход по НС и вычисление выходных значений НС
        e = y - x [-1] # ошибка
        delta = e*df(y) # локальный градиент

        W3[0] = W3[0] - lmd * delta * out[4] # корректировка первой связи 3 слоя
        W3[1] = W3[1] - lmd * delta * out[3] # корректировка второй связи 3 слоя

        delta2 = W3*delta*df(out[3:]) # вектор из 2-х величин локальных градиентов

        # корректировка связей второго слоя
        W2[0, :] = W2[0, :] - out[2] * delta2[1] * lmd
        W2[1, :] = W2[1, :] - out[1] * delta2[0] * lmd
        '''
        print("___________")
        print(W2)
        print("___________")
        print(df(out[0:3]))
        print("___________")
        print(delta2)
        print("___________")
        print(W2*df(out[0:3]))
        '''
        s = W2*df(out[0:3])
        delta3 = delta2 * s.T # вектор из 3-х величин локальных градиентов

        # корректировка связей первого слоя
        W1[0, :] = W1[0, :] - np.array(x[0:2]) * delta3[1] * lmd
        W1[1, :] = W1[1, :] - np.array(x[0:2]) * delta3[2] * lmd
        W1[2, :] = W1[2, :] - np.array(x[0:2]) * delta3[0] * lmd


data = [(8, 8, 95), (1, 6, 43), (11, 5, 52)]
dd = [[8, 8, 95], [1, 6, 43], [11, 5, 52]]
mc = 0
mb= 0
for i in range(len(dd)):
    if max(dd[i][0:2]) > mc:
        mc = max(dd[i][0:2])
    if dd[i][2] > mb:
        mb = dd[i][2]

for i in range(len(dd)):
    dd[i][0] = dd[i][0]/mc
    dd[i][1] = dd[i][1] / mc
    dd[i][2] = dd[i][2] / mb

train(dd)
for x in dd:
    y, _ = go_forward(x[0:2])
    print(f"Предсказанное значение: {y} => Ожидаемое значение: {x[-1]}")
test = [[5, 8, 71]]
mc = 0
mb= 0
for i in range(len(test)):
    if max(test[i][0:2]) > mc:
        mc = max(test[i][0:2])
    if test[i][2] > mb:
        mb = test[i][2]
for i in range(len(test)):
    test[i][0] = test[i][0]/mc
    test[i][1] = test[i][1] / mc
    test[i][2] = test[i][2] / mb

for x in test:
    y, _ = go_forward(x[0:2])
    print(f"Предсказанное значение: {y} => Ожидаемое значение: {x[-1]}")