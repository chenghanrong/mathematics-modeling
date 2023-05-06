import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题


# 灰色预测函数
def GM11(x):
    n = len(x)
    x1 = x.cumsum()
    z1 = (x1[:n - 1] + x1[1:]) / 2.0
    z1 = z1.reshape((n - 1, 1))
    B = np.append(-z1, np.ones_like(z1), axis=1)
    Y = x[1:].reshape((n - 1, 1))
    [[a], [b]] = np.dot(np.dot(np.linalg.inv(np.dot(B.T, B)), B.T), Y)
    f = lambda k: (x[0] - b / a) * np.exp(-a * (k - 1)) - (x[0] - b / a) * np.exp(-a * (k - 2))
    delta = np.abs(x - np.array([f(i) for i in range(1, n + 1)]))
    C = delta.std() / x.std()
    if C <= 0.35:
        result = np.array([f(i) for i in range(n)])
    else:
        result = None
    return result


# 定义输入数据
x = np.array([200, 205, 210, 218, 236, 257, 277, 295, 308, 325])

# 进行灰色预测
result = GM11(x)


# 计算评价指标
def evaluate(y_true, y_pred):
    mae = np.mean(np.abs(y_pred - y_true))
    mse = np.mean(np.square(y_pred - y_true))
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_pred - y_true) / y_true)) * 100
    return mae, mse, rmse, mape


mae, mse, rmse, mape = evaluate(x, result)
print('MAE:', mae)
print('MSE:', mse)
print('RMSE:', rmse)
print('MAPE:', mape)

# 绘制预测结果和原始数据的图像
x_range = range(1, len(x) + 1)
plt.plot(x_range, x, 'ro-', label='原始数据')
plt.plot(x_range, result, 'b*-', label='预测数据')
plt.legend(loc='upper left')
plt.grid(True)
plt.show()