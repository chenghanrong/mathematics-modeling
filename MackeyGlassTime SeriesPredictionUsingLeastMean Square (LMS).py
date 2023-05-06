import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 加载时间序列数据
# 读取 Excel 文件，第 1 列为时间步，第 2 列为时间序列数据
data = pd.read_excel('Dataset/Data.xlsx', header=None, names=['Time', 'Data'])
time_steps = 2  # 每隔几个时间步进行一次训练
teacher_forcing = 1  # 是否强制使用目标值作为输入，进行"老师引导"

# 对于训练和测试数据集
# 对于训练
Tr = data['Data'][99:2499]  # 选择一个时间序列数据区间 t = 100~2500

# 对于测试
Ts = data['Data'][2499:2999]  # 选择一个时间序列数据区间t = 2500~3000
Ys = data['Data'][2499:2999].values.flatten()  # 选择一段时间序列数据 y(t)

# 将数据转换为 NumPy 数组
Tr = np.array(Tr)
Ts = np.array(Ts)

# LMS 参数
eta = 5e-3  # 学习率
M = 1  # LMS 滤波器的阶数
MSE = []  # 初始均方误差(MSE)

# 训练 LMS 的权重
U = np.zeros(M + 1)
W = np.random.randn(M + 1)  # 初始化权重
Yp = np.zeros_like(Tr)  # 初始化生成序列
E = np.zeros_like(Tr)
for t in range(Tr[0], Tr[-1] - time_steps):
    U[:-1] = U[1:]  # 将滑动窗口向前移动一格

    # 如果选择"teacher_forcing"，则根据每个时间步的情况选择输入
    if (teacher_forcing == 1):
        if (t % time_steps == 0) or (t == Tr[0]):
            U[-1] = Tr[t]  # 类似于 RNN 中的过去和现在的输入信号
        else:
            U[-1] = Yp[t - 1]  # 如果不是第一个时间步，直接使用上一个预测值作为输入
    else:
        U[-1] = Tr[t]  # 没有"teacher_forcing"，则直接使用训练数据作为输入

    e = Tr[t + time_steps] - np.dot(W, U)  # 预测值与目标值的误差
    W = W + eta * e * U  # LMS 权重更新规则
    E[t] = e ** 2  # mse

training_time = time.process_time()  # 训练和计算 MSE 的时间

# 预测下一个序列值（测试）
U *= 0  # 重置滑动窗口
for t in range(Ts[0], Ts[-1] - time_steps + 1):
    U[:-1] = U[1:]  # 将滑动窗口向前移动一格

    # 如果选择"老师引导"，则根据每个时间步的情况选择输入
    if (teacher_forcing == 1):
        if (t % time_steps == 0) or (t == Ts[0]):
            U[-1] = Ys[t]  # 类似于 RNN 中的过去和现在的输入信号
        else:
            U[-1] = Yp[t - 1]  # 如果不是第一个时间步，直接使用上一个预测值作为输入
    else:
        U[-1] = Ys[t]  # 没有"teacher_forcing"，则直接使用测试数据作为输入

    Yp[t] = np.dot(W, U)  # 预测输出
    e = Ys[t + time_steps - 1] - Yp[t]  # 预测值与目标值的误差
    E[t] = e ** 2  # mse

testing_time = time.process_time()  # 测试和计算 MSE 的时间

# 结果展示
plt.figure(1)
plt.plot(Tr, 10 * np.log10(E[Tr]))  # MSE 曲线
plt.plot(Ts[:-time_steps + 1], 10 * np.log10(E[Ts[:-time_steps + 1]]), 'r')  # MSE 曲线
plt.grid(True, which='both', axis='both', linestyle='--', linewidth=0.5)

plt.title('损失函数')
plt.xlabel('迭代次数')
plt.ylabel('均方误差 (MSE)')
plt.legend(['训练阶段', '测试阶段'])

plt.figure(2)
plt.plot(Tr[2 * M:], data['Data'][99 + 2 * M:, 1])  # Mackey Glass 系列的实际值
plt.plot(Tr[2 * M:], Yp[2 * M:], 'r')  # 训练期间的预测值
plt.plot(Ts, Ys, '--b')  # 实际的未知数据
plt.plot(Ts[:-time_steps + 1], Yp[Ts[:-time_steps + 1]], '--r')  # 对未知数据集进行的预测值
plt.xlabel('时间 t')
plt.ylabel('输出 Y(t)')
plt.title('使用最小均方(LMS)算法预测Mackey Glass时间序列')
plt.ylim([np.min(Ys) - 0.5, np.max(Ys) + 0.5])
plt.legend(['训练阶段（目标值）', '训练阶段（预测值）', '测试阶段（目标值）', '测试阶段（预测值）'])

mitr = 10 * np.log10(np.mean(E[Tr]))  # 训练中的最小均方误差
mits = 10 * np.log10(np.mean(E[Ts[:-time_steps + 1]]))  # 测试中的最小均方误差

print('总训练时间为 {:.5f}'.format(training_time))
print('总测试时间为 {:.5f}'.format(testing_time))
print('训练期间的 MSE 值为 {:.3f} (dB)'.format(mitr))
print('测试期间的 MSE 值为 {:.3f} (dB)'.format(mits))
