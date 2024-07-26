import numpy as np
from matplotlib import pyplot as plt

prediction = np.load('../out/yeWei_result_withnoNorm_TCN&LSTM3.npy')
true = np.load('../out/yeWei_true_withnoNorm.npy')
# prediction = np.load('../out/yaLi_result_withNorm_TCN&LSTM3.npy')
# true = np.load('../out/yaLi_true_withNorm.npy')
# 绘制单次长预测结果对比
plt.plot(range(120), prediction[1,:,0])
plt.plot(range(120), true[1,:,0])
# 添加图例
plt.legend()
# 显示图表
plt.show()

#绘制第n步结果对比
plt.plot(range(len(prediction)), prediction[:,1,0])
plt.plot(range(len(prediction)), true[:,1,0])
# 添加图例
plt.legend()
# 显示图表
plt.show()