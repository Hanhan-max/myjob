import pandas as pd
import pywt
import numpy as np
import matplotlib.pyplot as plt

signal = pd.read_csv('../data/processe_data/yeWei.csv')['值']
time = range(len(signal))

# 选择小波函数
wavelet = 'db1'  # Daubechies小波

# 进行小波变换
coeffs = pywt.wavedec(signal, wavelet, level=4)

# 提取分解结果
cA4, cD4, cD3, cD2, cD1 = coeffs

# 合成短周期成分
short_period = pywt.waverec([None, cD4, cD3, cD2, cD1], wavelet)[:len(signal)]

# 计算残差
reconstructed_signal = pywt.waverec(coeffs, wavelet)
residual = signal - reconstructed_signal

plt.figure(figsize=(12, 8))

plt.subplot(4, 1, 1)
plt.plot(time, signal)
plt.title('Original Signal')

plt.subplot(4, 1, 2)
plt.plot(time[:len(cA4)], cA4)
plt.title('Approximation Coefficients (Level 4)')

plt.subplot(4, 1, 3)
plt.plot(time[:len(short_period)], short_period)
plt.title('Short Period Component')

plt.subplot(4, 1, 4)
plt.plot(time, residual)
plt.title('Residual Component')

plt.tight_layout()
plt.show()

