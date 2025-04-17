import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import invgamma
from matplotlib.pylab import mpl
mpl.use('Qt5Agg')  #解决plt函数卡住的问题：切换Matplotlib的后端渲染引擎


###################################################
"""
调参 inverse Gamme 的分布曲线
"""
# 定义 x 值范围
x = np.linspace(0.01, 5, 500)

# 定义几个不同的形状参数 a
shape_params = [0.5, 1, 2, 3, 5]
scale = 1  # 缩放参数 β

plt.figure(figsize=(10, 6))
for a in shape_params:
    pdf = invgamma.pdf(x, a, scale=scale)
    plt.plot(x, pdf, label=f'a={a}')
plt.title('Inverse Gamma Distribution PDF')
plt.xlabel('x')
plt.ylabel('Probability Density')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

###################################################
shape = 24
scale_params = [0.5,1,2,3]  # 缩放参数 β
plt.figure(figsize=(10, 6))
for a in scale_params:
    pdf = invgamma.pdf(x, shape, scale=a)
    plt.plot(x, pdf, label=f'a={a}')
plt.title('Inverse Gamma Distribution PDF')
plt.xlabel('x')
plt.ylabel('Probability Density')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

###################################################
"""
调参 f 的分布曲线
"""
theta, epsilon = 5, 3
x = np.linspace(0.01, 5, 500)
y = [theta * xx + epsilon for xx in x]
shape = 0.5
scale = 1  # 缩放参数 β
plt.figure(figsize=(10, 6))
pdf = invgamma.pdf(x, shape, scale=scale)
plt.plot(x, pdf, color='red', label='x')
plt.plot(y, pdf, label='y')
plt.title('Inverse Gamma Distribution PDF')
plt.xlabel('x')
plt.ylabel('Probability Density')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

###################################################
"""
调整shape参数
"""
theta, epsilon = 5, 0
x = np.linspace(0.01, 5, 500)
y = [theta * xx + epsilon for xx in x]

shape_params = [0.5, 1]
scale = 1

plt.figure(figsize=(10, 6))
for a in shape_params:
    pdf = invgamma.pdf(y, a, scale=scale)
    plt.plot(y, pdf, label=f'a={a}')
plt.title('Inverse Gamma Distribution PDF')
plt.xlabel('x')
plt.ylabel('Probability Density')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

##################################################
"""
调整scale参数
"""
theta, epsilon = 5, 0
x = np.linspace(0.01, 5, 500)
y = [theta * xx + epsilon for xx in x]

shape = 1
scale_params = [1, 2, 3]  # 缩放参数 β
plt.figure(figsize=(10, 6))
for a in scale_params:
    pdf = invgamma.pdf(y, shape, scale=a)
    plt.plot(y, pdf, label=f'a={a}')
plt.title('Inverse Gamma Distribution PDF')
plt.xlabel('x')
plt.ylabel('Probability Density')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

###################################################
"""
产生随机数
"""
a = 0.5        # 形状参数 (shape)
scale = 1    # 缩放参数 (scale = β)
size = 1000    # 样本数量
samples = invgamma.rvs(a=a, scale=scale, size=size)
plt.figure(figsize=(8, 5))
plt.hist(samples, bins=40, density=True, alpha=0.6, edgecolor='black')
plt.title(f'Inverse Gamma Distribution Samples (a={a}, scale={scale})')
plt.xlabel('Value')
plt.ylabel('Density')
plt.grid(True)
plt.tight_layout()
plt.show()
