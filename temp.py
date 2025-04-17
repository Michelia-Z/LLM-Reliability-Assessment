import matplotlib.pyplot as plt
import numpy as np

# 模拟数据
x = np.linspace(0, 10, 100)
y1 = np.sin(x)                 # 第一组数据（左轴）
y2 = np.exp(0.3 * x)           # 第二组数据（右轴）

# 创建图形和主坐标轴
fig, ax1 = plt.subplots(figsize=(10, 6))

# 绘制第一个折线图（左纵轴）
ax1.plot(x, y1, 'b-', label='Sine Wave')
ax1.set_xlabel('X')
ax1.set_ylabel('Amplitude', color='b')
ax1.tick_params(axis='y', labelcolor='b')

# 创建第二个坐标轴，共享x轴
ax2 = ax1.twinx()
ax2.plot(x, y2, 'r--', label='Exponential Growth')
ax2.set_ylabel('Growth', color='r')
ax2.tick_params(axis='y', labelcolor='r')

# 添加图例（合并两个图例）
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines + lines2, labels + labels2, loc='upper left')

# 设置标题和网格
plt.title('Dual Y-Axis Line Plot Example')
ax1.grid(True)
plt.tight_layout()
plt.show()
