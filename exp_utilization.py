"""
    Effect of GPU number/uShape/uScale on reliability & utilization
"""
from Genetic import genetic_algorithm
import matplotlib.pyplot as plt
from matplotlib.pylab import mpl
#mpl.use('Qt5Agg')  # 解决plt函数卡住的问题：切换Matplotlib的后端渲染引擎

fileDataPath = 'data/'
fileResPath = 'result/'

population_size = 10
mutation_rate = 0.2
crossover_rate1 = 0.2
crossover_rate2 = 0.2
max_generations = 5
iterations = 100

def plot1():
    # Figure 1:
    # x-axis: gpu number; y-axis: R & Utilization
    result = {}
    for k in k_range:
        uShape = [1 for _ in range(k)]  # 参数可以调整
        uScale = [1 for _ in range(k)]  # 参数可以调整
        final_score, utilization_ratio, input_matrix1, input_matrix21, input_matrix22 = genetic_algorithm(n, i, j, k, v,
                                                                                                          alpha, c_f,
                                                                                                          c_b,
                                                                                                          population_size,
                                                                                                          mutation_rate,
                                                                                                          crossover_rate1,
                                                                                                          crossover_rate2,
                                                                                                          max_generations,
                                                                                                          iterations,
                                                                                                          standard,
                                                                                                          uShape,
                                                                                                          uScale,
                                                                                                          theta,
                                                                                                          epsilon)
        result[(k, 'k', 'reliability')] = final_score
        result[(k, 'k', 'utilization')] = utilization_ratio

    y1 = [result[(k, 'k', 'reliability')] for k in k_range]
    y2 = [result[(k, 'k', 'utilization')] for k in k_range]

    fig, ax1 = plt.subplots(figsize=(7, 5))
    # 绘制左纵轴折线图
    ax1.plot(k_range, y1, 'b-', marker = 'o',label='Reliability')
    ax1.set_xlabel(r'$K$',fontsize = 15)
    ax1.set_ylabel('Reliability', color='b',fontsize = 15)
    ax1.tick_params(axis='y', labelcolor='b',labelsize=12)
    # 绘制右纵轴折线图，共享x轴
    ax2 = ax1.twinx()
    ax2.plot(k_range, y2, 'r--', marker = 's',label='Utilization')
    ax2.set_ylabel('GPU Utilization', color='r',fontsize = 15)
    ax2.tick_params(axis='y', labelcolor='r',labelsize = 12)
    # 添加图例（合并两个图例）
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper left')
    # 设置标题和网格
    plt.title('Reliability/Utilization under different GPU number')
    plt.tight_layout()
    plt.show()
    plt.savefig(fileResPath + "gpu-num-r-utilization.png")


def plot23():
    k = k_range[0]
    uShape_range = [[a / 10 for _ in range(k)] for a in range(8, 13)]
    uScale_range = [[a / 10 for _ in range(k)] for a in range(10, 16)]

    # Figure 2:
    # x-axis: uShape; y-axis: R & Utilization
    result = {}
    uScale = uScale_range[0]
    for uShape in uShape_range:
        final_score, utilization_ratio, input_matrix1, input_matrix21, input_matrix22 = genetic_algorithm(n, i, j, k, v,
                                                                                                          alpha, c_f,
                                                                                                          c_b,
                                                                                                          population_size,
                                                                                                          mutation_rate,
                                                                                                          crossover_rate1,
                                                                                                          crossover_rate2,
                                                                                                          max_generations,
                                                                                                          iterations,
                                                                                                          standard,
                                                                                                          uShape,
                                                                                                          uScale,
                                                                                                          theta,
                                                                                                          epsilon)
        result[(uShape[0], 'uShape', 'reliability')] = final_score
        result[(uShape[0], 'uShape', 'utilization')] = utilization_ratio

    # Figure 3:
    # x-axis: uScale; y-axis: R & Utilization
    uShape = uShape_range[0]
    for uScale in uScale_range:
        final_score, utilization_ratio, input_matrix1, input_matrix21, input_matrix22 = genetic_algorithm(n, i, j, k, v,
                                                                                                          alpha, c_f,
                                                                                                          c_b,
                                                                                                          population_size,
                                                                                                          mutation_rate,
                                                                                                          crossover_rate1,
                                                                                                          crossover_rate2,
                                                                                                          max_generations,
                                                                                                          iterations,
                                                                                                          standard,
                                                                                                          uShape,
                                                                                                          uScale,
                                                                                                          theta,
                                                                                                          epsilon)
        result[(uScale[0], 'uScale', 'reliability')] = final_score
        result[(uScale[0], 'uScale', 'utilization')] = utilization_ratio

    # Plot Figure 2:
    xAxis = [uShape[0] for uShape in uShape_range]
    y1 = [result[(uShape[0], 'uShape', 'reliability')] for uShape in uShape_range]
    y2 = [result[(uShape[0], 'uShape', 'utilization')] for uShape in uShape_range]
    fig, ax1 = plt.subplots(figsize=(7, 5))
    ax1.plot(xAxis, y1, 'b-',marker = 'o', label='Reliability')
    ax1.set_xlabel(r'$\alpha$')
    ax1.set_ylabel('Reliability', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax2 = ax1.twinx()
    ax2.plot(xAxis, y2, 'r--', marker = 's', label='Utilization')
    ax2.set_ylabel('GPU Utilization', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper left')
    plt.title(f'Reliability/Utilization under different GPU performance ($\\alpha$)')
    plt.tight_layout()
    plt.show()
    plt.savefig(fileResPath + "gpu-shape-r-utilization.png")

    # Plot Figure 3:
    xAxis = [uScale[0] for uScale in uScale_range]
    y1 = [result[(uScale[0], 'uScale', 'reliability')] for uScale in uScale_range]
    y2 = [result[(uScale[0], 'uScale', 'utilization')] for uScale in uScale_range]
    fig, ax1 = plt.subplots(figsize=(7, 5))
    ax1.plot(xAxis, y1, 'b-', marker='o', label='Reliability')
    ax1.set_xlabel(r'$\beta$')
    ax1.set_ylabel('Reliability', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax2 = ax1.twinx()
    ax2.plot(xAxis, y2, 'r--', marker='s', label='Utilization')
    ax2.set_ylabel('GPU Utilization', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper left')
    plt.title(f'Reliability/Utilization under different GPU performance ($\\beta$)')

    plt.tight_layout()
    plt.show()
    plt.savefig(fileResPath + "gpu-scale-r-utilization.png")


if __name__ == '__main__':
    n = 2   # GPU最多可以执行的layer数
    i = 6   # batch数 128
    j = 4   # layer数 16
    k = 4   # GPU数 20
    v = 1
    alpha = 1
    c_f = 6
    c_b = 12
    standard = 2000
    theta = 5
    epsilon = 0
    k_range = [k for k in range(5, 10)]  # gpu number
    plot1()
    plot23()









