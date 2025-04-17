import numpy as np
import matplotlib.pyplot as plt
from Evaluate import adaptability_assessment
from Genetic import genetic_algorithm
from matplotlib.pylab import mpl
#mpl.use('Qt5Agg')  #解决plt函数卡住的问题：切换Matplotlib的后端渲染引擎

fileDataPath = 'data/'
fileResPath = 'result/'

population_size = 10
mutation_rate = 0.2
crossover_rate1 = 0.8
crossover_rate2 = 0.8
max_generations = 2  #GA迭代次数
iterations = 10   #评价函数sample个数


def plot1(n, i, j, k, v, alpha, c_f, c_b, theta, epsilon, standard_range, uShape_range, R_range, gpuNum_range):
    # Figure 1:
    # x-axis: reliability; y-axis: GPU number; legend: standard
    result = {}
    uShape, uScale = uScale_range[0], uScale_range[-1]
    for R in R_range:  # 遍历R的值
        for standard in standard_range[::-1]:
            # gpu_num遍历范围缩小：从下一个standard对应的最小GPU数开始遍历；或者上一个R对应的最小GPU数开始遍历
            n1 = standard_range.index(standard)
            n2 = R_range.index(R)
            if n1 != len(standard_range) - 1 and n2 != 0:
                gpu_num_min = max(result[(standard_range[n1 + 1], R, uShape[0], uScale[0])],
                                  result[(standard, R_range[n2 - 1], uShape[0], uScale[0])])
            elif n1 == len(standard_range) - 1 and n2 != 0:
                gpu_num_min = result[(standard, R_range[n2 - 1], uShape[0], uScale[0])]
            elif n1 != len(standard_range) - 1 and n2 == 0:
                gpu_num_min = result[(standard_range[n1 + 1], R, uShape[0], uScale[0])]
            else:  # n1==len(standard_range)-1 and n2==0
                gpu_num_min = gpuNum_range[0]

            for gpu_num in range(gpu_num_min, gpuNum_range[-1] + 1):
                final_score, utilization_ratio, input_matrix1, input_matrix21, input_matrix22 = genetic_algorithm(n, i,
                                                                                                                  j,
                                                                                                                  k, v,
                                                                                                                  alpha,
                                                                                                                  c_f,
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
                if final_score >= R:
                    result[(standard, R, uShape[0], uScale[0])] = gpu_num
                    break
            if final_score < R:  # 异常值
                result[(standard, R, uShape[0], uScale[0])] = 1

    xAxis = [R for R in R_range]


    markers = ['o', 's', '^', 'D', 'p', 'h', 'v' ]  # 10种不同标记
    colors = ['g', 'b', 'orange', 'purple', 'r' ]

    plt.figure(figsize=(6, 5))
    for i, standard in enumerate(standard_range):
        data = [result[(standard, R, uShape[0], uScale[0])] for R in R_range]
        plt.plot(
            xAxis, data,
            marker=markers[i % len(markers)],  # 循环使用预定义的标记
            markersize=8,
            color=colors[i],
            linestyle='-',
            linewidth=1.5,
            label=f'$\\bar{{\\tau}}_1={standard}$'
        )
    '''
    for standard in standard_range:
        data = [result[(standard, R, uShape[0], uScale[0])] for R in R_range]
        plt.plot(xAxis, data, marker='o', label=f'$\\bar{{\\tau}}_1={standard}$')
    '''
    plt.xlabel(r'$R_0$',fontsize=15)
    plt.xticks(fontsize=12)
    plt.ylabel('minimal GPU number',fontsize=15)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()
    plt.savefig(fileResPath + "gpu-num-tau.png")


def plot2(n, i, j, k, v, alpha, c_f, c_b, theta, epsilon, standard_range, uShape_range, R_range, gpuNum_range):
    ########################################################
    # Figure 2:
    # x-axis: reliability; y-axis: GPU number; legend: uShape
    result = {}
    standard, uScale = standard_range[0], uScale_range[-1]
    for R in R_range:  # 遍历R的值
        for uShape in uShape_range:
            # gpu_num遍历范围缩小：从上一个R对应的最小GPU数开始遍历
            n2 = R_range.index(R)
            if n2 != 0:
                gpu_num_min = result[(standard, R_range[n2 - 1], uShape[0], uScale[0])]
            else:  # if n2==0
                gpu_num_min = gpuNum_range[0]

            for gpu_num in range(gpu_num_min, gpuNum_range[-1] + 1):
                final_score, utilization_ratio, input_matrix1, input_matrix21, input_matrix22 = genetic_algorithm(n, i, j,
                                                                                                                  k, v,
                                                                                                                  alpha,
                                                                                                                  c_f,
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
                if final_score >= R:
                    result[(standard, R, uShape[0], uScale[0])] = gpu_num
                    break
            if final_score < R:  # 异常值
                result[(standard, R, uShape[0], uScale[0])] = 1

    xAxis = [R for R in R_range]


    markers = ['o', 's', '^', 'D', 'p', 'h', 'v']  # 10种不同标记
    colors = ['g', 'b', 'orange', 'purple', 'r'] # 使用colormap生成颜色

    plt.figure(figsize=(6, 5))

    for i, uShape in enumerate(uShape_range):
        data = [result[(standard, R, uShape[0], uScale[0])] for R in R_range]
        plt.plot(
            xAxis, data,
            marker=markers[i % len(markers)],  # 循环使用预定义的标记
            markersize=8,
            color=colors[i],
            linestyle='-',
            linewidth=1.5,
            label=f'$\\bar{{\\alpha}}={uShape[0]}$'
        )
    '''
    for uShape in uShape_range:
        data = [result[(standard, R, uShape[0], uScale[0])] for R in R_range]
        plt.plot(xAxis, data, marker='o', label=f'$\\bar{{\\alpha}}={uShape[0]}$')
    '''
    plt.xlabel(r'$R_0$',fontsize = 15)
    plt.xticks(fontsize=12)
    plt.ylabel('minimal GPU number',fontsize = 15)
    plt.yticks(fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.savefig(fileResPath + "gpu-num-shape.png")


def plot3(n, i, j, k, v, alpha, c_f, c_b, theta, epsilon, standard_range, uShape_range, R_range, gpuNum_range):
    # Figure 3:
    # x-axis: reliability; y-axis: GPU number; legend: uScale
    result = {}
    standard, uShape = standard_range[0], uShape_range[0]
    for R in R_range:  # 遍历R的值
        for uScale in uScale_range:
            # gpu_num遍历范围缩小：从上一个R对应的最小GPU数开始遍历
            n2 = R_range.index(R)
            if n2 != 0:
                gpu_num_min = result[(standard, R_range[n2 - 1], uShape[0], uScale[0])]
            else:  # if n2==0
                gpu_num_min = gpuNum_range[0]

            for gpu_num in range(gpu_num_min, gpuNum_range[-1] + 1):
                final_score, utilization_ratio, input_matrix1, input_matrix21, input_matrix22 = genetic_algorithm(n, i, j,
                                                                                                                  k, v,
                                                                                                                  alpha,
                                                                                                                  c_f,
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
                if final_score >= R:
                    result[(standard, R, uShape[0], uScale[0])] = gpu_num
                    break
            if final_score < R:  # 异常值
                result[(standard, R, uShape[0], uScale[0])] = 1

    xAxis = [R for R in R_range]

    markers = ['o', 's', '^', 'D', 'p', 'h', 'v']  # 10种不同标记
    colors = ['g', 'b', 'orange', 'purple', 'r']   # 使用colormap生成颜色

    plt.figure(figsize=(6, 5))

    for i, uScale in enumerate(uScale_range):
        data = [result[(standard, R, uShape[0], uScale[0])] for R in R_range]
        # plt.plot(xAxis, data, label=f'$\\bar{{\\beta}}={uScale[0]}$')
        plt.plot(
            xAxis, data,
            marker=markers[i % len(markers)],  # 循环使用预定义的标记
            markersize=8,
            color=colors[i],
            linestyle='-',
            linewidth=1.5,
            label=f'$\\bar{{\\beta}}={uScale[0]}$'
        )

    plt.xlabel(r'$R_0$',fontsize = 15)
    plt.xticks(fontsize=12)
    plt.ylabel('minimal GPU number',fontsize = 15)
    plt.yticks(fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.savefig(fileResPath + "gpu-num-scale.png")


if __name__ == "__main__":
    n = 2   # GPU最多可以执行的layer数
    i = 6   # batch数
    j = 4   # layer数
    k = 4   # GPU数
    v = 1
    alpha = 1
    c_f = 6
    c_b = 12
    theta = 5
    epsilon = 0

    # parameter settings
    standard_range = [1500, 2000, 2500]
    uShape_range = [[a / 10 for _ in range(k)] for a in range(11, 13)]
    uScale_range = [[a / 10 for _ in range(k)] for a in range(10, 14)]
    R_range = [r/10 for r in range(8, 10)]
    gpuNum_range = range(2, 10)

    plot1(n, i, j, k, v, alpha, c_f, c_b, theta, epsilon, standard_range, uShape_range, R_range, gpuNum_range)
    plot2(n, i, j, k, v, alpha, c_f, c_b, theta, epsilon, standard_range, uShape_range, R_range, gpuNum_range)
    plot3(n, i, j, k, v, alpha, c_f, c_b, theta, epsilon, standard_range, uShape_range, R_range, gpuNum_range)

