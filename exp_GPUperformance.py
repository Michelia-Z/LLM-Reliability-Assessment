"""
    Goal: given the policy, analysis the reliability under different parameters of GPU performance
    Parameters: uShape; uScale; time threshold
"""

import matplotlib.pyplot as plt
from Evaluate import adaptability_assessment
from Genetic import genetic_algorithm
from matplotlib.pylab import mpl
#mpl.use('Qt5Agg')  #解决plt函数卡住的问题：切换Matplotlib的后端渲染引擎


population_size = 10
mutation_rate = 0.2
crossover_rate1 = 0.2
crossover_rate2 = 0.2
max_generations = 5  #GA迭代次数
iterations = 100   #评价函数sample个数

def compare(fileResPath, input_matrix1, input_matrix21, input_matrix22, i, j, k, n, v, alpha, c_f, c_b, standard_range, iterations, uShape_range, uScale_range, theta, epsilon):
    """
        (input_matrix1, input_matrix21, input_matrix22):    given policy
        iterations:                                         sample number to assess the reliability
        standard_range:                                     time threshold
        uShape_range, uScale_range:                         parameter range
        theta, epsilon:                                     parameter of function f=theta*u+epsilon
    """
    result = {}
    # 可调整参数画图
    for standard in standard_range:
        for uShape in uShape_range:
            for uScale in uScale_range:
                print(f"Experiment for {standard, uShape, uScale}")
                adaptability_score1, utilization_ratio1 = adaptability_assessment(input_matrix1,
                                                                                  input_matrix21,
                                                                                  input_matrix22,
                                                                                  i, j, k, n, v, alpha, c_f,
                                                                                  c_b,
                                                                                  iterations,
                                                                                  standard, uShape,
                                                                                  uScale, theta, epsilon)
                result[(standard, uShape[0], uScale[0])] = adaptability_score1

    # Figure 1:
    # x-axis: uShape; y-axis: Reliability; legend: standard
    uScale = uScale_range[0]
    xAxis = [uShape[0] for uShape in uShape_range]
    plt.figure(figsize=(6, 5))
    markers = ['o', 's', '^', 'D', 'p', 'h', 'v']  # 10种不同标记
    colors = ['g', 'b', 'orange', 'purple', 'r'] # 使用colormap生成颜色

    for i, standard in enumerate(standard_range):
        data = [result[(standard, uShape_range[n][0], uScale[0])] for n in range(len(uShape_range))]
        plt.plot(
            xAxis, data,
            marker=markers[i % len(markers)],  # 循环使用预定义的标记
            markersize=8,
            color=colors[i],
            linestyle='-',
            linewidth=1.5,
            label=f'$\\bar{{\\alpha}}={uShape[0]}$'
        )


    # for standard in standard_range:
    #     data = [result[(standard, uShape_range[n][0], uScale[0])] for n in
    #             range(len(uShape_range))]
    #     plt.plot(xAxis, data, marker='o', label=f'$\\bar{{\\tau}}_1={standard}$')
    plt.xlabel(r'$\alpha$',fontsize=15)
    plt.xticks(fontsize=12)
    plt.ylabel('Reliability',fontsize=15)
    plt.yticks(fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.savefig(fileResPath+"gpu-r-shape.png")

    # Figure 2:
    # x-axis: uScale; y-axis: Reliability; legend: standard
    uShape = uShape_range[0]
    xAxis = [uScale[0] for uScale in uScale_range]
    markers = ['o', 's', '^', 'D', 'p', 'h', 'v']  # 10种不同标记
    colors = ['g', 'b', 'orange', 'purple', 'r'] # 使用colormap生成颜色
    plt.figure(figsize=(6, 5))

    for i, standard in enumerate(standard_range):
        data = [result[(standard, uShape[0], uScale_range[n][0])] for n in
                range(len(uScale_range))]
        plt.plot(
            xAxis, data,
            marker=markers[i % len(markers)],  # 循环使用预定义的标记
            markersize=8,
            color=colors[i],
            linestyle='-',
            linewidth=1.5,
            label=f'$\\bar{{\\alpha}}={uShape[0]}$'
        )


    # for standard in standard_range:
    #     data = [result[(standard, uShape[0], uScale_range[n][0])] for n in
    #             range(len(uScale_range))]
    #     plt.plot(xAxis, data, marker='o', label=f'$\\bar{{\\tau}}_1={standard}$')
    plt.xlabel(r'$\beta$', fontsize=15)
    plt.xticks(fontsize=12)
    plt.ylabel('Reliability',fontsize=15)
    plt.yticks(fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.savefig(fileResPath+"gpu-r-scale.png")

if __name__ == "__main__":
    fileDataPath = 'data/'
    fileResPath = 'result/'

    n = 2   # GPU最多可以执行的layer数
    i = 6   # batch数
    j = 4   # layer数
    k = 4   # GPU数
    v = 1
    alpha = 1
    c_f = 6
    c_b = 12

    # parameter settings
    theta = 5
    epsilon = 0
    standard_range = [1500, 2000, 2500]  # time threshold
    uShape_range = [[a/10 for _ in range(k)] for a in range(11, 13)]
    uScale_range = [[a/10 for _ in range(k)] for a in range(10, 16)]

    # Fixed the policy
    standard = standard_range[0]
    uShape = uShape_range[0]
    uScale = uScale_range[0]
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

    # Given the policy --> analysis R under different parameters
    compare(fileResPath, input_matrix1, input_matrix21, input_matrix22, i, j, k, n, v, alpha, c_f, c_b,
            standard_range, iterations, uShape_range, uScale_range, theta, epsilon)


