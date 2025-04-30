"""
    Goal: given the policy, analysis the reliability under different parameters of GPU performance
    Parameters: uShape; uScale; time threshold
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from Evaluate import adaptability_assessment, adaptability_assessment_time, GPU_time
from Genetic import genetic_algorithm, genetic_algorithm_time
from matplotlib.pylab import mpl
mpl.use('Qt5Agg')  #解决plt函数卡住的问题：切换Matplotlib的后端渲染引擎


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
            label=f'$\\bar{{\\tau}}_1={standard}$'
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
            label=f'$\\bar{{\\tau}}_1={standard}$'
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


# 仿真sampleNum次 --> 计算时间的分布
def computeTimeSample(sto_matrix1, sto_matrix21, sto_matrix22, sampleNum, i, j, k, n, v, alpha, c_f, c_b, standard, uShape, uScale, theta, epsilon):
    """
    多次采样，得到每个sample的总计算时间
    Return:
        computeTimeSet:     (vector) all samples' computation times
        reliaiblity:        (scalar) reliability of this policy
    Parameters:
        sampleNum:      采样sample次数
        sto_matrix1...: 给定的policy
    """
    f_sample = np.zeros((j,k))
    b_sample = np.zeros((j,k))
    computeTimeSet = []
    for num in range(sampleNum):
        for jj in range(j):
            for kk in range(k):
                f_sample[jj,kk] = GPU_time(uShape[kk], uScale[kk], theta, epsilon)
                b_sample[jj,kk] = GPU_time(uShape[kk], uScale[kk], theta, epsilon)

        time, utilization = adaptability_assessment_time(sto_matrix1, sto_matrix21, sto_matrix22, i, j, k, n, v, alpha, c_f, c_b, f_sample, b_sample)
        computeTimeSet.append(time)
    reliaiblity = sum([a < standard for a in computeTimeSet]) / sampleNum
    return computeTimeSet, reliaiblity


# plot figure
def plot1(fileResPath, sampleNum, input_matrix1, input_matrix21, input_matrix22, i, j, k, n, v, alpha, c_f, c_b, standard, uShape_range, uScale_range, theta, epsilon):
    """
    绘制图：在不同alpha下，给定策略的总计算时间sample的箱线图 & 可靠性折线图。
    Figure 1: x-axis: uShape; y-axis: out-of-sample of computeTime / Reliability
    """
    result_r = []
    result_sample = []
    uScale = uScale_range[0]
    for uShape in uShape_range:
        print(f"Experiment for {uShape[0], uScale[0]}")
        computeTimeSet, reliaiblity = computeTimeSample(input_matrix1, input_matrix21, input_matrix22, sampleNum, i, j, k, n, v,
                                           alpha, c_f, c_b, standard, uShape, uScale, theta, epsilon)
        result_r.append(reliaiblity)
        result_sample += [[uShape[0], uScale[0], t] for t in computeTimeSet]
    df = pd.DataFrame(result_sample, columns=['uShape', 'uScale', 'computeTime'])  # 格式转为pd.DataFrame格式

    # ------------------- plot box of out-of-sample -------------------
    # 在同一个图中,采用双纵轴显示
    # 创建左轴
    fig, ax1 = plt.subplots(figsize=(8, 5))
    sns.set(style="ticks")  # style："white", "whitegrid", "ticks"
    sns.boxplot(data=df, x='uShape', y='computeTime', ax=ax1, width=.5, showfliers=False, showcaps=True, showmeans=True,
                meanprops={'marker': 'x', 'markeredgecolor': 'black', 'markerfacecolor': 'black'}, palette="Set2")
    # 设置左轴标签
    ax1.set_ylabel("frequency of computation times")
    ax1.set_xlabel(r"$\alpha$")
    # 获取分类的 x 轴位置，用于右轴x坐标对齐
    xticks = ax1.get_xticks()
    xticklabels = [float(label.get_text()) for label in ax1.get_xticklabels()]
    # 创建右轴
    ax2 = ax1.twinx()
    ax2.plot(xticks, result_r, color='blue', marker='^',  markersize=8, linestyle='--', linewidth=1.5)
    ax2.set_ylabel('reliability', color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')
    # 保持X轴标签一致
    ax1.set_xticks(xticks)
    ax1.set_xticklabels(xticklabels)
    # 调整y轴显示范围
    # ax1.set_ylim(0, 2000)
    # ax2.set_ylim(0.5, 1)
    plt.tight_layout()
    plt.show() #plt.show(block=True)
    plt.savefig(fileResPath + '1-gpu-shape-r.png', dpi=800)


def plot2(fileResPath, sampleNum, input_matrix1, input_matrix21, input_matrix22, i, j, k, n, v, alpha, c_f, c_b, standard, uShape_range, uScale_range, theta, epsilon):
    """
    绘制图：在不同beta下，给定策略的总计算时间sample的箱线图 & 可靠性折线图。
    Figure 2: x-axis: uSacle; y-axis: out-of-sample of computeTime / Reliability
    """
    result_r = []
    result_sample = []
    uShape = uShape_range[0]
    for uScale in uScale_range:
        print(f"Experiment for {uShape[0], uScale[0]}")
        computeTimeSet, reliaiblity = computeTimeSample(input_matrix1, input_matrix21, input_matrix22, sampleNum, i, j, k, n, v,
                                           alpha, c_f, c_b, standard, uShape, uScale, theta, epsilon)
        result_r.append(reliaiblity)
        result_sample += [[uShape[0], uScale[0], t] for t in computeTimeSet]
    df = pd.DataFrame(result_sample, columns=['uShape', 'uScale', 'computeTime'])  # 格式转为pd.DataFrame格式

    # ------------------- plot box of out-of-sample -------------------
    # 在同一个图中,采用双纵轴显示
    # 创建左轴
    fig, ax1 = plt.subplots(figsize=(8, 5))
    sns.set(style="ticks")
    sns.boxplot(data=df, x='uScale', y='computeTime', ax=ax1, width=.5, showfliers=False, showcaps=True, showmeans=True,
                meanprops={'marker': 'x', 'markeredgecolor': 'black', 'markerfacecolor': 'black'}, palette="Set2")
    # 设置左轴标签
    ax1.set_ylabel("frequency of computation times")
    ax1.set_xlabel(r"$\beta$")
    # 获取分类的 x 轴位置，用于右轴x坐标对齐
    xticks = ax1.get_xticks()
    xticklabels = [float(label.get_text()) for label in ax1.get_xticklabels()]
    # 创建右轴
    ax2 = ax1.twinx()
    ax2.plot(xticks, result_r, color='blue', marker='^',  markersize=8, linestyle='--', linewidth=1.5)
    ax2.set_ylabel('reliability', color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')
    # 保持X轴标签一致
    ax1.set_xticks(xticks)
    ax1.set_xticklabels(xticklabels)
    # 调整y轴显示范围
    # ax1.set_ylim(0, 2000)
    # ax2.set_ylim(0.5, 1)
    plt.tight_layout()
    plt.show() #plt.show(block=True)
    plt.savefig(fileResPath + '1-gpu-scale-r.png', dpi=800)


# plot figure (其他展现形式)
def plot0():
    result_r = []
    result_sample = []
    uScale = uScale_range[0]
    for uShape in uShape_range:
        print(f"Experiment for {uShape[0], uScale[0]}")
        computeTimeSet, reliaiblity = computeTimeSample(input_matrix1, input_matrix21, input_matrix22, sampleNum, i, j,
                                                        k, n, v,
                                                        alpha, c_f, c_b, standard, uShape, uScale, theta, epsilon)

        result_r.append(reliaiblity)
        result_sample += [[uShape[0], uScale[0], t] for t in computeTimeSet]  # ['uShape', 'uScale', 'computeTime']
    df = pd.DataFrame(result_sample, columns=['uShape', 'uScale', 'computeTime'])  # 格式转为pd.DataFrame格式

    # ------------------- plot box of out-of-sample -------------------
    # 上图是箱线图,下图是可靠性折线图
    # 创建上下两个子图
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(8, 8), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    # 上图：箱线图
    sns.boxplot(data=df, x='uShape', y='computeTime', ax=ax1, width=.5, showfliers=False, showcaps=True, showmeans=True,
                meanprops={'marker': 'x', 'markeredgecolor': 'black', 'markerfacecolor': 'black'}, palette="Set2")
    ax1.set_ylabel("frequency of computation times")
    ax1.set_xlabel(r"$\alpha$")
    # ax1.set_ylim(0, 2*standard)
    # 添加参考线
    # ax1.axhline(y=standard, color='red', linestyle='--', linewidth=1.5)
    # ax1.text(ax1.get_xlim()[1]-0.3, standard+50,  f'$\\bar{{\\tau}}_1$ = {standard}', color='red', ha='right', fontsize=12)
    # 获取分类的x轴位置，用于两个图的x坐标对齐
    xticks = ax1.get_xticks()
    xticklabels = [float(label.get_text()) for label in ax1.get_xticklabels()]
    # 下图：可靠性折线图
    ax2.plot(xticks, result_r, color='blue', marker='^',  markersize=8, linestyle='--', linewidth=1.5)
    ax2.set_ylabel('reliability')
    # ax2.set_ylim(0.5, 1)
    ax2.tick_params(axis='y')
    ax2.set_xlabel(r"$\alpha$")
    # 保持 X 轴标签一致
    ax2.set_xticks(xticks)
    ax2.set_xticklabels(xticklabels)
    # 美化布局
    plt.tight_layout()
    plt.show()
    plt.savefig(fileResPath + '1-gpu-shape-r.png', dpi=800)

    # ------------------- plot hist of out-of-sample -------------------
    # 每个shape对应的out-of-sample图
    plt.figure(figsize=(6, 5))
    sns.histplot(data=df, x='computeTime', hue='uShape', kde=False, bins=1000, alpha=0.6, legend='full')
    plt.axvline(x=standard, color='red', linestyle='--', linewidth=1.5)
    plt.text(standard + 20, plt.ylim()[1] * 0.9, f'$\\bar{{\\tau}}_1$ = {standard}', color='red', fontsize=12)
    plt.title('Frequency Distribution of Compute Time by uShape')
    plt.xlabel('Compute Time')
    plt.ylabel('Frequency')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xlim(0, 2*standard)  # 控制x轴范围
    plt.show()
    plt.savefig(fileResPath+'1-gpu-shape-r-2.png', dpi=800)

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
    uShape_range = [[a/10 for _ in range(k)] for a in range(11, 17, 2)]
    uScale_range = [[a/10 for _ in range(k)] for a in range(10, 14)]
    sampleNum = 5000 # number of simulating sample

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

    # Given the policy --> analysis R under different parameters (plot: out-of-sample & R)
    plot1(fileResPath, sampleNum, input_matrix1, input_matrix21, input_matrix22, i, j, k, n, v, alpha, c_f, c_b,
              standard, uShape_range, uScale_range, theta, epsilon)
    plot2(fileResPath, sampleNum, input_matrix1, input_matrix21, input_matrix22, i, j, k, n, v, alpha, c_f, c_b,
          standard, uShape_range, uScale_range, theta, epsilon)


