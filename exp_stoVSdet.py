"""
    compare policy via stochastic model & deterministic model
    f和b取均值，优化最大完成时间，得到策略 \pi_1
    考虑f和b的随机性，优化可靠性，得到策略 \pi_2
    仿真，生成f和b的样本，在1000个采样下，比较 \pi_1和\pi_2的效果（R和最大完成时间）

    Out-of-sample experiment
    1. Generate out-of-sample data
    2. use the same sample, and get the total training time under different policy
"""

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from Genetic import genetic_algorithm, genetic_algorithm_time
from Evaluate import adaptability_assessment_time, GPU_time


fileDataPath = 'data/'
fileResPath = 'result/'
population_size = 10
mutation_rate = 0.2
crossover_rate1 = 0.2
crossover_rate2 = 0.2
max_generations = 2  #GA迭代次数
iterations = 100   #评价函数sample个数


def outOFsample(sto_matrix1, sto_matrix21, sto_matrix22, det_matrix1, det_matrix21, det_matrix22, sampleNum, i, j, k, n, v, alpha, c_f, c_b, uShape, uScale, theta, epsilon):
    f_sample = np.zeros((j,k))
    b_sample = np.zeros((j,k))
    computeTime_sto = []
    computeTime_det = []
    for num in range(sampleNum):
        for jj in range(j):
            for kk in range(k):
                f_sample[jj,kk] = GPU_time(uShape[kk], uScale[kk], theta, epsilon)
                b_sample[jj,kk] = GPU_time(uShape[kk], uScale[kk], theta, epsilon)

        sto_time, sto_utilization = adaptability_assessment_time(sto_matrix1, sto_matrix21, sto_matrix22, i, j, k, n, v, alpha, c_f, c_b, f_sample, b_sample)
        det_time, det_utilization = adaptability_assessment_time(det_matrix1, det_matrix21, det_matrix22, i, j, k, n, v, alpha, c_f, c_b, f_sample, b_sample)
        computeTime_sto.append(sto_time)
        computeTime_det.append(det_time)

    return computeTime_sto, computeTime_det


# plot out-of-sample figure
def plotHistogram(computeTime_sto, computeTime_det):
    """
    绘制对比图：随机模型（sto）与确定性模型（det）的训练时间分布直方图。
    Parameters:
        computeTime_sto: 随机模型的训练时间数据
        computeTime_det: 确定性模型的训练时间数据
    """
    ## 将所有latency数据转为pd.DataFrame格式
    dataAll = [['Stochastic', a] for a in computeTime_sto]
    dataAll += [['Deterministic', a] for a in computeTime_det]
    df = pd.DataFrame(dataAll, columns=['Model', 'Time'])

    # 绘图设置
    # sns.set(style="ticks", palette="muted")
    sns.set(style="ticks")  # style："white", "whitegrid", "ticks"
    plt.figure(figsize=(7, 5))
    sns.histplot(data=df, x="Time", hue="Model", kde=False, bins=70, alpha=0.75)
    # plt.xlim(1600, 2400)
    plt.xlabel("Out-of-sample training time", fontsize=13)
    plt.ylabel("Frequency of out-of-sample data", fontsize=13)
    plt.title("Training Time Distribution: Stochastic vs Deterministic Models", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(fileResPath+'exp3-sp-det.png', dpi=800)
    plt.show(block=True)


    # statistical info for out-of-sample data
    computeTime_sto, computeTime_det = np.array(computeTime_sto), np.array(computeTime_det)
    min = [computeTime_sto.min(), computeTime_det.min()]
    max = [computeTime_sto.max(), computeTime_det.max()]
    aver = [computeTime_sto.mean(), computeTime_det.mean()]
    std = [computeTime_sto.std(), computeTime_det.std()]
    quaL = [np.percentile(computeTime_sto, 25), np.percentile(computeTime_det, 25)]
    quaU = [np.percentile(computeTime_sto, 75), np.percentile(computeTime_det, 75)]
    print('\nmin=', min, '\nmax=', max, '\naver=', aver, '\nquaL=', quaL, '\nquaU=', quaU, '\nstd=', std)


if __name__ == '__main__':
    sampleNum = 100  # num of sample for the out-of-sample experiment

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
    standard = 2000
    uShape = [1.5 for _ in range(k)]
    uScale = [1 for _ in range(k)]

    # policy under stochastic case
    print('\n\n', '==== DRO model ====')
    sto_score, sto_utilization, sto_matrix1, sto_matrix21, sto_matrix22 = genetic_algorithm(n, i, j, k, v,
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
    # policy under deterministic case
    print('\n\n', '==== Deterministic model ====')
    det_score, det_utilization, det_matrix1, det_matrix21, det_matrix22 = genetic_algorithm_time(n, i, j, k,
                                                                                                 v, alpha,
                                                                                                 c_f, c_b,
                                                                                                 population_size,
                                                                                                 mutation_rate,
                                                                                                 crossover_rate1,
                                                                                                 crossover_rate2,
                                                                                                 max_generations,
                                                                                                 uShape,
                                                                                                 uScale,
                                                                                                 theta,
                                                                                                 epsilon)
    # Generate out-of-sample data; assess all samples given two policies.
    computeTime_sto, computeTime_det = outOFsample(sto_matrix1, sto_matrix21, sto_matrix22, det_matrix1, det_matrix21,
                                                   det_matrix22, sampleNum, i, j, k, n, v, alpha, c_f, c_b, uShape,
                                                   uScale, theta, epsilon)
    plotHistogram(computeTime_sto, computeTime_det)