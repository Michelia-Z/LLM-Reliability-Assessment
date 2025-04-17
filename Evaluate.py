import random
import numpy as np
from scipy.stats import invgamma

#将层-GPU编码还原为初始编码
#输入：n*k维的GPU分布矩阵（染色体矩阵部分）
#输出：j*k维的层-GPU矩阵，每行代表对应的Transformer层可以由哪些GPU训练
def map_numbers_to_rows(input_matrix, n, k, j):
    """
    将输入矩阵映射到行索引。
    确保每列的非零值不重复，重复值只保留一个，其他重复值置为 0。
    goal： n*k矩阵 -> j*k矩阵
    input_matrix n*k矩阵
    n 每个GPU上的最大layer数
    k GPU数量
    j layer数
    """
    # 按列检查，移除重复的非零值
    for col in range(k):
        seen_values = set()  # 用于记录已经遇到的非零值
        for row in range(n):
            value = input_matrix[row][col]
            if value != 0:
                if value in seen_values:  # 如果发现重复值
                    input_matrix[row][col] = 0  # 将重复值置为 0
                else:
                    seen_values.add(value)  # 记录当前值
    # 初始化输出矩阵
    output_matrix = np.zeros((j, k), dtype=int)

    # 映射输入矩阵到输出矩阵
    for col in range(k):
        for row in range(n):
            value = input_matrix[row][col]
            if 1 <= value <= j:
                for i in range(k):
                    if output_matrix[value - 1][i] == 0:
                        output_matrix[value - 1][i] = col + 1
                        break
    return output_matrix

# 统计每一层可处理GPU数量
def count_nonzero_per_row(input_matrix):
    return np.count_nonzero(input_matrix, axis=1)

# 获取 GPU_task 矩阵（任务的每个层在哪个GPU上计算）
def trans_GPU_task(input_matrix1, input_matrix21, input_matrix22, gpu_num, i, j):
    # task vector & GPU vector 转化 task-GPU matrix
    task_gpu = np.zeros((i, j), dtype=int)
    task_check = np.zeros((i), dtype=int)
    for index in range(len(input_matrix21)):
        task = input_matrix21[index]
        GPU = input_matrix22[index]
        if task_check[task - 1] == j:
            continue
        else:
            task_check[task - 1] += 1
            GPU = GPU_mod(GPU, gpu_num[task_check[task - 1] - 1])
            task_gpu[task - 1][task_check[task - 1] - 1] = input_matrix1[task_check[task - 1] - 1][GPU - 1]
    return task_gpu

# 定义获取GPU的取余函数
def GPU_mod(gpu, gpu_num):
    if gpu_num == 0:
        return 0
    if gpu > gpu_num:
        gpu = gpu % gpu_num
        if gpu == 0:
            gpu = gpu_num
    elif gpu <= 0:
        gpu = 1
    return gpu

def adaptability_assessment(input_matrix1, input_matrix21, input_matrix22, i, j, k, n, v, alpha, c_f,c_b, iterations,
                            standard, uShape, uScale, theta, epsilon):
    #参数iteration: 评价函数sample个数
    task_matrix = np.zeros((i, 2 * j), dtype=int)
    time_matrix = np.zeros((k, n * 2 * i), dtype=int)
    work_time_matrix = np.zeros((k,iterations), dtype=int)
    utilization_matrix = np.zeros((iterations), dtype=float)
    input_matrix1 = map_numbers_to_rows(input_matrix1, n, k, j)
    gpu_num = count_nonzero_per_row(input_matrix1)
    task_gpu = trans_GPU_task(input_matrix1, input_matrix21, input_matrix22, gpu_num, i, j)

    # 用于保存每次迭代的 task_matrix 最后一列
    statistics_matrix = np.zeros((i, iterations), dtype=int)  #行表示batch，列表示sample，value表示结束时间

    for iteration in range(iterations):  # 循环 `iterations` 次
        # 重置 check 和时间矩阵（如果需要在每次迭代中重新计算）
        check = np.zeros(i, dtype=int)
        time_matrix.fill(0)

        for index in range(2 * i * j):
            task = input_matrix21[index]
            check[task - 1] += 1
            trans = check[task - 1]

            # 保留原来的 f 和 b，不做改变
            if trans == 1:
                gpu = task_gpu[task - 1][trans - 1]
                f_with_variation = GPU_time(uShape[gpu - 1], uScale[gpu - 1], theta, epsilon)
                work_time_matrix[gpu-1,iteration] = work_time_matrix[gpu-1,iteration] + f_with_variation
                gpu_index_array = np.where(time_matrix[gpu - 1] == 0)[0]
                gpu_index = gpu_index_array[0] if gpu_index_array.size > 0 else -1
                time = f_with_variation if gpu_index == 0 else time_matrix[gpu - 1][gpu_index - 1] + f_with_variation
            elif trans <= j:
                t = alpha + c_f / v
                gpu = task_gpu[task - 1][trans - 1]
                #这里需要调整GPU效率的影响
                f_with_variation = GPU_time(uShape[gpu - 1], uScale[gpu - 1], theta, epsilon)
                work_time_matrix[gpu - 1, iteration] = work_time_matrix[gpu - 1, iteration] + f_with_variation
                gpu_index_array = np.where(time_matrix[gpu - 1] == 0)[0]
                gpu_index = gpu_index_array[0] if gpu_index_array.size > 0 else -1
                time = max(
                    time_matrix[gpu - 1][gpu_index - 1] if gpu_index > 0 else 0,
                    task_matrix[task - 1][trans - 2] + t * int(
                        task_gpu[task - 1][trans - 1] != task_gpu[task - 1][trans - 2])
                ) + f_with_variation
            elif trans == j + 1:
                gpu = task_gpu[task - 1][2 * j - trans]
                #这里需要调整GPU效率的影响
                b_with_variation = GPU_time(uShape[gpu - 1], uScale[gpu - 1], theta, epsilon)
                work_time_matrix[gpu - 1, iteration] = work_time_matrix[gpu - 1, iteration] + b_with_variation
                gpu_index_array = np.where(time_matrix[gpu - 1] == 0)[0]
                gpu_index = gpu_index_array[0] if gpu_index_array.size > 0 else -1
                time = max(
                    time_matrix[gpu - 1][gpu_index - 1] if gpu_index > 0 else 0,
                    task_matrix[task - 1][trans - 2]
                ) + b_with_variation
            else:
                t = alpha + c_b / v
                gpu = task_gpu[task - 1][2 * j - trans]
                #这里需要调整GPU效率的影响
                b_with_variation = GPU_time(uShape[gpu - 1], uScale[gpu - 1], theta, epsilon)
                work_time_matrix[gpu - 1, iteration] = work_time_matrix[gpu - 1, iteration] + b_with_variation
                gpu_index_array = np.where(time_matrix[gpu - 1] == 0)[0]
                gpu_index = gpu_index_array[0] if gpu_index_array.size > 0 else -1
                time = max(
                    time_matrix[gpu - 1][gpu_index - 1] if gpu_index > 0 else 0,
                    task_matrix[task - 1][trans - 2] +
                    t * int(task_gpu[task - 1][2 * j - trans] != task_gpu[task - 1][2 * j - trans - 1])
                ) + b_with_variation
            if gpu_index >= 0:
                time_matrix[gpu - 1][gpu_index] = time
            task_matrix[task - 1][trans - 1] = time

        # 提取每次迭代的 task_matrix 最后一列，并添加到统计矩阵
        statistics_matrix[:, iteration] = task_matrix[:, -1]  # 取每个任务的最后一列
        sum_time = np.max(task_matrix[:, -1])
        sum_work_time = np.sum(work_time_matrix[:,iteration])
        utilization_matrix[iteration] = sum_work_time / (sum_time * k)

    # 计算每行小于 standard 的比例
    less_than_standard_proportion = np.sum(statistics_matrix < standard, axis=1) / iterations
    # print('time', np.average(np.max(statistics_matrix, axis=0))) # sample的最大完成时间-->average

    # 计算适应性评估分数（所有比例相乘）
    adaptability_score = np.prod(less_than_standard_proportion)
    utilization_ratio = np.average(utilization_matrix)

    return adaptability_score, utilization_ratio

def adaptability_assessment_time(input_matrix1, input_matrix21, input_matrix22, i, j, k, n, v, alpha, c_f, c_b, f, b):
    task_matrix = np.zeros((i, 2 * j), dtype=int)
    time_matrix = np.zeros((k, n * 2 * i), dtype=int)
    work_time_matrix = np.zeros(k, dtype=int)
    input_matrix1 = map_numbers_to_rows(input_matrix1,n,k,j)
    gpu_num = count_nonzero_per_row(input_matrix1)
    task_gpu = trans_GPU_task(input_matrix1, input_matrix21, input_matrix22, gpu_num, i, j)
    check = np.zeros(i, dtype=int)

    for index in range(2 * i * j):
        task = input_matrix21[index]
        check[task - 1] += 1
        trans = check[task - 1]
        if trans == 1:
            gpu = task_gpu[task - 1][trans - 1]
            f_with_variation = f[trans - 1][gpu - 1]
            work_time_matrix[gpu - 1] = work_time_matrix[gpu - 1] + f_with_variation
            gpu_index_array = np.where(time_matrix[gpu - 1] == 0)[0]
            gpu_index = gpu_index_array[0] if gpu_index_array.size > 0 else -1
            time = f_with_variation if gpu_index == 0 else time_matrix[gpu - 1][gpu_index - 1] + f_with_variation
        elif trans <= j:
            t = alpha + c_f / v
            gpu = task_gpu[task - 1][trans - 1]
            # 这里需要调整GPU效率的影响
            f_with_variation = f[trans -1][gpu - 1]
            work_time_matrix[gpu - 1] = work_time_matrix[gpu - 1] + f_with_variation
            gpu_index_array = np.where(time_matrix[gpu - 1] == 0)[0]
            gpu_index = gpu_index_array[0] if gpu_index_array.size > 0 else - 1
            time = max(
                time_matrix[gpu - 1][gpu_index - 1] if gpu_index > 0 else 0,
                task_matrix[task - 1][trans - 2] + t * int(
                    task_gpu[task - 1][trans - 1] != task_gpu[task - 1][trans - 2])
            ) + f_with_variation
        elif trans == j + 1:
            gpu = task_gpu[task - 1][2 * j - trans]
            # 这里需要调整GPU效率的影响
            b_with_variation = b[2 * j - trans][gpu - 1]
            work_time_matrix[gpu - 1] = work_time_matrix[gpu - 1] + b_with_variation
            gpu_index_array = np.where(time_matrix[gpu - 1] == 0)[0]
            gpu_index = gpu_index_array[0] if gpu_index_array.size > 0 else - 1
            time = max(
                time_matrix[gpu - 1][gpu_index - 1] if gpu_index > 0 else 0,
                task_matrix[task - 1][trans - 2]
            ) + b_with_variation
        else:
            t = alpha + c_b / v
            gpu = task_gpu[task - 1][2 * j - trans]
            # 这里需要调整GPU效率的影响
            b_with_variation = b[2 * j - trans][gpu - 1]
            work_time_matrix[gpu - 1] = work_time_matrix[gpu - 1] + b_with_variation
            gpu_index_array = np.where(time_matrix[gpu - 1] == 0)[0]
            gpu_index = gpu_index_array[0] if gpu_index_array.size > 0 else - 1
            time = max(
                time_matrix[gpu - 1][gpu_index - 1] if gpu_index > 0 else 0,
                task_matrix[task - 1][trans - 2] +
                t * int(task_gpu[task - 1][2 * j - trans] != task_gpu[task - 1][2 * j - trans - 1])
            ) + b_with_variation
        if gpu_index >= 0:
            time_matrix[gpu - 1][gpu_index] = time
        task_matrix[task - 1][trans - 1] = time
    sum_time = np.max(task_matrix[:, -1])
    sum_work_time = np.sum(work_time_matrix)
    utilization_ratio = sum_work_time / (sum_time * k)
    return sum_time, utilization_ratio

# 适应度评估：选择适应度最小的个体
def evaluate_population(population, i, j, k, n, v, alpha, c_f, c_b, iterations, standard, uShape, uScale, theta, epsilon):
    """
    评估种群中每个个体的适应度分数。
    """
    scores = []
    utilization_ratios = []
    for matrix, vector1, vector2 in population:
        # 调用 adaptability_assessment 计算适应度分数
        adaptability_score, utilization_ratio = adaptability_assessment(matrix, vector1, vector2, i, j, k, n, v, alpha,
                                                                        c_f, c_b, iterations,
                                                                        standard, uShape, uScale, theta,
                                                                        epsilon)
        scores.append(adaptability_score)
        utilization_ratios.append(utilization_ratio)
    return scores, utilization_ratios

def evaluate_population_time(population, i, j, k, n, v, alpha, c_f, c_b, f, b):
    scores = []
    utilization_ratios = []
    for matrix, vector1, vector2 in population:
        adaptability_score,utilization_ratio = adaptability_assessment_time(matrix, vector1, vector2, i, j, k, n, alpha, v, c_f, c_b, f, b)
        scores.append(adaptability_score)
        utilization_ratios.append(utilization_ratio)
    return scores, utilization_ratios

def GPU_time(uShape, uScale, theta, epsilon):
    # generate random number f = theta * u + epsilon
    # u ~ inverse gamma distribution
    time = epsilon + theta * invgamma.rvs(a=uShape, scale=uScale)
    while time <= 1 or time >= 10e+3: #异常值
        time = epsilon + theta * invgamma.rvs(a=uShape, scale=uScale)
    return time