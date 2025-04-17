import numpy as np
import random
from Evaluate import evaluate_population
from Evaluate import evaluate_population_time
import copy

#Task Vector交叉:子代任务顺序交叉函数
#通过交叉映射生成新的子代染色体
def generate_new_chromosome(chromosome1,chromosome2):
    # 确保两条染色体的长度相同
    if len(chromosome1) != len(chromosome2):
        raise ValueError("两条染色体的长度必须相同")
    crossover_point = random.randint(1, len(chromosome1) - 1)
    # 将染色体1分为A和B，染色体2分为C和D
    A = chromosome1[:crossover_point]
    B = chromosome1[crossover_point:]
    C = chromosome2[:crossover_point]
    D = chromosome2[crossover_point:]
    arrayB_2d = [(B[i], i + 1) for i in range(len(B))]
    arrayD_2d = [(D[i], i + 1) for i in range(len(D))]
    sorted_arrayB_2d = sorted(arrayB_2d, key=lambda x: x[0])
    sorted_arrayD_2d = sorted(arrayD_2d, key=lambda x: x[0])

    # 生成子染色体3和子染色体4
    child_chromosome3 = chromosome_mapping(sorted_arrayB_2d, sorted_arrayD_2d)
    child_chromosome4 = chromosome_mapping(sorted_arrayD_2d, sorted_arrayB_2d)
    # 生成子代1：将染色体A和子染色体3拼接
    child_gen1 = np.concatenate((A, child_chromosome3), axis=0)
    child_gen2 = np.concatenate((C, child_chromosome4), axis=0)
    # 生成子代2：将染色体B和子染色体4拼接

    return child_gen1, child_gen2

#交叉中间函数（交叉映射函数）
def chromosome_mapping(sorted_array1_2d, sorted_array2_2d):
    # 初始化新的染色体
    new_chromosome = []
    length = len(sorted_array2_2d)

    for i in range(1, length + 1):  # 遍历次序编码1到n
        # 在子染色体2中找到次序编码为i的数组的位置索引
        index_in_array2 = next(idx for idx, val in enumerate(sorted_array2_2d) if val[1] == i)

        # 在子染色体1中找到对应位置的值
        value_from_array1 = sorted_array1_2d[index_in_array2][0]

        # 添加值到新染色体中
        new_chromosome.append(value_from_array1)

    return new_chromosome

#交叉中间函数（随机选择一个交叉点，将两条染色体分为四部分）
def crossover(chromosome1, chromosome2):
    # 确保两条染色体的长度相同
    if len(chromosome1) != len(chromosome2):
        raise ValueError("两条染色体的长度必须相同")
    crossover_point = random.randint(1, len(chromosome1) - 1)
    # 将染色体1分为A和B，染色体2分为C和D
    A = chromosome1[:crossover_point]
    B = chromosome1[crossover_point:]
    C = chromosome2[:crossover_point]
    D = chromosome2[crossover_point:]

    return A, B, C, D#

#GPU交叉函数（选择一个点，交叉两条染色体）
def gpu_crossover(chromosome1, chromosome2):
    A, B, C, D = crossover(chromosome1, chromosome2)
    # 将 A 和 D 拼接
    child_gen1 = np.concatenate((A, D), axis=0)

    # 将 B 和 C 拼接
    child_gen2 = np.concatenate((B, C), axis=0)
    return child_gen2, child_gen1

#修复交叉后存在Transformer层没有分配给GPU的问题
def repair(matrix, j):
    """
    确保矩阵中每个整数 1 到 j 至少出现一次。
    如果有缺失的值，则随机选择一个至少出现两次的值，将其替换为缺失的值。
    如果没有可替换的值，则随机选择一个值为 0 的位置进行替换。

    参数：
        matrix (list or numpy.ndarray): n*K 维的二维矩阵。
        j (int): 检查的范围是 1 到 j 的整数。

    返回：
        numpy.ndarray: 调整后的矩阵。
    """
    # 将矩阵转换为 numpy 数组（如果尚未是 numpy 数组）
    if not isinstance(matrix, np.ndarray):
        matrix = np.array(matrix)

    # 找出矩阵中出现的所有值
    present_values = set(matrix.flatten())

    # 找出缺失的值
    missing_values = [value for value in range(1, j + 1) if value not in present_values]

    # 如果没有缺失的值，直接返回矩阵
    if missing_values:
        for missing_value in missing_values:
            # 找出矩阵中出现次数大于 1 的值
            unique, counts = np.unique(matrix, return_counts=True)
            value_counts = dict(zip(unique, counts))
            candidates = [val for val, count in value_counts.items() if count > 1 and val > 0]

            if candidates:
                # 随机选择一个候选值进行替换
                value_to_replace = random.choice(candidates)

                # 在矩阵中找到该值的位置
                positions = np.argwhere(matrix == value_to_replace)

                # 随机选择一个位置进行替换
                replace_position = random.choice(positions)
                matrix[replace_position[0], replace_position[1]] = missing_value
            else:
                # 找出矩阵中值为 0 的位置
                zero_positions = np.argwhere(matrix == 0)

                if zero_positions.size > 0:
                    # 随机选择一个零值位置
                    replace_position = random.choice(zero_positions)
                    matrix[replace_position[0], replace_position[1]] = missing_value
                else:
                    raise ValueError("Matrix has no duplicate values or zeros to replace.")



    # 统计每列出现了哪些数字并进行升序排列
    for col in range(matrix.shape[1]):
        # 获取该列的非零值
        column_values = matrix[:, col][matrix[:, col] != 0]

        # 获取该列的唯一值并升序排序
        sorted_values = np.sort(np.unique(column_values))

        # 将升序排序后的值放回该列，余下的位置补充为0
        matrix[:, col] = 0  # 清空该列
        matrix[:len(sorted_values), col] = sorted_values  # 将排序后的值放回

    return matrix

#变异中间函数：随机交换一条染色体两点的值
def mutation(vector):
    """
    随机选择两个点交换其位置，模拟变异操作。

    Args:
        vector: list or numpy.ndarray, 输入的1D向量（长度为2*i*j）

    Returns:
        list or numpy.ndarray: 变异后的向量
    """
    # 确保输入为可变的list类型
    if isinstance(vector, np.ndarray):
        vector = vector.tolist()

    # 随机选择两个不同的索引
    idx1, idx2 = random.sample(range(len(vector)), 2)

    # 交换位置
    vector[idx1], vector[idx2] = vector[idx2], vector[idx1]

    return vector

#变异中间函数：块交叉，交叉两个矩阵，将两个矩阵按列切割成小块，等概率选择每一块形成新的矩阵
def gpu_block_crossover(parent1, parent2):
    """
    GPU 杂交函数（块交换法）：通过两个父代矩阵生成一个子代矩阵。

    Args:
        parent1: numpy.ndarray, 父代1 (n x K)
        parent2: numpy.ndarray, 父代2 (n x K)

    Returns:
        numpy.ndarray: 子代矩阵 (n x K)
    """
    n, k = parent1.shape
    child = np.zeros((n, k), dtype=int)
    block_size = 2  # 2 列为一个块

    # 遍历每个块，按 2 列分块
    for j in range(0, k, block_size):
        if random.random() < 0.5:  # 随机选择父代1或父代2
            child[:, j:j + block_size] = parent1[:, j:j + block_size]
        else:
            child[:, j:j + block_size] = parent2[:, j:j + block_size]

    return child


#生成初始的Transformer matrix
def generate_matrix(n, k, j):
    # 随机生成 n x k 的矩阵，值范围为 1 到 j
    matrix = np.random.randint(1, j+1, size=(n, k))

    # 每列的非零值升序排列
    for col in range(k):
        non_zero_values = matrix[:, col][matrix[:, col] > 0]  # 获取非零值
        sorted_non_zero = np.sort(non_zero_values)  # 对非零值排序
        matrix[:, col][matrix[:, col] > 0] = sorted_non_zero  # 替换为排序后的值
    repair(matrix, j) #修复交叉后存在Transformer层没有分配给GPU的问题
    return matrix

# 初始化Task Vector
def generate_initial_task_sequence(i, j):
    """
    task vector
    生成长度为 2*i*j 的一维向量，其中数字 1 到 i 各出现 2*j 次。

    参数：
        i (int): 数字的范围为 1 到 i。
        j (int): 每个数字出现的次数为 2*j。

    返回：
        numpy.ndarray: 生成的一维向量。
    """
    # 每个数字出现 2*j 次
    sequence = np.repeat(np.arange(1, i + 1), 2 * j)
    # 随机打乱顺序
    np.random.shuffle(sequence)
    return sequence


# 初始化GPU Vector
def generate_machine_sequence(i, j, k):
    """
    GPU vector
    生成长度为 2*i*j 的机器序列，数字范围为 1 到 k。

    参数：
        i (int): 任务编号的范围为 1 到 i。
        j (int): 每个任务的重复次数。
        k (int): 机器编号的范围为 1 到 k。

    返回：
        numpy.ndarray: 生成的机器序列，一维向量。
    """
    # 确定序列长度
    sequence_length = 2 * i * j

    # 随机生成范围为 1 到 k 的数字序列
    machine_sequence = np.random.randint(1, k + 1, size=sequence_length)

    return machine_sequence

# 初始化种群
def generate_population(population_size, n, k, i, j):
    population = []
    for _ in range(population_size):
        matrix = generate_matrix(n, k, j)  # 初始化Transformer matrix
        vector1 = generate_initial_task_sequence(i, j)  # 初始化task vector
        vector2 = generate_machine_sequence(i, j, k)  # 初始化GPU vector
        population.append((matrix, vector1, vector2))
    return population


# 精英选择：选择适应度分数最小的个体
def select_elite(population, scores, elite_fraction=0.2):
    """
    选择适应度分数最高的个体作为精英。

    参数：
        population (list): 种群，包含多个个体。
        scores (list): 每个个体的适应度分数。
        elite_fraction (float): 精英比例，默认为 0.2。

    返回：
        list: 精英个体列表。
    """
    elite_size = int(len(population) * elite_fraction)  # 计算精英个体数量
    elite_indices = np.argsort(scores)[-elite_size:]  # 选择分数最高的个体
    elite_population = [population[i] for i in elite_indices]  # 获取精英个体
    return elite_population

def select_elite_time(population, scores, elite_fraction=0.2):
    """
    选择适应度分数最高的个体作为精英。

    参数：
        population (list): 种群，包含多个个体。
        scores (list): 每个个体的适应度分数。
        elite_fraction (float): 精英比例，默认为 0.2。

    返回：
        list: 精英个体列表。
    """
    elite_size = int(len(population) * elite_fraction)  # 计算精英个体数量
    elite_indices = np.argsort(scores)[elite_size:]  # 选择分数最低的个体
    elite_population = [population[i] for i in elite_indices]  # 获取精英个体
    return elite_population

# 交叉：根据交叉率1对向量1进行交叉，交叉率2对向量2进行交叉
def genetic_crossover(population, crossover_rate1, crossover_rate2, i, j, k):
    new_population = []
    for _ in range(len(population) // 2):
        parent1, parent2 = random.sample(population, 2)
        matrix1, vector1_1, vector2_1 = parent1
        matrix2, vector1_2, vector2_2 = parent2

        # 交叉向量1
        if random.random() < crossover_rate1:
            new_vector1_1, new_vector1_2 = generate_new_chromosome(vector1_1, vector1_2)
        else:
            new_vector1_1, new_vector1_2 = vector1_1, vector1_2

        # 交叉向量2
        if random.random() < crossover_rate2:
            new_vector2_1, new_vector2_2 = gpu_crossover(vector2_1, vector2_2)
        else:
            new_vector2_1, new_vector2_2 = vector2_1, vector2_2

        new_population.append((matrix1, new_vector1_1, new_vector2_1))
        new_population.append((matrix2, new_vector1_2, new_vector2_2))

    return new_population


# 变异-总函数：对向量1和向量2进行变异，矩阵部分进行变异
def mutate(population, mutation_rate, i, j, k):
    for i in range(len(population)):
        matrix, vector1, vector2 = population[i]

        # 变异向量1
        if random.random() < mutation_rate:
            vector1 = mutation(vector1)

        # 变异向量2
        if random.random() < mutation_rate:
            vector2 = mutation(vector2)

        # 变异矩阵
        if random.random() < mutation_rate:
            matrix = gpu_block_crossover(matrix, matrix)  # 假设变异采用块交换法
            matrix = repair(matrix,j)
        population[i] = (matrix, vector1, vector2)

    return population


# 更新种群：保留精英个体并将交叉和变异后的个体加入
def update_population(elite_population, new_population, population_size):
    combined_population = elite_population + new_population
    return combined_population[:population_size]


def genetic_algorithm(n, i, j, k,  v, alpha, c_f,c_b, population_size, mutation_rate, crossover_rate1, crossover_rate2,
                      max_generations, iterations, standard, uShape, uScale, theta, epsilon):
    population = generate_population(population_size, n, k, i, j)

    # 用于记录每代的适应度分数
    best_scores = []
    best_utilization_ratios = []

    # 用于记录每代最优个体的任务矩阵
    best_individuals = []

    for generation in range(max_generations):
        print(f"Generation {generation + 1}/{max_generations}")

        # 适应度评估
        scores, utilizations = evaluate_population(population, i, j, k, n, v, alpha, c_f, c_b, iterations, standard,
                                                   uShape, uScale, theta, epsilon)

        # 记录当前代的最优适应度分数
        best_score = max(scores)
        best_scores.append(best_score)  # 适应度评估分数越小越好
        best_index = np.argmax(scores)
        if best_score == 1:
            ratios = []
            for a in range(population_size):
                if scores[a] == 1:
                    ratios.append(utilizations[a])
            best_utilization_ratios.append(np.average(ratios))
        else:
            best_utilization_ratios.append(utilizations[best_index])

        # 记录当前代的最优个体及其任务矩阵
        best_individual_index = np.argmax(scores)
        best_individual = population[best_individual_index]
        best_individuals.append(best_individual)

        print(f"Generation {generation + 1}: Best Fitness Score = {best_scores[-1]}")
        print(f"Generation {generation + 1}: Utilization Ratio = {best_utilization_ratios[-1]}")


        # 精英选择
        elite_population = select_elite(population, scores)

        # 交叉
        new_population = genetic_crossover(population, crossover_rate1, crossover_rate2, i, j, k)

        # 变异
        new_population = mutate(new_population, mutation_rate, i, j, k)

        # 更新种群
        population = update_population(elite_population, new_population, population_size)

    # 获取最终的最优个体
    final_scores, utilization_ratios = evaluate_population(population, i, j, k, n, v, alpha, c_f,c_b, iterations, standard, uShape, uScale, theta, epsilon)
    final_score = np.max(final_scores)
    best_individual_index = np.argmax(final_scores)
    # 输出任务矩阵
    best_individual = population[best_individual_index]
    input_matrix1 = best_individual[0]
    input_matrix21 = best_individual[1]
    input_matrix22 = best_individual[2]
    utilization_ratio = utilization_ratios[best_individual_index]
    return final_score, utilization_ratio, input_matrix1, input_matrix21, input_matrix22

def genetic_algorithm_time(n, i, j, k, v, alpha, c_f, c_b, population_size, mutation_rate, crossover_rate1, crossover_rate2,
                      max_generations, uShape, uScale, theta, epsilon):

    # 根据uShape, uScale, theta, epsilon得到f和b (取分布的均值)
    f = [[0 for kk in range(k)] for jj in range(j)]
    for jj in range(j):
        for kk in range(k):
            fjk = uScale[kk] / (uShape[kk] - 1) * theta + epsilon  # mean of inverse Gamma = uScale / (uShape - 1)
            f[jj][kk] = fjk
    b = copy.copy(f)

    population = generate_population(population_size, n, k, i, j)

    # 用于记录每代的适应度分数
    best_scores = []
    best_utilization_ratios = []

    # 用于记录每代最优个体的任务矩阵
    best_individuals = []

    for generation in range(max_generations):
        print(f"Generation {generation + 1}/{max_generations}")

        # 适应度评估
        scores, utilizations = evaluate_population_time(population, i, j, k, n, v, alpha, c_f, c_b, f, b)

        # 记录当前代的最优适应度分数
        best_score = min(scores)
        best_scores.append(best_score)  # 适应度评估分数越小越好
        best_index = np.argmin(scores)
        best_utilization_ratios.append(utilizations[best_index])

        # 记录当前代的最优个体及其任务矩阵
        best_individual_index = np.argmin(scores)
        best_individual = population[best_individual_index]
        best_individuals.append(best_individual)

        print(f"Generation {generation + 1}: Best Fitness Score = {best_scores[-1]}")
        print(f"Generation {generation + 1}: Utilization Ratio = {best_utilization_ratios[-1]}")


        # 精英选择
        elite_population = select_elite_time(population, scores)

        # 交叉
        new_population = genetic_crossover(population, crossover_rate1, crossover_rate2, i, j, k)

        # 变异
        new_population = mutate(new_population, mutation_rate, i, j, k)

        # 更新种群
        population = update_population(elite_population, new_population, population_size)

    # 获取最终的最优个体
    final_scores, utilization_ratios = evaluate_population_time(population, i, j, k, n, v, alpha, c_f, c_b, f, b)
    final_score = np.min(final_scores)
    best_individual_index = np.argmin(final_scores)
    # 输出任务矩阵
    best_individual = population[best_individual_index]
    input_matrix1 = best_individual[0]
    input_matrix21 = best_individual[1]
    input_matrix22 = best_individual[2]
    utilization_ratio = utilization_ratios[best_individual_index]
    return final_score, utilization_ratio, input_matrix1, input_matrix21, input_matrix22