20240321 整体思路 新

1. 网络可靠性优化 (network reliability optimization)
  1. 网络可靠性优化常见指标
  2. 模型范式
2. 数据中心网络（2023.12.6：论文思路整理，林科PPT）
  1. 传统数据中心网络常见的架构
  2. 数据中心网络的常见指标与所面临的可靠性问题
3. 大模型计算集群（yanzi，ppt论文，新的）
  1. 大模型越来越火：
    Transformer-based language models [13, 27, 33–35, 42, 46] in Natural Language Processing (NLP) have driven rapid progress in recent years as computation at scale has become more available and 
    datasets have become larger.（来自*）
    1. 42：2017-Attention Is All You Need ：GOOGLE提出transformer模型vaswani2017attention 
    2. 13：BERT：Pre-training of Deep Bidirectional Transformers for Language Understanding：GOOGLE的BERT模型devlin2018bert 
    3. Language Models are Few-Shot Learners：Open AI GPT-3brown2020language 
    4. Swin Transformer Hierarchical Vision Transformer using Shifted Windows：微软liu2021swin 
    5. Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer：GOOGLE raffel2020exploring 
  2. 除了大模型的模型架构……，其中另外一个重要的research 方向是大模型训练集群基于并行策略的调度方法以及相关的性能保障。
  3. Parallelization is a key strategy on training large models at scale，而并行策略恰好在可靠性上很重要，现在并行策略研究都集中在性能上，几乎没有人关注可靠性上。我们的文章也将从并行策略入手分析大模型计算集群的可靠性 。
  4. 并行策略（不带优化）
    1. 数据并行：
      1. In DP, model parameters are replicated on each device.（来自&）
        1. 鼻祖：A bridging model for parallel computation-1990 valiant1990bridging 
        2. PyTorch Distributed Experiences on Accelerating Data Parallel Training. li2020pytorch 
    2. 张量并行：
      1.  recent work [39, 40] has shown how tensor (intra-layer) model parallelism, where matrix multiplications within each transformer layer are split over multiple GPUs, can be used to overcome these limitations.（来自*）
        1. 39：NeurIPS-2018-mesh-tensorflow-deep-learning-for-supercomputers-Paper shazeer2018mesh 
        2. 40：Megatron-LM Training Multi-Billion Parameter Language Models Using Model Parallelism shoeybi2019megatron 
    3. 管道并行：
      1. Pipeline model parallelism [14, 20, 23, 29, 30, 45] is another technique to support the training of large models, where layers of a model are striped over multiple GPUs. A batch is split into smaller microbatches, and execution is pipelined across these microbatches.（来自*）
        1. 29：PipeDream Fast and Efficient Pipeline Parallel DNN Training harlap2018pipedream 
    4. ZeRO:（不写）
      ZeRO: Memory Optimizations Toward Training Trillion Parameter Models 
      1.  We develop ZeRO— Zero Redundancy Optimizer — to optimize memory efficiency on both while obtaining high compute and communication efficienc
    5. 专家并行（不写，但有优化）
  5. 并行策略（性能相关分析）
    1. 数据并行
      1. Fault-Tolerant Hybrid-Parallel Training at Scale with Reliable and Efficient In-memory Checkpointing: we identify the frequency of existing checkpoint-based fault-tolerance being significantly limited by the storage I/O overheads, which results in hefty re-training costs on restarting from the nearest checkpoint. wang2023reliable 优化容错率，没有模型
    2. 张量并行
      1. TAP: Accelerating Large-Scale DNN Training Through Tensor Automatic Parallelisation:we present a model parallelism framework TAP that automatically searches for the best data and tensor parallel schedules. 没有模型shi2023tap 
      2. A Linear Algebraic Approach to Model Parallelism in Deep Learning：We propose a linear-algebraic approach to model parallelism in deep learning, which allows parallel distribution of any tensor in the DNN.全是线性代数的模型，不算建模hewett2020linear 
    3. 管道并行
      1. GPipe Easy Scaling with Micro-Batch Pipeline Parallelism：  huang2019gpipe 为了解决对高效和独立于任务的模型并行性的需求，提高灵活性，没有模型
      2. 23：MLSys-2021-pipelined-backpropagation-at-scale-training-large-models-without-batches-Paper kosson2021pipelined 提出小批量、细粒度的管道反向传播，提高效率，不算有模型
    4. PTD-P
      1. *Efficient Large-Scale Language Model Training on GPU Clusters：introduction部分有介绍前三种并行，组合数据并行、管道张量并行，narayanan2021efficient ：将张量、管道和数据并行度扩展到数千个gpu。我们提出了一种新的交错流水线调度PTD-P，与现有的方法相比，其内存占用可以提高10+%的吞吐量(提出PTD-P (inter-node pipeline parallelism, intra-node tensor parallelism, and data parallelism)，improves efficiency at small batch sizes )
  6. 并行策略带性能优化（架构调度相关）
  7. 可靠性：进一步的，考虑时延特征来完成性能优化这部分文献大都是基于动态规划来完成优化求解的，也有少数基于整数规划（每个一句话总结）
    1. Pipeline parallel：14:DAPPLE: A pipelined data parallel approch for training large models（结合数据并行和管道并行）fan2021dapple 动态规划减小延时
    2. Pipeline parallel：29：PipeDream Fast and Efficient Pipeline Parallel DNN Training harlap2018pipedream 对DNN延时建模了，算法上的分析，也是动态规划
    3. Pipeline parallel：NeurIPS-2020-efficient-algorithms-for-device-placement-of-dnn-graph-operators-Paper，tarnawski2020efficient ， 本文给出了DNN工作负载的模型划分问题的算法。它们以推理和训练为目标，并优化最小化延迟或最大化吞吐量的目标。用动态规划、整数规划
    4. Pipeline parallel：Pipe-Torch_Pipeline-Based_Distributed_Deep_Learning_in_a_GPU_Cluster_with_Heterogeneous_Networking，zhan2019pipe，管道火炬算法
4. 其他不知道有没有用的
  1. 2017-Bringing HPC Techniques to Deep Learning：在深度学习中引入high-performance computing (HPC)，提高compute power
  2. 2017-Bringing HPC Techniques to Deep Learning：在深度学习中引入high-performance computing (HPC)，提高compute power
  3. 2015-GeePS Scalable deep learning on distributed GPUs with a GPU-specialized parameter server：介绍支持大规模深度学习的GeePS
  4. Parameter Server for Distributed Machine Learning：介绍 a parameter server framework to solve distributed machine learning problems
  5. 2021-An Image Is Worth 16×16 WordsTransformers for Image Recognition at Scale：计算机视觉的贡献

