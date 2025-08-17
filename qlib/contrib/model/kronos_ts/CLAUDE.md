<!--
 * @Author: Hugo
 * @Date: 2025-08-15 15:19:36
 * @LastEditors: shen.lan123@gmail.com
 * @LastEditTime: 2025-08-17 14:12:39
 * @Description: 
-->
# CLAUDE.md

本文档为Claude Code (claude.ai/code)在本代码库中工作时提供指导。必须使用中文对话。

## 代码库概述

本代码库包含了基于Kronos架构的时间序列预测模型KronosTS的实现。该模型专为金融时间序列预测设计，并与Qlib框架集成。

### 架构

主要组件包括：

1. **KronosTS** (`core.py`): 与Qlib模型接口集成的主模型包装器
2. **KronosTokenizer** (`kronos_model/kronos.py`): 使用二进制球面量化进行数据压缩的分词器
3. **Kronos** (`kronos_model/kronos.py`): 基于Transformer的时间序列预测主模型
4. **BSQuantizer** (`kronos_model/module.py`): 二进制球面量化的实现
5. **支持模块** (`kronos_model/module.py`): 包含Transformer块、注意力机制、嵌入层和其他组件
6. **案例模块**(`prediction_example.py`):更好的理解传入的数据结构和模型的交互
7. **TSDatasetH**(路径`../data/loader`):数据预处理的算法

### 关键特性

- 使用二进制球面量化进行高效的数据表示
- 支持分层标记化，具有s1_bits和s2_bits参数
- 基于Transformer的架构，带有旋转位置嵌入
- 专为金融时间序列预测设计（OHLCV数据）
- 与Qlib框架集成用于量化金融

## 开发指南

### 代码结构

- `core.py`: 与Qlib接口的主KronosTS模型实现
- `prediction_example.py`: 用于演示模型预测的示例代码
- `kronos_model/`: 包含核心Kronos模型组件的目录
  - `kronos.py`: 主模型和分词器实现
  - `module.py`: 支持神经网络模块
  - `__init__.py`: 模块导出和模型字典

### 依赖项

关键依赖包括：
- torch
- pandas
- numpy
- huggingface_hub
- einops

## 常见开发任务

### 使用模型

KronosTS模型初始化参数包括：
- `max_context`: 模型的最大上下文长度
- `clip`: 输入归一化的裁剪值
- `model_name`: 预训练权重的Hugging Face模型名称
- `tokenizer_name`: Hugging Face分词器名称
- `pred_len`: 预测长度
- `temperature`, `top_k`, `top_p`: 采样参数

### 模型组件

修改模型时，需要理解数据流：
1. 输入数据使用KronosTokenizer进行分词
2. 标记由Kronos Transformer模型处理
3. 通过自回归推理生成预测
4. 结果解码回原始空间


### 优化任务

感觉模型预测部分写的并不好。并没有使用gpu的并行。仅能处理单个股票的单个截面的时序数据，比如特征为N，需要的时序为X，则仅能处理(X,N)形状的数据。且需要x_timestamp和y_timestamp。如果需要滚动预测股票时序则需要处理(R,X,N)处理。能否对其进行深度改造使其可以完成这种预期的操作？感觉难点在于x_timestamp和y_timestamp的处理？可以看看prediction_example来帮助你理解每个参数的数据结构。think harder
