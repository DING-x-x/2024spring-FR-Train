# FR-train 公平鲁棒训练方法

2023学年下学期高级软件工程课程项目

## 项目介绍
本项目基于 2020 年发表于 ICML 上的论文：**FR-Train: A Mutual Information-Based Approach to Fair and Robust Training**

论文作者: Yuji Roh, Kangwook Lee, Steven Euijong Whang, and Changho Suh

论文链接：[FR-Train: A Mutual Information-Based Approach to Fair and Robust Training](https://proceedings.mlr.press/v119/roh20a/roh20a.pdf)

## 项目结构
```
fr-train
├─ main.py  项目运行入口
├─ Args.py  参数解析
├─ data_loader.py  数据加载
├─ model_arch.py  分类器与鉴别器结构
├─ training_process.py  模型训练过程定义  
├─ utils.py  工具类函数
├─ requirement.txt  环境配置文件
├─ datasets 数据集
│  ├─ public_dataset  公开数据集
│  │  ├─ adult  
│  │  └─ german
│  └─ synthetic_dataset  原论文合成数据集
└── README.md
```

## 环境配置
项目所需要的依赖
```shell
pip install -r requirement.txt
```

## 运行方法
运行不同数据集与不同数据类型的示例

论文原有合成数据集
```shell
python main.py --dataname synthetic --type clean  
python main.py --dataname synthetic --type poisoned
```
adult数据集
```shell
python main.py --dataname adult --type clean
python main.py --dataname adult --type poisoned
```

german数据集
```shell
python main.py --dataname german --type clean
python main.py --dataname german --type poisoned
```
