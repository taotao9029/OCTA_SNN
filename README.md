# Stroke GPU Classification Project

本项目用于从小鼠眼底/OCTA 视频中提取血管微循环事件特征，并训练/推理脑卒中二分类模型。当前整理后的主线流程是：视频取清晰帧 -> 血管分割 -> 视频事件流生成 -> 血管区域事件过滤 -> SNN网络 。

## 当前目录结构

```
.
├── README.md                   #
├── seg.py                      # 眼底血管分割
├── event_filter.py             # 血管区域事件流过滤
├── Video2Events.py             # 视频提取事件流
├── train_group_out.py          # 训练group_out整体流程
├── train_snn.py                # 训练五折整体流程
├── requirements.txt            # 运行所需要的环境
├── data/                       # 当前数据存放位置
├── log/                        # 当前训练日志存放位置
├── output                      # 当前结果存放位置
└── 
```


## 训练整体流程

当前血管视频转换成事件流：

```bash
python Video2Events.py 
```

脚本会执行：

将整段视频转换为临时事件流 。


当前血管分割区域：

```bash
python seg.py
```

脚本会执行：

得到每一个视频的血管分割区域 。


当前视频流的血管区域过滤：

```bash
python event_filter.py
```

脚本会执行：

过滤得到每一个事件流的的血管区域 。


SNN的五折整体流程：

```bash
python train_snn.py
```

脚本会执行：

五折训练的每个epoch的结果以及训练过程 。


RF的group_out整体流程：

```bash
python train_group_out.py
```

脚本会执行：

snn的group_out的训练结果以及中间结果 。

## 主要依赖

项目有独立的 `requirements.txt`，请执行 pip install -r requirements.txt 安装当前环境：


## 维护建议

1. 后续新增实验脚本建议放入 `experiments/`，不要继续堆在项目根目录。
2. 生成图片、临时 CSV、解释结果建议统一输出到 `outputs/`。

