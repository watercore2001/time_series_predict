# time_series_predict
Time  series prediction for water

## Preprocess
- Station: 在原始数据上 -1, 使其取值范围为 [0, 147) 的整数
- Watershed: 在原始数据上 -1, 使其取值范围为 [0, 9) 的整数
- WeekOfYear: 表示采集日期在一年中所处的第几周，其取值范围为 [0, 53) 的整数
  - 原始数据为采集日期，经过数据分析，我们的数据并没有覆盖一年的所有天数。
  - 而且考虑到原始数据就是周数据，我们决定仅使用 WeekOfYear 代表时间信息作为输入
- 其他浮点型数据： 进行标准归一化。 x=(x-mean)/std，归一化后平均值为0, 标准差为1.

## 数据集
数据集的最小单元为站点(Station), 对于每个站点, 遍历每一个时间点 t, 得到一个样本：
- 将 [t, t+L1) 时间段作为模型的输入
- 将 [t+L1, t+L1+L2) 时间段作为模型的输出
- 相当于使用前 L1 个值预测后面的 L2 个值
- 我们选择L1=32：有些站点的时间序列长度只有50多，设太大了就没法针对这些站点进行训练了
- 我们选择L2=1：
  - 原因1是我们后续实际使用模型的时候也只预测一个值
  - 原因2是我们这个数据集规律不太明显，预测L2个值太难了

需要重点说明的是，按照我们的模型设计，我们的模型具有较强的适用性。
**对于同一个模型**，不论L1是多少，不论L2是多少，模型都是能运行。
至于怎么实现的，具体看模型结构的部分


每个样本包括输入(x)和真值(y)两部分，模型接收x作为输入，输出预测值y_hat, 通过比较 y_hat 和 y 的差别，让模型进步：

以输入序列L1,输出序列L2举例，那么一个样本的形状如下所示:

```
**这里我们同样用输入序列L1,输出序列L2为例子**

输入 x 包括了很多信息:
- feature: 形状(L1,4),代表 L1 个时间点里面的 4 个预测值.
- x_week_of_years: 形状(L1), 代表 L1 个时间点中每个时间点的周数
- y_week_of_years: 形状(L2), 代表要预测的时间点的周数
- station_id: 形状(1), 代表所在的站点 id
- watershed_id: 形状(1), 代表所在的水域 id
- Lat_Lng: 形状(1), 代表所在的经纬度

真值 y
- feature：形状(L2,4), 代表真值

预测值y_hat：形状和真值完全相同
```


将一个站点的所有样本称为**站点数据集**，并划分为训练集和验证集。

将每个流域(Watershed)中的所有**站点数据集**整合起来称为**流域数据集**。

将所有**流域数据集**整合起来称为**整体数据集**。




## 模型结构

具体模型: 1D_CNN, Transformer

### Embedding 模块

```
**这里我们同样用输入序列L1,输出序列L2为例子**

输入 x 包括了很多信息:
- feature: 形状(L1,4),代表 L1 个时间点里面的 4 个预测值.
- x_week_of_years: 形状(L1), 代表 L1 个时间点中每个时间点的周数
- y_week_of_years: 形状(L2), 代表要预测的时间点的周数
- station_id: 形状(1), 代表所在的站点 id
- watershed_id: 形状(1), 代表所在的水域 id
- Lat_Lng: 形状(1), 代表所在的经纬度

真值 y
- feature：形状(L2,4), 代表真值

预测值y_hat：形状和真值完全相同
```

如前所述，**在同一个时间点**，模型的输入都包含了很多信息。经纬度是两个浮点数，站点id和流域id和日期是整数..... 如何将这些信息融合起来? 

这就是Embedding模块做的事情。**我们给所有模型设计了相同的 Embedding 模块**。

Embedding 模块具体分为以下部分，其中d代表模型所使用的维度，d越大代表模型的参数越多。
- FeatureEmbed：一层1D卷积，输入形状(L1,4), 输出形状(L1,d)
- WeekEmbed: 该模块的形状为(53,d),53为所有可能的周数,为每个周都准备好一个对应的数据。输入(L1),按照索引取出对应的数据， 输出(L1,d)
- StationEmbed: 和WeekEmbed同理，该模块的形状为(147,d)，147为所有可能的站点数
- WatershedEmbed：和WeekEmbed同理，该模块的形状为(9,d)，9为所有可能的流域数
- LatLngEmbed：一层线性层，输入为2, 输出为d

将的所有信息 Embed 之后，他们都是维度为d的数据，怎么融合起来呢？我们这里采取**直接相加**的方式进行融合

**值得注意的是, 我们需要分别embed输入x和输出y两个部分**。

输入时间序列x是完整的，我们可以直接将其 embed 为 x_embed，形状为(L1, d)

但是模型看不到输出时间序列的四个参数，所以我们只能将四个参数的值都设为0, 然后将其 embed 为 y_embed，形状为(L2,d)

因此后续模型的任务就是如何从 x_embed, y_embed 中预测出四个参数



### 模型1： 1DCNN

在我们的设计中，一维卷积层并不会改变输入的形状，只会将其中的信息做一些混合。
例如，输入形状为(L, d), 输出形状还是 (L,d)


我们的设计就是简单地将 depth 个一维卷积层堆叠起来，没有使用其他的池化层这种东西。

模型首先将 x_embed 和 y_embed 堆叠起来，变成(L1+L2, d)

然后经过这些一维卷积层，输出还是(L1+L2,d)，但是我们认为其中的信息已经做了传递

最后我们取出后面 L2 个输出，得到(L2, d), 将其线性映射为(L2,4), 就是最终输出了。

可以看出不管 L1, L2 是多大，我们的模型都能能跑的。


### 模型2： Transformer

可以将 Transformer 可以理解为多个 Attention 层的堆叠。

一个 Attention 层在干什么事情呢，简单来说？
- 输入为 q, k, v 三个*序列*
- 他会进行信息融合
- 输出 output 一个值，output 的形状和 q 的形状相同

模型的前几个Attention层干的事情，将其称之为 Encoder:
- 输入为 x_embed, x_embed, x_embd 三个序列
- 输出为 out1
- 将 out1, out1, out1 三个序列输入给下一层 Attention
- ......
- 最终的输出记做 enc_out

模型的后几个Attention层干的事情，将其称之为 Decoder:
- 输入为 y_embed, enc_out, enc_out 三个序列
- 输出形状和 y_embed 相同，为 (L2,d)

然后将 (L2,d)线性映射为 (L2,4)就是最终预测结果了


## 模型选型流程

使用**整体数据集**，因为我们需要一个比较大的数据集来选拔模型（数据集大有说服力）。

不同模型在整体数据集的训练集上训练，计算其在验证集上的精度，精度指标包括：R2, MAE, MSE. 

R2精度指标严重和 batch size 相关，参考性不大。

### 超参数选择

模型参数量都不大, 都是 500K 左右，太大了容易过拟合。

1DCNN:
- depth: 8
- d: 128

Transformer
- encoder depth: 2
- decoder depth: 1
- d: 128

在 Embed 模块：并没有使用 Lat_Lng 这个信息，我觉得在使用了 station_id, 和 watershed_id 的基础上，经纬度意义不大

优化器：AdamW

损失函数：MSELOSS

学习率：1e-5

batch_size: 128

模型结果对比

| modeL | R2 | MAE | MSE |
|-------|-----|-----|-----|
| 1D_CNN  | 0.6471 | 0.250L2 | 0.1624 |
| Transformer| 0.6743 | 0.2316 | 0.1496 |



## 模型预测

选定 Transformer 模型，使用训练好的模型对每个站点进行预测，具体操作如下：
- 遍历站点的每一个时间点，使用该时间点的前 32，来预测当前时间点的值
- 因此只能预测第 32 个时间点之后的时间

模型的输出按照如下方式组织，每个 csv 中的前 32 行和真值是相同的

```anguLar2htmL
output
|--0watershed
    |-- 008station.csv
    |-- 009station.csv
    |-- 013station.csv
    ...
|--1watershed
|--2watershed

```



## 可选的补充实验

按照重要性排序:
- 每个流域单独训练一个模型，效果不好
- 辅助信息的重要性





