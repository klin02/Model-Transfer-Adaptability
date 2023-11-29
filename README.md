# 改动说明

## update: 2023/06/05

+ 解决之前ResNet系列和MobileNetV2的模型结构错误
+ 将准确性和安全性框架融合入ALL文件夹，并添加部署说明（安全性评估尚未添加）
+ 改动matlab脚本适配当前框架
+ 针对准确性，单个模型拟合较好，整体较差，需要后期数据处理。

## update: 2023/05/29

+ GDFQ：结合之前框架，训练了所有模型的生成器。后续将进一步引入评估和决策边界样本增强。

## update2: 2023/05/26

+ 添加了cifar100数据集支持，详见ALL-cifar100。原先ALL文件夹重命名为ALL-cifar10
+ 在ALL-cifar10中罗列了多种训练策略的结果，并进行简要分析。针对cifar100，同样进行了多种尝试，展示了全精度准确度和拟合效果均较好的训练策略和结果。
+ 对poly2添加新的分段拟合matlab脚本，分界点等可自行指定。

## update1: 2023/05/26

+ 添加针对BasicBlock、BottleNeck、InvertedResidual的Makelayer方法和Relu6的框架。支持了当前所有不含LSTM的网络
+ 新的全精度模型训练方案，详见ALL/README.md
+ 更改了flop_ratio将relu层参数量加到conv层处的笔误。
+ conv层新增groups参数的cfg
+ ALL/README.md部分更换为集合所有当前模型，采用fakefreeze方法，且weight：bias由0.5:0.5变为1：1的拟合效果，并包含对计算量的log和cuberoot加权。之前的模型拟合结果和尝试分析移动至`fit_bkp.md`

## update: 2023/04/26

+ 将新框架应用于所有模型（AlexNet、AlexNet_BN、VGG_16、VGG_19、Inception_BN），并采用不同方式对单个模型拟合，以及对所有模型一同拟合。详见ALL

+ 更新Matlab脚本，支持单模型/多模型拟合，支持选择不同多项式次数的拟合模型。添加约束保证拟合曲线单调不下降

+ ALL的README所提到的POT点不协调的问题

  + 关于精度的讨论可见Inception_BN部分的README
  + 关于原因及方案讨论可见ALL的README

  

## update: 2023/04/22

+ 添加Inception BN模型，对框架改动如下
  + 使用cfg_table进行模型快速部署和量化（可应用于其他模型），cfg_table提供整体forward平面结构，包括inc模块。inc模块则由inc_ch_table和inc_cfg_table进行部署量化。其规则可详见文件
  + cfg_table每项对应一个可进行量化融合的模块，如相邻conv bn relu可融合，在cfg_table中表现为['C','BR',...]。从而可以更方便的从该表进行量化和flops/param权重提取
  + 更改fold_ratio方法，以支持cfg_table的同项融合，多考虑了relu层，摆脱原先临近依赖的限制。方案：读取到conv时，获取相同前后缀的层，并相加
  + 更改module，允许量化层传入层的bias为none。原fold_bn已经考虑了，只需要改动freeze即可。
    + 对于Conv，freeze前后都无bias。
    + 对于ConvBN，freeze前后有bias。forward使用临时值，inference使用固定值（固定到相应conv_module）。
  + 由于允许conv.bias=None,相应改变全精度模型fold_bn方法，从而保证量化前后可比参数相同。改写方式同量化层
  + 更改js_div计算方法，一个层如果同时有多个参数，例如weight和bias，应该总共加起来权重为1。当前直接简单取平均（即js除以该层参数量），后续考虑加权。PS: Inception_BN中，外层conv层有bias，Inception模块内由于后接bn层，bias为false
    + 由于named_parameters迭代器长度不固定，需要先将排成固定列表再处理，从而获得同一层参数数，改动见ptq.py。对全精度模型做此操作即可
  + Inception_BN框架中的model_utils方法可以通过调整反量化位置来进行bug的确定。经过当前实验，可以初步判断精度问题出现在inception结构中，具体信息见Inception_BN相关部分。经过排查，量化框架本身并未出现问题，问题可能在于该模型参数分布与POT量化点分布的不适配。

## update: 2023/04/17

+ 指定了新的梯度学习率方案，对全精度模型重新训练以达到更高的acc，并重新进行ptq和fit

  

## update: 2023/04/16

+ 添加了matlab的拟合及绘图脚本，支持模型分类标记，且曲线拟合相比cftool更加平滑
+ ptq.py中计算js_param笔误，应由flop_ratio改为par_ratio。否则flops和param拟合没有区别
+ module.py中bias_qmax方法，应当为float类型传参num_bits为16，e_bits为7.
  + 这里主要关注e_bits，拟合离群点主要为FLOAT_7_E5 / FLOAT_8_E5 / FLOAT_8_E6，其表现为bias两极分布，与之前int量化bias溢出的问题现象相似。
  + 原先指定e_bits为5，由于bias的scale为input和weight的scale乘积，bias量化范围应当大致为x和weight量化范围的平方倍。目前代码支持的最高x和weight量化范围大致为 2的2的6次方 ，因此bias范围应当近似取到 2的2的7次方，即将e_bits指定为7
  + 改动之后，离群点消失，拟合效果显著提高

