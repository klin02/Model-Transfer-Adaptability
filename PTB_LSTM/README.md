# PTB_LSTM量化说明

## 全精度模型

1. 数据集及预处理：该模型使用PTB数据集，里面包含大量的英文句子，与cifar10数据集以图像为样本不同，该数据集将句子切割作为样本，并构建词表将单词转化为数字索引，便于网络处理。
2. 模型目标及评价指标：预测某个单词/短句后面出现的单词。由于每个样本在相同单词/短句的后续单词未必一致，不能简单的使用acc进行评判。语言模型使用困惑度ppl作为评价指标，值越小表示后续单词的可能性越集中，表征模型性能约好。数学上ppl表现为loss的指数。
3. 模型结构的额外要求：由于PTB LSTM模型中的embedding层具有稀疏矩阵参数，且其中很多值为0，导致Adam优化器学习率调整机制失效。pytorch官方embedding类处说明，当前可支持的CUDA优化器只有SGD、sparseAdam（专为稀疏矩阵准备的Adam优化器）。另外，在训练结束，进行推理时，一般会对lstm层使用flatten_parameters（）方法，将其参数拉平为一维，方便并行计算。
4. 模型参数量和计算量的获取：由于PTB LSTM的输入并非floattensor，而是int类型的longTensor和hidden。之前所使用的ptflops需要使用其特殊输入构造API。在get_model_complexity_info方法中额外传入一个返回值为dict的输入构造器，从而获得特定的输入。



# ptq部分

### 量化层说明

#### embedding

+ 该层接受输入的每个元素均为int类型的索引，根据索引将embedding矩阵（即weight参数）的对应行取出，放置到对应位置。因此输入不需要进行量化，只需要对weight进行量化即可。
+ 注意到，输出的每个元素都来自于weight，因此只需要weight进行了量化。同时考虑到输出可能只包括了embedding矩阵的部分元素，设置了qo对输出进行了统计。并通过`self.M.data=self.qw.scale/self.qo.scale).data`进行rescale。



#### LSTM

+ 全精度层允许多层，为量化方便，我们将对多层LSTM进行拆分。

+ 该层接受输入x和隐层hidden。其中hidden是可选输入，可unpack为上一步隐层h和上一步状态c。如果未指定或输入为None时，nn.LSTM会自动初始化值为0的隐层作为输入。h,c形状为(nlayer,batch_sz, hidden_size)。考虑到第一层可能始终不接受hidden输入，或在第一个batch输入为None，后续输入为非零值。量化时指定参数has_hidden表明是否需要对输入hidden作为统计。

  当has_hidden为true时，表示需要统计，设置相应的统计值qih和qic。考虑到第一个batch输入的hidden可能为None，只在hidden不为None时进行qih和qic的更新以及hidden的反量化，防止因为scale为0导致量化值出现nan。

  当has_hidden为false时，表示该层始终不接受非零的hidden，无需进行统计，并在相应方法中进行检查。

+ 对于单层的lstm_module，参数主要有weight_ih_l0，weight_hh_l0，bias_ih_l0，bias_hh_l0。其中后缀为ih的表示输入x和输出hidden之间的关系，后缀为hh表示输入hidden和输出hidden之间的关系。简便起见，我们当前不对每个矩阵进行进一步分拆。

  另外，由于该层运算并非简单的加减、乘法和卷积运算，不能很方便的进行rescale。当前仍然使用伪量化来进行推理。后续的rescale可以考虑用一个相近的线性函数来模拟。

+ 在quantize_forward时，与其他层直接调用toch.nn.Functional不同，仍然使用nn.LSTM进行。这是因为直接调用函数还涉及到flatten_weight等操作，为了简化调用逻辑，新建一个临时的LSTM层，并修改其参数与量化值一致来进行运算。