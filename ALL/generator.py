import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

class Generator(nn.Module):
	def __init__(self, options=None, teacher_weight=None, freeze=True):
		super(Generator, self).__init__()
		# 记录额外信息，帮助模型边界测试的辅助
		self.target_test_acc = None #训练目标在测试集上取得的acc
		self.target_gen_acc = None #伪数据对训练目标边界的拟合效果

		self.settings = options
		# 注意这里有embedding层，两个分别是词典大小和向量长度
		# 用于将标签映射为向量
		if teacher_weight==None:
			self.label_emb = nn.Embedding(self.settings.nClasses, self.settings.latent_dim)
		else:
			#当randemb为False时，要求latentdim与输出层输入通道一致
			self.label_emb = nn.Embedding.from_pretrained(teacher_weight, freeze=freeze)

		self.embed_normalizer = nn.BatchNorm1d(self.label_emb.weight.T.shape,affine=False,track_running_stats=False)

		if not self.settings.no_DM:
			self.fc_reducer = nn.Linear(in_features=self.label_emb.weight.shape[-1], out_features=self.settings.intermediate_dim)

			self.init_size = self.settings.img_size // 4
			self.l1 = nn.Sequential(nn.Linear(self.settings.intermediate_dim, 128 * self.init_size ** 2))
		else:
			self.init_size = self.settings.img_size // 4
			self.l1 = nn.Sequential(nn.Linear(self.settings.latent_dim, 128 * self.init_size ** 2))

		self.conv_blocks0 = nn.Sequential(
			nn.BatchNorm2d(128),
		)

		self.conv_blocks1 = nn.Sequential(
			nn.Conv2d(128, 128, 3, stride=1, padding=1),
			nn.BatchNorm2d(128, 0.8),
			nn.LeakyReLU(0.2, inplace=True),
		)
		self.conv_blocks2 = nn.Sequential(
			nn.Conv2d(128, 64, 3, stride=1, padding=1),
			nn.BatchNorm2d(64, 0.8),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Conv2d(64, self.settings.channels, 3, stride=1, padding=1),
			nn.Tanh(),
			nn.BatchNorm2d(self.settings.channels, affine=False)
		)

	def forward(self, z, labels, linear=None, z2=None):
		# GDFQ此处为随机噪声乘label
		if linear == None:
			gen_input = self.embed_normalizer(torch.add(self.label_emb(labels),self.settings.noise_scale*z).T).T 

			if not self.settings.no_DM:
				gen_input = self.fc_reducer(gen_input)

		else:
			embed_norm = self.embed_normalizer(torch.add(self.label_emb(labels),self.settings.noise_scale*z).T).T 

			if not self.settings.no_DM:
				gen_input = self.fc_reducer(embed_norm)
			else:
				gen_input = embed_norm

			gen_input = (gen_input * linear.unsqueeze(2)).sum(dim=1)

		out = self.l1(gen_input)
		out = out.view(out.shape[0], 128, self.init_size, self.init_size)
		img = self.conv_blocks0(out)
		img = nn.functional.interpolate(img, scale_factor=2)
		img = self.conv_blocks1(img)
		img = nn.functional.interpolate(img, scale_factor=2)
		img = self.conv_blocks2(img)
		return img


class Generator_imagenet(nn.Module):
	def __init__(self, options=None, teacher_weight=None, freeze=True):
		super(Generator_imagenet, self).__init__()
		self.settings = options
		if teacher_weight==None:
			self.label_emb = nn.Embedding(self.settings.nClasses, self.settings.latent_dim)
		else:
			self.label_emb = nn.Embedding.from_pretrained(teacher_weight, freeze=freeze)

		self.embed_normalizer = nn.BatchNorm1d(self.label_emb.weight.T.shape,affine=False,track_running_stats=False)

		if not self.settings.no_DM:
			self.fc_reducer = nn.Linear(in_features=self.label_emb.weight.shape[-1], out_features=self.settings.intermediate_dim)

			self.init_size = self.settings.img_size // 4
			self.l1 = nn.Sequential(nn.Linear(self.settings.intermediate_dim, 128 * self.init_size ** 2))
		else:
			self.init_size = self.settings.img_size // 4
			self.l1 = nn.Sequential(nn.Linear(self.settings.latent_dim, 128 * self.init_size ** 2))

		self.conv_blocks0_0 = CategoricalConditionalBatchNorm2d(1000, 128)

		self.conv_blocks1_0 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
		self.conv_blocks1_1 = CategoricalConditionalBatchNorm2d(1000, 128, 0.8)
		self.conv_blocks1_2 = nn.LeakyReLU(0.2, inplace=True)

		self.conv_blocks2_0 = nn.Conv2d(128, 64, 3, stride=1, padding=1)
		self.conv_blocks2_1 = CategoricalConditionalBatchNorm2d(1000, 64, 0.8)
		self.conv_blocks2_2 = nn.LeakyReLU(0.2, inplace=True)
		self.conv_blocks2_3 = nn.Conv2d(64, self.settings.channels, 3, stride=1, padding=1)
		self.conv_blocks2_4 = nn.Tanh()
		self.conv_blocks2_5 = nn.BatchNorm2d(self.settings.channels, affine=False)

	def forward(self, z, labels, linear=None):
		if linear == None:
			gen_input = self.embed_normalizer(torch.add(self.label_emb(labels),z).T).T
			if not self.settings.no_DM:
				gen_input = self.fc_reducer(gen_input)
		else:
			embed_norm = self.embed_normalizer(torch.add(self.label_emb(labels),z).T).T
			if not self.settings.no_DM:
				gen_input = self.fc_reducer(embed_norm)
			else:
				gen_input = embed_norm
			gen_input = (gen_input * linear.unsqueeze(2)).sum(dim=1)

		out = self.l1(gen_input)
		out = out.view(out.shape[0], 128, self.init_size, self.init_size)
		img = self.conv_blocks0_0(out, labels, linear=linear)
		img = nn.functional.interpolate(img, scale_factor=2)
		img = self.conv_blocks1_0(img)
		img = self.conv_blocks1_1(img, labels, linear=linear)
		img = self.conv_blocks1_2(img)
		img = nn.functional.interpolate(img, scale_factor=2)
		img = self.conv_blocks2_0(img)
		img = self.conv_blocks2_1(img, labels, linear=linear)
		img = self.conv_blocks2_2(img)
		img = self.conv_blocks2_3(img)
		img = self.conv_blocks2_4(img)
		img = self.conv_blocks2_5(img)
		return img

class ConditionalBatchNorm2d(nn.BatchNorm2d):
    """Conditional Batch Normalization"""
    def __init__(self, num_features, eps=1e-05, momentum=0.1,
                 affine=False, track_running_stats=True):
        super(ConditionalBatchNorm2d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats
        )

    def forward(self, input, weight, bias, **kwargs):
        self._check_input_dim(input)

        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            self.num_batches_tracked += 1
            if self.momentum is None:  # use cumulative moving average
                exponential_average_factor = 1.0 / self.num_batches_tracked.item()
            else:  # use exponential moving average
                exponential_average_factor = self.momentum

        output = F.batch_norm(input, self.running_mean, self.running_var,
                              self.weight, self.bias,
                              self.training or not self.track_running_stats,
                              exponential_average_factor, self.eps)
        if weight.dim() == 1:
            weight = weight.unsqueeze(0)
        if bias.dim() == 1:
            bias = bias.unsqueeze(0)
        size = output.size()
        weight = weight.unsqueeze(-1).unsqueeze(-1).expand(size)
        bias = bias.unsqueeze(-1).unsqueeze(-1).expand(size)
        return weight * output + bias

class CategoricalConditionalBatchNorm2d(ConditionalBatchNorm2d):

    def __init__(self, num_classes, num_features, eps=1e-5, momentum=0.1,
                 affine=False, track_running_stats=True):
        super(CategoricalConditionalBatchNorm2d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats
        )
        self.weights = nn.Embedding(num_classes, num_features)
        self.biases = nn.Embedding(num_classes, num_features)

        self._initialize()

    def _initialize(self):
        init.ones_(self.weights.weight.data)
        init.zeros_(self.biases.weight.data)

    def forward(self, input, c, linear=None,**kwargs):
        weight = self.weights(c)
        bias = self.biases(c)

        if linear != None:
            weight = (weight * linear.unsqueeze(2)).mean(dim=1)
            bias = (bias * linear.unsqueeze(2)).mean(dim=1)

        return super(CategoricalConditionalBatchNorm2d, self).forward(input, weight, bias)
