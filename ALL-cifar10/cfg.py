# conv: 'C',''/'B'/'BRL'/'BRS',qi,in_ch,out_ch,kernel_size,stirde,padding,bias
# relu: 'RL'
# relu6: 'RS'
# inception: 'Inc'
# maxpool: 'MP',kernel_size,stride,padding
# adaptiveavgpool: 'AAP',output_size
# view: 'VW': 
#   dafault: x = x.view(x.size(0),-1)
# dropout: 'D'
# MakeLayer: 'ML','BBLK'/'BTNK'/'IRES', ml_idx, blocks
# softmax: 'SM'
# class 10
ResNet_18_cfg_table = [
    ['C','BRL',True,3,16,3,1,1,True],
    ['ML','BBLK',0,2],
    ['ML','BBLK',1,2],
    ['ML','BBLK',2,2],
    ['ML','BBLK',3,2],
    ['AAP',1],
    ['VW'],
    ['FC',128,10,True],
    ['SM']
]
ResNet_50_cfg_table = [
    ['C','BRL',True,3,16,3,1,1,True],
    ['ML','BTNK',0,3],
    ['ML','BTNK',1,4],
    ['ML','BTNK',2,6],
    ['ML','BTNK',3,3],
    ['AAP',1],
    ['VW'],
    ['FC',512,10,True],
    ['SM']
]
ResNet_152_cfg_table = [
    ['C','BRL',True,3,16,3,1,1,True],
    ['ML','BTNK',0,3],
    ['ML','BTNK',1,8],
    ['ML','BTNK',2,36],
    ['ML','BTNK',3,3],
    ['AAP',1],
    ['VW'],
    ['FC',512,10,True],
    ['SM']
]
MobileNetV2_cfg_table = [
    ['C','BRS',True,3,32,3,1,1,True],
    ['ML','IRES',0,1],
    ['ML','IRES',1,2],
    ['ML','IRES',2,3],
    ['ML','IRES',3,3],
    ['ML','IRES',4,3],
    ['ML','IRES',5,1],
    ['C','',False,320,1280,1,1,0,True],
    ['AAP',1],
    ['VW'],
    ['FC',1280,10,True]
]
AlexNet_cfg_table = [
    ['C','',True,3,32,3,1,1,True],
    ['RL'],
    ['MP',2,2,0],
    ['C','',False,32,64,3,1,1,True],
    ['RL'],
    ['MP',2,2,0],
    ['C','',False,64,128,3,1,1,True],
    ['RL'],
    ['C','',False,128,256,3,1,1,True],
    ['RL'],
    ['C','',False,256,256,3,1,1,True],
    ['RL'],
    ['MP',3,2,0],
    ['VW'],
    ['D',0.5],
    ['FC',2304,1024,True],
    ['RL'],
    ['D',0.5],
    ['FC',1024,512,True],
    ['RL'],
    ['FC',512,10,True]
]
AlexNet_BN_cfg_table = [
    ['C','BRL',True,3,32,3,1,1,True],
    ['MP',2,2,0],
    ['C','BRL',False,32,64,3,1,1,True],
    ['MP',2,2,0],
    ['C','BRL',False,64,128,3,1,1,True],
    ['C','BRL',False,128,256,3,1,1,True],
    ['C','BRL',False,256,256,3,1,1,True],
    ['MP',3,2,0],
    ['VW'],
    ['D',0.5],
    ['FC',2304,1024,True],
    ['RL'],
    ['D',0.5],
    ['FC',1024,512,True],
    ['RL'],
    ['FC',512,10,True]
]
VGG_16_cfg_table = [
    ['C','BRL',True,3,64,3,1,1,True],
    ['C','BRL',False,64,64,3,1,1,True],
    ['MP',2,2,0],
    ['C','BRL',False,64,128,3,1,1,True],
    ['C','BRL',False,128,128,3,1,1,True],
    ['MP',2,2,0],
    ['C','BRL',False,128,256,3,1,1,True],
    ['C','BRL',False,256,256,3,1,1,True],
    ['C','BRL',False,256,256,3,1,1,True],
    ['MP',2,2,0],
    ['C','BRL',False,256,512,3,1,1,True],
    ['C','BRL',False,512,512,3,1,1,True],
    ['C','BRL',False,512,512,3,1,1,True],
    ['MP',2,2,0],
    ['C','BRL',False,512,512,3,1,1,True],
    ['C','BRL',False,512,512,3,1,1,True],
    ['C','BRL',False,512,512,3,1,1,True],
    ['MP',2,2,0],
    ['VW'],
    ['FC',512,4096,True],
    ['RL'],
    ['D',0.5],
    ['FC',4096,4096,True],
    ['RL'],
    ['D',0.5],
    ['FC',4096,10,True]
]
VGG_19_cfg_table = [
    ['C','BRL',True,3,64,3,1,1,True],
    ['C','BRL',False,64,64,3,1,1,True],
    ['MP',2,2,0],
    ['C','BRL',False,64,128,3,1,1,True],
    ['C','BRL',False,128,128,3,1,1,True],
    ['MP',2,2,0],
    ['C','BRL',False,128,256,3,1,1,True],
    ['C','BRL',False,256,256,3,1,1,True],
    ['C','BRL',False,256,256,3,1,1,True],
    ['C','BRL',False,256,256,3,1,1,True],
    ['MP',2,2,0],
    ['C','BRL',False,256,512,3,1,1,True],
    ['C','BRL',False,512,512,3,1,1,True],
    ['C','BRL',False,512,512,3,1,1,True],
    ['C','BRL',False,512,512,3,1,1,True],
    ['MP',2,2,0],
    ['C','BRL',False,512,512,3,1,1,True],
    ['C','BRL',False,512,512,3,1,1,True],
    ['C','BRL',False,512,512,3,1,1,True],
    ['C','BRL',False,512,512,3,1,1,True],
    ['MP',2,2,0],
    ['VW'],
    ['FC',512,4096,True],
    ['RL'],
    ['D',0.5],
    ['FC',4096,4096,True],
    ['RL'],
    ['D',0.5],
    ['FC',4096,10,True]
]
Inception_BN_cfg_table = [
    ['C','',True,3,64,3,1,1,True],
    ['RL'],
    ['C','',False,64,64,3,1,1,True],
    ['RL'],
    ['Inc',0],
    ['Inc',1],
    ['MP',3,2,1],
    ['Inc',2],
    ['Inc',3],
    ['Inc',4],
    ['Inc',5],
    ['Inc',6],
    ['MP',3,2,1],
    ['Inc',7],
    ['Inc',8],
    ['AAP',1],
    ['C','',False,1024,10,1,1,0,True],
    ['VW']
]

model_cfg_table = {
    'AlexNet'       : AlexNet_cfg_table,
    'AlexNet_BN'    : AlexNet_BN_cfg_table,
    'VGG_16'        : VGG_16_cfg_table,
    'VGG_19'        : VGG_19_cfg_table,
    'Inception_BN'  : Inception_BN_cfg_table,
    'ResNet_18'     : ResNet_18_cfg_table,
    'ResNet_50'     : ResNet_50_cfg_table,
    'ResNet_152'    : ResNet_152_cfg_table,
    'MobileNetV2'   : MobileNetV2_cfg_table
}

#每行对应一个Inc结构(channel)的参数表
inc_ch_table=[
    [ 64, 64, 96,128, 16, 32, 32],#3a
    [256,128,128,192, 32, 96, 64],#3b
    [480,192, 96,208, 16, 48, 64],#4a
    [512,160,112,224, 24, 64, 64],#4b
    [512,128,128,256, 24, 64, 64],#4c
    [512,112,144,288, 32, 64, 64],#4d
    [528,256,160,320, 32,128,128],#4e
    [832,256,160,320, 32,128,128],#5a
    [832,384,192,384, 48,128,128] #5b
]

# br0,br1,br2,br3 <- br1x1,br3x3,br5x5,brM
# 每个子数组对应Inc结构中一个分支的结构，均默认含'BRL'参数，bias为False
# Conv层第2、3个参数是对应Inc结构(即ch_table中的一行)中的索引
# 由于每个Inc结构操作一致，只有权重不同，使用索引而非具体值，方便复用
# 各分支后还有Concat操作，由于只有唯一结构，未特殊说明
# conv: 'C', ('BRL' default), in_ch_idex, out_ch_idx, kernel_size, stride, padding, (bias: True default)
# maxpool: 'MP', kernel_size, stride, padding
# relu: 'RL'
inc_cfg_table = [
    [
        ['C',0,1,1,1,0]
    ],
    [
        ['C',0,2,1,1,0],
        ['C',2,3,3,1,1]
    ],
    [
        ['C',0,4,1,1,0],
        ['C',4,5,5,1,2]
    ],
    [
        ['MP',3,1,1],
        ['RL'],
        ['C',0,6,1,1,0]
    ]
]

# ml_cfg_table = []

#BasicBlock
#value: downsample,inplanes,planes,planes*expansion,stride,1(dafault stride and group) 
bblk_ch_table = [
    [False, 16, 16, 16,1,1], #layer1,first
    [False, 16, 16, 16,1,1], #       other
    [True,  16, 32, 32,2,1], #layer2
    [False, 32, 32, 32,1,1], 
    [True,  32, 64, 64,2,1], #layer3
    [False, 64, 64, 64,1,1], 
    [True,  64,128,128,2,1], #layer4
    [False,128,128,128,1,1]  
]
#conv: 'C','B'/'BRL'/'BRS', in_ch_idx, out_ch_idx, kernel_sz, stride_idx, padding, groups_idx (bias: True default)
#add: 'AD', unconditonal. unconditonal为true或flag为true时将outs中两元素相加
bblk_cfg_table = [
    [
        ['C','BRL',1,2,3,4,1,5],
        ['C','B'  ,2,2,3,5,1,5],
    ],
    # downsample, 仅当downsample传入为True时使用
    [
        ['C','B'  ,1,3,1,4,0,5]
    ],
    # 分支交汇后动作
    [
        ['AD',True],
        ['RL']
    ]
]

#BottleNeck
#value: downsample,inplanes,planes,planes*expansion,stride,1(dafault stride and group) 
btnk_ch_table = [
    [True,  16, 16, 64,1,1], #layer1,first
    [False, 64, 16, 64,1,1], #       other 
    [True,  64, 32,128,2,1], #layer2
    [False,128, 32,128,1,1], 
    [True, 128, 64,256,2,1], #layer3
    [False,256, 64,256,1,1],  
    [True, 256,128,512,2,1], #layer4
    [False,512,128,512,1,1]   
]
#conv: 'C','B'/'BRL'/'BRS', in_ch_idx, out_ch_idx, kernel_sz, stride_idx, padding, groups_idx (bias: True default)
#add: 'AD', unconditonal. unconditonal为true或flag为true时将outs中两元素相加
btnk_cfg_table = [
    [
        ['C','BRL',1,2,1,5,0,5],
        ['C','BRL',2,2,3,4,1,5],
        ['C','B'  ,2,3,1,5,0,5]
    ],
    # downsample, 仅当downsample传入为True时使用
    [
        ['C','B'  ,1,3,1,4,0,5]
    ], 
    # 分支交汇后动作
    [
        ['AD',True],
        ['RL']
    ]
]

#InvertedResidual
#value: identity_flag, in_ch, out_ch, in_ch*expand_ratio, stride, 1(dafault stride and group) 
ires_ch_table = [
    [False, 32, 16,  32,1,1], #layer1,first
    [ True, 16, 16,  16,1,1], #       other
    [False, 16, 24,  96,2,1], #layer2
    [ True, 24, 24, 144,1,1],
    [False, 24, 32, 144,2,1], #layer3
    [ True, 32, 32, 192,1,1],
    [False, 32, 96, 192,1,1], #layer4
    [ True, 96, 96, 576,1,1],
    [False, 96,160, 576,2,1], #layer5
    [ True,160,160, 960,1,1],
    [False,160,320, 960,1,1], #layer6
    [ True,320,320,1920,1,1]
]

#conv: 'C','B'/'BRL'/'BRS', in_ch_idx, out_ch_idx, kernel_sz, stride_idx, padding, groups_idx (bias: True default)
#add: 'AD', unconditonal. unconditonal为true或flag为true时将outs中两元素相加
ires_cfg_table = [
    [
        ['C','BRS',1,3,1,5,0,5],
        ['C','BRS',3,3,3,4,1,3],
        ['C','B'  ,3,2,1,5,0,5]
    ],
    # identity_br empty
    [

    ],
    # 分支汇合后操作
    [
        ['AD',False] #有条件的相加
    ]
    
]