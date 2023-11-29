import sys
import os


# 从get_param.py输出重定向文件val.txt中提取参数量和计算量
def extract_ratio(model_name):
    fr = open('param_flops/'+model_name+'.txt','r')
    lines = fr.readlines()

    Mac = lines[1].split('Mac,')[0].split(',')[-1]
    if 'M' in Mac:
        Mac = Mac.split('M')[0]
        Mac = float(Mac)
    elif 'G' in Mac:
        Mac = Mac.split('G')[0]
        Mac = float(Mac)
        Mac *= 1024
    
    Param = lines[1].split('M,')[0]
    Param = float(Param)
    
    layer = []
    par_ratio = []
    flop_ratio = []
    weight_ratio = []
    for line in lines:
        if '(' in line and ')' in line:
            layer.append(line.split(')')[0].split('(')[1])
            r1 = line.split('%')[0].split(',')[-1]
            r1 = float(r1)
            par_ratio.append(r1)
            r2 = line.split('%')[-2].split(',')[-1]
            r2 = float(r2)
            flop_ratio.append(r2)
            if 'conv' in line:
                #无论是否bias=false都计算，fold之后直接使用conv的近似计算
                inch = line.split(',')[4]
                # outch = line.split(',')[5]
                klsz = line.split(',')[6].split('(')[-1]
                inch = float(inch)
                # outch = float(outch)
                klsz = float(klsz)
                wr = inch * klsz * klsz
                wr = wr / (1+wr)
                weight_ratio.append(wr)
            elif 'fc' in line:
                inch = line.split(',')[4].split('=')[-1]
                inch = float(inch)
                wr = inch / (1+inch)
                weight_ratio.append(wr)
            else:
                weight_ratio.append(0)


    return Mac, Param, layer, par_ratio, flop_ratio, weight_ratio


if __name__ == "__main__":
    Mac, Param, layer, par_ratio, flop_ratio, weight_ratio = extract_ratio('Inception_BN')
    print(Mac)
    print(Param)
    print(layer)
    print(par_ratio)
    print(flop_ratio)
    print(weight_ratio)