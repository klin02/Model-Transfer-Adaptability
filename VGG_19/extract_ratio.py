import sys
import os


# 从get_param.py输出重定向文件val.txt中提取参数量和计算量
def extract_ratio():
    fr = open('param_flops.txt','r')
    lines = fr.readlines()
    layer = []
    par_ratio = []
    flop_ratio = []
    for line in lines:
        if '(' in line and ')' in line:
            layer.append(line.split(')')[0].split('(')[1])
            r1 = line.split('%')[0].split(',')[-1]
            r1 = float(r1)
            par_ratio.append(r1)
            r2 = line.split('%')[-2].split(',')[-1]
            r2 = float(r2)
            flop_ratio.append(r2)

    return layer, par_ratio, flop_ratio


if __name__ == "__main__":
    layer, par_ratio, flop_ratio = extract_ratio()
    print(layer)
    print(par_ratio)
    print(flop_ratio)