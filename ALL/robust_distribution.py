"""
    利用PCA降维可视化伪数据所得特征与模型分类结果/生成器输入标签关系，从而观测决策边界
    模型分类结果out表示该模型的决策边界
    生成器输入标签label表示生成器所拟合的边界
    针对cifar10数据集下ResNet_18生成可视化图像
    以ResNet18为例，利用PCA降维可视化测试集/未调整生成器和调整后生成器的数据分布
"""
from model import Model
from dataloader import DataLoader
from gen_one import GenOption

import os
import argparse
import numpy as np
import pandas as pd

from scipy.linalg import eigh

import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms

import seaborn as sns
from matplotlib import pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


parser = argparse.ArgumentParser(description='PCA features')
# 获取测试集在model上分布图
parser.add_argument('--pca_src', action='store_true')
# 未调整生成器伪数据的分布图
parser.add_argument('--pca_gen_org', action='store_true')
# 调整后生成器伪数据的分布图
parser.add_argument('--pca_gen_adjust', action='store_true')

# # 指定选择哪个模型对应的生成器
# parser.add_argument('--quant',action='store_true')
# # 当选择的为量化网络时，下述参数才生效
# parser.add_argument('--quant_type',type=str,default='NONE')
# parser.add_argument('--num_bits',type=int,default=0)
# parser.add_argument('--e_bits',type=int,default=0)


def reduce_df(dataframe, num_per_class):
    df_list = []
    for i in range(10):
        df_list.append(dataframe.iloc[i * 1000: i * 1000 + num_per_class])
    df = pd.concat(df_list)
    return df

if __name__ == '__main__':
    args = parser.parse_args()
    
    # 确定模型及生成器路径
    model_name = 'ResNet_18'
    # model_name = 'ResNet_50'
    model_file = 'ckpt_full/cifar10/'+model_name+'.pt'
    if args.pca_gen_org:
        gen_org_file = 'ckpt_gen/cifar10/'+model_name+'.pt'
        gen_org = torch.load(gen_org_file)
        gen_org.eval()
    if args.pca_gen_adjust:
        gen_adjust_file = 'ckpt_gen/cifar10/'+model_name+'_adjust.pt'
        gen_adjust = torch.load(gen_adjust_file)
        gen_adjust.eval()
    pca_dir = 'pca_'+model_name+'_cifar10/'


    if True in [args.pca_src, args.pca_gen_org, args.pca_gen_adjust]:
        os.makedirs(pca_dir, exist_ok=True)
        model = Model(model_name=model_name,dataset='cifar10').cuda()
        model.load_state_dict(torch.load(model_file))
        model.eval()
        print('Model Ready')

        dataloader = DataLoader('cifar10',200)
        _,_,test_loader = dataloader.getloader()
        
        features = []
        targets = []
        with torch.no_grad():
            for i, (data, target) in enumerate(test_loader):
                data = data.cuda()
                _, feature = model(data, out_feature=True)
                features.append(feature.cpu())
                targets.append(target)

        features_cat = torch.cat(features, dim=0)
        targets_cat = torch.cat(targets, dim=0)

        feature_scaler = StandardScaler().fit(features_cat)
        standardized_data = feature_scaler.transform(features_cat)
        
        covar_matrix = np.matmul(standardized_data.T , standardized_data)
        values, vectors = eigh(covar_matrix, eigvals=(510,511))
        vectors = vectors.T

        if args.pca_src:
            print('pca of source data')
            new_coordinates = np.matmul(vectors, standardized_data.T)
            new_coordinates = np.vstack((new_coordinates, targets_cat)).T
            dataframe = pd.DataFrame(data=new_coordinates, columns=("1st_principal", "2nd_principal", "label"))
            dataframe.sort_values(by=['label'], axis=0, inplace=True)
            df = reduce_df(dataframe, 1000)

            pca_result = sns.FacetGrid(df, hue="label", height=10, hue_kws={'marker':['x'] * 10}).map(plt.scatter, '1st_principal', '2nd_principal')

            pca_result.set(xticks=[], yticks=[], xlabel='', ylabel='')
            plt.savefig(pca_dir+'pca_src.png')

        if args.pca_gen_org:
            print('pca of gen before adjust')

            features_gen = []
            targets_gen = []

            with torch.no_grad():
                for i in range(50):
                    z = torch.randn(200, 512).cuda()
                    labels = (torch.ones(200) * (i // 5)).type(torch.LongTensor)
                    targets_gen.append(labels)
                    labels = labels.cuda()
                    z = z.contiguous()
                    labels = labels.contiguous()
                    images = gen_org(z, labels)
                    _, feature = model(images, out_feature=True)
                    features_gen.append(feature.cpu())
        
            features_cat_gen = torch.cat(features_gen, dim=0)
            targets_cat_gen = torch.cat(targets_gen, dim=0)

            standardized_data_gen = feature_scaler.transform(features_cat_gen)

            new_coordinates_gen = np.matmul(vectors, standardized_data_gen.T)
            new_coordinates_gen = np.vstack((new_coordinates_gen, targets_cat_gen)).T

            dataframe_gen = pd.DataFrame(data=new_coordinates_gen, columns=("1st_principal", "2nd_principal", "label"))
        
            pca_result = sns.FacetGrid(dataframe_gen, hue="label", height=10, hue_kws={'marker':['x'] * 10}).map(plt.scatter, '1st_principal', '2nd_principal')
            pca_result.set(xticks=[], yticks=[], xlabel='', ylabel='')
            plt.savefig(pca_dir+'pca_gen_org.png')

        if args.pca_gen_adjust:
            print('pca of gen after adjust')

            features_gen = []
            targets_gen = []

            with torch.no_grad():
                for i in range(50):
                    z = torch.randn(200, 512).cuda()
                    labels = (torch.ones(200) * (i // 5)).type(torch.LongTensor)
                    targets_gen.append(labels)
                    labels = labels.cuda()
                    z = z.contiguous()
                    labels = labels.contiguous()
                    images = gen_adjust(z, labels)
                    _, feature = model(images, out_feature=True)
                    features_gen.append(feature.cpu())
        
            features_cat_gen = torch.cat(features_gen, dim=0)
            targets_cat_gen = torch.cat(targets_gen, dim=0)

            standardized_data_gen = feature_scaler.transform(features_cat_gen)

            new_coordinates_gen = np.matmul(vectors, standardized_data_gen.T)
            new_coordinates_gen = np.vstack((new_coordinates_gen, targets_cat_gen)).T

            dataframe_gen = pd.DataFrame(data=new_coordinates_gen, columns=("1st_principal", "2nd_principal", "label"))
        
            pca_result = sns.FacetGrid(dataframe_gen, hue="label", height=10, hue_kws={'marker':['x'] * 10}).map(plt.scatter, '1st_principal', '2nd_principal')
            pca_result.set(xticks=[], yticks=[], xlabel='', ylabel='')
            plt.savefig(pca_dir+'pca_gen_adjust.png')