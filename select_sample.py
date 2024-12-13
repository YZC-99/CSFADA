import argparse
import os
import random
import shutil
import sys
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
import torch.nn as nn
from utils.clustering import run_kmeans
from utils.correlation_loss import correlation_loss
from dataloaders.npc import get_npc_dataset
from dataloaders.fundus import get_fundus_dataset
from dataloaders.prostate import get_prostate_dataset
from networks.net_factory import net_factory
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.decomposition import PCA
import faiss

# 设置路径
_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'utils')
sys.path.append(_path)


def compute_correlation_loss(model, data_loader):
    """
    计算每个样本的相关性损失，并将其归一化到 [0, 1] 之间
    返回一个字典，字典的key是样本的名称，value是归一化后的correlation loss
    """
    model.eval()  # 设置模型为评估模式
    correlation_loss_scores = {}

    all_losses = []  # 用于存储所有样本的correlation loss，之后进行归一化处理

    # 遍历数据加载器
    for batch_idx, data in enumerate(data_loader):
        inputs_weak, inputs_strong, sample_names = data['image'].cuda(), data['image_s'].cuda(), data['name']
        # 获取弱增强和强增强的输出
        outputs_weak = model(inputs_weak)  # 弱增强的输出
        outputs_strong = model(inputs_strong)  # 强增强的输出

        # 计算相关性损失
        loss = correlation_loss(outputs_weak, outputs_strong)

        # 将每个样本的损失加入all_losses列表
        all_losses.append(loss.cpu().detach().item())

        # 将损失与样本名称一起存储
        for i, name in enumerate(sample_names):
            correlation_loss_scores[name] = loss.cpu().detach().item()

    # 归一化所有样本的correlation loss到[0, 1]之间
    min_loss = min(all_losses)
    max_loss = max(all_losses)

    # 对字典中的损失进行归一化
    for name, loss in correlation_loss_scores.items():
        normalized_loss = (loss - min_loss) / (max_loss - min_loss)  # 归一化
        correlation_loss_scores[name] = normalized_loss

    return correlation_loss_scores


def extract_features(model, data_loader, return_name=False):
    """
    提取模型的弱增强和强增强特征。

    参数：
        model: 目标模型。
        data_loader: 数据加载器，包含输入图像。
        return_name: 是否返回样本名称。
                   如果为 False，返回原始特征矩阵；
                   如果为 True，返回一个字典，key 为样本名称，value 为对应的特征。
    输出：
        features_weak, features_strong 或 feature_dict (根据 return_name 确定)
    """
    model.eval()
    features_weak = []
    features_strong = []
    feature_dict = {}

    with torch.no_grad():
        for data in data_loader:
            image_weak, image_strong = data['image'].cuda(), data['image_s'].cuda()
            sample_names = data['name']  # 假设 data_loader 提供每个样本的名称

            weak_feature, _ = model.forward_feat_out(image_weak)  # 提取弱增强特征
            strong_feature, _ = model.forward_feat_out(image_strong)  # 提取强增强特征

            if return_name:
                # 如果需要返回样本名称，将名称和特征对应起来
                for i, name in enumerate(sample_names):
                    feature_dict[name] = {
                        'weak': weak_feature[i].cpu().numpy(),
                        'strong': strong_feature[i].cpu().numpy()
                    }
            else:
                # 否则，将特征添加到列表中
                features_weak.append(weak_feature.cpu().numpy())
                features_strong.append(strong_feature.cpu().numpy())

    if return_name:
        return feature_dict
    else:
        # 合并所有特征到一个矩阵中
        features_weak = np.concatenate(features_weak, axis=0)
        features_strong = np.concatenate(features_strong, axis=0)
        return features_weak, features_strong


def compute_distance_scores(model, target_train_loader, weight_weak=0.5, weight_strong=0.5, n_clusters=30):
    """
    计算每个样本的综合距离得分，结合弱增强和强增强的特征距离，并将得分归一化到[0, 1]之间。

    输入：
        model: 训练好的模型，用于提取特征。
        target_train_loader: 数据加载器，包含待计算距离的样本。
        weight_weak: 弱增强特征的权重，默认0.5。
        weight_strong: 强增强特征的权重，默认0.5。
        n_clusters: K-means 聚类的簇数，默认30。

    输出：
        combined_scores: 包含每个样本名称及其归一化后的综合得分的字典。
    """

    # 提取弱增强和强增强的特征，返回的是一个字典，key是样本名称，value是包含弱增强和强增强特征的字典
    features_w_s_name = extract_features(model, target_train_loader, return_name=True)

    # 提取所有弱增强和强增强特征
    features_weak = np.array([features['weak'] for features in features_w_s_name.values()])
    features_strong = np.array([features['strong'] for features in features_w_s_name.values()])

    # 如果特征是四维张量 (batch_size, channels, height, width)，需要展平成二维
    if features_weak.ndim > 2:
        features_weak = features_weak.reshape(features_weak.shape[0], -1)  # 展平成 (n_samples, n_features)
    if features_strong.ndim > 2:
        features_strong = features_strong.reshape(features_strong.shape[0], -1)

    # 对弱增强和强增强特征进行降维处理
    pca = PCA(n_components=50)
    features_weak_reduced = pca.fit_transform(features_weak)
    features_strong_reduced = pca.fit_transform(features_strong)

    # 确保特征数组是 C 连续的
    features_weak_reduced = np.ascontiguousarray(features_weak_reduced)
    features_strong_reduced = np.ascontiguousarray(features_strong_reduced)

    # 使用 MiniBatchKMeans 聚类
    kmeans_weak = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=256).fit(features_weak_reduced)
    kmeans_strong = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=256).fit(features_strong_reduced)
    centroids_weak = np.ascontiguousarray(kmeans_weak.cluster_centers_)
    centroids_strong = np.ascontiguousarray(kmeans_strong.cluster_centers_)

    # 使用 faiss 进行距离计算
    index_weak = faiss.IndexFlatL2(features_weak_reduced.shape[1])
    index_weak.add(centroids_weak)
    _, min_dist_weak = index_weak.search(features_weak_reduced, 1)
    min_dist_weak = min_dist_weak.flatten()

    index_strong = faiss.IndexFlatL2(features_strong_reduced.shape[1])
    index_strong.add(centroids_strong)
    _, min_dist_strong = index_strong.search(features_strong_reduced, 1)
    min_dist_strong = min_dist_strong.flatten()

    # 合并弱增强和强增强的距离，使用加权和
    combined_distances = weight_weak * min_dist_weak + weight_strong * min_dist_strong

    # 归一化距离到 [0, 1] 区间
    min_dist = np.min(combined_distances)
    max_dist = np.max(combined_distances)

    normalized_distances = (combined_distances - min_dist) / (max_dist - min_dist)

    # 创建一个字典，返回样本名称及其归一化后的代表性度量
    combined_scores = {name: score for name, score in zip(features_w_s_name.keys(), normalized_distances)}

    return combined_scores


# 选择代表性样本：结合弱增强和强增强的聚类结果
def select_representative_samples(correlation_loss_scores, combined_distances, beta=0.5):
    """
    根据综合得分公式选择代表性样本。

    输入：
        correlation_loss_scores: 字典，样本名称为key，相关性损失为value，已经归一化到[0, 1]之间。
        combined_distances: 字典，样本名称为key，综合距离为value，已经归一化到[0, 1]之间。
        beta: 权重参数，用于平衡相关性损失和距离，范围是[0, 1]，默认是0.5。

    输出：
        representative_samples: 按照综合得分从高到低排序的样本名称列表。
    """

    final_scores = {}

    # 计算每个样本的最终得分
    for sample_name in correlation_loss_scores.keys():
        L_corr = correlation_loss_scores[sample_name]
        D_combined = combined_distances[sample_name]

        # 根据公式计算 S_final
        S_final = beta * (1 - L_corr) + (1 - beta) * (1 / (D_combined + 1e-8))  # 防止除零错误
        final_scores[sample_name] = S_final

    # 按照最终得分从高到低排序
    representative_samples = sorted(final_scores.keys(), key=lambda x: final_scores[x], reverse=True)

    return representative_samples


def YZC(cfg, per=0.2):
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    # 选择数据集和类别
    if cfg.task == 'fundus':
        sites_dict = get_fundus_dataset(cfg)

        def create_model():
            model = net_factory(net_type="unet", in_chns=3,
                                class_num=3)
            return model
    elif cfg.task == 'npc':
        sites_dict = get_npc_dataset(cfg)

        def create_model():
            model = net_factory(net_type="unet", in_chns=1,
                                class_num=2)
            return model
    elif cfg.task == 'prostate':
        sites_dict = get_prostate_dataset(cfg)

        def create_model():
            model = net_factory(net_type="unet", in_chns=1,
                                class_num=2)
            return model
    # 加载目标站点数据
    target_train_ds = sites_dict[args.target_site]['all']
    target_train_loader = DataLoader(target_train_ds, batch_size=8, shuffle=False, num_workers=8, pin_memory=True)

    # 初始化模型
    model = create_model()
    model.load_state_dict(torch.load(args.ckpt_dir + "/last.pth"))
    model = model.cuda()
    model.eval()

    # Stage 1 计算每个样本的correlation loss,把所有样本的correlation loss归一化到[0,1]之间，返回一个字典，这个字典的key是样本的名称，value是归一化后的correlation loss
    correlation_loss_scores = compute_correlation_loss(model, target_train_loader)

    # Stage 2 根据代表性度量选择样本
    # 提取弱增强和强增强的特征
    # 对弱增强和强增强的特征进行K-means聚类，分别获得质心
    # 合并弱增强和强增强的距离。最终需要将他们的得分归一化到[0,1]之间
    distance_scores = compute_distance_scores(model, target_train_loader)

    # Stage 3
    representative_samples = select_representative_samples(correlation_loss_scores, distance_scores, beta=args.beta)
    # 取前per% /2 的样本和后per% /2 的样本
    representative_samples = representative_samples[:int(len(representative_samples) * per / 2)] + representative_samples[-int(len(
                                                                                                            representative_samples) * per / 2):]
    # file_path = os.path.join(cfg.ckpt_dir, 'YZC', f'{cfg.target_site}_%.2f_.txt' % per)
    file_path = os.path.join(cfg.ckpt_dir, 'YZC', f'{cfg.target_site}_%.2f_beta_{args.beta}.txt' % per)

    # representative_samples = representative_samples[:int(len(representative_samples) * per)]
    # file_path = os.path.join(cfg.ckpt_dir, 'YZC', f'{cfg.target_site}_%.2f_top.txt' % per)

    os.makedirs(os.path.dirname(file_path), exist_ok=True)  # 确保文件夹存在
    with open(file_path, 'w') as f:
        for sample_name in representative_samples:
            f.write(f"{sample_name}\n")

    print(f"代表性样本的名称已保存到 {file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patch_size", type=list, default=[256, 256])
    parser.add_argument("--source_site", type=str, default="SiteA")
    parser.add_argument("--target_site", type=str, default="SiteB")
    parser.add_argument("--task", type=str, default="npc")
    parser.add_argument("--ckpt_dir", type=str, default="/dk2/yzc/continual_seg_exp/npc/correlation-epoch10/SiteA_2_B/")
    parser.add_argument("--per", type=float, default=0.2, help="比例：选择的代表性样本比例")
    parser.add_argument("--beta", type=float, default=0.5, help="比例：选择的代表性样本比例")
    args = parser.parse_args()
    YZC(args, per=args.per)

    # args.per = 0.4
    # YZC(args, per=args.per)
    #
    # args.per = 0.6
    # YZC(args, per=args.per)
    #
    # args.per = 0.8
    # YZC(args, per=args.per)
    #
    # args.per = 1.0
    # YZC(args, per=args.per)
    #
    # args.per = 0.2
    # betas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    # for beta in betas:
    #     args.beta = beta
    #     YZC(args, per=args.per)
