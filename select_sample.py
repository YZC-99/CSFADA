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
from utils.correlation_loss import correlation_loss
from dataloaders.npc import get_npc_dataset
from dataloaders.fundus import get_fundus_dataset
from dataloaders.prostate import get_prostate_dataset
from networks.net_factory import net_factory
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.decomposition import PCA
import faiss

# Set path
_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'utils')
sys.path.append(_path)


def compute_correlation_loss(model, data_loader):
    """
    Compute the correlation loss for each sample and normalize it to [0, 1].
    Returns a dictionary where the key is the sample name and the value is the normalized correlation loss.
    """
    model.eval()  # Set the model to evaluation mode
    correlation_loss_scores = {}

    all_losses = []  # List to store the correlation losses of all samples for later normalization

    # Iterate over the data loader
    for batch_idx, data in enumerate(data_loader):
        inputs_weak, inputs_strong, sample_names = data['image'].cuda(), data['image_s'].cuda(), data['name']
        # Get the outputs for weak and strong augmentations
        outputs_weak = model(inputs_weak)  # Output for weak augmentation
        outputs_strong = model(inputs_strong)  # Output for strong augmentation

        # Compute the correlation loss
        loss = correlation_loss(outputs_weak, outputs_strong)

        # Add the loss of each sample to the all_losses list
        all_losses.append(loss.cpu().detach().item())

        # Store the loss along with the sample names
        for i, name in enumerate(sample_names):
            correlation_loss_scores[name] = loss.cpu().detach().item()

    # Normalize the correlation loss of all samples to the range [0, 1]
    min_loss = min(all_losses)
    max_loss = max(all_losses)

    # Normalize the losses in the dictionary
    for name, loss in correlation_loss_scores.items():
        normalized_loss = (loss - min_loss) / (max_loss - min_loss)  # Normalize
        correlation_loss_scores[name] = normalized_loss

    return correlation_loss_scores


def extract_features(model, data_loader, return_name=False):
    """
    Extract weak and strong augmentation features from the model.

    Arguments:
        model: The target model.
        data_loader: Data loader containing input images.
        return_name: Whether to return sample names.
                   If False, return the original feature matrix;
                   If True, return a dictionary with the sample names as keys and corresponding features as values.
    Output:
        features_weak, features_strong or feature_dict (depending on return_name)
    """
    model.eval()
    features_weak = []
    features_strong = []
    feature_dict = {}

    with torch.no_grad():
        for data in data_loader:
            image_weak, image_strong = data['image'].cuda(), data['image_s'].cuda()
            sample_names = data['name']  # Assume data_loader provides the sample names

            weak_feature, _ = model.forward_feat_out(image_weak)  # Extract weak augmentation features
            strong_feature, _ = model.forward_feat_out(image_strong)  # Extract strong augmentation features

            if return_name:
                # If names are needed, map the names to the features
                for i, name in enumerate(sample_names):
                    feature_dict[name] = {
                        'weak': weak_feature[i].cpu().numpy(),
                        'strong': strong_feature[i].cpu().numpy()
                    }
            else:
                # Otherwise, append the features to the lists
                features_weak.append(weak_feature.cpu().numpy())
                features_strong.append(strong_feature.cpu().numpy())

    if return_name:
        return feature_dict
    else:
        # Concatenate all features into one matrix
        features_weak = np.concatenate(features_weak, axis=0)
        features_strong = np.concatenate(features_strong, axis=0)
        return features_weak, features_strong


def compute_distance_scores(model, target_train_loader, weight_weak=0.5, weight_strong=0.5, n_clusters=30):
    """
    Compute the combined distance score for each sample by combining the weak and strong augmentation feature distances,
    and normalize the scores to the range [0, 1].

    Arguments:
        model: The trained model used to extract features.
        target_train_loader: Data loader containing the samples for distance computation.
        weight_weak: The weight for weak augmentation features, default is 0.5.
        weight_strong: The weight for strong augmentation features, default is 0.5.
        n_clusters: The number of clusters for K-means clustering, default is 30.

    Output:
        combined_scores: A dictionary containing each sample's name and its normalized combined score.
    """

    # Extract weak and strong augmentation features, returning a dictionary where key is sample name and value contains weak and strong features
    features_w_s_name = extract_features(model, target_train_loader, return_name=True)

    # Extract all weak and strong augmentation features
    features_weak = np.array([features['weak'] for features in features_w_s_name.values()])
    features_strong = np.array([features['strong'] for features in features_w_s_name.values()])

    # If the features are 4D tensors (batch_size, channels, height, width), reshape them into 2D
    if features_weak.ndim > 2:
        features_weak = features_weak.reshape(features_weak.shape[0], -1)  # Flatten to (n_samples, n_features)
    if features_strong.ndim > 2:
        features_strong = features_strong.reshape(features_strong.shape[0], -1)

    # Perform dimensionality reduction using PCA
    pca = PCA(n_components=50)
    features_weak_reduced = pca.fit_transform(features_weak)
    features_strong_reduced = pca.fit_transform(features_strong)

    # Ensure the feature arrays are C-contiguous
    features_weak_reduced = np.ascontiguousarray(features_weak_reduced)
    features_strong_reduced = np.ascontiguousarray(features_strong_reduced)

    # Use MiniBatchKMeans for clustering
    kmeans_weak = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=256).fit(features_weak_reduced)
    kmeans_strong = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=256).fit(features_strong_reduced)
    centroids_weak = np.ascontiguousarray(kmeans_weak.cluster_centers_)
    centroids_strong = np.ascontiguousarray(kmeans_strong.cluster_centers_)

    # Use faiss for distance computation
    index_weak = faiss.IndexFlatL2(features_weak_reduced.shape[1])
    index_weak.add(centroids_weak)
    _, min_dist_weak = index_weak.search(features_weak_reduced, 1)
    min_dist_weak = min_dist_weak.flatten()

    index_strong = faiss.IndexFlatL2(features_strong_reduced.shape[1])
    index_strong.add(centroids_strong)
    _, min_dist_strong = index_strong.search(features_strong_reduced, 1)
    min_dist_strong = min_dist_strong.flatten()

    # Combine the weak and strong distances using a weighted sum
    combined_distances = weight_weak * min_dist_weak + weight_strong * min_dist_strong

    # Normalize the distances to the range [0, 1]
    min_dist = np.min(combined_distances)
    max_dist = np.max(combined_distances)

    normalized_distances = (combined_distances - min_dist) / (max_dist - min_dist)

    # Create a dictionary to return the sample names and their normalized representative metrics
    combined_scores = {name: score for name, score in zip(features_w_s_name.keys(), normalized_distances)}

    return combined_scores


# Select representative samples: combine weak and strong augmentation clustering results
def select_representative_samples(correlation_loss_scores, combined_distances, beta=0.5):
    """
    Select representative samples based on a combined score formula.

    Arguments:
        correlation_loss_scores: A dictionary with sample names as keys and normalized correlation loss as values.
        combined_distances: A dictionary with sample names as keys and normalized combined distance as values.
        beta: A weight parameter for balancing correlation loss and distance, in the range [0, 1], default is 0.5.

    Output:
        representative_samples: A list of sample names sorted by their combined score from high to low.
    """

    final_scores = {}

    # Compute the final score for each sample
    for sample_name in correlation_loss_scores.keys():
        L_corr = correlation_loss_scores[sample_name]
        D_combined = combined_distances[sample_name]

        # Calculate the final score S_final using the formula
        S_final = beta * (1 - L_corr) + (1 - beta) * (1 / (D_combined + 1e-8))  # Prevent division by zero
        final_scores[sample_name] = S_final

    # Sort the samples by their final score from high to low
    representative_samples = sorted(final_scores.keys(), key=lambda x: final_scores[x], reverse=True)

    return representative_samples


def select_run(cfg, per=0.2):
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    # Select dataset and task
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
    # Load target site data
    target_train_ds = sites_dict[args.target_site]['all']
    target_train_loader = DataLoader(target_train_ds, batch_size=8, shuffle=False, num_workers=8, pin_memory=True)

    # Initialize model
    model = create_model()
    model.load_state_dict(torch.load(args.ckpt_dir + "/last.pth"))
    model = model.cuda()
    model.eval()

    # Stage 1: Compute the correlation loss for each sample and normalize it to [0, 1]
    correlation_loss_scores = compute_correlation_loss(model, target_train_loader)

    # Stage 2: Select samples based on representative metrics
    # Extract weak and strong augmentation features
    # Perform K-means clustering on weak and strong features to obtain centroids
    # Combine weak and strong distances. Finally, normalize the scores to [0, 1]
    distance_scores = compute_distance_scores(model, target_train_loader)

    # Stage 3: Select representative samples
    representative_samples = select_representative_samples(correlation_loss_scores, distance_scores, beta=args.beta)
    # Take the top per% /2 samples and the bottom per% /2 samples
    representative_samples = representative_samples[:int(len(representative_samples) * per / 2)] + representative_samples[-int(len(
                                                                                                            representative_samples) * per / 2):]
    # file_path = os.path.join(cfg.ckpt_dir, 'YZC', f'{cfg.target_site}_%.2f_.txt' % per)
    file_path = os.path.join(cfg.ckpt_dir, 'YZC', f'{cfg.target_site}_%.2f_beta_{args.beta}.txt' % per)

    # representative_samples = representative_samples[:int(len(representative_samples) * per)]
    # file_path = os.path.join(cfg.ckpt_dir, 'YZC', f'{cfg.target_site}_%.2f_top.txt' % per)

    os.makedirs(os.path.dirname(file_path), exist_ok=True)  # Ensure the directory exists
    with open(file_path, 'w') as f:
        for sample_name in representative_samples:
            f.write(f"{sample_name}\n")

    print(f"Representative sample names have been saved to {file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patch_size", type=list, default=[256, 256])
    parser.add_argument("--source_site", type=str, default="SiteA")
    parser.add_argument("--target_site", type=str, default="SiteB")
    parser.add_argument("--task", type=str, default="npc")
    parser.add_argument("--ckpt_dir", type=str, default="/dk2/yzc/continual_seg_exp/npc/correlation-epoch10/SiteA_2_B/")
    parser.add_argument("--per", type=float, default=0.2, help="The percentage of selected samples")
    parser.add_argument("--beta", type=float, default=0.5, help="Weight parameter for balancing correlation loss and distance")
    args = parser.parse_args()
    select_run(args, per=args.per)
