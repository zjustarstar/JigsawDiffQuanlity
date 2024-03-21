import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from matplotlib.patches import Rectangle
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from torchvision import models


# 定义模型
class FeatureExtractor(nn.Module):
    """
    使用指定的预训练模型从图像中提取特征的类。

    参数:
    - model_type (str): 要使用的预训练模型类型（'vit'代表视觉变换器，'inception'代表Inception v3）。
    - device (str): 运行模型的设备（'cpu'或'cuda'）。

    方法:
    - __init__(self, model_type='vit', device='cpu'): 初始化基于指定类型和设备的模型。
    - forward(self, x): 处理输入图像并返回提取的特征。
    """

    def __init__(self, model_type='vit', device='cpu'):
        super(FeatureExtractor, self).__init__()
        self.device = device
        self.model_type = model_type

        if model_type == 'vit':
            self.model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT).to(device)
            self.model.heads = nn.Identity()  # 移除分类头部
            self.preprocess = transforms.Compose([
                transforms.Lambda(lambda x: x.convert('RGB')),  # 确保图像为RGB
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])
        elif model_type == 'inception':
            self.model = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT).to(device)
            self.model.eval()  # 设置为评估模式
            self.preprocess = transforms.Compose([
                transforms.Lambda(lambda x: x.convert('RGB')),  # 确保图像为RGB
                transforms.Resize((299, 299)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def forward(self, x):
        x = self.preprocess(x).to(self.device)
        x = x.unsqueeze(0)  # 增加批量维度
        with torch.no_grad():
            if self.model_type == 'inception':
                # Inception模型的输出是一个tuple
                x = self.model(x)[0]
            else:
                x = self.model(x)
        return x


def cluster_features(features, method, n_clusters, **kwargs):
    """
    Cluster features using specified clustering method.

    Parameters:
    - features: np.array, features to cluster.
    - method: str, clustering method ('kmeans', 'agglomerative', 'dbscan', 'gmm', 'spectral').
    - n_clusters: int, number of clusters (ignored for DBSCAN).
    - **kwargs: additional keyword arguments to pass to the clustering algorithm.

    Returns:
    - cluster_ids: np.array, cluster labels for each feature.
    """
    if method == 'kmeans':
        cluster_model = KMeans(n_clusters=n_clusters, **kwargs)
    elif method == 'agglomerative':
        cluster_model = AgglomerativeClustering(n_clusters=n_clusters, **kwargs)
    elif method == 'dbscan':
        cluster_model = DBSCAN(**kwargs)
    elif method == 'gmm':
        cluster_model = GaussianMixture(n_components=n_clusters, **kwargs)
    elif method == 'spectral':
        cluster_model = SpectralClustering(n_clusters=n_clusters, **kwargs)
    else:
        raise ValueError(f"Unsupported clustering method: {method}")

    # GMM uses 'fit' and 'predict' separately
    if method == 'gmm':
        cluster_model.fit(features)
        cluster_ids = cluster_model.predict(features)
    else:
        cluster_ids = cluster_model.fit_predict(features)

    return cluster_ids


def visualize_feature_space(features, cluster_ids, method, title='Feature space visualization'):
    """
    可视化特征空间。

    参数:
    - features: 特征数组。
    - cluster_ids: 聚类标签数组。
    - method: 降维方法，'pca' 或 'tsne'。
    - title: 图表标题。
    """
    if method == 'pca':
        reducer = PCA(n_components=2)
    elif method == 'tsne':
        reducer = TSNE(n_components=2, learning_rate='auto', init='random')
    else:
        raise ValueError("Unsupported method. Use 'pca' or 'tsne'.")

    reduced_features = reducer.fit_transform(features)

    plt.figure(figsize=(10, 6))
    for i in range(np.max(cluster_ids) + 1):
        plt.scatter(reduced_features[cluster_ids == i, 0], reduced_features[cluster_ids == i, 1], label=f'Cluster {i}')
    plt.legend()
    plt.title(title)
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.show()


# 图像预处理和分块
def preprocess_image(image):
    # 确保图像只有三个通道
    if image.mode != 'RGB':
        image = image.convert('RGB')
    return image


def get_image_patches(image, transform, grid_size, exclude_edge=False):
    """
    从给定图像中提取正方形图块，可选地排除边缘的图块。

    参数:
    - image (PIL.Image): 要从中提取图块的输入图像。
    - grid_size (int): 划分图像的网格大小。例如，grid_size为10意味着图像将被分成10x10的图块。
    - exclude_edges (bool, 可选): 是否排除图像边缘的图块，默认为True。

    返回:
    - List[PIL.Image]: 提取的图块列表。
    """
    W, H = image.size
    patch_W, patch_H = W // grid_size, H // grid_size
    patches = []

    for i in range(grid_size):
        for j in range(grid_size):
            if exclude_edge and (i == 0 or i == grid_size - 1 or j == 0 or j == grid_size - 1):
                continue
            patch = image.crop((j * patch_W, i * patch_H, (j + 1) * patch_W, (i + 1) * patch_H))
            patches.append(patch)

    processed_patches = [transform(patch) for patch in patches]
    return patches, torch.stack(processed_patches)


def extract_color_histogram_features(patches, bins=32):
    """
    提取颜色直方图特征。
    参数:
    - patches: 图块列表。
    - bins: 直方图中的柱数。
    返回:
    - features: 颜色直方图特征数组。
    """
    features = []
    for patch in patches:
        # 将图块转换为NumPy数组
        np_patch = np.array(patch)
        # 计算每个颜色通道的直方图
        hist_r = np.histogram(np_patch[:, :, 0], bins=bins, range=(0, 255))[0]
        hist_g = np.histogram(np_patch[:, :, 1], bins=bins, range=(0, 255))[0]
        hist_b = np.histogram(np_patch[:, :, 2], bins=bins, range=(0, 255))[0]
        # 合并三个颜色通道的直方图作为特征
        hist_features = np.concatenate((hist_r, hist_g, hist_b))
        features.append(hist_features)
    return np.array(features)


# 图像处理和特征提取
def extract_features(vit_extractor, inception_extractor, image_path, color_bins, grid_size):
    """
    使用指定的特征提取器从图块列表中提取特征。

    参数:
    - patches (List[PIL.Image]): 图块列表。
    - extractors (List[FeatureExtractor]): 用于提取特征的特征提取器实例列表。

    返回:
    - np.array: 提取的特征数组，每行对应一个图块的特征。
    """
    # Code for extracting features from patches
    image = Image.open(image_path)
    original_patches, processed_patches = get_image_patches(image, vit_extractor.preprocess, grid_size)  # 获取原始和处理后的图块

    color_features = extract_color_histogram_features(original_patches, bins=color_bins)

    # 使用ViT模型提取特征
    vit_features = []
    for patch in original_patches:
        feature = vit_extractor(patch).cpu().numpy().flatten()
        vit_features.append(feature)
    vit_features = np.vstack(vit_features)

    # 使用Inception模型提取特征
    inception_features = []
    for patch in original_patches:
        feature = inception_extractor(patch).cpu().numpy().flatten()
        inception_features.append(feature)
    inception_features = np.vstack(inception_features)

    # 合并ViT特征、Inception特征和颜色特征
    combined_features = np.hstack((vit_features, inception_features, color_features))

    # 应用特征缩放
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(combined_features)

    return scaled_features, original_patches


# 打印相同类别的图块
def print_clustered_patches(cluster_ids, patches, max_patches=25):
    unique_clusters = set(cluster_ids)

    for cluster in unique_clusters:
        cluster_patches = [patch for patch, cluster_id in zip(patches, cluster_ids) if cluster_id == cluster]

        if len(cluster_patches) > max_patches:
            print(f"Cluster {cluster} has more than {max_patches} patches. Skipping...")
            continue

        n_patches = len(cluster_patches)
        n_cols = n_rows = min(5, max(1, int(np.sqrt(n_patches))))  # Ensure at least 1 and no more than 5

        fig, axs = plt.subplots(n_rows, n_cols, figsize=(10, 10))
        fig.suptitle(f'Cluster {cluster}')

        # When there's a single subplot, ensure axs is an array
        if n_rows == 1 and n_cols == 1:
            axs = np.array([axs])  # Wrap single ax in an array
        elif n_rows == 1 or n_cols == 1:
            axs = axs.flatten()  # Flatten to handle 1D array of axs
        else:
            axs = axs.ravel()  # Flatten the axs array for easy iteration

        for ax, patch in zip(axs, cluster_patches):
            ax.imshow(patch)
            ax.axis('off')

        # Hide any unused subplots
        for ax in axs[len(cluster_patches):]:
            ax.axis('off')

        plt.show()


def merge_clusters_based_on_linkage(features, cluster_ids, distance_threshold):
    """
    使用 linkage 和 fcluster 根据距离阈值合并聚类。

    参数:
    - features: 特征数组。
    - cluster_ids: 初始聚类标签数组。
    - distance_threshold: 合并聚类的距离阈值。

    返回:
    - new_cluster_ids: 合并后的聚类标签数组。
    """
    # 计算聚类中心
    n_clusters = np.max(cluster_ids) + 1
    cluster_centers = np.array([features[cluster_ids == i].mean(axis=0) for i in range(n_clusters)])

    # 使用 linkage 函数进行层次聚类
    Z = linkage(cluster_centers, 'ward')

    # 使用 fcluster 函数根据距离阈值确定最终的聚类标签
    merged_cluster_labels = fcluster(Z, t=distance_threshold, criterion='distance')

    # 创建一个新的聚类标签映射，以保持标签的连续性
    new_cluster_ids_map = {old_id: new_id for old_id, new_id in zip(np.unique(cluster_ids), merged_cluster_labels)}

    # 应用新的聚类标签映射
    new_cluster_ids = np.array([new_cluster_ids_map[cluster_id] for cluster_id in cluster_ids])

    return new_cluster_ids


def recluster_within_clusters(features, merged_cluster_ids, n_clusters, method, **kwargs):
    """
    只对合并后的聚类进行再次聚类。

    参数:
    - features: 特征数组。
    - merged_cluster_ids: 合并后的聚类标签数组。
    - n_clusters: 再聚类的聚类数量。
    - method: 再聚类的方法。
    """
    unique_clusters = np.unique(merged_cluster_ids)
    final_cluster_ids = np.copy(merged_cluster_ids)

    for cluster_id in unique_clusters:
        # 选择当前大聚类中的特征
        current_features = features[merged_cluster_ids == cluster_id]

        # 只对那些包含多于一个初始聚类的样本的聚类进行再次聚类
        if len(current_features) > 1:  # 这里假设“合并的聚类”意味着它包含多个原始聚类的样本
            # 对当前大聚类进行细分聚类
            current_cluster_ids = cluster_features(current_features, method=method, n_clusters=n_clusters, **kwargs)
            unique_new_ids = np.unique(current_cluster_ids)

            # 生成新的聚类标签，确保它们是唯一的
            new_labels = cluster_id * 10 + unique_new_ids  # 这里使用*10是为了生成新的、独特的聚类标签
            for i, new_label in enumerate(new_labels):
                final_cluster_ids[merged_cluster_ids == cluster_id][
                    current_cluster_ids == unique_new_ids[i]] = new_label

    return final_cluster_ids


def annotate_and_display_image_with_clusters(image_path, cluster_ids, grid_size, edge_label="Edge", figsize=(10, 10)):
    # 加载图像并获取尺寸
    original_image = Image.open(image_path)
    original_width, original_height = original_image.size

    # 计算单个图块的宽度和高度
    patch_width = original_width // grid_size
    patch_height = original_height // grid_size

    # 初始化绘图
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(original_image)

    cluster_index = 0  # 跟踪非边缘图块的索引
    for i in range(grid_size):
        for j in range(grid_size):
            # 计算当前图块的左上角坐标
            x = j * patch_width
            y = i * patch_height

            # 确定标签：边缘图块使用edge_label，非边缘图块使用聚类标签
            # if i == 0 or i == grid_size - 1 or j == 0 or j == grid_size - 1:
            #     label = edge_label
            # else:
            if cluster_index < len(cluster_ids):
                label = str(cluster_ids[cluster_index])
                cluster_index += 1
            else:
                continue  # 如果超出了cluster_ids的长度，跳过剩余的绘制

            # 在图块中心放置标签文本
            ax.text(x + patch_width / 2, y + patch_height / 2, label, color='red', ha='center', va='center',
                    fontsize=12,
                    weight='bold')

            # 绘制图块的边框，所有图块都绘制边框，包括边缘图块
            rect = Rectangle((x, y), patch_width, patch_height, linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect)

    ax.axis('off')
    plt.tight_layout()
    plt.show()



def main(image_path, grid_size=8, n_clusters=10, color_bins=32, distance_threshold=27, recluster_n_clusters=2):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 初始化 FeatureExtractor 实例
    vit_extractor = FeatureExtractor(model_type='vit', device=device)
    inception_extractor = FeatureExtractor(model_type='inception', device=device)

    # 提取特征和原始图块
    features, original_patches = extract_features(vit_extractor, inception_extractor, image_path, color_bins,
                                                  grid_size)

    # 进行聚类
    cluster_ids = cluster_features(features, method='agglomerative', n_clusters=n_clusters)
    # merged_cluster_ids = merge_clusters_based_on_linkage(features, cluster_ids, distance_threshold)
    #
    # final_cluster_ids = recluster_within_clusters(features, merged_cluster_ids, n_clusters=recluster_n_clusters,
    #                                               method='agglomerative')
    #
    # # 可视化
    # visualize_feature_space(features, final_cluster_ids, method='tsne', title='Feature Space Visualization with t-SNE')
    # # print_clustered_patches(final_cluster_ids, original_patches)
    # annotate_and_display_image_with_clusters(image_path, final_cluster_ids, grid_size=grid_size, edge_label="Edge",
    #                                          figsize=(10, 10))

    return cluster_ids

if __name__ == "__main__":
    image_path = './bus.jpg'
    grid_size = 10
    main(image_path, grid_size=grid_size)
