import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from sklearn.manifold import TSNE
import seaborn as sns

# 加载CUI embedding
cui_embed = torch.load('embeddings/cui-embedding-500.pt')
cui_codes = cui_embed['cui']
embeddings = cui_embed['data']

# 转换为NumPy数组以便使用scipy
embeddings_np = embeddings.numpy()

linked = linkage(embeddings_np, method='ward')

# 绘制聚类树状图
plt.figure(figsize=(15, 10))
dendrogram(linked,
           orientation='top',
           labels=cui_codes,
           distance_sort='descending',
           show_leaf_counts=True)
plt.title('Hierarchical Clustering Dendrogram of CUI Codes')
plt.xlabel('CUI Codes')
plt.ylabel('Distance')
plt.xticks(rotation=90, fontsize=8)  # 旋转标签以便阅读
plt.tight_layout()
plt.savefig('cluster_results/cui_dendrogram.png', dpi=300)

# 确定合适的聚类数量
# 可以从树状图视觉上确定，或者使用不同的阈值进行实验
# 这里我们尝试使用距离阈值或指定聚类数量

# 方法1：基于距离阈值划分聚类
max_dist = 5.0  # 可以根据树状图结果调整这个值
clusters_by_distance = fcluster(linked, max_dist, criterion='distance')

# 方法2：直接指定聚类数量
num_clusters = 100  # 可以根据业务需求或树状图结果调整
clusters_by_num = fcluster(linked, num_clusters, criterion='maxclust')

# 创建结果数据框
result_df = pd.DataFrame({
    'cui_code': cui_codes,
    'cluster_by_distance': clusters_by_distance,
    'cluster_by_number': clusters_by_num
})

# 展示每个聚类的CUI代码
print("基于距离阈值的聚类结果:")
for cluster_id in np.unique(clusters_by_distance):
    cluster_members = result_df[result_df['cluster_by_distance'] == cluster_id]['cui_code'].tolist()
    print(f"聚类 {cluster_id}: {cluster_members}")

print("\n基于指定聚类数量的聚类结果:")
for cluster_id in np.unique(clusters_by_num):
    cluster_members = result_df[result_df['cluster_by_number'] == cluster_id]['cui_code'].tolist()
    print(f"聚类 {cluster_id}: {cluster_members}")

# 保存结果
result_df.to_csv('cluster_results/cui_clustering_results.csv', index=False)

# 使用t-SNE进行降维可视化
tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings_np)-1))
embeddings_2d = tsne.fit_transform(embeddings_np)

# 创建可视化DataFrame
viz_df = pd.DataFrame({
    'cui_code': cui_codes,
    'x': embeddings_2d[:, 0],
    'y': embeddings_2d[:, 1],
    'cluster': clusters_by_num
})

# 可视化聚类结果
plt.figure(figsize=(12, 10))
sns.scatterplot(x='x', y='y', hue='cluster', data=viz_df, palette='viridis', s=100)

# 标注CUI代码
for i, row in viz_df.iterrows():
    plt.text(row['x']+0.01, row['y']+0.01, row['cui_code'], fontsize=8)

plt.title('t-SNE Visualization of CUI Codes Clustering')
plt.xlabel('t-SNE dimension 1')
plt.ylabel('t-SNE dimension 2')
plt.legend(title='Cluster')
plt.savefig('cluster_results/cui_clustering_tsne.png', dpi=300)

# 计算聚类的解释性 - 找出每个聚类的中心点
cluster_centers = {}
for cluster_id in np.unique(clusters_by_num):
    cluster_indices = np.where(clusters_by_num == cluster_id)[0]
    cluster_embeddings = embeddings_np[cluster_indices]
    center = np.mean(cluster_embeddings, axis=0)
    cluster_centers[cluster_id] = center

# 找出与中心最接近的CUI代码作为聚类代表
cluster_representatives = {}
for cluster_id, center in cluster_centers.items():
    cluster_indices = np.where(clusters_by_num == cluster_id)[0]
    distances = np.linalg.norm(embeddings_np[cluster_indices] - center, axis=1)
    closest_idx = cluster_indices[np.argmin(distances)]
    cluster_representatives[cluster_id] = cui_codes[closest_idx]

print("\n每个聚类的代表性CUI代码:")
for cluster_id, rep_cui in cluster_representatives.items():
    print(f"聚类 {cluster_id}: {rep_cui}")

# 如果数据规模较大，可以补充计算轮廓系数等指标来评估聚类质量