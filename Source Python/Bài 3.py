import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Đọc dữ liệu từ file 'results.csv'
data = pd.read_csv('results.csv')

# Chọn các chỉ số đặc trưng
features = [
    'stats_passing_Passes Completed',  # Kiểm soát bóng
    'stats_standard_Goals',            # Số bàn thắng
    'stats_defense_Tackles',           # Khả năng tắc bóng
    'stats_standard_Assists',          # Số đường kiến tạo
    'stats_shooting_Shots on target/90' # Số cú sút trúng đích mỗi 90 phút
]

# Lọc dữ liệu với các chỉ số đặc trưng và xử lý giá trị thiếu
data_features = data[features].fillna(0)

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_features)

# Phương pháp Elbow để xác định số lượng cụm tối ưu
inertia = []
k_range = range(1, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data_scaled)
    inertia.append(kmeans.inertia_)

# Vẽ biểu đồ Elbow
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(k_range, inertia, marker='o')
plt.title('Phương pháp Elbow')
plt.xlabel('Số lượng cụm (k)')
plt.ylabel('Inertia')
plt.grid(True)
plt.savefig('elbow_plot.png')

# Giả sử số cụm tối ưu là 4 (cần xem biểu đồ để xác định)
optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
clusters = kmeans.fit_predict(data_scaled)

# Giảm chiều dữ liệu bằng PCA xuống 2 chiều
pca = PCA(n_components=2)
data_pca = pca.fit_transform(data_scaled)

# Tạo DataFrame cho dữ liệu PCA
df_pca = pd.DataFrame(data_pca, columns=['PC1', 'PC2'])
df_pca['Cluster'] = clusters

# Vẽ biểu đồ phân cụm 2D
plt.subplot(1, 2, 2)
sns.scatterplot(x='PC1', y='PC2', hue='Cluster', data=df_pca, palette='viridis', s=100)
plt.title('Biểu đồ phân cụm 2D')
plt.xlabel('Thành phần chính 1 (PC1)')
plt.ylabel('Thành phần chính 2 (PC2)')
plt.savefig('cluster_plot.png')

# Lưu toàn bộ hình ảnh
plt.tight_layout()
plt.savefig('combined_plots.png')
