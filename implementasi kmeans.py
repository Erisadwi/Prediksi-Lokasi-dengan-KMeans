import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------- DATA DESA ---------------------
data = pd.DataFrame({
    'nama_desa': ['Baosan Kidul', 'Wonodadi', 'Sendang', 'Mrayan', 'Binade', 'Baosan Lor', 'Ngrayun', 'Temon', 'Selur', 'Cepoko', 'Gedangan'],
    'jumlah_penduduk': [6861, 4569, 3647, 7028, 2907, 7987, 7455, 3471, 6897, 6251, 4712],
    'aksesibilitas': [65, 76, 77, 59, 65, 56, 48, 32, 54, 52, 80],
    'jarak_rs_terdekat': [33.3, 33.8, 38.7, 30.2, 31.7, 28.5, 27, 19.5, 30.4, 23.2, 37.9],
    'fasilitas_kesehatan': [0, 0, 0, 0, 0, 3, 1, 0, 0, 1, 0],
})

features_kmeans = data[['jumlah_penduduk', 'aksesibilitas', 'jarak_rs_terdekat', 'fasilitas_kesehatan']]

silhouette_scores = []
range_clusters = range(2, 6)

for k in range_clusters:
    kmeans = KMeans(n_clusters=k, random_state=42)
    cluster_labels = kmeans.fit_predict(features_kmeans)
    score = silhouette_score(features_kmeans, cluster_labels)
    silhouette_scores.append(score)
    print(f"Jumlah Cluster: {k} - Silhouette Score: {score:.4f}")

silhouette_df = pd.DataFrame({
    'Jumlah Cluster': list(range_clusters),
    'Silhouette Score': silhouette_scores
})

plt.plot(range_clusters, silhouette_scores, marker='o')
plt.title('Silhouette Score vs Jumlah Cluster')
plt.xlabel('Jumlah Cluster')
plt.ylabel('Silhouette Score')
plt.grid(True)
plt.show()

best_k = range_clusters[np.argmax(silhouette_scores)]
print(f"\nJumlah cluster terbaik berdasarkan silhouette score: {best_k}")

kmeans_final = KMeans(n_clusters=best_k, random_state=42)
data['cluster'] = kmeans_final.fit_predict(features_kmeans)

for i in range(best_k):
    print(f"\nKlaster {i}:")
    print(data[data['cluster'] == i]['nama_desa'].tolist())

silhouette_html = silhouette_df.to_html(index=False)

cluster_html = data.sort_values(by='cluster')[[
    'nama_desa', 'jumlah_penduduk', 'aksesibilitas', 'jarak_rs_terdekat', 'fasilitas_kesehatan', 'cluster'
]].to_html(index=False)


with open("hasil_klaster.html", "w") as file:
    file.write("<h2>Hasil Silhouette Score untuk Berbagai Jumlah Cluster</h2>")
    file.write(silhouette_html)
    file.write("<h2>Hasil Klastering K-Means Desa di Kecamatan Ngrayun</h2>")
    file.write(cluster_html)

print("\nTabel HTML telah disimpan sebagai 'hasil_klaster.html'")
