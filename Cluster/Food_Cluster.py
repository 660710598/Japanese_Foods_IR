import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

filename = 'Japan_Food_Ingredients_Cleaned.csv'
try:
    df = pd.read_csv(filename, encoding='utf-8-sig')
    df['Cleaned Ingredients'] = df['Cleaned Ingredients'].fillna('')
except FileNotFoundError:
    print(f"❌ ไม่พบไฟล์ {filename} กรุณาตรวจสอบชื่อไฟล์อีกครั้ง")
    exit()

print("TF-IDF Matrix...")
vectorizer = TfidfVectorizer()
df['Search_Text'] = (df['Recipe Title'].astype(str) + " ") * 3 + df['Cleaned Ingredients'].astype(str)
tfidf_matrix = vectorizer.fit_transform(df['Search_Text'])


true_k = 6
print(f"🤖 กำลังจัดกลุ่มเมนูอาหารเป็น {true_k} กลุ่ม ด้วย K-Means...")
kmeans_model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1, random_state=0)
kmeans_model.fit(tfidf_matrix)

df['Cluster'] = kmeans_model.labels_

print("\n" + "="*60)
print("✅ ผลลัพธ์การจัดกลุ่ม (คำศัพท์ที่เป็นตัวแทนของแต่ละหมวดหมู่):")
print("="*60)

order_centroids = kmeans_model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names_out()

for i in range(true_k):
    top_words = [terms[ind] for ind in order_centroids[i, :7]]
    print(f"🍲 Cluster {i}: {', '.join(top_words)}")
print("="*60)

print("\n🎨 กำลังวาดกราฟ Scatter Plot เพื่อดูการกระจายตัวของข้อมูล...")
pca = PCA(n_components=2, random_state=0)
reduced_features = pca.fit_transform(tfidf_matrix.toarray())
centroids_2d = pca.transform(kmeans_model.cluster_centers_)

df_pca = pd.DataFrame(reduced_features, columns=['PCA1', 'PCA2'])
df_pca['Cluster'] = df['Cluster']
df_pca['Recipe Title'] = df['Recipe Title']

plt.figure(figsize=(10, 7))
sns.set_theme(style="whitegrid")

sns.scatterplot(
    x='PCA1', 
    y='PCA2',
    hue='Cluster', 
    palette=sns.color_palette('Set2', true_k), 
    data=df_pca,
    legend='full',
    alpha=0.7, 
    s=100 
)

plt.scatter(
    centroids_2d[:, 0], 
    centroids_2d[:, 1], 
    marker='X',        
    s=250,             
    color='red',        
    edgecolor='black',  
    label='Centroids'   
)

plt.title(f'Japanese Recipes Clustering (K-Means, K={true_k}) with PCA', fontsize=16, fontweight='bold', pad=15)
plt.xlabel('PCA Component 1', fontsize=12)
plt.ylabel('PCA Component 2', fontsize=12)

plt.legend(title='Cluster & Centroid', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

print("✅ วาดกราฟเสร็จสมบูรณ์! (หน้าต่างรูปภาพกำลังแสดงขึ้นมา)")
plt.show()