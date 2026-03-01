import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Evaluation Metrics
def precision_at_k(retrieved, relevant, k):
    retrieved_k = retrieved[:k]
    if not retrieved_k: return 0.0
    relevant_and_retrieved = set(retrieved_k).intersection(set(relevant))
    return len(relevant_and_retrieved) / k

def recall_at_k(retrieved, relevant, k):
    retrieved_k = retrieved[:k]
    if not relevant: return 0.0
    relevant_and_retrieved = set(retrieved_k).intersection(set(relevant))
    return len(relevant_and_retrieved) / len(relevant)

def average_precision(retrieved, relevant):
    if not relevant: return 0.0
    score = 0.0
    num_hits = 0.0
    for i, p in enumerate(retrieved):
        if p in relevant:
            num_hits += 1.0
            score += num_hits / (i + 1.0)
    return score / len(relevant)

# Load dataset
filename = 'Japan_Food_Ingredients_Cleaned.csv'
try:
    df = pd.read_csv(filename, encoding='utf-8-sig')
    df['Cleaned Ingredients'] = df['Cleaned Ingredients'].fillna('')
except FileNotFoundError:
    print(f"{filename} not found.")
    exit()

vectorizer = TfidfVectorizer()
# สร้างคอลัมน์ 'Search_Text' โดยรวมชื่อเมนูและส่วนผสมที่ทำความสะอาดแล้วเข้าด้วยกัน เพื่อให้การค้นหามีประสิทธิภาพมากขึ้น
# ชื่อเมนูถูกคูณด้วย 3 เพื่อเพิ่มน้ำหนักให้กับชื่อเมนูในการคำนวณความคล้ายคลึง เพราะผู้ใช้มักจะใส่ชื่อเมนูหรือวัตถุดิบหลักในการค้นหา
df['Search_Text'] = (df['Recipe Title'].astype(str) + " ") * 3 + df['Cleaned Ingredients'].astype(str)
# สร้างเมทริกซ์ TF-IDF จากคอลัมน์ 'Search_Text' ซึ่งจะใช้ในการคำนวณความคล้ายคลึงระหว่างคำค้นหาของผู้ใช้กับข้อมูลในฐานข้อมูล
tfidf_matrix = vectorizer.fit_transform(df['Search_Text'])

def search_recipes(query, top_n=5):
    query_vec = vectorizer.transform([query])
    similarity_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = similarity_scores.argsort()[-top_n:][::-1]
    
    # สร้างลิสต์สำหรับเก็บผลลัพธ์ที่มีคะแนนความคล้ายคลึงมากกว่า 0 เท่านั้น 
    for idx in top_indices:
        score = similarity_scores[idx]
        # กรองเอาเฉพาะอันที่คะแนนมากกว่า 0 
        if score > 0:
            results.append({
                'Index': idx,#เก็บดัชนีของเอกสารที่ถูกดึงมาแสดงผลลัพธ์ เพื่อใช้ในการคำนวณ Precision และ Recall ในภายหลัง
                'Title': df.iloc[idx]['Recipe Title'],
                'Score': round(score * 100, 2), # แปลงเป็นเปอร์เซ็นต์ให้ดูง่าย
                'URL': df.iloc[idx]['Recipe URL']
            })
    return results

#สร้างตัวแปร List สำหรับเก็บ "ประวัติ" การประเมินผลของทุกคำค้นหา
session_history = []
while True:
    print("Welcome to Japanese Foods🍣")
    print("(type 'q' or 'exit' to quit)")
    user_query = input("\n  Enter your search query: ")
    
    if user_query.lower() in ['q', 'exit', 'quit']:
        print("\n Thank you for using the search engine.")
        break
        
    if not user_query.strip():
        print("Please enter a valid search query. ")
        continue

    # ค้นหาข้อมูล
    K=5
    search_results = search_recipes(user_query, top_n=K)

    # แสดงผลลัพธ์
    if search_results:
        print(f"\n  find {len(search_results)} Menu that matches '{user_query}' ")
        for i, result in enumerate(search_results):
            print(f"[{i+1}] {result['Title']}")
            print(f"     Similarity: {result['Score']}% | URL: {result['URL']}")
            
            # คำนวณเอกสารที่เกี่ยวข้องโดยการตรวจสอบว่าชื่อเมนูหรือส่วนผสมที่ทำความสะอาดแล้วมีคำค้นหาของผู้ใช้หรือไม่
        relevant_docs = df[
            df['Recipe Title'].str.contains(user_query, case=False, na=False) | 
            df['Cleaned Ingredients'].str.contains(user_query, case=False, na=False)
        ].index.tolist()
            
        # ดึงดัชนีของเอกสารที่ถูกดึงมาแสดงผลลัพธ์ เพื่อใช้ในการคำนวณ Precision และ Recall ในภายหลัง
        retrieved_indices = [res['Index'] for res in search_results]

        p_score = precision_at_k(retrieved_indices, relevant_docs, K)
        r_score = recall_at_k(retrieved_indices, relevant_docs, K)
        ap_score = average_precision(retrieved_indices, relevant_docs)

        # เก็บผลลัพธ์ของคำค้นหานี้เข้าประวัติรวม
        session_history.append({
            'query': user_query,
            'p': p_score,
            'r': r_score,
            'ap': ap_score
        })
        print("\n" + "="*60)
        print(f"     Precision@{K}: {p_score:.2f} | Recall@{K}: {r_score:.2f} | Average Precision: {ap_score:.2f}")
        print("\n" + "="*60)
        ap_list = []
        # วนลูปปริ้นท์ประวัติการค้นหาทั้งหมดที่เคยค้นมา
        for history in session_history:
            print(f"{history['query']:<15} | {history['p']:<12.2f} | {history['r']:<10.2f} | {history['ap']:.2f}")
            ap_list.append(history['ap'])
        print("-" * 60)
        map_score = np.mean(ap_list)
        print(f"🏆 MAP (Mean Average Precision) : {map_score:.2f}")
    else:
        print(f"\n  No results found for '{user_query}' ")
        
    print("-" * 50)