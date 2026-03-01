import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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
    
    results = []
    for idx in top_indices:
        score = similarity_scores[idx]
        # กรองเอาเฉพาะอันที่คะแนนมากกว่า 0 
        if score > 0:
            results.append({
                'Title': df.iloc[idx]['Recipe Title'],
                'Score': round(score * 100, 2), # แปลงเป็นเปอร์เซ็นต์ให้ดูง่าย
                'URL': df.iloc[idx]['Recipe URL']
            })
            
    return results

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
    search_results = search_recipes(user_query, top_n=5)

    # แสดงผลลัพธ์
    if search_results:
        print(f"\n พบ {len(search_results)} เมนูที่ตรงกับ '{user_query}' ")
        for i, result in enumerate(search_results):
            print(f"[{i+1}] {result['Title']}")
            print(f"     Similarity: {result['Score']}%")
            print(f"     Link: {result['URL']}")
    else:
        print(f"\n  No results found for '{user_query}' ")
        
    print("-" * 50)