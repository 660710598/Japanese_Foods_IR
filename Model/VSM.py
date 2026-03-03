import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

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

# Build TF-IDF Matrix
vectorizer = TfidfVectorizer()
# Combine 'Cleaned Title' and 'Cleaned Ingredients' for better search relevance
df['Search_Text'] = (df['Cleaned Title'].astype(str) + " ") * 3 + df['Cleaned Ingredients'].astype(str)
# Create TF-IDF matrix
tfidf_matrix = vectorizer.fit_transform(df['Search_Text'])
# Build Inverted Index
feature_names = vectorizer.get_feature_names_out()
inverted_index = {}

# Build Inverted Index
for col_idx, term in enumerate(feature_names):
    doc_indices = tfidf_matrix[:, col_idx].nonzero()[0]
    postings_list = [f"Doc{doc_id}" for doc_id in doc_indices]
    inverted_index[term] = {
        'DF': len(doc_indices),
        'Postings': postings_list
    }

# Define synonyms for query expansion
synonyms_dict = {
    'meat': ['pork', 'beef', 'chicken', 'meatball', 'belly', 'thigh', 'breast', 'wing'],
    'pork': ['bacon', 'belly', 'ham', 'shoulder', 'tonkatsu'],
    'beef': ['steak', 'lean', 'cow'],
    'chicken': ['poultry', 'thigh', 'breast', 'wing', 'karaage'],
    'fish': ['salmon', 'tuna', 'cod', 'bonito', 'filet'],
    'seafood': ['fish', 'shrimp', 'salmon', 'tuna', 'crab', 'squid'],
    'seaweed': ['nori', 'kombu', 'kelp', 'wakame'],
    'noodle': ['ramen', 'udon', 'soba', 'pasta', 'macaroni', 'spaghetti'],
    'rice': ['grain', 'bowl', 'donburi', 'sushi'],
    'bread': ['bun', 'loaf', 'sando', 'sandwich', 'dough', 'pizza', 'breadcrumb'],
    'veg': ['vegetable', 'carrot', 'cabbage', 'onion', 'garlic', 'potato', 'tomato', 'cucumber', 'broccoli', 'eggplant', 'spinach', 'zucchini'],
    'vegetable': ['carrot', 'cabbage', 'onion', 'garlic', 'potato', 'tomato', 'cucumber', 'broccoli', 'eggplant', 'spinach', 'zucchini'],
    'mushroom': ['shiitake', 'fungus', 'shimeji', 'enoki'],
    'onion': ['scallion', 'shallot', 'leek', 'chive'],
    'sauce': ['soy', 'mayo', 'mayonnaise', 'ketchup', 'mustard', 'worcestershire', 'teriyaki', 'dressing'],
    'soup': ['broth', 'stock', 'dashi', 'bouillon', 'miso'],
    'oil': ['fat', 'olive', 'sesame', 'frying'],
    'spicy': ['chili', 'pepper', 'curry', 'hot', 'wasabi', 'gochujang'],
    'sweet': ['sugar', 'honey', 'syrup', 'dessert', 'mirin', 'sweetener'],
    'herb': ['parsley', 'basil', 'cilantro', 'shiso', 'mint'],
    'spice': ['cumin', 'paprika', 'pepper', 'ginger'],
    'dairy': ['milk', 'butter', 'cheese', 'mozzarella', 'cream', 'cottage']
}

# Function to expand query using synonyms
def expand_query(query):
    expanded_terms = []
    for word in query.lower().split():
        expanded_terms.append(word) 
        
        if word in synonyms_dict:
            expanded_terms.extend(synonyms_dict[word])
    
    return " ".join(expanded_terms)

N=15
#Search Function
def search_recipes(original_query, top_n=N):
    expanded_q = expand_query(original_query)
    if expanded_q != original_query.lower():
        print(f"\n   [Query Expansion] : '{expanded_q}'")

    query_vec = vectorizer.transform([expanded_q])
    similarity_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = similarity_scores.argsort()[-top_n:][::-1]
    
    results = []
    for idx in top_indices:
        score = similarity_scores[idx]
        if score > 0:
            results.append({
                'Index': idx,
                'Title': df.iloc[idx]['Recipe Title'],
                'Score': round(score * 100, 2),
                'URL': df.iloc[idx]['Recipe URL'],
                'Cluster': df.iloc[idx]['Cluster_ID']+1 
            })
    return results

print("Grouping menu items with K-Means Clustering")
true_k = 6 
#K-Means Clustering
kmeans_model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1, random_state=0)
kmeans_model.fit(tfidf_matrix)
df['Cluster_ID'] = kmeans_model.labels_

order_centroids = kmeans_model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names_out()

print(" Top terms per cluster:")
for i in range(true_k):
    top_words = [terms[ind] for ind in order_centroids[i, :7]]
    print(f"   🍲 Cluster {i+1}: {', '.join(top_words)}")
print("="*60)

session_history = []
# Interactive Search Loop
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

    query_terms = user_query.lower().split() 
    print("\n" + "-"*60)
    print(f" Inverted Index : '{user_query}'")
    print("-" * 60)
    print(f"{'Term':<15} | {'DF':<5} | {'Postings List '}")
    print("-" * 60)
    # Display DF and Postings List for each term in the query
    for term in query_terms:
        if term in inverted_index:
            df_count = inverted_index[term]['DF']
            postings = inverted_index[term]['Postings'][:10] 
            postings_str = ", ".join(postings)
            if df_count > 10:
                postings_str += ", ..."
            print(f"{term:<15} | {df_count:<5} | [{postings_str}]")
        else:
            print(f"{term:<15} | 0     | [Not Found]")
    print("-" * 60)

    # Perform search and evaluate results
    search_results = search_recipes(user_query, top_n=N)
    if search_results:
        print(f"\n  find {len(search_results)} Menu that matches '{user_query}' ")
        for i, result in enumerate(search_results):
            print(f"[{i+1}] {result['Title']}")
            print(f"     Similarity: {result['Score']}% | Category: Cluster {result['Cluster']} | URL: {result['URL']}")

        expanded_eval_query = expand_query(user_query)
        eval_pattern = '|'.join(expanded_eval_query.split())

        relevant_docs = df[
            df['Cleaned Title'].str.contains(eval_pattern, case=False, na=False) | 
            df['Cleaned Ingredients'].str.contains(eval_pattern, case=False, na=False)
        ].index.tolist()
            
        retrieved_indices = [res['Index'] for res in search_results]
        p_score = precision_at_k(retrieved_indices, relevant_docs, N)
        r_score = recall_at_k(retrieved_indices, relevant_docs, N)
        ap_score = average_precision(retrieved_indices, relevant_docs)

        session_history.append({
            'query': user_query,
            'p': p_score,
            'r': r_score,
            'ap': ap_score
        })

        print("\n" + "="*60)
        print(f"Precision_{N}: {p_score:.2f} | Recall_{N}: {r_score:.2f} | Average Precision: {ap_score:.2f}")
        print("" + "="*60)
        print("\n")
        ap_list = []
        print("menu            | precision    | recall     | average precision")
        print("-" * 60)
        for history in session_history:
           
            print(f"{history['query']:<15} | {history['p']:<12.2f} | {history['r']:<10.2f} | {history['ap']:.2f}")
            ap_list.append(history['ap'])
        print("-" * 60)
        map_score = np.mean(ap_list)
        print(f"🏆 MAP (Mean Average Precision) : {map_score:.2f}")
    else:
        print(f"\n  No results found for '{user_query}' ")
        
    print("-" * 60)