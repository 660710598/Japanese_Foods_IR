import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

input_filename = 'Japan_Food_Ingredients_Full.csv'
output_filename = 'Japan_Food_Ingredients_Cleaned.csv'

try:
    df = pd.read_csv(input_filename, encoding='utf-8-sig')
except FileNotFoundError:
    print(f"{input_filename} not found.")
    exit()

lemmatizer = WordNetLemmatizer()
stopwords = set(stopwords.words('english'))

custom_culinary_stopwords = {
    'cup', 'cups', 'teaspoon', 'teaspoons', 'tsp', 'tablespoon', 'tablespoons', 'tbsp',
    'ounce', 'ounces', 'oz', 'gram', 'grams', 'g', 'ml', 'liter', 'liters',
    'pinch', 'dash', 'piece', 'pieces', 'slice', 'slices', 'clove', 'cloves',
    'chopped', 'minced', 'sliced', 'diced', 'peeled', 'grated', 'fresh', 'dried',
    'to', 'taste', 'optional', 'divided', 'large', 'small', 'medium', 'whole', 'half',
    'serving', 'minute', 'min', 'tb', 'lb', 'pound', 
    'people', 'package', 'packet', 'bag', 'substitute'
}

all_stopwords = stopwords.union(custom_culinary_stopwords)

def clean_ingredients(text): 
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = word_tokenize(text)

    cleaned_tokens = []
    for token in tokens:
        if token not in all_stopwords and len(token) > 1:
            # Lemmatization เพื่อให้คำที่มีรูปแบบต่างๆ กลับมาเป็นรูปแบบพื้นฐาน เช่น "chopped" -> "chop"
            lemma = lemmatizer.lemmatize(token)
            cleaned_tokens.append(lemma)
    return ' '.join(cleaned_tokens)

# ใช้ฟังก์ชัน clean_ingredients กับคอลัมน์ 'Ingredients' เพื่อสร้างคอลัมน์ใหม่ 'Cleaned Ingredients'
df['Cleaned Ingredients'] = df['Ingredients'].apply(clean_ingredients)

df.to_csv(output_filename, index=False, encoding='utf-8-sig')
print(f"Data cleaning completed. Cleaned data saved to {output_filename}.")