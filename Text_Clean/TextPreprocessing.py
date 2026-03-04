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
    'cup', 'cups', 'teaspoon', 'teaspoons', 'tsp', 'tablespoon', 'tablespoons', 'tbsp', 'tbs', 'tb',
    'ounce', 'ounces', 'oz', 'gram', 'grams', 'g', 'gr', 'ml', 'liter', 'liters', 'kg', 'lb', 'pound', 'pounds',
    'pinch', 'dash', 'piece', 'pieces', 'pcs', 'slice', 'slices', 'clove', 'cloves',
    'package', 'packages', 'packet', 'packets', 'pack', 'packs', 'bag', 'bags', 'box', 'boxes',
    'chunk', 'chunks', 'drop', 'drops', 'handful',
    'serving', 'servings', 'people', 'yield', 'yields',
    'minute', 'minutes', 'min', 'mins', 'hour', 'hours', 'hr', 'hrs',
    'chopped', 'minced', 'sliced', 'diced', 'peeled', 'grated', 'shredded', 'crushed', 'mashed',
    'cut', 'ground', 'fresh', 'dried', 'frozen', 'cooked', 'fried', 'steamed', 'roasted', 'baked',
    'pickled', 'marinated', 'finely', 'thinly', 'lightly', 'beaten', 'melted', 'softened',
    'room', 'temp', 'temperature', 'cold', 'hot', 'warm', 'boiling',
    'large', 'small', 'medium', 'whole', 'half', 'quarter', 'thick', 'thin',
    'optional', 'divided', 'substitute', 'taste', 'needed', 'used', 'required',
    'purpose', 'all', 'instant', 'quality', 'extra', 'plus', 'pure', 'raw',
    'ingredient', 'ingredients', 'recipe', 'recipes', 'direction', 'directions',
    'instruction', 'instructions', 'method', 'step', 'steps', 
    'garnish', 'topping', 'toppings', 'base', 'cooking', 'baking', 'food'
}

all_stopwords = stopwords.union(custom_culinary_stopwords)

def clean_title(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', ' ', text) # เก็บไว้แค่ตัวอักษรภาษาอังกฤษ
    tokens = word_tokenize(text) # แปลงข้อความเป็น tokens 
    
    cleaned_tokens = []
    for token in tokens:
        # ตัด stopwords พื้นฐาน และแปลงพหูพจน์เป็นเอกพจน์ (lemmatization)
        if token not in stopwords and len(token) > 1:
            lemma = lemmatizer.lemmatize(token)
            cleaned_tokens.append(lemma)
    return ' '.join(cleaned_tokens)

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

# ใช้ฟังก์ชัน clean_title กับคอลัมน์ 'Recipe Title' เพื่อสร้างคอลัมน์ใหม่ 'Cleaned Title'
df['Cleaned Title'] = df['Recipe Title'].apply(clean_title)

# ใช้ฟังก์ชัน clean_ingredients กับคอลัมน์ 'Ingredients' เพื่อสร้างคอลัมน์ใหม่ 'Cleaned Ingredients'
df['Cleaned Ingredients'] = df['Ingredients'].apply(clean_ingredients)

df.to_csv(output_filename, index=False, encoding='utf-8-sig')
print(f"Data cleaning completed. Cleaned data saved to {output_filename}.")