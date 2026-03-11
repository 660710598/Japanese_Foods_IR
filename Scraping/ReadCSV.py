import csv
import requests
import re
import time
from bs4 import BeautifulSoup


input_filename = 'Japan_Food_Links_100.csv' 
output_filename = 'Japan_Food_Ingredients_Full.csv'

full_recipe_data = [["Recipe Title", "Recipe URL", "Ingredients"]]


print("Loading recipe links from CSV file...")


#  เปิดไฟล์ CSV เพื่ออ่านข้อมูล
try:
    with open(input_filename, 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        header = next(reader) 
        recipes_to_scrape = list(reader) 
except FileNotFoundError:
    print(f"{input_filename} not found.")
    exit()

for index, row in enumerate(recipes_to_scrape):
    if len(row) < 2:
        continue
        
    title = row[0]
    url = row[1]
    
    print(f"[{index + 1}/{len(recipes_to_scrape)}] From: {title}")
    
    if url == "No URL found":
        full_recipe_data.append([title, url, "No Data"])
        continue

    try:
        res = requests.get(url, timeout=10) 
        res.encoding = "utf-8"
        soup = BeautifulSoup(res.text, 'html.parser')
        
        ingredients_list = []
        ingredient_tags = soup.find_all(class_=re.compile(r'ingredient', re.IGNORECASE))
        
        for tag in ingredient_tags:
            text = tag.get_text(separator=' ', strip=True)
            
            if text and len(text) > 1 and text not in ingredients_list:
                ingredients_list.append(text)
                
        ingredients_text = " ".join(ingredients_list)
        
        if not ingredients_text:
             ingredients_text = "No Ingredients Found"
        full_recipe_data.append([title, url, ingredients_text])
        
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        full_recipe_data.append([title, url, "Error"])
    time.sleep(1)

# บันทึกข้อมูลทั้งหมดลงไฟล์ CSV ใหม่
print("Saving full recipe data to CSV file...")
with open(output_filename, 'w', newline='', encoding='utf-8-sig') as f:
    writer = csv.writer(f)
    writer.writerows(full_recipe_data)

print(f"Data scraping completed. Full recipe data saved to {output_filename}.")