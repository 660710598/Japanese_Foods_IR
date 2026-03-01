import csv
import requests
from bs4 import BeautifulSoup
import time
import re

input_filename = 'Japan_Food_Links_50.csv' 
output_filename = 'Japan_Food_Ingredients_Full.csv'

full_recipe_data = [["Recipe Title", "Recipe URL", "Ingredients"]]

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

print("Loading recipe links from CSV file...")


#  เปิดไฟล์ CSV เพื่ออ่านข้อมูล
try:
    with open(input_filename, 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        header = next(reader) # ข้ามบรรทัด Header (Recipe Title, Recipe URL)
        recipes_to_scrape = list(reader) # เก็บข้อมูลที่เหลือลง List
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
        res = requests.get(url, headers=headers, timeout=10) # ใส่ timeout กันเว็บค้าง
        res.encoding = "utf-8"
        soup = BeautifulSoup(res.text, 'html.parser')
        
        ingredients_list = []
        # ค้นหาแท็กที่มีคลาสชื่อที่เกี่ยวข้องกับส่วนผสม (ingredient) โดยใช้ regex เพื่อความยืดหยุ่น
        ingredient_tags = soup.find_all(class_=re.compile(r'ingredient', re.IGNORECASE))
        
        for tag in ingredient_tags:
            # ดึงข้อความจากแท็ก และทำความสะอาดข้อมูลด้วย regex เพื่อเอาเฉพาะตัวอักษรและตัวเลข (รวมถึงเครื่องหมายที่จำเป็น)
            text = tag.get_text(separator=' ', strip=True)
            text = re.sub(r'[^\w\sก-๙/.,-]', '', text) 
            
            if text and len(text) > 1 and text not in ingredients_list:
                ingredients_list.append(text)
                
        # นำส่วนผสมทั้งหมดมาต่อกันเป็นข้อความเดียว คั่นด้วยช่องว่าง เพื่อให้ง่ายต่อการทำ IR Tokenize
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