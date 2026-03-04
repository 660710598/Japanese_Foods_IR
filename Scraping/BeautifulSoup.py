from bs4 import BeautifulSoup
import requests
import csv
from urllib.parse import urljoin


base_url = "https://cookpad.com"
search_base_url = "https://cookpad.com/eng/search/japan"

recipe_data = [["Recipe Title", "Recipe URL"]] 
amount = 100 
count = 0 
page = 1 

print("Fetching recipe links from Cookpad website...")

while count < amount:
    current_url = f"{search_base_url}?page={page}"
    res = requests.get(current_url)
    res.encoding = "utf-8"
    
    if res.status_code != 200:
        print(f"Failed to retrieve {current_url} (Status code: {res.status_code})")
        break
        
    soup = BeautifulSoup(res.text, 'html.parser')
    courses = soup.find_all('h2')
        
    for course in courses:
        if count >= amount:
            break 
            
        title = course.get_text(strip=True)
        if not title:
            continue
            
        a_tag = course.find('a') 
        if not a_tag:
            a_tag = course.find_parent('a')
            
        if a_tag and 'href' in a_tag.attrs:
            recipe_url = urljoin(base_url, a_tag['href'])
        else:
            recipe_url = "No URL found"

        recipe_data.append([title, recipe_url])
        count += 1
    # จบ 1 หน้า ให้บวกเลขหน้าเพิ่ม เพื่อเตรียมไปดึงหน้าต่อไป
    page += 1

# 2. บันทึกข้อมูลลง CSV
filename = 'Japan_Food_Links_50.csv'
with open(filename, 'w', newline='', encoding='utf-8-sig') as f:
    writer = csv.writer(f)
    writer.writerows(recipe_data)
print(f"Successfully scraped {count} recipes! Data saved to file {filename}.")

