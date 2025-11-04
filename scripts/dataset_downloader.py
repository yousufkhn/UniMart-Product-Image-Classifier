import os
import requests
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()
ACCESS_KEY = os.getenv("UNSPLASH_ACCESS_KEY")

categories = [
    "electronics",
    "books",
    "clothing",
    "footwear",
    "accessories",
    "stationery",
    "home decor",
    "sports and fitness",
    "personal care",
    "bags and luggage",
    "furniture",
    "food items"
]

def download_images(query, folder, count=200):
    os.makedirs(folder, exist_ok=True)
    url = f"https://api.unsplash.com/search/photos"
    headers = {"Accept-Version": "v1", "Authorization": f"Client-ID {ACCESS_KEY}"}

    page = 1
    downloaded = 0

    print(f"\nDownloading {query} images...")
    while downloaded < count:
        params = {"query": query, "page": page, "per_page": 30}
        response = requests.get(url, headers=headers, params=params)
        data = response.json()

        for i, photo in enumerate(data["results"]):
            img_url = photo["urls"]["small"]
            img_data = requests.get(img_url).content
            file_path = os.path.join(folder, f"{query}_{page}_{i}.jpg")
            with open(file_path, "wb") as f:
                f.write(img_data)
            downloaded += 1
            if downloaded >= count:
                break
        page += 1
        if not data["results"]:
            break
    print(f"✅ Downloaded {downloaded} images for {query}")

def main():
    for category in categories:
        safe_name = category.replace(" ", "_")
        folder = f"dataset/{safe_name}"
        download_images(category, folder, count=200)  # 200 per category

if __name__ == "__main__":
    main()
