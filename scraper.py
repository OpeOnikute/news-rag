import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
from datetime import datetime
import json
import os

from dotenv import load_dotenv

from tags import strip_tags

class WordpressScraper:
    def __init__(self):
        self.base_url = os.environ.get("WP_BASE_URL")
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        self.data = []

    def get_total_pages(self, per_page=100):
        """Get total number of available pages"""
        url = f"{self.base_url}/posts"
        params = {
            'per_page': per_page,
            'page': 1
        }
        response = requests.get(url, headers=self.headers, params=params)
        total_posts = int(response.headers.get('X-WP-Total', 0))
        total_pages = int(response.headers.get('X-WP-TotalPages', 0))
        return total_posts, total_pages
    
    def get_author_name(self, author_id):
        """Get author name from author ID"""
        url = f"{self.base_url}/users/{author_id}"
        response = requests.get(url, headers=self.headers)
        if response.status_code == 200:
            return response.json().get('name', '')
        return ''

    def get_categories(self, category_ids):
        """Get category names from category IDs"""
        categories = []
        for cat_id in category_ids:
            url = f"{self.base_url}/categories/{cat_id}"
            response = requests.get(url, headers=self.headers)
            if response.status_code == 200:
                categories.append(response.json().get('name', ''))
        return ",".join(categories)
    
    def process_article(self, article):
        """Process individual article JSON data"""

        # Write to a text file and reference the name in the JSON
        content = article.get('content', {}).get('rendered', '')
        filename = f"posts/{article.get('slug')}.txt"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(strip_tags(content))

        article_data = {
            'id': article.get('id'),
            'url': article.get('link', ''),
            'title': article.get('title', {}).get('rendered', ''),
            'date': article.get('date', ''),
            'modified_date': article.get('modified', ''),
            'author': article.get('author_string', ''),
            'content_file': filename,
            'excerpt': strip_tags(article.get('excerpt', {}).get('rendered', '')),
            'categories': self.get_categories(article.get('categories', [])),
            'featured_media_url': article.get('featured_media_url', ''),
            'scraped_at': datetime.now().isoformat()
        }
        return article_data

    def scrape_articles(self, start_page=1, end_page=None, per_page=100):
        """Scrape articles using WordPress REST API"""
        total_posts, total_pages = self.get_total_pages(per_page)
        print(f"Found {total_posts} total posts across {total_pages} pages")
        
        if end_page is None or end_page > total_pages:
            end_page = total_pages

        for page in range(start_page, end_page + 1):
            print(f"Scraping page {page} of {end_page}...")
            url = f"{self.base_url}/posts"
            params = {
                'page': page,
                'per_page': per_page,
                'context': 'view',
                '_embed': 1  # Include embedded content like featured images
            }
            
            try:
                response = requests.get(url, headers=self.headers, params=params)
                response.raise_for_status()
                articles = response.json()
                
                for article in articles:
                    article_data = self.process_article(article)
                    self.data.append(article_data)
                    
                time.sleep(1)  # Be nice to the API
                
            except requests.exceptions.RequestException as e:
                print(f"Error on page {page}: {str(e)}")
                continue
    
    def save_to_json(self, filename='data/total.json'):
        """Save scraped data to JSON file"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)
    
def main():
    scraper = WordpressScraper()
    scraper.scrape_articles(1, 2, 1)
    scraper.save_to_json()

if __name__ == "__main__":
    load_dotenv()
    main()