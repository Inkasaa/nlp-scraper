"""
scraper_news.py
Purpose: Script for scraping news articles from the web.
"""

import requests
from bs4 import BeautifulSoup
import json
import re
import pandas as pd
import uuid
from datetime import datetime


def build_dataset(target_size=300):
    all_articles = []
    seen_urls = set()

    print("Collecting URLs...")

    aland_urls = get_aland_article_links()
    yle_urls = get_yle_article_links(limit=300)

    all_urls = aland_urls + yle_urls

    print(f"Total URLs collected: {len(all_urls)}")

    for i, url in enumerate(all_urls):
        if url in seen_urls:
            continue

        print(f"\n[{len(all_articles)+1}] Scraping: {url}")

        # Choose correct scraper
        if "yle.fi" in url:
            data = scrape_yle_article(url)
        else:
            data = scrape_article(url)

        if not data:
            print("Failed → skipping")
            continue

        # Add metadata
        data["id"] = str(uuid.uuid4())
        data["scraped_date"] = datetime.now().strftime("%Y-%m-%d")

        all_articles.append(data)
        seen_urls.add(url)

        # Stop when enough data
        if len(all_articles) >= target_size:
            break

    print(f"\nCollected {len(all_articles)} articles")

    return all_articles

BASE_URL = "https://yle.fi"

START_PAGES = [
   #"https://yle.fi/tuoreimmat",
    "https://yle.fi/uutiset",
    "https://yle.fi/kulttuuri",
    "https://yle.fi/urheilu",
    "https://yle.fi/uutiset/paikallisuutiset",
    "https://yle.fi/t/18-220306/fi", # Politiikka 
    "https://yle.fi/t/18-204933/fi", # Talous
    "https://yle.fi/t/18-186623/fi", # MOT
    "https://yle.fi/t/18-215452/fi", # Luonto
    "https://yle.fi/t/18-212923/fi", # Tiede
    "https://yle.fi/t/18-34837/fi", # Kotimaa
    "https://yle.fi/t/18-34953/fi", # Ulkomaat

]


def get_yle_article_links(limit=300):
    unique_urls = set()

    for page_url in START_PAGES:
        try:
            response = requests.get(page_url, timeout=10)
            response.raise_for_status()
        except requests.RequestException as e:
            print(f"Error fetching {page_url}: {e}")
            continue

        soup = BeautifulSoup(response.text, "lxml")

        for a in soup.find_all("a"):
            href = a.get("href")

            if not href:
                continue

            # Convert relative → absolute
            if href.startswith("/"):
                href = BASE_URL + href

            # Filtering rules:
            if (
                href.startswith("https://yle.fi")
                and "/a/" in href
                and "?" not in href
                and "#" not in href
            ):
                unique_urls.add(href)

            if len(unique_urls) >= limit:
                print(f"Found {len(unique_urls)} unique YLE article links")
                return list(unique_urls)
            
    print(f"Found {len(unique_urls)} unique YLE article links")
    return list(unique_urls)

def scrape_yle_article(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Error fetching article: {e}")
        return None

    soup = BeautifulSoup(response.text, "lxml")

    # Find script containing the initial state
    scripts = soup.find_all("script")

    target_script = None
    for s in scripts:
        if s.string and "window.__INITIAL__STATE__" in s.string:
            target_script = s.string
            break

    if not target_script:
        print("No data script found")
        return None

    # Extract JSON safely (remove JS prefix)
    start = target_script.find("{")
    end = target_script.rfind("}")

    if start == -1 or end == -1:
        print("Could not locate JSON boundaries")
        return None

    json_text = target_script[start:end + 1]

    try:
        data = json.loads(json_text)
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        return None

    # Navigate safely through structure
    try:
        article = data["pageData"]["article"]

        headline = article.get("headline", {}).get("full", "")
        date = article.get("datePublished", "")

        paragraphs = []
        for item in article.get("content", []):
            if item.get("type") == "text":
                paragraphs.append(item.get("text", ""))

        body = " ".join(paragraphs)

        return {
            "url": url,
            "headline": headline,
            "date": date,
            "body": body
        }

    except Exception as e:
        print(f"Error parsing article structure: {e}")
        return None

def get_aland_article_links():
    """
    Scrape Ålands radio front page and return up to 100 (probably max 10) unique, clean article URLs.
    """

    BASE_URL = "https://www.alandsradio.ax"
    url = "https://www.alandsradio.ax/nyheter/"

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Error fetching news: {e}")
        return []

    soup = BeautifulSoup(response.text, "lxml")
    links = soup.find_all("a", href=True)
    article_urls = set()

    for link in links:

        href = link.get("href")
        if not href:
            continue

        # Convert relative URLs to absolute
        if href.startswith("/"):
            href = BASE_URL + href

        # --- FILTERING RULES ---
        if (
            href.startswith(BASE_URL)
            and "/nyheter/" in href
            and "?" not in href
            and "#" not in href
        ):
            article_urls.add(href)
  
        
        if len(article_urls) >= 100:
            break

    print(f"Found {len(article_urls)} unique Ålands radio article links")
    return list(article_urls)


def scrape_article(url):
    """
    Visit an article URL and extract the headline and body text.
    Returns a dict with 'url', 'headline', 'body', or None if extraction fails.
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Error fetching article: {url}\n{e}")
        return None

    soup = BeautifulSoup(response.text, "lxml")
    # Extract headline
    headline_tag = soup.find("h1")
    if not headline_tag or not headline_tag.text.strip():
        print(f"No headline found for: {url}")
        return None
    headline = headline_tag.text.strip()

    # Extract all paragraph text for body
    paragraphs = soup.find_all("p")
    body = " ".join(p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True))
    if not body:
        print(f"No body text found for: {url}")
        return None

    return {
        "url": url,
        "headline": headline,
        "body": body
    }

def save_dataset(articles):
    df = pd.DataFrame(articles)

    # Reorder columns nicely
    df = df[["id", "url", "scraped_date", "headline", "body"]]

    df.to_csv("data/raw_articles.csv", index=False)

    print("Dataset saved to data/raw_articles.csv")


def main():
    articles = build_dataset(target_size=300)
    save_dataset(articles)


if __name__ == "__main__":
    main()
