"""
scraper_news.py
Purpose: Script for scraping news articles from the web.
"""

import requests
from bs4 import BeautifulSoup


def get_article_links():
    """
    Scrape BBC News front page and return up to 100 unique, clean article URLs.
    """
    url = "https://www.bbc.com/news"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Error fetching BBC News: {e}")
        return []

    soup = BeautifulSoup(response.text, "lxml")
    links = soup.find_all("a", href=True)
    article_urls = set()

    for link in links:
        href = link["href"]
        # Skip links with '#' or '?', or unwanted sections
        if (
            "#" in href or
            "?" in href or
            href.startswith("/news/topics/") or
            href.startswith("/news/live/")
        ):
            continue
        # Only keep /news/ articles
        if href.startswith("/news/") and not href.startswith("/news/av/"):
            full_url = f"https://www.bbc.com{href}"
            article_urls.add(full_url)
        elif href.startswith("https://www.bbc.com/news/") and not href.startswith("https://www.bbc.com/news/av/"):
            article_urls.add(href)
        if len(article_urls) >= 100:
            break

    print(f"Found {len(article_urls)} unique article links")
    return list(article_urls)


def scrape_article(url):
    """
    Visit a BBC article URL and extract the headline and body text.
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


def main():
    # Get up to 100 unique BBC article URLs
    article_urls = get_article_links()
    # For testing, scrape only 5 articles
    for i, url in enumerate(article_urls[:5]):
        print(f"\nScraping article {i+1}: {url}")
        article_data = scrape_article(url)
        if article_data:
            print(f"HEADLINE: {article_data['headline']}")
            print(f"BODY (first 200 chars): {article_data['body'][:200]}...")
        else:
            print("Failed to extract article.")


if __name__ == "__main__":
    main()
