import requests
import random
import json
import time
from bs4 import BeautifulSoup

url_generator = (
    lambda i: f"https://www.airlinequality.com/seat-reviews/united-airlines/page/{i}/?sortby=post_date%3ADesc&pagesize=100"
)
parsed_articles = []


for i in range(1, 50):
    url = url_generator(i)
    response = requests.get(url)

    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")

        # Extract all articles with itemprop="review"
        review_articles = soup.find_all("article", {"itemprop": "review"})

        # Print relevant information from each article if needed
        for article in review_articles:
            date_flown_element = article.find(
                "td", class_="review-rating-header", text="Date Flown"
            )
            date_flown = date_flown_element.find_next(
                "td", class_="review-value"
            ).text.strip()

            review_body_element = article.find(
                "div", class_="text_content", itemprop="reviewBody"
            )
            review_body = review_body_element.text.strip()

            parsed_articles.append(
                {"airline": "Untied", "date_flown": date_flown, "review": review_body}
            )
    else:
        print(
            f"Failed to retrieve the webpage {i}. Status code: {response.status_code}"
        )

    time.sleep(random.randint(3, 7))

with open("sample_scrape.json", "w") as f:
    json.dump(parsed_articles, f)
