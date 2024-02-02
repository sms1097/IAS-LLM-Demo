import requests
import random
import json
import time
from bs4 import BeautifulSoup

url_generator = (
    lambda i: f"https://www.airlinequality.com/airline-reviews/united-airlines/page/{i}/?sortby=post_date%3ADesc&pagesize=100"
)
parsed_articles = []

metric_categories = [
    "seat_comfort",
    "cabin_staff_service",
    "food_and_beverages",
    "inflight_entertainment",
    "ground_service",
    "wifi_and_connectivity",
    "value_for_money",
    "seat_comfort",
    "recommended"
]


for i in range(1, 50):
    url = url_generator(i)
    response = requests.get(url)

    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")

        review_articles = soup.find_all("article", {"itemprop": "review"})

        for article in review_articles:
            review_body_element = article.find(
                "div", class_="text_content", itemprop="reviewBody"
            )
            review_body = review_body_element.text.strip()
            customer_feedback = {
                "airline": "Untied",
                "review": review_body,
            }

            category_reviews = article.find_all(
                class_=lambda x: x and "review-rating-header" in x
            )
            for review in category_reviews:
                category_name = review["class"][-1]
                if category_name not in metric_categories:
                    review_value_element = review.find_next("td", class_="review-value")
                    review_value = review_value_element.contents[0]
                elif category_name == "recommended":
                    recommended = review.find_next(
                        "td", class_="review-value rating-yes"
                    )
                    review_value = True if recommended else False
                else:
                    stars_element = review.find_next("td", class_="review-rating-stars")

                    filled_stars = stars_element.find_all("span", class_="star fill")
                    review_value = len(filled_stars)
                customer_feedback[category_name] = review_value

            parsed_articles.append(customer_feedback)
    else:
        print(
            f"Failed to retrieve the webpage {i}. Status code: {response.status_code}"
        )

    time.sleep(random.randint(1, 7))

with open("sample_scrape.json", "w") as f:
    json.dump(parsed_articles, f)
