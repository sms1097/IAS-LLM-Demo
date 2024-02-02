import os
import tiktoken
import hashlib
from peft import AutoPeftModelForCausalLM
import pandas as pd
import numpy as np
from openai import OpenAI
import os
from transformers import AutoTokenizer, GenerationConfig, pipeline

from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_fixed

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(".env")

OPENAI_KEY = os.environ["OPENAI_KEY"]
TOGETHER_API_KEY = os.environ["TOGETHER_API_KEY"]

TOGETHER_CLIENT = OpenAI(
    api_key=TOGETHER_API_KEY,
    base_url="https://api.together.xyz",
)
OPENAI_CLIENT = OpenAI(api_key=OPENAI_KEY)


GPT3 = "gpt-3.5-turbo-1106"
GPT4 = "gpt-4-1106-preview"
MIXTRAL = "mistralai/Mixtral-8x7B-Instruct-v0.1"

CATEGORIES = [
    "seat_comfort",
    "cabin_staff_service",
    "food_and_beverages",
    "inflight_entertainment",
    "ground_service",
    "wifi_and_connectivity",
    "value_for_money",
]


def num_tokens_from_string(string: str) -> int:
    encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = len(encoding.encode(string))
    return num_tokens


@retry(wait=wait_fixed(15), stop=stop_after_attempt(4))
def run_gpt(user_prompt, model, temperature=0.3):
    model_kwargs = {"temperature": temperature}
    user_prompt = user_prompt[:3700]

    response = OPENAI_CLIENT.chat.completions.create(
        messages=[{"role": "user", "content": user_prompt}],
        stream=False,
        model=model,
        **model_kwargs,
    )

    output = response.choices[0].message.content
    return output


def run_mixtral(user_prompt, temperature=0.1):
    model_kwargs = {"temperature": temperature}
    user_prompt = user_prompt[:3700]

    chat_completion = TOGETHER_CLIENT.chat.completions.create(
        messages=[
            {"role": "user", "content": user_prompt},
        ],
        model=MIXTRAL,
        **model_kwargs,
    )
    output = chat_completion.choices[0].message.content
    return output


def llm(prompt, model):
    if model in [GPT3, GPT4]:
        output = run_gpt(prompt, model)
    elif model == MIXTRAL:
        output = run_mixtral(prompt)
    else:
        raise Exception(f"Invalid Model {model}")
    return output


def _review2sentiment(rating):
    if rating <= 3:
        sentiment = "Negative"
    else:
        sentiment = "Positive"

    return sentiment


def _make_cid(input_string):
    sha256_hash_object = hashlib.sha256()
    sha256_hash_object.update(input_string.encode("utf-8"))
    hash_result = sha256_hash_object.hexdigest()

    return hash_result


def load_sample_scrape():
    df = pd.read_json("scripts/sample_scrape.json")
    for category_name in CATEGORIES:
        df[category_name] = df[category_name].apply(
            lambda x: _review2sentiment(x) if not np.isnan(x) else "Neutral"
        )
    df = df.sample(500)
    df["cid"] = df.review.apply(lambda x: _make_cid(x))
    df = pd.melt(
        df, id_vars=["cid", "review"], var_name="category", value_name="sentiment"
    )
    return df


def parse_output(llm_output, categories):
    """Used to parse output of `tag_prompt_template`"""
    output = {}
    category_reviews = [x.lstrip("- ") for x in llm_output.split("\n")]
    for category_name, review in zip(categories, category_reviews):
        # format the information
        rating = review.replace(f"{category_name}: ", "").split("(")[0]
        output[category_name] = rating

    return output


def tag_prompt_template(review):
    return f"""Here's a customer review for an experience they had on an airline.
For each of the following categories decide if the customer's sentiment is Positive, Negative, or Neutral.
If the category is not mentioned, return "N/A".
The intended airline is Untied Airlines.

Return using ONLY the following output schema.
- seat_comfort: <sentiment>
- cabin_staff_service: <sentiment>
- food_and_beverages: <sentiment>
- inflight_entertainment: <sentiment>
- ground_service: <sentiment>
- wifi_and_connectivity: <sentiment>
- value_for_money: <sentiment>

Return only in the above format.

Review: {review}
Output: """


def run_model_sim(reviews, model):
    records = []
    for obs in tqdm(reviews.to_records(index=False)):
        cid, review = obs[0], obs[1]
        prompt = tag_prompt_template(review)
        output = llm(prompt, model)
        category_predictions = parse_output(output, CATEGORIES)
        records += [
            {"cid": cid, "category": category_name, f"{model}_pred": sentiment}
            for category_name, sentiment in category_predictions.items()
        ]

    review_df = pd.DataFrame.from_records(records)
    return review_df


def sentiment_prompt_template(review, category):
    return f"""Here is a customer review of an airline experience.
Did the customer have a Positive, Negative, or Neutral experience with the specific category?
Only evaluate the sentiment for the experience for the category mentioned below.

Review:
{review}

Category:
{category}

Sentiment:"""


def chat_template(question):
    return f"<|user|>\n{question}</s>\n<|assistant|>"


def run_inference_small_model(df):
    model = AutoPeftModelForCausalLM.from_pretrained(
        "sms1097/tinyllama-airline-reviews", load_in_4bit=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "sms1097/tinyllama-airline-reviews", padding_side="left"
    )
    prompts = df.apply(
        lambda row: chat_template(
            sentiment_prompt_template(row["review"], row["category"])
        ),
        axis=1,
    ).tolist()
    generation_config = GenerationConfig(
        penalty_alpha=0.6,
        do_sample=True,
        top_k=5,
        temperature=0.3,
        repetition_penalty=1.2,
        max_new_tokens=4,
        pad_token_id=tokenizer.eos_token_id,
    )
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        generation_config=generation_config,
    )
    outputs = pipe(prompts, batch_size=16)
    sentiment = [x.split("<|assistant|>")[-1].split("\n") for x in outputs]
    df['small_model_predictions'] = sentiment
    df.to_csv('predictions.csv', index=False)


def main(models=[GPT3, GPT4, MIXTRAL]):
    """Generate Outputs for GPT4, GPT3, and Mixtral"""
    df = load_sample_scrape()
    reviews = df[["cid", "review"]].drop_duplicates()  # {id: review}
    for model in models:
        predictions = run_model_sim(reviews, model)
        df = pd.merge(df, predictions, on=["cid", "category"], how="inner")

    df.to_csv("predictions.csv", index=False)


if __name__ == "__main__":
    main()
