import pandas as pd
import time
import numpy as np
import os
from openai import OpenAI
from .prompts import problem_extraction_prompt_template, problem_summarization_prompt_template
from stqdm import stqdm
from tqdm import tqdm

TOGETHER_API_KEY = os.environ["TOGETHER_API_KEY"]
OPENAI_KEY = os.environ["OPENAI_KEY"]

def llm(user_prompt, temperature=0.1):
    model_kwargs = {"temperature": temperature}
    user_prompt = user_prompt[:3700]

    client = OpenAI(api_key=OPENAI_KEY)
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": user_prompt
            },
        ],
        # model="mistralai/Mixtral-8x7B-Instruct-v0.1",
        model= "gpt-3.5-turbo-1106",
        **model_kwargs
    )
    output = chat_completion.choices[0].message.content
    return output 



def load_human_labeled_data():
    def review2sentiment(rating):
        if rating <= 3:
            sentiment = 0
        else:
            sentiment = 1

        return sentiment

    column_rename = {
        "Seat Comfort": "seat_comfort",
        "Cabin Staff Service": "cabin_staff_service",
        "Food & Beverages": "food_and_beverages",
        "Ground Service": "ground_service",
        "Inflight Entertainment": "inflight_entertainment",
        "Wifi & Connectivity": "wifi_and_connectivity",
        "Value For Money": "value_for_money",
    }
    hf_df = pd.read_csv("data/human_labeled_data.csv")
    hf_df.rename(columns=column_rename, inplace=True)
    for col in column_rename.values():
        hf_df[col] = hf_df[col].apply(review2sentiment)
        hf_df[f"{col}_pred"] = hf_df[f"{col}_pred"].apply(
            lambda x: 1 if x == "Positive" else (0 if x == "Negative" else np.nan)
        )

    columns2keep = (
        ["cid", "Review"]
        + list(column_rename.values())
        + [f"{x}_pred" for x in column_rename.values()]
    )
    return hf_df[columns2keep]


def draft_report(df, category, sentiment, sample_size=40):
    reviews = df[(df[category] == sentiment)].Review.sample(sample_size).tolist()
    problem_prompts = [problem_extraction_prompt_template(r, category, sentiment) for r in reviews]

    problems = [llm(p) for p in stqdm(problem_prompts, desc='Extracting problems')]
    problem_str = '\n'.join(problems)
    report_prompt = problem_summarization_prompt_template(problem_str, sentiment, category)
    report = llm(report_prompt)
    return report

def stream_data(output_text):
    for line in output_text.split('\n'):
        for word in line.split(' '):
            yield word + " "
            time.sleep(0.02)
        yield '\n'
