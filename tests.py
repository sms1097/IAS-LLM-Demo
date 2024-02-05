from src.util import load_human_labeled_data , draft_report
from src.ppi import calculate_ppi
from openai import OpenAI
import pandas as pd
import numpy as np
import os


def test_ppi():
    category="cabin_staff_service"
    hf_df = load_human_labeled_data()
    pred_df = pd.read_csv("data/predictions_large.csv")

    pred_df['model_prediction'] = pred_df["model_prediction"].apply(
        lambda x: 1 if x == "Positive" else (0 if x == "Negative" else np.nan)
    )

    category_df = hf_df[
        (~hf_df[category].isna()) & (~hf_df[f"{category}_pred"].isna())
    ].sample(100)
    y_labeled = category_df[category].to_numpy()
    yhat_labeled = category_df[f"{category}_pred"].to_numpy()
    category_pred_df =pred_df[(pred_df["categoy"] == category) & (~pred_df.model_prediction.isna())]
    yhat_unlabeled =  category_pred_df["model_prediction"].to_numpy()

    ci, ci_classical, ci_imputed, ns = calculate_ppi(
        y_labeled, yhat_labeled, yhat_unlabeled, alpha=0.05, num_trials=1000
    )


def test_data_chat():
    df = load_human_labeled_data()
    category="cabin_staff_service"
    sentiment = 1
    sample_size=40

    report = draft_report(df, category, sentiment, sample_size)

    print(report)
    

if __name__ == '__main__':
    # test_ppi()
    test_data_chat()
