import os

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from ppi_py import ppi_mean_ci, classical_mean_ci
import pandas as pd
import seaborn as sns
import streamlit as st
from openai import OpenAI
from src.util import load_human_labeled_data, draft_report, stream_data, llm




CATEGORIES = [
    "seat_comfort",
    "cabin_staff_service",
    "food_and_beverages",
    "inflight_entertainment",
    "ground_service",
    "wifi_and_connectivity",
    "value_for_money",
]

st.title("Airline Open Source LLM Demo")

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "mistralai/Mistral-7B-Instruct-v0.2"

if "messages" not in st.session_state:
    st.session_state.messages = []

tab1, tab2, tab3 = st.tabs(
    ["Build a Classification Model", "Make an Inference", "Understand the Problems"]
)

with tab1:
    temp_prompt = """Here's a customer review for an experience they had on an airline.
For each of the following categories decide if the customer's sentiment is Positive, Negative, or Neutral.
If a category is not mentioned return "N/A".

Return using ONLY the following output schema.
- seat_comfort: <sentiment>
- cabin_staff_service: <sentiment>
- food_and_beverages: <sentiment>
- inflight_entertainment: <sentiment>
- ground_service: <sentiment>
- wifi_and_connectivity: <sentiment>
- value_for_money: <sentiment>

Return only in the above format."""
    tab1_form = st.form(key='classifier')
    prompt = tab1_form.text_area('Prompt', value=temp_prompt)
    tab1_submit = tab1_form.form_submit_button('Run Prompt')

    if tab1_submit:
        output = llm(prompt)
        streamed_output = stream_data(output)
        st.write(streamed_output)


with tab2:
    category="cabin_staff_service"
    hf_df = load_human_labeled_data()
    pred_df = pd.read_csv("data/predictions_large.csv")

    pred_df['model_prediction'] = pred_df["model_prediction"].apply(
        lambda x: 1 if x == "Positive" else (0 if x == "Negative" else np.nan)
    )

    tab2_form = st.form(key="Preferences")
    category = tab2_form.radio(
        "Select category to study...",
        CATEGORIES,
    )
    tab2_submit = tab2_form.form_submit_button("Run Inference")

    if tab2_submit:
        category_df = hf_df[
            (~hf_df[category].isna()) & (~hf_df[f"{category}_pred"].isna())
        ].sample(250, replace=False, random_state=433)
        y_labeled = category_df[category].to_numpy()
        yhat_labeled = category_df[f"{category}_pred"].to_numpy()
        category_pred_df = pred_df[(pred_df["categoy"] == category) & (~pred_df.model_prediction.isna())]
        yhat_unlabeled =  category_pred_df.sample(frac=1, replace=False)["model_prediction"].to_numpy()

        # sns.set_theme(style='white', font_scale=1.5, font="DejaVu Sans")
        avg_ci = ppi_mean_ci(
            y_labeled, yhat_labeled, yhat_unlabeled, alpha=0.05
        )
        avg_ci_classical = classical_mean_ci(y_labeled)
        ci_imputed = classical_mean_ci(yhat_unlabeled)

        x_lim_ci_min = min(avg_ci[0], avg_ci_classical[0], ci_imputed[0]) * 0.9
        x_lim_ci_max = max(avg_ci[1], avg_ci_classical[1], ci_imputed[1]) * 1.1

        # ylim_trials = min()
        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(13, 3))

        # Intervals
        axs.plot(
            avg_ci,
            [0.4, 0.4],
            linewidth=20,
            color="#DAF3DA",
            path_effects=[
                pe.Stroke(linewidth=22, offset=(-1, 0), foreground="#71D26F"),
                pe.Stroke(linewidth=22, offset=(1, 0), foreground="#71D26F"),
                pe.Normal(),
            ],
            label=" prediction-powered",
            solid_capstyle="butt",
        )
        axs.plot(
            avg_ci_classical,
            [0.25, 0.25],
            linewidth=20,
            color="#EEEDED",
            path_effects=[
                pe.Stroke(linewidth=22, offset=(-1, 0), foreground="#BFB9B9"),
                pe.Stroke(linewidth=22, offset=(1, 0), foreground="#BFB9B9"),
                pe.Normal(),
            ],
            label=" classical",
            solid_capstyle="butt",
        )
        axs.plot(
            ci_imputed,
            [0.1, 0.1],
            linewidth=20,
            color="#FFEACC",
            path_effects=[
                pe.Stroke(linewidth=22, offset=(-1, 0), foreground="#FFCD82"),
                pe.Stroke(linewidth=22, offset=(1, 0), foreground="#FFCD82"),
                pe.Normal(),
            ],
            label=" imputed",
            solid_capstyle="butt",
        )
        # axs[1].axvline(0.3, ymin=0.0, ymax=1, linestyle="dotted", linewidth=3, label="reported election result", color="#F7AE7C")
        axs.set_xlabel(f"Percent Happy for {category}")
        axs.set_yticks([])
        axs.set_yticklabels([])
        axs.xaxis.set_tick_params()
        axs.set_ylim([0, 0.5])
        axs.set_xlim([x_lim_ci_min, x_lim_ci_max])
        axs.locator_params(nbins=3)
        axs.legend(borderpad=1, labelspacing=1, bbox_to_anchor=(1, 1))
        sns.despine(ax=axs, top=True, right=True, left=True)
        plt.tight_layout()

        st.pyplot(fig)


with tab3:
    hf_df = load_human_labeled_data()
    tab3_form = st.form(key="Produce a report")
    category = tab3_form.radio(
        "Select category...",
        CATEGORIES,
    )

    sentiment = tab3_form.radio(
        "Select sentiment...",
        ['Positive', 'Negative'],
    )

    num_examples = tab3_form.slider('Number of examples', min_value=10, max_value=100)
    tab3_submit = tab3_form.form_submit_button("Generate Report")

    if tab3_submit:
        sentiment = 1 if sentiment == 'Positive' else 0
        report = draft_report(hf_df, category, sentiment, num_examples)
        output = stream_data(report)

        st.write_stream(output)
        