import os

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from openai import OpenAI
from src.util import load_human_labeled_data, draft_report, stream_data
from src.ppi import calculate_ppi




CATEGORIES = [
    "seat_comfort",
    "cabin_staff_service",
    "food_and_beverages",
    "inflight_entertainment",
    "ground_service",
    "wifi_and_connectivity",
    "value_for_money",
]
TOGETHER_API_KEY = os.environ["TOGETHER_API_KEY"]

st.title("Airline Open Source LLM Demo")

client = OpenAI(api_key=TOGETHER_API_KEY, base_url="https://api.together.xyz")

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "mistralai/Mistral-7B-Instruct-v0.2"

if "messages" not in st.session_state:
    st.session_state.messages = []

tab1, tab2, tab3 = st.tabs(
    ["Build a Classification Model", "Make an Inference", "Understand the Problems"]
)

with tab1:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Message the bot..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            stream = client.chat.completions.create(
                model=st.session_state["openai_model"],
                messages=[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ],
                stream=True,
            )
            response = st.write_stream(stream)
        st.session_state.messages.append({"role": "assistant", "content": response})


with tab2:
    category="cabin_staff_service"
    hf_df = load_human_labeled_data()
    pred_df = pd.read_csv("data/predictions_large.csv")

    pred_df['model_prediction'] = pred_df["model_prediction"].apply(
        lambda x: 1 if x == "Positive" else (0 if x == "Negative" else np.nan)
    )

    form = st.form(key="Preferences")
    category = form.radio(
        "Select category to study...",
        CATEGORIES,
    )
    submit = form.form_submit_button("Run Inference")

    if submit:
        category_df = hf_df[
            (~hf_df[category].isna()) & (~hf_df[f"{category}_pred"].isna())
        ].sample(250, replace=False)
        y_labeled = category_df[category].to_numpy()
        yhat_labeled = category_df[f"{category}_pred"].to_numpy()
        category_pred_df = pred_df[(pred_df["categoy"] == category) & (~pred_df.model_prediction.isna())]
        yhat_unlabeled =  category_pred_df.sample(frac=1, replace=False)["model_prediction"].to_numpy()

        # sns.set_theme(style='white', font_scale=1.5, font="DejaVu Sans")
        ci, ci_classical, ci_imputed, ns = calculate_ppi(
            y_labeled, yhat_labeled, yhat_unlabeled, alpha=0.05, num_trials=100
        )
        avg_ci = ci.mean(axis=0)[-1]
        avg_ci_classical = ci_classical.mean(axis=0)[-1]

        x_lim_ci_min = min(avg_ci[0], avg_ci_classical[0], ci_imputed[0]) * 0.9
        x_lim_ci_max = max(avg_ci[1], avg_ci_classical[1], ci_imputed[1]) * 1.1

        y_lim_trial_max = max((ci_classical.mean(axis=0)[:, 1] - ci_classical.mean(axis=0)[:, 0])[2:]) * 1.1

        # ylim_trials = min()
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(13, 3))

        # Intervals
        axs[1].plot(
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
        axs[1].plot(
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
        axs[1].plot(
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
        axs[1].set_xlabel(f"Percent Happy for {category}")
        axs[1].set_yticks([])
        axs[1].set_yticklabels([])
        axs[1].xaxis.set_tick_params()
        axs[1].set_ylim([0, 0.5])
        axs[1].set_xlim([x_lim_ci_min, x_lim_ci_max])
        axs[1].locator_params(nbins=3)
        axs[1].legend(borderpad=1, labelspacing=1, bbox_to_anchor=(1, 1))
        sns.despine(ax=axs[1], top=True, right=True, left=True)

        # Lines
        axs[0].plot(
            ns,
            ci.mean(axis=0)[:, 1] - ci.mean(axis=0)[:, 0],
            label="prediction-powered",
            color="#71D26F",
            linewidth=3,
        )
        axs[0].plot(
            ns,
            ci_classical.mean(axis=0)[:, 1] - ci_classical.mean(axis=0)[:, 0],
            label="classical",
            color="#BFB9B9",
            linewidth=3,
        )
        axs[0].locator_params(axis="y", tight=None, nbins=6)
        axs[0].set_ylabel("width")
        axs[0].set_xlabel("n")
        axs[0].set_ylim([0, y_lim_trial_max])
        sns.despine(ax=axs[0], top=True, right=True)
        plt.tight_layout()

        st.pyplot(fig)


with tab3:
    hf_df = load_human_labeled_data()
    form = st.form(key="Produce a report")
    category = form.radio(
        "Select category...",
        CATEGORIES,
    )

    sentiment = form.radio(
        "Select sentiment...",
        ['Positive', 'Negative'],
    )

    num_examples = form.slider('Number of examples', min_value=10, max_value=100)
    submit = form.form_submit_button("Generate Report")

    if submit:
        sentiment = 1 if sentiment == 'Positive' else 0
        report = draft_report(hf_df, category, sentiment, num_examples)
        output = stream_data(report)

        st.write_stream(output)
        