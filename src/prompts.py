problem_extraction_prompt_template = lambda review, category, sentiment: f"""This customer had a {sentiment} experience regarding {category}.
In less than 10 words describe the problem related to {category}.

Review:
{review}

Short Review:"""

problem_summarization_prompt_template = lambda problem_str, sentiment, category: f"""Here are some statements from customers that visited our airline.
Please break down the main themes that summarize what was {sentiment} about their experience with {category}.
Provide a bulleted list with approximate counts and order by most important. Produce 10 themes.


Problems:
{problem_str}

Report:"""