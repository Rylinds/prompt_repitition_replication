from data.arc_loader import ARCLoader, format_mcq_prompt

"""Rough Idea"""

# experiment init
loader = ARCLoader(split="test")
examples = loader.load_subset(n=100, seed=42) # piloting on 100

# experiment loop
for example in examples:
    for ordering in ["question_first", "options_first"]:
        prompt_base = format_mcq_prompt(example, ordering)

        # baseline
        result_baseline = agent.query(f"{prompt_base}The answer is")

        # repeated
        result_repeated = agent.query(f"{prompt_base}{prompt_base}The answer is")

    # store results with question_id, variant, ordering etc. for checks later