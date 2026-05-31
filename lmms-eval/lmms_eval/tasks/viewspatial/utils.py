import os
import re


def viewspatial_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    """Extracts the text prompt from a viewspatial dataset sample.

    Supports post-prompt injection via environment variable
    VIEWSPATIAL_MCA_POST_PROMPT or lmms_eval_specific_kwargs.

    Args:
        doc (dict): A viewspatial dataset sample dictionary.
        lmms_eval_specific_kwargs (dict, optional): Task-specific kwargs.

    Returns:
        str: The input prompt text.
    """
    question = doc["question"]
    choices = doc["choices"]

    # prompt format from viewspatial bench paper
    question_text = f"Question: {question}\n"
    choices_text = f"Choices: {choices}\n"

    # Environment variable takes priority, then lmms_eval_specific_kwargs, then default
    post_prompt = os.getenv("VIEWSPATIAL_MCA_POST_PROMPT", "")
    if not post_prompt and lmms_eval_specific_kwargs:
        post_prompt = lmms_eval_specific_kwargs.get("default", {}).get("post_prompt", "")
    if not post_prompt:
        post_prompt = "Reply only to the corresponding option.\nAnswer:"

    prompt = question_text + choices_text + post_prompt
    return prompt


def viewspatial_doc_to_visual(doc):
    """Extracts and converts visual data from a viewspatial dataset sample.

    This function iterates over the 'images' key in the dataset sample, calling
    the .convert("RGB") method on each item (presumed to be a PIL/Pillow Image).

    Args:
        doc (dict): A viewspatial dataset sample dictionary.

    Returns:
        list: A list of visual elements (e.g., PIL Images) converted to 'RGB' format.
    """
    return [visual.convert("RGB") for visual in doc["images"]]


def extract_option(text):
    # Highest priority: extract from <Answer>...</Answer> tag
    answer_tag_match = re.search(r"<Answer>\s*(.*?)\s*</Answer>", text, re.DOTALL)
    if answer_tag_match:
        tag_content = answer_tag_match.group(1).strip()
        letter_match = re.search(r"\b([A-D])\b", tag_content, re.IGNORECASE)
        if letter_match:
            return letter_match.group(1).upper()

    match = re.search(r"\b([A-D])\b", text, re.IGNORECASE)
    return match.group(1).upper() if match else None


def viewspatial_process_results(doc, results):
    """Processes the model's output for a single viewspatial document."""
    # extract grounded answer
    grounded_output = doc["answer"]
    grounded_option = extract_option(grounded_output)

    # extract predicted answer: try full response first (for <Answer> tag),
    # then fallback to last-line extraction
    pred = results[0]
    pred_answer = extract_option(pred)
    if pred_answer is None:
        pred_answer = extract_option(pred.split("\n")[-1])

    score = 1.0 if pred_answer == grounded_option else 0.0
    # eval_logger.info(f"Score: {score}")

    return {"overall_accuracy": {"score": score}}


def viewspatial_aggregate_results(results):
    """Aggregates the 'overall_accuracy' results.

    Calculates the mean score from a list of result dictionaries.

    Args:
        results (list[dict]): A list of dictionaries, where each dict has
                              a 'score' key (e.g., [{'score': 1.0}, ...]).

    Returns:
        float: The average score (mean accuracy).
    """
    # --- Compute the total score across all results ---
    total_score = 0.0
    for res in results:
        total_score += res["score"]

    # --- Compute average score safely ---
    avg_score = total_score / len(results) if results else 0.0
    return avg_score
