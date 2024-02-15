import re
import random
import difflib
import pandas as pd
import logging

from seqlbtoolkit.data import txt_to_token_span

logger = logging.getLogger(__name__)


def locate_phrase_in_text(phrase: str, n_phrase_tks: int, text_tks: list[str]):
    """
    Locate a phrase in a list of tokenized text,
    returning the start and end indices of the phrase in the text.

    If the phrase is not found, return None.
    """
    if not phrase or phrase.lower() in ["none", "n/a", "null", "-"]:
        return []
    tk_spans, _ = find_substring(phrase, n_phrase_tks, text_tks)
    return tk_spans


def merge_overlapping_spans(spans):
    if not spans or len(spans) == 1:
        return spans

    # Sort spans based on the start position
    spans.sort(key=lambda x: x[0])

    merged_spans = []
    current_span = spans[0]

    for next_span in spans[1:]:
        # Check for overlap
        if overlaps(current_span[:2], next_span[:2]):
            # If overlapping, keep the larger span
            if (next_span[1] - next_span[0]) > (current_span[1] - current_span[0]):
                current_span = next_span
        else:
            # If no overlap, add the current span to the result and move to the next span
            merged_spans.append(current_span)
            current_span = next_span

    # Add the last span
    merged_spans.append(current_span)

    return merged_spans


def match_target_in_tokens(target, tokens):
    text = " ".join(tokens)
    text = f" {text} "
    pattern = re.compile(re.escape(rf" {target.strip()} "))

    txt_matches = [(match.start() + 1, match.end() - 1) for match in pattern.finditer(text)]
    if not txt_matches:
        return None

    tk_matches = txt_to_token_span(tokens, text, txt_matches)
    tk_matches = [(start, end, " ".join(tokens[start:end])) for start, end in tk_matches if start != end]
    return tk_matches


def match_target_in_tokens_no_space(target, tokens):
    text = "".join(tokens)
    pattern = re.compile(re.escape(target.strip().replace(" ", "")))

    txt_matches = [(match.start(), match.end()) for match in pattern.finditer(text)]
    if not txt_matches:
        return None

    tk_matches = txt_to_token_span(tokens, text, txt_matches)
    tk_matches = [(start, end, " ".join(tokens[start:end])) for start, end in tk_matches if start != end]
    return tk_matches


# Function to reconstruct phrase from tokens
def reconstruct_phrase(window):
    reconstructed = ""
    for i, token in enumerate(window):
        # Add space before the token if it's not a punctuation or part of a word (like "high-")
        if i > 0 and not (token in {",", ".", "-", ";", "!"} or window[i - 1][-1] == "-"):
            reconstructed += " "
        reconstructed += token
    return reconstructed


def find_substring(target, n_tgt_tks, paragraph_tokens, threshold=0.75):
    """
    Find the closest match for a target string in a list of paragraph tokens.
    """
    best_matches = list()
    best_score = 0.0

    # Check if the target is a substring of the text
    tk_matches = match_target_in_tokens(target, paragraph_tokens)
    if tk_matches:
        return tk_matches, 1.0

    if len(target) > 15:
        tk_matches = match_target_in_tokens_no_space(target, paragraph_tokens)
        if tk_matches:
            return tk_matches, 1.0

    # If not, try to find the closest match
    n_tk_range = (max(1, n_tgt_tks - 2), n_tgt_tks + 2)
    for i in range(len(paragraph_tokens) - n_tgt_tks + 2):
        for j in range(i + n_tk_range[0], min(len(paragraph_tokens), i + n_tk_range[1])):
            window = paragraph_tokens[i:j]
            reconstructed = reconstruct_phrase(window)
            # Calculate similarity score
            score = difflib.SequenceMatcher(None, target, reconstructed).ratio()

            if score > best_score and score >= threshold:
                best_score = score
                best_matches = [(i, j, reconstructed)]
            elif score > threshold and score == best_score:
                # If there's a tie, choose the shorter phrase
                if len(reconstructed) < len(best_matches[-1][2]):
                    best_matches = [(i, j, reconstructed)]
                elif len(reconstructed) == len(best_matches[-1][2]):
                    best_matches.append((i, j, reconstructed))

    return best_matches, best_score


def find_closest_column_name(df: pd.DataFrame, keyword: str, threshold: float = 0.8):
    column_names = df.columns

    # Use get_close_matches to find the closest match
    # n=1 means return the best match
    # cutoff is the minimum similarity ratio
    closest_matches = difflib.get_close_matches(keyword, column_names, n=1, cutoff=threshold)

    if closest_matches:
        return closest_matches[0]
    else:
        return None


def extract_markdown_tables(markdown_text):
    """
    Regex pattern for matching markdown tables
    """
    # This pattern looks for lines starting and ending with a pipe symbol, and containing at least one dash-separated line
    table_pattern = r"(\|.*\|\s*\n\|[-| :]+\|.*\n(?:\|.*\|\n?)*)"

    # Find all matches in the markdown text
    tables = re.findall(table_pattern, markdown_text, re.MULTILINE)

    return tables


def markdown_table_to_dataframe(markdown_table):
    """
    Convert a markdown table to a pandas DataFrame

    Parameters
    ----------
    markdown_table : str
        Markdown table (with irrelevant content removed) to convert

    Returns
    -------
    df : pandas.DataFrame
        DataFrame containing the data from the markdown table
    """
    # Split the table into lines
    lines = markdown_table.strip().split("\n")

    # Extract headers
    headers = lines[0].split("|")[1:-1]  # Remove the outer empty strings
    headers = [header.strip() for header in headers]

    # Extract rows
    rows = []
    for line in lines[2:]:  # Skip the first two lines (headers and dashes)
        row = line.split("|")[1:-1]  # Remove the outer empty strings
        row = [cell.strip() for cell in row]
        rows.append(row)

    # Create DataFrame
    df = pd.DataFrame(rows, columns=headers)
    return df


def get_data_downsampling_ids(n_data, n_samples: int = 1000):
    if n_data <= n_samples:
        logger.warning(f"Number of samples ({n_samples}) is greater than number of data ({n_data}).")
        return list(range(n_data))

    # Downsample the data
    data_ids = sorted(random.sample(list(range(n_data)), n_samples))
    return data_ids


def overlaps(a: tuple[int, int], b: tuple[int, int]) -> bool:
    """Check if two spans partially overlap with each other."""
    return (
        (a[0] < b[1] and a[0] >= b[0])
        or (a[1] <= b[1] and a[1] > b[0])
        or (b[0] < a[1] and b[0] >= a[0])
        or (b[1] <= a[1] and b[1] > a[0])
    )
