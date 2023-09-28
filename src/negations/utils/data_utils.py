import os
import json
import time

import pandas as pd


def get_type_hint(q_list: list):
    """
    Uses the first word of the question to return the likely response format
    of the question's answer.

    Args:
    -----
        q_list: a list of strings containing the questions

    Returns:
    --------
        A list of strings containing whether the questions are extraction or classification
    """
    hints = []
    for q in q_list:
        first_word = q.split(" ")[0].lower()
        if first_word in [
            "what's",
            "which",
            "what",
            "when",
            "where",
            "how",
            "on",
            "who",
            "in",
            "per",
            "the",
            "at",
            "could",
            "aside",
            "during",
            "how's",
            "pertaining",
        ]:
            hints.append("extraction")
        elif first_word in [
            "does",
            "are",
            "is",
            "am",
            "can",
            "do",
            "did",
            "was",
            "were",
            "should",
            "has",
            "have",
            "will",
            "while",
        ]:
            hints.append("classification")
        else:
            raise Exception("Did not expect Q: ", q)
    return hints


def is_na(x: str):
    """
    Given an answer, checks if it is NaN.  Handles structure and non-structure cases

    Args:
    -----
        x: a string of the answer or predictions

    Returns:
    --------
        A boolean indicating whether the correct answer is NaN or not
    """
    if type(x) in [list, pd.Series]:
        if type(x) == pd.Series:
            import pdb

            pdb.set_trace()
        return sum([is_na(item) for item in x]) == len(x)

    if type(x) == str and "{" in x:
        try:
            struct = json.loads(x.replace("nan", "'n/a'"))
        except Exception as e:
            print(
                "### failed to parse {} due to {} in `is_na` ###".format(x, e)
            )
            return False
        all_nans = True
        for dict_item in struct:
            for value in dict_item.values():
                if not is_na(value):
                    all_nans = False
        return all_nans
    else:
        if pd.isnull(x) or (
            type(x) == str
            and (x.lower() == "n/a" or x.lower() == "na" or x.lower() == "nan")
        ):
            return True
        elif type(x) != str:
            raise NotImplementedError(
                "Can't evaluate non-structure answer that is not a string"
            )
        else:
            return False


def flatten(x: list) -> list:
    """
    A helper function to flatten a list of lists to only a list
    """
    return [a for i in x for a in flatten(i)] if isinstance(x, list) else [x]
