"""The shared choice task and answer parser."""

import hashlib
import json
import re
from typing import Any

ORDER_ARMS = ("original", "reversed", "shuffle", "shuffle2", "shuffle3",
              "shuffle4", "shuffle5", "shuffle6")
REPEAT_ARMS = ("original",) + tuple(f"original_repeat{i}" for i in range(1, 8))
ALL_ARMS = ORDER_ARMS + REPEAT_ARMS[1:]
PROMPT_PREFIX = (
    "Solve the multiple-choice question in the following JSON task. "
    'Return only a JSON object with exactly one field, "choice", whose value '
    "is one exact candidate key from criteria. "
    "Do not include an explanation, Markdown, or additional fields.\n\n"
)


class InputError(ValueError):
    """An actionable input error whose message contains no supplied values."""


def input_sha256(question: dict) -> str:
    fields = ("state", "question", "options", "gold_index", "candidate_ids", "source_labels")
    content = json.dumps({key: question[key] for key in fields}, ensure_ascii=False,
                         sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def wire(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def unique_object(pairs: list[tuple[str, Any]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def keys_for(question: dict) -> list[str]:
    keys = question.get("source_labels") or question["candidate_ids"]
    if (len(keys) != len(question["options"]) or
            any(not isinstance(key, str) for key in keys) or len(set(keys)) != len(keys)):
        raise ValueError("candidate keys must be unique strings")
    return keys


def validate_question(question: dict) -> None:
    if not isinstance(question.get("id"), str):
        raise ValueError("question ID must be a string")
    options = question.get("options")
    if not isinstance(options, list) or len(options) < 2 or any(not isinstance(x, str) for x in options):
        raise ValueError("options must contain at least two strings")
    n = len(options)
    if question.get("option_count", n) != n:
        raise ValueError("option_count differs from the option list")
    ids = question.get("candidate_ids")
    if not isinstance(ids, list) or len(ids) != n or any(not isinstance(x, str) for x in ids) or len(set(ids)) != n:
        raise ValueError("candidate_ids must contain one unique string per option")
    gold = question.get("gold_index")
    if type(gold) is not int or not 0 <= gold < n:
        raise ValueError("gold_index must index an option")
    if not isinstance(question.get("question"), str) or not isinstance(question.get("state"), str):
        raise ValueError("question and state must be strings")
    keys_for(question)


def validate_order(order: list[int], n: int) -> None:
    if not isinstance(order, list) or any(type(i) is not int for i in order) or sorted(order) != list(range(n)):
        raise ValueError("order must be a permutation of original option indices")


def task_json(question: dict, order: list[int]) -> dict:
    """Reorder complete candidates while retaining their original keys."""
    validate_order(order, len(question["options"]))
    keys = keys_for(question)
    return {"state": question["state"], "questions": {"q": {
        "type": "choice", "instructions": question["question"],
        "criteria": {keys[i]: question["options"][i] for i in order},
    }}}


def decode_answer(text: str, keys: list[str], option_texts: list[str] | None = None) -> str:
    """Accept an unambiguous choice JSON, including an exact unique option text."""
    if not isinstance(text, str):
        raise ValueError("final answer is not text")
    candidate = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*\n?([\s\S]*?)\n?```", candidate, flags=re.IGNORECASE)
    if fenced:
        candidate = fenced.group(1).strip()
    try:
        obj = json.loads(candidate, object_pairs_hook=unique_object)
    except ValueError:
        matches = [m for m in re.finditer(r"\{[^{}]*\}", candidate)
                   if re.search(r'"choice"\s*:', m.group())]
        if not matches or not re.fullmatch(r"\s*(?:\*\*|```)?\s*", candidate[matches[-1].end():]):
            raise ValueError("no unique terminal choice JSON") from None
        objects = [json.loads(m.group(), object_pairs_hook=unique_object) for m in matches]
        if any(obj != objects[0] for obj in objects[1:]):
            raise ValueError("conflicting choice objects")
        obj = objects[0]
    if not isinstance(obj, dict) or set(obj) != {"choice"} or not isinstance(obj["choice"], str):
        raise ValueError("expected one exact candidate key")
    choice = obj["choice"]
    if choice not in keys and option_texts is not None:
        matches = [key for key, value in zip(keys, option_texts) if value == choice]
        if len(matches) != 1:
            raise ValueError("choice is not a unique candidate")
        choice = matches[0]
    if choice not in keys:
        raise ValueError("unknown candidate key")
    return choice
