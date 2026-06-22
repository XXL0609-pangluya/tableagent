"""Python 3 port of the official WikiTableQuestions evaluator (v1.0.2).

Faithfully reproduces the matching semantics of WikiTableQuestions/evaluator.py
so our reported accuracy equals the official one. Changes vs the original are
purely Python 2 -> 3 (unicode/long/basestring/ur-strings/print), plus a
programmatic ``evaluate`` API on top of the original batch logic.

Reference: https://github.com/ppasupat/WikiTableQuestions
"""
from __future__ import annotations

import os
import re
import unicodedata
from abc import ABCMeta, abstractmethod
from math import isinf, isnan
from typing import Optional


# ---------------- String Normalization ----------------

def normalize(x: str) -> str:
    if not isinstance(x, str):
        x = x.decode("utf8", errors="ignore")
    # Remove diacritics
    x = "".join(
        c for c in unicodedata.normalize("NFKD", x) if unicodedata.category(c) != "Mn"
    )
    # Normalize quotes and dashes
    x = re.sub(r"[‘’´`]", "'", x)
    x = re.sub(r"[“”]", '"', x)
    x = re.sub(r"[‐‑‒–—−]", "-", x)
    while True:
        old_x = x
        # Remove citations
        x = re.sub(r"((?<!^)\[[^\]]*\]|\[\d+\]|[•♦†‡*#+])*$", "", x.strip())
        # Remove details in parenthesis
        x = re.sub(r"(?<!^)( \([^)]*\))*$", "", x.strip())
        # Remove outermost quotation mark
        x = re.sub(r'^"([^"]*)"$', r"\1", x.strip())
        if x == old_x:
            break
    # Remove final '.'
    if x and x[-1] == ".":
        x = x[:-1]
    # Collapse whitespaces and convert to lower case
    x = re.sub(r"\s+", " ", x, flags=re.U).lower().strip()
    return x


# ---------------- Value Types ----------------

class Value(metaclass=ABCMeta):
    _normalized: Optional[str] = None

    @abstractmethod
    def match(self, other: "Value") -> bool:
        ...

    @property
    def normalized(self) -> Optional[str]:
        return self._normalized


class StringValue(Value):
    def __init__(self, content: str):
        assert isinstance(content, str)
        self._normalized = normalize(content)
        self._hash = hash(self._normalized)

    def __eq__(self, other):
        return isinstance(other, StringValue) and self.normalized == other.normalized

    def __hash__(self):
        return self._hash

    def __str__(self):
        return "S" + str([self.normalized])

    __repr__ = __str__

    def match(self, other: Value) -> bool:
        assert isinstance(other, Value)
        return self.normalized == other.normalized


class NumberValue(Value):
    def __init__(self, amount, original_string: Optional[str] = None):
        assert isinstance(amount, (int, float))
        if abs(amount - round(amount)) < 1e-6:
            self._amount = int(amount)
        else:
            self._amount = float(amount)
        if not original_string:
            self._normalized = str(self._amount)
        else:
            self._normalized = normalize(original_string)
        self._hash = hash(self._amount)

    @property
    def amount(self):
        return self._amount

    def __eq__(self, other):
        return isinstance(other, NumberValue) and self.amount == other.amount

    def __hash__(self):
        return self._hash

    def __str__(self):
        return ("N(%f)" % self.amount) + str([self.normalized])

    __repr__ = __str__

    def match(self, other: Value) -> bool:
        assert isinstance(other, Value)
        if self.normalized == other.normalized:
            return True
        if isinstance(other, NumberValue):
            return abs(self.amount - other.amount) < 1e-6
        return False

    @staticmethod
    def parse(text: str):
        """Try to parse into a number; return int/float or None."""
        try:
            return int(text)
        except Exception:
            try:
                amount = float(text)
                assert not isnan(amount) and not isinf(amount)
                return amount
            except Exception:
                return None


class DateValue(Value):
    def __init__(self, year: int, month: int, day: int, original_string: Optional[str] = None):
        """Create a new DateValue. Placeholders are marked as -1."""
        assert isinstance(year, int)
        assert isinstance(month, int) and (month == -1 or 1 <= month <= 12)
        assert isinstance(day, int) and (day == -1 or 1 <= day <= 31)
        assert not (year == month == day == -1)
        self._year = year
        self._month = month
        self._day = day
        if not original_string:
            self._normalized = "{}-{}-{}".format(
                year if year != -1 else "xx",
                month if month != -1 else "xx",
                day if day != "-1" else "xx",  # kept faithful to original
            )
        else:
            self._normalized = normalize(original_string)
        self._hash = hash((self._year, self._month, self._day))

    @property
    def ymd(self):
        return (self._year, self._month, self._day)

    def __eq__(self, other):
        return isinstance(other, DateValue) and self.ymd == other.ymd

    def __hash__(self):
        return self._hash

    def __str__(self):
        return ("D(%d,%d,%d)" % (self._year, self._month, self._day)) + str([self._normalized])

    __repr__ = __str__

    def match(self, other: Value) -> bool:
        assert isinstance(other, Value)
        if self.normalized == other.normalized:
            return True
        if isinstance(other, DateValue):
            return self.ymd == other.ymd
        return False

    @staticmethod
    def parse(text: str):
        """Try to parse into a date tuple (year, month, day) or None."""
        try:
            ymd = text.lower().split("-")
            assert len(ymd) == 3
            year = -1 if ymd[0] in ("xx", "xxxx") else int(ymd[0])
            month = -1 if ymd[1] == "xx" else int(ymd[1])
            day = -1 if ymd[2] == "xx" else int(ymd[2])
            assert not (year == month == day == -1)
            assert month == -1 or 1 <= month <= 12
            assert day == -1 or 1 <= day <= 31
            return (year, month, day)
        except Exception:
            return None


# ---------------- Value Instantiation ----------------

def to_value(original_string, corenlp_value=None) -> Value:
    if isinstance(original_string, Value):
        return original_string
    if not corenlp_value:
        corenlp_value = original_string
    amount = NumberValue.parse(corenlp_value)
    if amount is not None:
        return NumberValue(amount, original_string)
    ymd = DateValue.parse(corenlp_value)
    if ymd is not None:
        if ymd[1] == ymd[2] == -1:
            return NumberValue(ymd[0], original_string)
        else:
            return DateValue(ymd[0], ymd[1], ymd[2], original_string)
    return StringValue(original_string)


def to_value_list(original_strings, corenlp_values=None) -> list[Value]:
    assert isinstance(original_strings, (list, tuple, set))
    if corenlp_values is not None:
        assert isinstance(corenlp_values, (list, tuple, set))
        assert len(original_strings) == len(corenlp_values)
        return list(set(to_value(x, y) for (x, y) in zip(original_strings, corenlp_values)))
    return list(set(to_value(x) for x in original_strings))


# ---------------- Denotation check ----------------

def check_denotation(target_values: list[Value], predicted_values: list[Value]) -> bool:
    if len(target_values) != len(predicted_values):
        return False
    for target in target_values:
        if not any(target.match(pred) for pred in predicted_values):
            return False
    return True


# ---------------- TSV helpers ----------------

def tsv_unescape(x: str) -> str:
    return x.replace(r"\n", "\n").replace(r"\p", "|").replace("\\\\", "\\")


def tsv_unescape_list(x: str) -> list[str]:
    return [tsv_unescape(y) for y in x.split("|")]


# ---------------- Target loading ----------------

def load_targets_from_tagged(tagged_path: str) -> dict[str, list[Value]]:
    """Authoritative targets using targetValue + targetCanon from a .tagged file."""
    targets: dict[str, list[Value]] = {}
    with open(tagged_path, "r", encoding="utf8") as fin:
        header = fin.readline().rstrip("\n").split("\t")
        for line in fin:
            stuff = dict(zip(header, line.rstrip("\n").split("\t")))
            ex_id = stuff["id"]
            original_strings = tsv_unescape_list(stuff["targetValue"])
            canon_strings = tsv_unescape_list(stuff["targetCanon"])
            targets[ex_id] = to_value_list(original_strings, canon_strings)
    return targets


def load_targets_from_tsv(tsv_path: str) -> dict[str, list[Value]]:
    """Targets using only targetValue (no CoreNLP canon). Slightly more lenient
    than the official path; prefer load_targets_from_tagged when available."""
    targets: dict[str, list[Value]] = {}
    with open(tsv_path, "r", encoding="utf8") as fin:
        header = fin.readline().rstrip("\n").split("\t")
        for line in fin:
            stuff = dict(zip(header, line.rstrip("\n").split("\t")))
            ex_id = stuff["id"]
            original_strings = tsv_unescape_list(stuff["targetValue"])
            targets[ex_id] = to_value_list(original_strings)
    return targets


# ---------------- Programmatic evaluate API ----------------

def evaluate(
    predictions: dict[str, list[str]],
    targets: dict[str, list[Value]],
    exclude_ids: Optional[set] = None,
) -> dict:
    """Score predictions against targets.

    Args:
        predictions: ex_id -> list of predicted item strings (empty list = no prediction)
        targets: ex_id -> list[Value] (from load_targets_*)
        exclude_ids: ids of disputed examples to also report an 'adjusted' score for.
    Returns:
        dict with raw accuracy + (if exclude_ids) adjusted accuracy (disputed removed),
        plus per-example correctness.
    """
    exclude_ids = exclude_ids or set()
    num_examples = 0
    num_correct = 0
    num_correct_adj = 0
    num_examples_adj = 0
    num_excluded = 0
    per_example: dict[str, bool] = {}
    missing: list[str] = []

    for ex_id, target_values in targets.items():
        num_examples += 1
        pred_items = predictions.get(ex_id)
        if pred_items is None:
            missing.append(ex_id)
            per_example[ex_id] = False
            correct = False
        else:
            predicted_values = to_value_list(pred_items) if pred_items else []
            correct = check_denotation(target_values, predicted_values)
            per_example[ex_id] = correct
        if correct:
            num_correct += 1
        if ex_id in exclude_ids:
            num_excluded += 1
        else:
            num_examples_adj += 1
            if correct:
                num_correct_adj += 1

    accuracy = (num_correct + 1e-9) / (num_examples + 1e-9) if num_examples else 0.0
    accuracy_adj = (num_correct_adj + 1e-9) / (num_examples_adj + 1e-9) if num_examples_adj else 0.0
    return {
        "accuracy": round(accuracy, 4),
        "num_examples": num_examples,
        "num_correct": num_correct,
        "num_missing_predictions": len(missing),
        "accuracy_adjusted": round(accuracy_adj, 4),
        "num_examples_adjusted": num_examples_adj,
        "num_correct_adjusted": num_correct_adj,
        "num_excluded_disputed": num_excluded,
        "per_example": per_example,
    }


def find_tagged_path(dataset_root: str, split_basename: str) -> Optional[str]:
    """Map a split (e.g. 'training', 'pristine-unseen-tables') to its .tagged file.
    random-split-* dev/train splits are subsets of training, so they fall back to
    training.tagged for authoritative canon."""
    direct = os.path.join(dataset_root, "tagged", "data", split_basename + ".tagged")
    if os.path.exists(direct):
        return direct
    if split_basename.startswith("random-split"):
        fallback = os.path.join(dataset_root, "tagged", "data", "training.tagged")
        if os.path.exists(fallback):
            return fallback
    return None
