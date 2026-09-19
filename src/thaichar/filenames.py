"""Filename parser for the Thai character corpus."""

import re

# Pattern: optional "Copy of " prefix, then <prefix>_<docid><subset>_<line>_<idx>.jpg
_RE = re.compile(
    r"^(?:Copy of )?"              # optional "Copy of "
    r"(?P<prefix>be|bc|bl|ce)"     # prefix
    r"_(?P<docid>\d{3})"           # _NNN doc id
    r"(?P<subset>sg|tg)"           # subset
    r"_(?P<line>\d+)"              # _line
    r"_(?P<idx>\d+)"               # _idx
    r"\.jpg$",                     # .jpg extension
    re.IGNORECASE,
)


def parse_filename(name: str) -> dict:
    """Parse a Thai character image filename into its components.

    Returns a dict with keys: prefix, doc_id, subset, line, idx, is_copy,
    group, parsed. Non-matching names get None fields and parsed=False.
    """
    is_copy = name.startswith("Copy of ")
    m = _RE.match(name)
    if m is None:
        return {
            "prefix": None,
            "doc_id": None,
            "subset": None,
            "line": None,
            "idx": None,
            "is_copy": is_copy,
            "group": None,
            "parsed": False,
        }
    return {
        "prefix": m.group("prefix"),
        "doc_id": int(m.group("docid")),
        "subset": m.group("subset"),
        "line": int(m.group("line")),
        "idx": int(m.group("idx")),
        "is_copy": is_copy,
        "group": f"{m.group('prefix')}_{m.group('docid')}",
        "parsed": True,
    }
