from __future__ import annotations

import re


SENTIMENT_LABELS = {"negative", "neutral", "positive"}
DIRECTION_LABELS = {"down", "flat", "up"}


def parse_outlook(text: str) -> dict[str, str]:
    """Extract labels and narrative from a structured outlook completion."""
    value = str(text or "").strip()
    sentiment = _extract_label(value, "Sentiment", SENTIMENT_LABELS) or _extract_tag_label(
        value,
        "SENTIMENT",
        SENTIMENT_LABELS,
    )
    direction = _extract_label(value, "Direction", DIRECTION_LABELS) or _extract_tag_label(
        value,
        "DIRECTION",
        DIRECTION_LABELS,
    )
    narrative = _extract_narrative(value)
    return {
        "sentiment": sentiment,
        "direction": direction,
        "narrative": narrative,
    }


def format_outlook(sentiment: str, direction: str, narrative: str) -> str:
    return "\n".join(
        [
            f"[SENTIMENT={sentiment}]",
            f"[DIRECTION={direction}]",
            "[NARRATIVE]",
            narrative.strip(),
        ]
    ).strip()


def direction_to_signal(label: str) -> int:
    mapping = {"up": 1, "down": -1, "flat": 0}
    return mapping.get(str(label).lower(), 0)


def _extract_label(text: str, field: str, allowed: set[str]) -> str:
    pattern = rf"(?im)^\s*{re.escape(field)}\s*:\s*([A-Za-z]+)"
    match = re.search(pattern, text)
    if not match:
        return ""
    label = match.group(1).lower()
    return label if label in allowed else ""


def _extract_tag_label(text: str, field: str, allowed: set[str]) -> str:
    pattern = rf"(?i)\[{re.escape(field)}\s*=\s*([A-Za-z]+)\]"
    match = re.search(pattern, text)
    if not match:
        return ""
    label = match.group(1).lower()
    return label if label in allowed else ""


def _extract_narrative(text: str) -> str:
    tag_match = re.search(r"(?ims)\[NARRATIVE\]\s*(.*)$", text)
    if tag_match:
        return tag_match.group(1).strip()

    match = re.search(r"(?ims)^\s*Narrative\s*:\s*(.*)$", text)
    if match:
        return match.group(1).strip()

    lines = []
    for line in text.splitlines():
        if re.match(r"(?i)^\s*(Sentiment|Direction)\s*:", line):
            continue
        if re.match(r"(?i)^\s*\[(SENTIMENT|DIRECTION|NARRATIVE)(\s*=.*)?\]\s*$", line):
            continue
        if line.strip():
            lines.append(line)
    return "\n".join(lines).strip()
