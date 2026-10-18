from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_yaml_text(text: str) -> Any:
    """Parse YAML text into Python objects.

    Uses `yaml.safe_load` when PyYAML is available, and falls back to a clean
    block-YAML + JSON parser so the eval harness works in environments without PyYAML.
    """
    try:
        import yaml  # type: ignore[import-untyped]

        return yaml.safe_load(text)
    except ImportError:
        pass

    stripped = text.strip()
    if stripped.startswith("{") or stripped.startswith("["):
        return json.loads(stripped)
    return _parse_block_yaml(text)


def load_yaml_file(path: str | Path) -> Any:
    file_path = Path(path)
    return load_yaml_text(file_path.read_text(encoding="utf-8"))


def _parse_scalar(raw: str) -> Any:
    val = raw.strip()
    if not val or val in ("null", "~", "None"):
        return None
    if val in ("true", "True"):
        return True
    if val in ("false", "False"):
        return False
    if (val.startswith('"') and val.endswith('"')) or (
        val.startswith("{") and val.endswith("}")
    ) or (val.startswith("[") and val.endswith("]")):
        try:
            return json.loads(val)
        except json.JSONDecodeError:
            if val.startswith('"') and val.endswith('"'):
                return val[1:-1]
    if val.startswith("'") and val.endswith("'"):
        return val[1:-1].replace("''", "'")
    try:
        return int(val)
    except ValueError:
        pass
    try:
        return float(val)
    except ValueError:
        pass
    return val


def _preprocess_lines(text: str) -> list[tuple[int, str]]:
    out: list[tuple[int, str]] = []
    for raw_line in text.splitlines():
        if not raw_line.strip():
            continue
        lstripped = raw_line.lstrip(" ")
        if lstripped.startswith("#"):
            continue
        indent = len(raw_line) - len(lstripped)
        out.append((indent, lstripped.rstrip()))
    return out


def _split_key_val(content: str) -> tuple[str, str]:
    in_single = False
    in_double = False
    for idx, ch in enumerate(content):
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif ch == ":" and not in_single and not in_double:
            if idx + 1 == len(content) or content[idx + 1] == " ":
                return content[:idx].strip(), content[idx + 1 :].strip()
    raise ValueError(f"Invalid YAML mapping line: {content!r}")


def _parse_block_yaml(text: str) -> Any:
    lines = _preprocess_lines(text)
    if not lines:
        return {}
    value, _ = _parse_node(lines, 0, lines[0][0])
    return value


def _parse_node(
    lines: list[tuple[int, str]],
    idx: int,
    base_indent: int,
) -> tuple[Any, int]:
    if idx >= len(lines):
        return None, idx

    indent, content = lines[idx]
    if content.startswith("- ") or content == "-":
        return _parse_sequence(lines, idx, base_indent)
    return _parse_mapping(lines, idx, base_indent)


def _parse_sequence(
    lines: list[tuple[int, str]],
    idx: int,
    base_indent: int,
) -> tuple[list[Any], int]:
    items: list[Any] = []
    n = len(lines)
    while idx < n:
        indent, content = lines[idx]
        if indent < base_indent:
            break
        if indent > base_indent:
            raise ValueError(f"Unexpected indentation at line {idx}: {content!r}")
        if not (content.startswith("- ") or content == "-"):
            break

        rest = content[2:].strip() if content.startswith("- ") else ""
        idx += 1

        if not rest:
            if idx < n and lines[idx][0] > base_indent:
                child, idx = _parse_node(lines, idx, lines[idx][0])
                items.append(child)
            else:
                items.append(None)
            continue

        # Check if `rest` is an inline mapping entry like `- user: "Hello"`
        if _looks_like_mapping_entry(rest):
            k, v_str = _split_key_val(rest)
            item_dict: dict[str, Any] = {}
            if v_str in ("|", ">"):
                scalar_text, idx = _collect_multiline_scalar(
                    lines, idx, base_indent + 2, folded=(v_str == ">")
                )
                item_dict[k] = scalar_text
            elif v_str == "":
                if idx < n and lines[idx][0] > base_indent:
                    child, idx = _parse_node(lines, idx, lines[idx][0])
                    item_dict[k] = child
                else:
                    item_dict[k] = None
            else:
                item_dict[k] = _parse_scalar(v_str)

            if idx < n and lines[idx][0] > base_indent:
                more_map, idx = _parse_mapping(lines, idx, lines[idx][0])
                if isinstance(more_map, dict):
                    item_dict.update(more_map)
            items.append(item_dict)
        else:
            items.append(_parse_scalar(rest))

    return items, idx


def _looks_like_mapping_entry(text: str) -> bool:
    if text.startswith(('"', "'", "{", "[")):
        return False
    try:
        _split_key_val(text)
        return True
    except ValueError:
        return False


def _collect_multiline_scalar(
    lines: list[tuple[int, str]],
    idx: int,
    min_indent: int,
    *,
    folded: bool,
) -> tuple[str, int]:
    parts: list[str] = []
    n = len(lines)
    while idx < n:
        indent, content = lines[idx]
        if indent < min_indent:
            break
        parts.append(content)
        idx += 1
    joiner = " " if folded else "\n"
    return joiner.join(parts).strip(), idx


def _parse_mapping(
    lines: list[tuple[int, str]],
    idx: int,
    base_indent: int,
) -> tuple[dict[str, Any], int]:
    mapping: dict[str, Any] = {}
    n = len(lines)
    while idx < n:
        indent, content = lines[idx]
        if indent < base_indent:
            break
        if indent > base_indent:
            raise ValueError(f"Unexpected mapping indentation at line {idx}: {content!r}")
        if content.startswith("- ") or content == "-":
            break

        key, val_str = _split_key_val(content)
        idx += 1

        if val_str in ("|", ">"):
            if idx < n and lines[idx][0] > base_indent:
                scalar_text, idx = _collect_multiline_scalar(
                    lines, idx, lines[idx][0], folded=(val_str == ">")
                )
                mapping[key] = scalar_text
            else:
                mapping[key] = ""
        elif val_str == "":
            if idx < n and lines[idx][0] > base_indent:
                child, idx = _parse_node(lines, idx, lines[idx][0])
                mapping[key] = child
            else:
                mapping[key] = {}
        else:
            mapping[key] = _parse_scalar(val_str)

    return mapping, idx
