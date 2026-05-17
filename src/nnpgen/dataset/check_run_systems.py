#!/usr/bin/env python3
"""Check DeePMD run.json systems, dataset coverage, and type_map.raw order."""

import argparse
import datetime as _datetime
import json
import os
import re
import socket
from difflib import get_close_matches
from collections import Counter, defaultdict


def load_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def normalize_path(path):
    return os.path.normpath(os.path.abspath(os.path.expanduser(path)))


def path_has_test_component(path):
    return any("test" in part.lower() for part in normalize_path(path).split(os.sep) if part)


def get_training_data(config):
    training = config.get("training")
    if isinstance(training, dict) and isinstance(training.get("training_data"), dict):
        return training["training_data"]
    if isinstance(config.get("training_data"), dict):
        return config["training_data"]
    raise ValueError("Cannot find training.training_data in run.json")


def parse_auto_prob(auto_prob):
    if not auto_prob:
        return []
    chunks = [chunk.strip() for chunk in str(auto_prob).split(";") if chunk.strip()]
    ranges = []
    for chunk in chunks:
        if chunk.startswith("prob_sys_size"):
            continue
        parts = [part.strip() for part in chunk.split(":")]
        if len(parts) != 3:
            ranges.append({"raw": chunk, "valid": False, "error": "expected start:end:prob"})
            continue
        try:
            start = int(parts[0])
            end = int(parts[1])
            prob = float(parts[2])
        except ValueError as exc:
            ranges.append({"raw": chunk, "valid": False, "error": str(exc)})
            continue
        ranges.append(
            {
                "raw": chunk,
                "valid": True,
                "start": start,
                "end": end,
                "count": end - start,
                "prob": prob,
            }
        )
    return ranges


def raw_system_groups(run_json_path):
    """Count systems separated by blank lines in the raw JSON systems array."""
    with open(run_json_path, encoding="utf-8") as handle:
        lines = handle.readlines()

    in_training_data = False
    in_systems = False
    bracket_depth = 0
    groups = []
    current = []

    for line in lines:
        if not in_training_data and '"training_data"' in line:
            in_training_data = True

        if in_training_data and not in_systems and re.search(r'"systems"\s*:', line):
            in_systems = True
            after_bracket = line.split("[", 1)[1] if "[" in line else ""
            bracket_depth = line.count("[") - line.count("]")
            line_to_process = after_bracket
        elif in_systems:
            line_to_process = line
            bracket_depth += line.count("[") - line.count("]")
        else:
            continue

        stripped = line_to_process.strip()
        string_values = re.findall(r'"([^"]+)"', line_to_process)
        if string_values:
            current.extend(string_values)
        elif stripped == "" and current:
            groups.append(current)
            current = []

        if in_systems and bracket_depth <= 0:
            if current:
                groups.append(current)
            break

    return groups


def parse_type_map_raw(path):
    values = []
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            values.extend(re.findall(r"[A-Za-z][A-Za-z0-9]*", line))
    return values


def check_type_map(values, expected):
    if not values:
        return False, "empty type_map.raw"
    if len(values) > len(expected):
        return False, "type_map.raw has more elements than run.json type_map"
    prefix = expected[: len(values)]
    if values == prefix:
        return True, "ok"
    unknown = [item for item in values if item not in expected]
    if unknown:
        return False, "unknown elements: {}".format(", ".join(unknown))
    return False, "expected prefix {}, got {}".format(prefix, values)


def looks_like_deepmd_system(path):
    return (
        os.path.isfile(os.path.join(path, "type.raw"))
        and os.path.isfile(os.path.join(path, "type_map.raw"))
    )


def scan_dataset_systems(dataset_root):
    systems = []
    skipped_test_dirs = 0
    for root, dirs, _files in os.walk(dataset_root):
        dirs[:] = [name for name in dirs if not name.startswith(".") and name != "__pycache__"]

        if path_has_test_component(root):
            skipped_test_dirs += 1
            dirs[:] = []
            continue

        if looks_like_deepmd_system(root):
            systems.append(normalize_path(root))
            dirs[:] = []
    return sorted(set(systems)), skipped_test_dirs


def format_list(items, limit=None):
    if not items:
        return ["- None"]
    shown = items if limit is None else items[:limit]
    lines = ["- `{}`".format(item) for item in shown]
    if limit is not None and len(items) > limit:
        lines.append("- ... {} more".format(len(items) - limit))
    return lines


def write_report(report_path, report):
    parent = os.path.dirname(report_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write(report)


def build_report(args):
    run_json = normalize_path(args.run_json)
    dataset_root = normalize_path(args.dataset_root)
    config = load_json(run_json)

    expected_type_map = config.get("model", {}).get("type_map")
    if not isinstance(expected_type_map, list):
        raise ValueError("Cannot find model.type_map list in run.json")

    training_data = get_training_data(config)
    systems = training_data.get("systems")
    if not isinstance(systems, list):
        raise ValueError("training_data.systems is not a list")

    normalized_systems = [normalize_path(path) for path in systems]
    listed_set = set(normalized_systems)
    auto_prob = training_data.get("auto_prob")
    auto_ranges = parse_auto_prob(auto_prob)
    raw_groups = raw_system_groups(run_json)

    duplicate_paths = sorted(path for path, count in Counter(normalized_systems).items() if count > 1)
    outside_dataset = sorted(path for path in normalized_systems if not path.startswith(dataset_root + os.sep))
    missing_dirs = sorted(path for path in normalized_systems if not os.path.isdir(path))
    existing_dirs = [path for path in normalized_systems if os.path.isdir(path)]

    missing_type_map = []
    bad_type_map = []
    ok_type_map_counter = Counter()
    type_map_by_sequence = defaultdict(list)
    for system_path in existing_dirs:
        type_map_path = os.path.join(system_path, "type_map.raw")
        if not os.path.isfile(type_map_path):
            missing_type_map.append(system_path)
            continue
        values = parse_type_map_raw(type_map_path)
        ok, reason = check_type_map(values, expected_type_map)
        if ok:
            ok_type_map_counter[tuple(values)] += 1
        else:
            bad_type_map.append((system_path, type_map_path, values, reason))
        type_map_by_sequence[tuple(values)].append(system_path)

    dataset_systems, skipped_test_dirs = scan_dataset_systems(dataset_root)
    unlisted_dataset_systems = sorted(path for path in dataset_systems if path not in listed_set)
    listed_missing_from_dataset_scan = sorted(path for path in listed_set if path.startswith(dataset_root + os.sep) and path not in dataset_systems)

    unlisted_bad_type_map = []
    unlisted_type_map_counter = Counter()
    for system_path in unlisted_dataset_systems:
        type_map_path = os.path.join(system_path, "type_map.raw")
        values = parse_type_map_raw(type_map_path)
        ok, reason = check_type_map(values, expected_type_map)
        if ok:
            unlisted_type_map_counter[tuple(values)] += 1
        else:
            unlisted_bad_type_map.append((system_path, type_map_path, values, reason))

    replacement_suggestions = []
    for missing_path in missing_dirs:
        matches = get_close_matches(missing_path, unlisted_dataset_systems, n=3, cutoff=0.72)
        replacement_suggestions.append((missing_path, matches))

    auto_prob_errors = []
    if any(not item.get("valid") for item in auto_ranges):
        auto_prob_errors.extend(item["raw"] for item in auto_ranges if not item.get("valid"))
    valid_ranges = [item for item in auto_ranges if item.get("valid")]
    expected_start = 0
    for item in valid_ranges:
        if item["start"] != expected_start:
            auto_prob_errors.append(
                "range {} starts at {}, expected {}".format(item["raw"], item["start"], expected_start)
            )
        if item["end"] < item["start"]:
            auto_prob_errors.append("range {} has negative count".format(item["raw"]))
        expected_start = item["end"]
    if valid_ranges and valid_ranges[-1]["end"] != len(systems):
        auto_prob_errors.append(
            "last auto_prob end is {}, systems length is {}".format(valid_ranges[-1]["end"], len(systems))
        )

    raw_group_counts = [len(group) for group in raw_groups]
    auto_counts = [item["count"] for item in valid_ranges]
    raw_group_match = raw_group_counts == auto_counts

    lines = []
    lines.append("# DeePMD run.json systems check")
    lines.append("")
    lines.append("- Generated: {}".format(_datetime.datetime.now().isoformat(timespec="seconds")))
    lines.append("- Host: {}".format(socket.gethostname()))
    lines.append("- run.json: `{}`".format(run_json))
    lines.append("- dataset root: `{}`".format(dataset_root))
    lines.append("- report scope: listed systems, auto_prob range coverage, non-test dataset omissions, type_map.raw prefix order")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("- run.json model.type_map: `{}`".format(expected_type_map))
    lines.append("- listed training systems: {}".format(len(systems)))
    lines.append("- duplicate listed systems: {}".format(len(duplicate_paths)))
    lines.append("- listed systems outside dataset root: {}".format(len(outside_dataset)))
    lines.append("- missing listed system directories: {}".format(len(missing_dirs)))
    lines.append("- missing type_map.raw among existing listed systems: {}".format(len(missing_type_map)))
    lines.append("- invalid type_map.raw order/content among listed systems: {}".format(len(bad_type_map)))
    lines.append("- non-test DeepMD-like dataset systems found under dataset root: {}".format(len(dataset_systems)))
    lines.append("- non-test dataset systems not listed in run.json: {}".format(len(unlisted_dataset_systems)))
    lines.append("- invalid type_map.raw order/content among unlisted non-test dataset systems: {}".format(len(unlisted_bad_type_map)))
    lines.append("- listed dataset paths not recognized by dataset scan: {}".format(len(listed_missing_from_dataset_scan)))
    lines.append("- skipped test-related directory roots during dataset scan: {}".format(skipped_test_dirs))
    lines.append("")
    lines.append("## auto_prob")
    lines.append("")
    lines.append("- raw auto_prob: `{}`".format(auto_prob))
    lines.append("- parsed range counts: `{}`".format(auto_counts))
    lines.append("- raw blank-line group counts in systems array: `{}`".format(raw_group_counts))
    lines.append("- blank-line groups match auto_prob ranges: `{}`".format(raw_group_match))
    lines.append("- range coverage status: `{}`".format("OK" if not auto_prob_errors else "ERROR"))
    if auto_prob_errors:
        lines.extend(format_list(auto_prob_errors))
    lines.append("")
    lines.append("| range | count | prob | status |")
    lines.append("|---|---:|---:|---|")
    for item in auto_ranges:
        if not item.get("valid"):
            lines.append("| `{}` |  |  | ERROR: {} |".format(item["raw"], item.get("error", "invalid")))
            continue
        status = "OK"
        if item["end"] > len(systems):
            status = "end exceeds systems length"
        lines.append("| {}:{} | {} | {} | {} |".format(item["start"], item["end"], item["count"], item["prob"], status))
    lines.append("")

    lines.append("## type_map.raw sequence counts")
    lines.append("")
    if ok_type_map_counter:
        lines.append("| count | sequence |")
        lines.append("|---:|---|")
        for seq, count in ok_type_map_counter.most_common():
            lines.append("| {} | `{}` |".format(count, list(seq)))
    else:
        lines.append("- None")
    lines.append("")

    lines.append("## Missing listed system directories")
    lines.append("")
    lines.extend(format_list(missing_dirs))
    lines.append("")

    lines.append("## Likely replacements for missing listed directories")
    lines.append("")
    if replacement_suggestions:
        lines.append("| missing listed path | close non-test dataset candidates |")
        lines.append("|---|---|")
        for missing_path, matches in replacement_suggestions:
            if matches:
                joined = "<br>".join("`{}`".format(match) for match in matches)
            else:
                joined = "None"
            lines.append("| `{}` | {} |".format(missing_path, joined))
    else:
        lines.append("- None")
    lines.append("")

    lines.append("## Missing type_map.raw in listed systems")
    lines.append("")
    lines.extend(format_list(missing_type_map))
    lines.append("")

    lines.append("## Invalid type_map.raw in listed systems")
    lines.append("")
    if bad_type_map:
        for system_path, type_map_path, values, reason in bad_type_map:
            lines.append("- system: `{}`".format(system_path))
            lines.append("  - type_map.raw: `{}`".format(type_map_path))
            lines.append("  - parsed: `{}`".format(values))
            lines.append("  - reason: {}".format(reason))
    else:
        lines.append("- None")
    lines.append("")

    lines.append("## Duplicate listed systems")
    lines.append("")
    lines.extend(format_list(duplicate_paths))
    lines.append("")

    lines.append("## Listed systems outside dataset root")
    lines.append("")
    lines.extend(format_list(outside_dataset))
    lines.append("")

    lines.append("## Non-test dataset systems not listed in run.json")
    lines.append("")
    lines.extend(format_list(unlisted_dataset_systems))
    lines.append("")

    lines.append("## type_map.raw sequence counts for unlisted non-test dataset systems")
    lines.append("")
    if unlisted_type_map_counter:
        lines.append("| count | sequence |")
        lines.append("|---:|---|")
        for seq, count in unlisted_type_map_counter.most_common():
            lines.append("| {} | `{}` |".format(count, list(seq)))
    else:
        lines.append("- None")
    lines.append("")

    lines.append("## Invalid type_map.raw in unlisted non-test dataset systems")
    lines.append("")
    if unlisted_bad_type_map:
        for system_path, type_map_path, values, reason in unlisted_bad_type_map:
            lines.append("- system: `{}`".format(system_path))
            lines.append("  - type_map.raw: `{}`".format(type_map_path))
            lines.append("  - parsed: `{}`".format(values))
            lines.append("  - reason: {}".format(reason))
    else:
        lines.append("- None")
    lines.append("")

    lines.append("## Listed dataset paths not recognized by dataset scan")
    lines.append("")
    lines.append("These paths are listed in run.json but were not counted as DeepMD-like dataset systems because they are missing a directory, type.raw, or type_map.raw.")
    lines.append("")
    lines.extend(format_list(listed_missing_from_dataset_scan))
    lines.append("")

    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-json", required=True, help="Path to run.json")
    parser.add_argument("--dataset-root", required=True, help="Dataset root to scan")
    parser.add_argument("--report", required=True, help="Output Markdown report path")
    args = parser.parse_args()

    report = build_report(args)
    write_report(normalize_path(args.report), report)
    print("Wrote report: {}".format(normalize_path(args.report)))


if __name__ == "__main__":
    main()
