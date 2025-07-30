#!/usr/bin/env python3
"""
format_check.py  –  structural-format checker for two-modes output

Usage
  $ python format_check.py path/to/output_dir/

The checker validates the file against the block-structure rules laid out in
FORMAT_SPEC (see `main.py`).  Each rule is implemented as an independent
function that can be switched on/off in the `CHECKS` list below.
"""

import argparse
import glob
import os
import re
import sys
from typing import Callable, Tuple, List, Dict

# --------------------------------------------------------------------------- #
#  utility helpers
# --------------------------------------------------------------------------- #

TAG_NAMES = ("PLANNING", "IMP", "VERIFY", "REVIEW", "ANSWER")
TAG_RE    = re.compile(r"\[(/?)(%s)\]" % "|".join(TAG_NAMES))

def find_blocks(tag: str, text: str) -> List[str]:
    """Return a list with the inner text of every `[tag] … [/{tag}]` block."""
    pattern = re.compile(rf"\[{tag}](.*?)\[/{tag}]", re.DOTALL | re.MULTILINE)
    return pattern.findall(text)

# --------------------------------------------------------------------------- #
#  individual check implementations
#   •  each returns (passed: bool, message: str)
# --------------------------------------------------------------------------- #

def check_last_verdict(text: str) -> Tuple[bool, str]:
    blocks = find_blocks("VERIFY", text)
    if not blocks:
        return False, "No VERIFY block found"
    last = blocks[-1].strip()
    if re.search(r"\\boxed\{(?:correct|wrong)\}\s*$", last):
        return True, ""
    return False, "Last VERIFY block lacks a boxed verdict"

def check_answer_logic(text: str) -> Tuple[bool, str]:
    verify_iter = list(re.finditer(r"\[VERIFY](.*?)\[/VERIFY]", text, re.DOTALL))
    if not verify_iter:
        return False, "No VERIFY block found"
    last_verify_match = verify_iter[-1]
    verdict_m = re.search(r"\\boxed\{(correct|wrong)\}", last_verify_match.group(1))
    if not verdict_m:
        return False, "Verdict missing in last VERIFY block"

    verdict = verdict_m.group(1)
    answer_iter = list(re.finditer(r"\[ANSWER](.*?)\[/ANSWER]", text, re.DOTALL))
    remainder   = text[last_verify_match.end():]

    if verdict == "correct":
        if len(answer_iter) != 1:
            return False, "Verdict is correct but ANSWER block count != 1"
        if not re.match(r"^\s*\[ANSWER]", remainder):
            return False, "ANSWER block not immediately after last VERIFY"
    else:  # ‘wrong’
        if answer_iter:
            return False, "ANSWER block present despite wrong verdict"

    return True, ""

def check_answer_count(text: str) -> Tuple[bool, str]:
    cnt = cnt = len(re.findall(r"\[ANSWER](.*?)\[/ANSWER]", text, re.DOTALL))
    if cnt == 1:
        return True, ""
    return False, f"Expected exactly 1 ANSWER block, found {cnt}"

def check_balanced_tags(text: str) -> Tuple[bool, str]:
    for tag in TAG_NAMES:
        opens  = len(re.findall(rf"\[{tag}]", text))
        closes = len(re.findall(rf"\[/{tag}]", text))
        if opens != closes:
            return False, f"Unbalanced tag {tag}: {opens} open vs {closes} close"
    return True, ""

def check_planning_header(text: str) -> Tuple[bool, str]:
    for block in find_blocks("PLANNING", text):
        first_line = block.lstrip().splitlines()[0]
        if not re.search(r"\(\s*(Planning|Implementation|Verification|Review)\s*\)$",
                         first_line):
            return False, "PLANNING block missing action header: " + repr(first_line)
    return True, ""

def check_planning_isolated(text: str) -> Tuple[bool, str]:
    pattern = re.compile(r"\[PLANNING](.*?)\[/PLANNING]", re.DOTALL)
    for m in pattern.finditer(text):
        inner = m.group(1)
        if re.search(r"\[(IMP|VERIFY|REVIEW)]", inner):
            return False, "Sub-tag found inside PLANNING-only block"
    return True, ""

def _check_tag_sequence(text: str, tag: str, keyword: str) -> Tuple[bool, str]:
    """
    Validate that **every** action block `[TAG] … [/TAG]`
    (where TAG is IMP, VERIFY or REVIEW) obeys two constraints:

    1.  It is *immediately* preceded—ignoring only whitespace and new‑lines—
        by a `[PLANNING] … [/PLANNING]` block.

    2.  The body of that planning block contains the required `keyword`
        (e.g. ``"Goal of this step:"``), case‑insensitively, if
        `keyword` is a non‑empty regex.

    Returns ``(True, "")`` when all such blocks conform; otherwise
    ``(False, <reason>)`` describing the first violation encountered.
    """
    # Compile once for speed and clarity.
    action_pat   = re.compile(rf"\[{tag}](.*?)\[/{tag}]", re.DOTALL)
    planning_pat = re.compile(r"\[PLANNING](.*?)\[/PLANNING]", re.DOTALL)

    # Iterate over **every** action block in the document
    for am in action_pat.finditer(text):
        action_start = am.start()
        action_line = text.count('\n', 0, action_start) + 1

        # -------------------------------------------------------------
        # Find the *nearest* preceding PLANNING block, if any
        # -------------------------------------------------------------
        preceding_text = text[:action_start]
        plan_match = None
        for pm in planning_pat.finditer(preceding_text):
            plan_match = pm  # last (nearest) match wins

        if plan_match is None:
            return False, f"{tag} block at line {action_line} has no preceding PLANNING block"

        # -------------------------------------------------------------
        # Ensure the action block comes *immediately* after [/PLANNING]
        # (only whitespace/newlines are allowed in between)
        # -------------------------------------------------------------
        interstitial = preceding_text[plan_match.end():]
        if re.search(r"\S", interstitial):
            return False, (
                f"{tag} block at line {action_line} is not immediately "
                "after its PLANNING block"
            )

        # -------------------------------------------------------------
        # Keyword / header check inside the planning body
        # -------------------------------------------------------------
        if keyword and not re.search(keyword, plan_match.group(1), re.IGNORECASE):
            return (
                False,
                f"PLANNING block before {tag} (at line {action_line}) "
                "is missing the required header",
            )

    # If we get here, *all* action blocks are correctly paired.
    return True, ""

def check_impl_pair(text: str) -> Tuple[bool, str]:
    ok, msg = _check_tag_sequence(text, "IMP", r"Goal\s+of\s+this\s+step:")
    return ok, msg or ""

def check_verif_pair(text: str) -> Tuple[bool, str]:
    ok, msg = _check_tag_sequence(text, "VERIFY",
                                  r"Scope\s+of\s+this\s+verification:")
    return ok, msg or ""

def check_review_pair(text: str) -> Tuple[bool, str]:
    ok, msg = _check_tag_sequence(text, "REVIEW", "")
    return ok, msg or ""

def check_verify_verdict_each(text: str) -> Tuple[bool, str]:
    for blk in find_blocks("VERIFY", text):
        if not re.search(r"\\boxed\{(?:correct|wrong)\}", blk):
            return False, "A VERIFY block is missing its boxed verdict"
    return True, ""

def check_numbered_list_in_planning_only(text: str) -> Tuple[bool, str]:
    """
    Check that every PLANNING-only block (i.e., one marked with '(Planning)' in
    the header) includes a numbered list beginning with '1.'.
    """
    for blk in find_blocks("PLANNING", text):
        lines = blk.lstrip().splitlines()
        if not lines:
            continue  # skip empty
        header = lines[0]
        if not re.search(r"\(\s*Planning\s*\)$", header, re.IGNORECASE):
            continue  # only check planning-only blocks
        if not re.search(r"^\s*1\.", blk, re.MULTILINE):
            return False, "Planning-only block without numbered list"
    return True, ""

def check_action_type_uniqueness(text: str) -> Tuple[bool, str]:
    segments = re.split(r"\[PLANNING]", text)[1:]           # first split is preamble
    for seg in segments:
        step = "[PLANNING]" + seg
        # look only up to the next [PLANNING] or end
        step_body = step.split("[PLANNING]", 1)[0]
        kinds = set(t for t in ("IMP", "VERIFY", "REVIEW") if f"[{t}]" in step_body)
        if len(kinds) > 1:
            return False, f"Multiple action tags {kinds} in one step"
    return True, ""

def check_no_answer_before_last_verify(text: str) -> Tuple[bool, str]:
    last_verify = list(re.finditer(r"\[VERIFY](.*?)\[/VERIFY]", text, re.DOTALL))
    if not last_verify:
        return False, "No VERIFY block found"
    last_end = last_verify[-1].end()
    first_answer = re.search(r"\[ANSWER]", text)
    if first_answer and first_answer.start() < last_end:
        return False, "ANSWER block appears before last VERIFY"
    return True, ""

# --------------------------------------------------------------------------- #
#  registry of checks  ——  toggle ‘enabled’ to False to skip a rule
# --------------------------------------------------------------------------- #

Check = Tuple[str, bool, Callable[[str], Tuple[bool, str]]]

CHECKS: List[Check] = [
    ("Last VERIFY ends with verdict",             False, check_last_verdict),
    ("ANSWER presence vs verdict",                False, check_answer_logic),
    ("Exactly one ANSWER block",                  True, check_answer_count),
    ("Balanced tag counts",                       True, check_balanced_tags),
    ("PLANNING header format",                    True, check_planning_header),
    ("PLANNING block isolation",                  True, check_planning_isolated),
    ("Implementation block pairing",              True, check_impl_pair),
    ("Implementation planning header",            True, check_impl_pair),     # part of same func
    ("Verification block pairing",                True, check_verif_pair),
    ("Verification planning header",              True, check_verif_pair),    # same func
    ("Each VERIFY has verdict",                   True, check_verify_verdict_each),
    ("Review block pairing",                      True, check_review_pair),
    ("Review planning header",                    True, check_review_pair),   # same func
    ("Numbered list in planning-only step",       True, check_numbered_list_in_planning_only),
    ("Unique action tag per step",                True, check_action_type_uniqueness),
    ("No ANSWER before last VERIFY",              False, check_no_answer_before_last_verify),
]

# --------------------------------------------------------------------------- #
#  main driver
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(description="Validate two-modes output format.")
    parser.add_argument("--directory", metavar="DIR",
                        help="directory with parsed_output_*.txt files")
    parser.add_argument("--disable", nargs="*", default=[],
                        help="names (or 1-based indices) of checks to disable")
    args = parser.parse_args()

    # disable requested checks by name or index
    for flag in args.disable:
        for i, (name, enabled, func) in enumerate(CHECKS):
            if flag == str(i + 1) or flag.lower() in name.lower():
                CHECKS[i] = (name, False, func)

    file_pattern = os.path.join(args.directory, "tmath_*.txt")
    files_to_check = glob.glob(file_pattern)

    if not files_to_check:
        sys.exit(f"No files found matching '{file_pattern}'")

    file_map = {}
    for f in files_to_check:
        match = re.search(r"tmath_(\d+)\.txt$", f)
        if match:
            file_map[int(match.group(1))] = f

    passed_indices = []
    failed_indices = []

    for idx in sorted(file_map.keys()):
        filepath = file_map[idx]
        print(f"--- Checking {os.path.basename(filepath)} ---")

        try:
            with open(filepath, "r", encoding="utf-8") as f_handle:
                text = f_handle.read()
        except OSError as e:
            print(f"Error reading {filepath}: {e}", file=sys.stderr)
            failed_indices.append(idx)
            continue

        failed: Dict[str, str] = {}
        for check_idx, (name, enabled, func) in enumerate(CHECKS, 1):
            if not enabled:
                continue
            passed, msg = func(text)
            if not passed:
                failed[f"{check_idx}. {name}"] = msg or "unnamed failure"

        if failed:
            failed_indices.append(idx)
            print("❌  Format check failed on:")
            for label, reason in failed.items():
                print(f"  • {label}: {reason}")
        else:
            passed_indices.append(idx)
            print("✅  All enabled format checks passed.")
        print()

    print("=" * 40)
    print("Summary:")
    print(f"Passed indices ({len(passed_indices)}): {passed_indices}")
    print(f"Failed indices ({len(failed_indices)}): {failed_indices}")

    if failed_indices:
        sys.exit(1)


if __name__ == "__main__":
    main()