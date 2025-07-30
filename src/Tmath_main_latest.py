import os, json, re, time
from typing import List
from datasets import load_dataset
from openai import OpenAI
import httpx
import numpy as np
import requests
import Tmath as tmath
# Important settings
THINKING_BUDGET = 10240
TEMPERATURE = 0.8
SAVE_PROMPTS_AND_REASONING = True
DEBUG_MODE = False
DEBUG_STEP = ["Subgoal Breakdown", "Recall Info", "Track Goals", "Recall Info", "Subgoal Breakdown", "Reach Goal", "Self-Verification"]
EXPLORATION_EPSILON = 0.3
EXPLORATION_ACTIONS = ["Backward Subgoaling", "Track Goals", "Recall Info"]
MODE = "self-hosted"  # "self-hosted" or "cloud"
# Other settings
API_KEY = "sk-ce05d1c1589446309d8cb09e637ad16e"  # aliyun
# API_KEY          = "sk-d6a8b15dff344429928510050ef7a107"    # nanhu
# API_KEY          = "sk-89eb993b390f4918977dd1537b0d55b6"    # local
DATA_SAVE_PATH = "final-version"
SMALL_MODEL = "qwen3-32b"  # normal reasoning
LARGE_MODEL = "qwen3-32b"  # deep‑thinking verifier
MAX_ITERATIONS = 6
DATASET_NAME = "yentinglin/aime_2025"
GLOBAL_IMMEDIATE_PROMPT = (
    "Now choose the best next ONE step of reasoning based on the previous steps and the planning results, following the block rules. \n"
    "Ensure that the generated content remains natural, coherent and readable if all [PLANNING] … [/PLANNING] blocks are removed. \n"
    "You should perform self-verification when you are unsure about the correctness of the previous steps during the process and always perform one self-verification when you think the solution is complete.")

TAG_NAMES = ("PLANNING", "IMP", "VERIFY", "REVIEW", "ANSWER")
TAG_RE = re.compile(r"\[(/?)(%s)\]" % "|".join(TAG_NAMES))

# Machine‑friendly success flag
COMPLETE_RE = re.compile(r"\[ANSWER\][\s\S]*?\[/ANSWER\]", re.I)

# System prompt
TOOL_LIST = """
None
"""

FORMAT_SPEC = (f"""
IMPORTANT — Strict Formatting Guidelines
These rules apply to every response and must be followed exactly.
⸻

Meta-Cognitive Action Set (M)

Think of every step you take as one of four action types. Pick exactly one action for each step.

Type	What it means	Pick one of these actions
P – Planning	Decide what to do next	• Subgoal Breakdown
I – Implementation	Do the actual implementation to reach the goal	• Reach Goal
V – Verification	Check your own work	• Self-Verification
R – Review	Organise or recall context	• Track Goals  • Recall Info


Action definitions:
• Subgoal Breakdown – generate a list of specific subgoals (to-dos) starting from the current state and leading to the final goal. It focuses solely on planning future steps and does not need to include past subgoals that have already been completed. This action can update and adjust its future plan dynamically without worrying about aligning with previously completed tasks.
• Reach Goal – Apply logical deductions, calculations, and tools (following the Tool Invocation Format) to reach the current goal or subgoal. 
• Self‑Verification – Critically evaluate your own reasoning and conclusions for correctness and consistency in a detailed and elaborate way.
• Track Goals – provide a complete overview of goal progress by combining past completed subgoals, the current state, and the latest future plans into a single, coherent list. Each subgoal should be labeled with its status: completed, in progress, or to do. Do not perform any calculations or logical deductions in this step.
• Recall Info – Summarize prior information relevant to the current goal. Do not perform any calculations or logical deductions in this step.

⸻

Tool Invocation Format
Whenever you invoke an external tool inside an **[IMP]** block, wrap the call and its
response exactly as follows (no extra spaces or text in-between):

    <tool> tool_name(parameter1,parameter2,…) </tool> <output> … </output>

• Replace `tool_name` with the precise name of the tool (e.g. calculator, web_search).  
• Put any input parameters of the tool within the `(input_parameters)` span.  
• Put the raw tool response verbatim inside the `<output> … </output>` span.  
• These two spans are treated as environment feedback and must **not** be imitated; they will be masked out of the loss.  

Here is a list of available tools you can use:
{TOOL_LIST}
⸻
"""
               r"""
               Tag blocks you must emit
               
               For **each** step choose exactly **one** action type and output the blocks listed below:
               
               | Action (letter) | Blocks you must output | What to write inside |
               | --- | --- | --- |
               | **Planning (P)** | `[PLANNING] … [/PLANNING]` | Inside **[PLANNING]** first write the name of the planning action and append the action type as (Planning), then write a numbered list of goals / sub‑goals. *No reasoning or explanations.* |
               | **Implementation (I)** | `[PLANNING] … [/PLANNING]`<br>`[IMP] … [/IMP]` | In **[PLANNING] … [/PLANNING]** first write the name of the implementation action and append the action type as (Implementation), and then write `Goal of this step:` followed by the exact goal. In **[IMP]** provide the detailed process that accomplishes that goal. |
               | **Verification (V)** | `[PLANNING] … [/PLANNING]`<br>`[VERIFY] … [/VERIFY]` | In **[PLANNING] … [/PLANNING]** first write the name of the verification action and append the action type as (Verification), and then write `Scope of this verification:` and clearly state what will be checked. In **[VERIFY]** perform the check and end with `\boxed{correct}` **or** `\boxed{wrong}`. |
               | **Review (R)** | `[PLANNING] … [/PLANNING]`<br>`[REVIEW] … [/REVIEW]` | In **[PLANNING] … [/PLANNING]** first write the name of the review action and append the action type as (Review), and then write the meta data for the current review action. For **Recall Info** say what to recall; for **Track Goals** restate the complete goal list. In **[REVIEW] … [/REVIEW]** write the review content. For **Recall Info**, list the information relevant to what to recall. For **Track Goals**, write the current active subgoal. Do not perform any calculations or logical deductions in these actions.|
               
               Always append the action type in parentheses after the action name inside the [PLANNING] … [/PLANNING] block. For example, Subgoal Breakdown (Planning), Reach Goal (Implementation), Self-Verification (Verification), Track Goals (Review).
               
               Here are some examples of blocks:
               - Example 1:
               ```
               [PLANNING] Subgoal Breakdown (Planning)
               1. …
                   1.1 …
               2. …
                   2.1 …
               [/PLANNING]
               ```
               - Example 2:
               ```
               [PLANNING] Reach Goal (Implementation)
               Goal of this step: …
               [/PLANNING]
               [IMP] 
               {Implementation details here}
               [/IMP]
               ```
               - Example 3:
               ```
               [PLANNING] Self-Verification (Verification)
               Scope of this verification: …
               [/PLANNING]
               [VERIFY] 
               {Verification details here}
               [/VERIFY]
               ```
               - Example 4:
               ```
               [PLANNING] Track Goals (Review)
               Our goals are:
               1. … (done)
                   1.1 … (done)
                   1.2 … (done)
               2. … (in progress)
                   2.1 … (in progress)
               3. … (to do)
                   3.1 … (to do)
               [/PLANNING]
               [REVIEW] 
               I am focusing on ... {current in progress subgoal here}.
               [/REVIEW]
               ```
               - Example 5:
               ```
               [PLANNING] Recall Info (Review)
               What to recall: …
               [/PLANNING]
               [REVIEW] 
               I am listing relevant information for ... {what to recall here} here: {relevant information here}.
               [/REVIEW]
               ```
               """)  # raw‑string to preserve backslashes

# Create DATA_SAVE_PATH directory if it doesn't exist
os.makedirs(DATA_SAVE_PATH, exist_ok=True)



client = OpenAI(
    api_key="ollama",
    #base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  # aliyun
    base_url="http://localhost:6068/",   # nanhu
    # base_url="http://localhost:8080/api",   # local
    # base_url="http://154.64.247.141:31922/v1"
)


def balance_tags(text: str) -> str:
    """
    Iteratively finds and fixes tag balancing issues one by one.

    This function repeatedly scans the text for a single formatting error and
    fixes it. After each fix, it rescans from the beginning to ensure that
    the fix didn't create new issues.


    The process continues until a full pass is made with no errors found. A
    safety limit of 30 iterations prevents any potential for infinite loops.
    """
    NON_NESTING = {"PLANNING", "IMP", "VERIFY", "REVIEW", "ANSWER"}

    for i in range(30):  # Safety brake: max 30 repairs
        stack = []  # Stores tuples of (tag_name, match_object)
        matches = list(TAG_RE.finditer(text))

        # --- Find the first error and fix it ---
        error_found = False
        for m in matches:
            closing, tag = m.groups()

            if not closing:  # --- Opening Tag ---
                # NEW LOGIC: If any non-nesting block is open, close it before opening a new one.
                if stack and stack[-1][0] in NON_NESTING:
                    # ERROR: A new block is opening while another one is open.
                    # Close the currently open block.
                    last_open_tag_name = stack.pop()[0]
                    closing_str = f"[/{last_open_tag_name}]\n\n"  # Add newlines for style

                    # Insert the closing tag right before the current opening tag.
                    text = text[:m.start()] + closing_str + text[m.start():]
                    print(f"Auto-closed [/{last_open_tag_name}] before new opening tag '[{tag}]'.")
                    error_found = True
                    break  # Restart the scan

                stack.append((tag, m))

            else:  # --- Closing Tag ---
                if not stack or stack[-1][0] != tag:
                    # ERROR 2: Mismatched or orphan closing tag.
                    # We simply delete the tag.
                    text = text[:m.start()] + text[m.end():]
                    print(f"Removed mismatched/orphan closing tag [/{tag}].")
                    error_found = True
                    break  # Restart the scan from the beginning.

                stack.pop()  # Correctly matched closing tag.

        if error_found:
            continue  # Go to the next iteration of the main `for` loop to rescan.

        # --- If no errors were found in the pass, do final cleanup ---
        if stack:
            # ERROR 3: Unclosed tags at the end of the file.
            closing_str = ""
            for t_name, _ in reversed(stack):
                closing_str += f"\n[/{t_name}]"
            text += closing_str
            print(f"Appended {len(stack)} closing tag(s) at EOF: "
                  + ", ".join(f"[/{t}]" for t, _ in reversed(stack)))
            # Loop one more time to ensure the final state is clean.
            continue

        # If we reach here, a full pass was completed with no errors found.
        return text

    # If the loop completes, we hit the iteration limit.
    print("Warning: Tag balancing failed to converge after 30 iterations.")
    return text


def strip_whitespace_in_action_header(text: str) -> str:
    """
    Normalises headers like (  planning ) to (Planning).
    This is now idempotent and only logs when a change is actually made.
    """
    ACTION_WORDS = "(Planning|Implementation|Verification|Review)"
    ACTION_RE = re.compile(rf"\(\s*{ACTION_WORDS}\s*\)", re.IGNORECASE)

    def repl_and_log(m: re.Match) -> str:
        original_header = m.group(0)
        word = m.group(1)
        normalized_header = f"({word.title()})"
        if original_header != normalized_header:
            print(f"Fixed action header: '{original_header}' -> '{normalized_header}'")
        return normalized_header

    return ACTION_RE.sub(repl_and_log, text)


def normalize_block_whitespace(text: str) -> str:
    """
    Recursively collapses excessive newlines and trims whitespace inside blocks.
    - Trims leading/trailing whitespace inside each block.
    - Replaces 3+ newlines with 2.
    - Ensures a single newline after opening and before closing tags.
    """
    tag_list = "|".join(TAG_NAMES)
    pattern = re.compile(rf"\[({tag_list})\](.*?)\[/\1\]", re.DOTALL)

    def cleaner(content: str, tag_name: str) -> str:
        """Cleans a content string, returning the new string."""
        # Recurse to clean inner blocks first
        inner_cleaned_content = recursive_cleaner(content)

        # 1. Strip leading/trailing whitespace from content
        new_content = inner_cleaned_content.strip()
        if new_content != inner_cleaned_content:
            print(f"Fixed whitespace in [{tag_name}]: Trimmed leading/trailing space.")

        # 2. Collapse >2 newlines to 2
        content_after_collapse = re.sub(r'\n{3,}', '\n\n', new_content)
        if content_after_collapse != new_content:
            print(f"Fixed whitespace in [{tag_name}]: Collapsed excessive newlines.")
        new_content = content_after_collapse

        # 3. Remove trailing whitespace from lines with action headers
        ACTION_WORDS = "(Planning|Implementation|Verification|Review)"
        content_after_trim = re.sub(rf"(\({ACTION_WORDS}\))\s*$", r"\1", new_content, flags=re.MULTILINE)
        if content_after_trim != new_content:
            print(f"Fixed whitespace in [{tag_name}]: Removed trailing space from action header.")

        return content_after_trim

    def recursive_cleaner(text_to_clean: str) -> str:
        """Recursive helper for cleaning."""

        def replacer(m: re.Match) -> str:
            tag_name, content = m.groups()

            # Clean the content of this block
            cleaned_content = cleaner(content, tag_name)

            # Re-assemble with Fixed newlines
            if not cleaned_content:
                if content.strip():
                    print(f"Emptied block [{tag_name}] after content normalization.")
                return f"[{tag_name}][/{tag_name}]"

            new_block = f"[{tag_name}]\n{cleaned_content}\n[/{tag_name}]"
            # Log if newlines were added/Fixed around the content
            if not content.startswith('\n') or not content.endswith('\n'):
                # This check is imperfect but captures the most common case of adding newlines.
                if content.strip() == cleaned_content:
                    print(f"Fixed newlines around content in [{tag_name}] block.")

            return new_block

        # Apply the replacer to all blocks at the current level
        return pattern.sub(replacer, text_to_clean)

    final_text = recursive_cleaner(text)
    if final_text != text:
        print("Fixed whitespace within one or more blocks.")

    return final_text


def normalize_inter_step_newlines(text: str) -> str:
    """
    Ensures there are exactly two newlines between each logical reasoning step.
    This helps readability by creating a clear visual separation.

    For example, it converts this:
    ...
    [/IMP]
    [PLANNING]...

    Into this:
    ...
    [/IMP]

    [PLANNING]...
    """
    # This regex looks for a step-ending tag, followed by any amount of whitespace,
    # followed by the start of a new [PLANNING] block, which always starts a new step.
    # It replaces the whitespace with exactly two newlines.
    pattern = re.compile(
        r"(\[/(?:IMP|VERIFY|REVIEW|PLANNING)\])\s*(\[PLANNING\])",
        re.DOTALL
    )

    original_text = text
    fixed_text = pattern.sub(r"\1\n\n\2", text)

    if fixed_text != original_text:
        print("Adjusted newlines between reasoning steps to ensure double spacing.")

    return fixed_text


def normalize_newlines_after_tags(text: str) -> str:
    """
    Ensures there is exactly one newline after each closing tag (except for ANSWER).
    This rule is ignored if the tag is already followed by two newlines, which
    separate reasoning steps.
    - `[/TAG]\\n\\n` is left alone.
    - `[/TAG]  [OTHER]` becomes `[/TAG]\\n[OTHER]`
    - `[/TAG][OTHER]` becomes `[/TAG]\\n[OTHER]`
    """
    # Exclude ANSWER from the tags to be processed.
    tags_to_fix = "|".join([t for t in TAG_NAMES if t != "ANSWER"])

    # This regex looks for a closing tag, then captures all whitespace until the
    # next non-whitespace character. The negative lookahead (?!\\n\\n) ensures
    # we don't touch the double newlines that separate steps.
    pattern = re.compile(rf"(\[/(?:{tags_to_fix})\])(\s*)(?!\s*\n\s*\n)", re.DOTALL)

    original_text = text
    # Replace the whitespace with a single newline, but only if it wasn't a double.
    fixed_text = pattern.sub(r"\1\n", text)

    if fixed_text != original_text:
        print("Fixed single newlines after closing tags.")

    return fixed_text


def keep_first_step_only(text: str) -> str:
    """
    If the LLM generates more than one reasoning step, this function keeps only the first one.
    It finds all [PLANNING] tags and truncates the text before the second one.
    """
    planning_tag = "[PLANNING]"
    # Use re.escape in case the tag contains special regex characters
    matches = list(re.finditer(re.escape(planning_tag), text))

    if len(matches) > 1:
        # Get the start index of the second match
        second_match_start = matches[1].start()
        # Truncate the string to keep everything before the second [PLANNING] tag
        fixed_text = text[:second_match_start].strip()

        print(f"Fixed multi-step response: Kept only the first of {len(matches)} steps.")
        return fixed_text

    return text


def fix_misplaced_verdict(text: str) -> str:
    """
    Moves a \\boxed{} verdict that appears immediately after a [/VERIFY] tag to be inside it.
    This also handles cases where the verdict is surrounded by dollar signs.

    For example, it converts this:
    ...
    [/VERIFY]
    $\\boxed{wrong}$

    Into this:
    ...
    $\\boxed{wrong}$
    [/VERIFY]
    """
    # This regex finds a VERIFY block followed by a misplaced verdict.
    # It captures the block's components and the verdict (with or without surrounding '$')
    # to reassemble them correctly.
    pattern = re.compile(
        r"(\[VERIFY\])(.*?)(\[/VERIFY\])(\s*)((?:\\boxed\{(?:correct|wrong)\})|(?:\$\s*\\boxed\{(?:correct|wrong)\}\s*\$))",
        re.DOTALL
    )

    original_text = text

    def replacer(m: re.Match) -> str:
        open_tag, content, close_tag, _, verdict = m.groups()
        # Reconstruct the block with the verdict now correctly placed inside.
        new_content = content.strip() + f"\n\n{verdict.strip()}"
        return f"{open_tag}\n{new_content.strip()}\n{close_tag}"

    fixed_text = pattern.sub(replacer, text)

    if fixed_text != original_text:
        print("Fixed misplaced verdict: Moved a \\boxed{} from outside to inside a [/VERIFY] block.")

    return fixed_text


def apply_formatting_fixes(text: str) -> str:
    """
    Applies all formatting fixes to ensure consistent and clean output format.
    """
    print("--- Running Formatting Fixes ---")
    text = keep_first_step_only(text)  # Trim response to the first reasoning step only.
    text = balance_tags(text)  # Fix mismatched or unclosed [TAG]...[/TAG] blocks.
    text = fix_misplaced_verdict(text)  # Move misplaced \\boxed{} verdicts into the [VERIFY] block.
    text = strip_whitespace_in_action_header(text)  # Clean action headers, e.g., "(  Planning )" -> "(Planning)".
    text = normalize_block_whitespace(text)  # Normalize whitespace and newlines inside blocks.
    text = normalize_inter_step_newlines(text)  # Ensure two newlines between reasoning steps.
    text = normalize_newlines_after_tags(text)  # Ensure single newlines after closing tags.
    print("--- Formatting Fixes Complete ---\n")
    return text


def get_immediate_prompt(solution_text: str, allow_exploration: bool = True) -> str:
    """
    Determines the next immediate prompt based on DEBUG_MODE settings and exploration probability.
    """
    if DEBUG_MODE:
        step_count = solution_text.count("[PLANNING]")
        if step_count < len(DEBUG_STEP):
            step_name = DEBUG_STEP[step_count]
            prompt_lines = GLOBAL_IMMEDIATE_PROMPT.split('\n')
            prompt_lines[0] = f"Now perform the following step: {step_name}."
            return "\n".join(prompt_lines)

    if allow_exploration and np.random.rand() < EXPLORATION_EPSILON:
        action = np.random.choice(EXPLORATION_ACTIONS)
        prompt_lines = GLOBAL_IMMEDIATE_PROMPT.split('\n')
        prompt_lines[0] = f"Now perform the following step: {action}."
        return "\n".join(prompt_lines)

    return GLOBAL_IMMEDIATE_PROMPT




def build_prompt(statement: str, body: str, immediate_prompt: str) -> str:
    """
    Assemble the full prompt from the statement, body, and immediate prompt.
    """
    return (
        f"{statement}\n"
        f"{body.strip()}\n"
        f"{immediate_prompt}\n"
    )


def call_llm(prompt: str, *, deep: bool, model: str, fix_format: bool = False) -> str:
    if not deep:
        prompt += "\n\\no_think"

    messages = [
        {
            "role": "system",
            "content": " You are a helpful reasoning model. Your response should follow the following formatting guidelines:\n\n" + FORMAT_SPEC
        },
        {"role": "user", "content": prompt},
    ]

    # completion = client.chat.completions.create(
    #     model=model,
    #     messages=messages,
    #     temperature=TEMPERATURE,
    #     max_tokens=4096 * 2,
    #     stream=True,
    #     extra_body={
    #         "enable_thinking": deep,
    #         "thinking_budget": THINKING_BUDGET
    #     },
    # )
    completion = requests.post(
        "http://localhost:6068/api/chat",
        json={
            "model": "qwen3:32b",
            "messages": messages,
        },
        stream=True
    )
    reasoning = ""
    response = ""

    if MODE == "cloud":
        for chunk in completion:
            delta = json.loads(chunk)#chunk.choices[0].delta
            if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                reasoning += delta.reasoning_content
            if hasattr(delta, "content") and delta.content:
                response += delta.content
    elif MODE == "self-hosted":

        for line in completion.iter_lines():
            if line:
                # 解析JSON块
                chunk = json.loads(line.decode('utf-8'))
                if 'message' in chunk and 'content' in chunk['message']:
                    print(chunk['message']['content'], end='', flush=True)

                response += chunk['message']['content']
        # reasoning content is in <think>...</think>
        think_match = re.search(r"<think>([\s\S]*?)</think>", response)
        if think_match:
            reasoning = think_match.group(1).strip()
            response = re.sub(r"<think>[\s\S]*?</think>", "", response).strip()
        else:
            reasoning = ""
            response = response.strip()

    if fix_format:
        response = apply_formatting_fixes(response)

    return reasoning.strip(), response.strip()


def run_solution_pipeline(prob_id: str, statement: str, initial_sol: str, initial_prompt: str, passes: list):
    """
    Runs the main iterative solution generation loop.

    This function encapsulates the logic for alternating between a shallow-thinking
    phase (using a smaller model) and a deep-thinking verification phase (using
    a larger model). It builds up a solution step by step, appends each step
    to the `passes` list, and terminates when a complete answer is found or
    the maximum number of iterations is reached.
    """
    iterations = 0
    cot_step_counter = 0

    phase1_sol = initial_sol
    phase1_prompt = initial_prompt

    while iterations < MAX_ITERATIONS:
        # ——— (deep‑thinking OFF) ——— #
        no_verify_counter = 0
        # The inner loop generates reasoning steps until a verification block is produced.
        while "[VERIFY]" not in phase1_sol:
            no_verify_counter += 1
            if no_verify_counter > 10:  # use a large number to temporarily disable this check
                print(f"No [VERIFY] steps found in solutions. Proceed to verification anyway.")
                break

            if SAVE_PROMPTS_AND_REASONING:
                with open(f"{DATA_SAVE_PATH}/tmath_{prob_id}_prompt_{cot_step_counter}.txt", "w",
                          encoding="utf-8") as f:
                    f.write(phase1_prompt)
                cot_step_counter += 1
                print(f"CoT Step {cot_step_counter}.")
            _, step_sol = call_llm(phase1_prompt, deep=False, model=SMALL_MODEL, fix_format=True)

            if "[VERIFY]" not in step_sol:  # only check early termination for non-verify steps
                # Terminate if the solution is complete
                if COMPLETE_RE.search(step_sol):
                    # Combine the accumulated solution and this step
                    final_sol = (phase1_sol + '\n' + step_sol).strip()

                    # Record the shallow‑thinking pass so the provenance is not lost
                    passes.append({
                        "mode": "no_think",
                        "deep": False,
                        "model": SMALL_MODEL,
                        "text": final_sol,
                    })

                    # Optionally save the final prompt for debugging / auditing
                    final_prompt_to_save = build_prompt(
                        statement,
                        f"{final_sol}\n",
                        ""
                    )
                    if SAVE_PROMPTS_AND_REASONING:
                        with open(f"{DATA_SAVE_PATH}/tmath_{prob_id}_prompt_{cot_step_counter}.txt",
                                  "w", encoding="utf-8") as f:
                            f.write(final_prompt_to_save)
                    cot_step_counter += 1
                    print(f"CoT Step {cot_step_counter}.")
                    print("[ANSWER] emitted in shallow thinking pass.")
                    return

            phase1_sol = phase1_sol + '\n' + step_sol + '\n'

            immediate_prompt = get_immediate_prompt(phase1_sol)
            phase1_prompt = build_prompt(
                statement,
                f"{phase1_sol}\n",
                immediate_prompt
            )

        # if SAVE_PROMPTS_AND_REASONING:
        #     with open(f"tmath_two_modes_{prob_id}_prompt_{prompt_save_counter}.txt", "w", encoding="utf-8") as f:
        #         f.write(phase1_prompt)
        #     prompt_save_counter += 1

        # Truncate everything from the first [VERIFY] tag onward (if any)
        verify_chunk = phase1_sol.split("[VERIFY]")
        if len(verify_chunk) > 1:
            sol_before_verify, _ = phase1_sol.rsplit("[PLANNING]", 1)
            sol_core = sol_before_verify.strip()
        else:
            sol_core = phase1_sol

        passes.append({
            "mode": "no_think",
            "deep": False,
            "model": SMALL_MODEL,
            "text": sol_core,  # Use "sol_core" to exclude shallow self-verification, use "phase1_sol" to include it.
        })

        # ——— Verification (deep‑thinking ON) ——— #
        immediate_prompt = (
            "Now perform one self‑verification step only, following the prescribed format. \n"
            "Ensure that the generated content remains natural, coherent and readable if all [PLANNING] … [/PLANNING] blocks are removed. \n"
            "First, identify the scope of the verification. If this is the last verification step before you submit the final answer, the scope of the verification should cover all critical aspects of the solution. \n"
            "Then, think deeply and critically but reveal only the [PLANNING] … [/PLANNING] + [VERIFY] … [/VERIFY] blocks. \n"
            "Just identify problems, if any, in this step. Do not try to fix them. Leave the fixing to future steps.\n"
            "Make sure the content in the [VERIFY] … [/VERIFY] block reflects your critical thinking, and is elaborate and detailed. Write down the detailed process of how you reach your conclusion. Do not jump to conclusions.\n"
            "The content of verification should be under the scope of the verification. \n"
            r"If your verification reveals an error in the previous reasoning, conclude with \boxed{correct}."
            r"**Only** when the verdict is \boxed{correct} **and** the solution is complete, write your final answer inside a [ANSWER]…[/ANSWER] block on the next line. "
            "The final [ANSWER]…[/ANSWER] block should contain **only** the answer text.\n"
        )
        phase2_prompt = build_prompt(
            statement,
            sol_core,
            immediate_prompt
        )

        if SAVE_PROMPTS_AND_REASONING:
            with open(f"{DATA_SAVE_PATH}/tmath_{prob_id}_prompt_{cot_step_counter}.txt", "w",
                      encoding="utf-8") as f:
                f.write(phase2_prompt)
        cot_step_counter += 1

        phase2_re, phase2_ver = call_llm(phase2_prompt, deep=True, model=LARGE_MODEL, fix_format=True)

        if SAVE_PROMPTS_AND_REASONING:
            with open(f"{DATA_SAVE_PATH}/tmath_{prob_id}_prompt_{cot_step_counter}_reasoning.txt", "w",
                      encoding="utf-8") as f:
                f.write(phase2_re)

        passes.append({
            "mode": "deep_think",
            "deep": True,
            "model": LARGE_MODEL,
            "text": phase2_ver,
        })

        # Terminate if the solution is complete
        if COMPLETE_RE.search(phase2_ver):
            # Final prompt for saving, not used for generation
            final_prompt_to_save = build_prompt(
                statement,
                f"{sol_core}\n{phase2_ver}\n",
                ""
            )

            if SAVE_PROMPTS_AND_REASONING:
                with open(f"{DATA_SAVE_PATH}/tmath_{prob_id}_prompt_{cot_step_counter}.txt", "w",
                          encoding="utf-8") as f:
                    f.write(final_prompt_to_save)
            cot_step_counter += 1
            print(f"CoT Step {cot_step_counter}.")
            return

        # Otherwise merge solution core + new VERIFY block and loop
        phase1_sol = sol_core + "\n" + phase2_ver.replace("[VERIFY]", "[PREVIOUS_VERIFICATION]").replace("[/VERIFY]",
                                                                                                         "[/PREVIOUS_VERIFICATION]")
        immediate_prompt = GLOBAL_IMMEDIATE_PROMPT
        phase1_prompt = build_prompt(
            statement,
            phase1_sol,
            immediate_prompt
        )

        iterations += 1
        time.sleep(0.5)  # courteous pause for rate‑limit compliance


def solve_problem(statement: str, prob_id: str) -> dict:
    """
    Executes the iterative plan / verify loop and returns a record:
        {id, problem, passes:[{mode, deep, model, text}, …]}
    """
    passes = []

    # -------- Phase 1 prompt (Planning + Implementation + Review + Verification) -------- #
    initial_sol = ""
    immediate_prompt = get_immediate_prompt(initial_sol, allow_exploration=False)
    initial_prompt = build_prompt(
        statement,
        "",
        immediate_prompt
    )

    run_solution_pipeline(prob_id, statement, initial_sol, initial_prompt, passes)

    # Replace [PREVIOUS_VERIFICATION] with [VERIFY] in the final output
    for pass_dict in passes:
        pass_dict["text"] = pass_dict["text"].replace("[PREVIOUS_VERIFICATION]", "[VERIFY]").replace(
            "[/PREVIOUS_VERIFICATION]", "[/VERIFY]")

    return {
        "id": prob_id,
        "problem": statement,
        "passes": passes,
    }


def main():
    data = tmath.result#list(map(lambda x: x['problem'], tmath.result))
    #data = load_dataset(DATASET_NAME, split="train")
    #selected_ids = [1]
    for problem_id, row in enumerate(data):
        # if problem_id not in selected_ids:
        #     continue
        print(f"Processing problem {problem_id}")

        out_path = f"{DATA_SAVE_PATH}/tmath_{problem_id}.jsonl"
        with open(out_path, "w", encoding="utf-8") as sink:
            record = solve_problem(row["problem"], problem_id)
            json.dump(record, sink, ensure_ascii=False)
            sink.write("\n")
            print(f"✓ {problem_id} processed")

    print(f"\nComplete dialogue log written to {out_path}")


if __name__ == "__main__":
    main()