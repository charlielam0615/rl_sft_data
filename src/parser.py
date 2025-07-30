import json
import argparse


def parse_jsonl_to_txt(jsonl_file_path, output_file_path, prompt_file_path, verbose_gen_mode=False):
    """
    Parse JSONL file and extract 'problem' and 'text' fields to a single text file.

    Args:
        jsonl_file_path: Path to the input JSONL file
        output_file_path: Path to the output text file
        prompt_file_path: Path to the prompt text file
        verbose_gen_mode: If True, write descriptive information.
    """
    with open(jsonl_file_path, 'r', encoding='utf-8') as jsonl_file:
        with open(output_file_path, 'w', encoding='utf-8') as output_file:
            for line_num, line in enumerate(jsonl_file, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    # Parse JSON object from line
                    data = json.loads(line)
                    if not verbose_gen_mode:
                        with open(prompt_file_path, 'w', encoding='utf-8') as prompt_file:
                            prompt_file.write(data.get('problem', ''))
                        passes = data.get('passes', [])
                        for pass_data in passes:
                            text = pass_data.get('text', '')
                            output_file.write(f"{text}\n")
                        continue

                    # Extract problem
                    problem = data.get('problem', '')

                    # Write header for this entry
                    output_file.write(f"=== Entry {line_num} (ID: {data.get('id', 'N/A')}) ===\n")
                    output_file.write(f"PROBLEM:\n{problem}\n\n")

                    # Extract text from passes
                    passes = data.get('passes', [])

                    for pass_num, pass_data in enumerate(passes, 1):
                        text = pass_data.get('text', '')
                        mode = pass_data.get('mode', 'unknown')
                        model = pass_data.get('model', 'unknown')

                        output_file.write(f"PASS {pass_num} (Mode: {mode}, Model: {model}):\n")
                        output_file.write(f"{text}\n\n")

                    output_file.write("-" * 80 + "\n\n")

                except json.JSONDecodeError as e:
                    print(f"Error parsing line {line_num}: {e}")
                    continue


def main():
    parser = argparse.ArgumentParser(description="Parse JSONL files to text.")
    parser.add_argument(
        '--verbose_gen_mode',
        action='store_true',
        default=False,
        help="Enable data generation mode. Do not write descriptive information."
    )
    args = parser.parse_args()

    for problem_id in [10]:
        input_file = f"aime2025_two_modes_{problem_id}.jsonl"
        output_file = f"parsed_output_{problem_id}.txt"
        prompt_file = f"parsed_input_{problem_id}.txt"

        try:
            parse_jsonl_to_txt(input_file, output_file, prompt_file, verbose_gen_mode=args.verbose_gen_mode)
            print(f"Successfully parsed {input_file} and saved to {output_file}")
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
