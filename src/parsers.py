import os
import json
import argparse
from parser import parse_jsonl_to_txt

def parse_folder(input_dir, output_dir, verbose_gen_mode=False):
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(input_dir):
        if filename.endswith('.jsonl'):
            input_path = os.path.join(input_dir, filename)
            base = os.path.splitext(filename)[0]
            output_path = os.path.join(output_dir, f"{base}.txt")
            prompt_path = os.path.join(output_dir, f"{base}_prompt.txt")
            try:
                parse_jsonl_to_txt(input_path, output_path, prompt_path, verbose_gen_mode=verbose_gen_mode)
                print(f"Parsed {input_path} -> {output_path}")
            except Exception as e:
                print(f"Error parsing {input_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Batch parse all JSONL files in a folder to TXT.")
    parser.add_argument('--input_dir', type=str, required=True, help='Input folder containing .jsonl files')
    parser.add_argument('--output_dir', type=str, required=True, help='Output folder for .txt files')
    parser.add_argument('--verbose_gen_mode', action='store_true', default=False, help='Enable verbose mode for output')
    args = parser.parse_args()
    parse_folder(args.input_dir, args.output_dir, verbose_gen_mode=args.verbose_gen_mode)

if __name__ == "__main__":
    main()