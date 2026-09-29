import sys
from pathlib import Path

# Fix p2_inference.py to properly load tokenizer
inference_file = Path("scripts/phase2/p2_inference.py")
content = inference_file.read_text(encoding="utf-8")

# Find and replace the tokenizer loading section
old_code = '''if self._tokenizer is None:
                    tok_dir = model_dir / "backbone"
                    if tok_dir.exists():
                        self._tokenizer = AutoTokenizer.from_pretrained(tok_dir)
                    else:
                        from scripts.phase2.p2_dataset import load_tokenizer
                        self._tokenizer = load_tokenizer()'''

new_code = '''if self._tokenizer is None:
                    for tok_path in [
                        model_dir / "tokenizer",
                        model_dir / "backbone",
                    ]:
                        if tok_path.exists():
                            try:
                                self._tokenizer = AutoTokenizer.from_pretrained(
                                    str(tok_path), local_files_only=True
                                )
                                logger.info(f"Tokenizer loaded from {tok_path}")
                                break
                            except Exception as e:
                                logger.warning(f"Could not load from {tok_path}: {e}")
                    if self._tokenizer is None:
                        from scripts.phase2.p2_dataset import load_tokenizer
                        self._tokenizer = load_tokenizer()'''

if old_code in content:
    content = content.replace(old_code, new_code)
    inference_file.write_text(content, encoding="utf-8")
    print("Fixed successfully.")
else:
    print("Pattern not found. Printing current tokenizer section...")
    for i, line in enumerate(content.split("\n")):
        if "tokenizer" in line.lower() and "none" in line.lower():
            start = max(0, i-2)
            end = min(len(content.split("\n")), i+15)
            for j, l in enumerate(content.split("\n")[start:end], start):
                print(f"{j}: {l}")
            break