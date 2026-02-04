import os
from pathlib import Path

def pack_context():
    output_file = Path("_context_for_ai.txt")
    project_root = Path(".") # Assumes script is run from project root
    src_dir = project_root / "src"
    
    # Files to include explicitly
    config_files = [
        project_root / "AI_CONTEXT.md",
        project_root / "pyproject.toml",
    ]
    
    print(f"Packing context from {project_root.absolute()}...")

    try:
        with open(output_file, "w", encoding="utf-8") as out:
            # 1. Add Config/Context Files
            for config_path in config_files:
                if config_path.exists():
                    out.write(f"vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv\n")
                    out.write(f"START FILE: {config_path.name}\n")
                    out.write(f"----------------------------------------\n")
                    try:
                        content = config_path.read_text(encoding="utf-8")
                        out.write(content)
                    except Exception as e:
                        out.write(f"Error reading file: {e}\n")
                    out.write(f"\n----------------------------------------\n")
                    out.write(f"END FILE: {config_path.name}\n")
                    out.write(f"^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n\n")
                else:
                    print(f"Skipping {config_path.name} (not found)")

            # 2. Add Source Code (Recursively)
            if src_dir.exists():
                for file_path in src_dir.rglob("*.py"):
                    # Calculate relative path for clarity (e.g., src/asr_stat_kit/cli.py)
                    rel_path = file_path.relative_to(project_root)
                    
                    out.write(f"vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv\n")
                    out.write(f"START FILE: {rel_path}\n")
                    out.write(f"----------------------------------------\n")
                    try:
                        content = file_path.read_text(encoding="utf-8")
                        out.write(content)
                    except Exception as e:
                        out.write(f"Error reading file: {e}\n")
                    out.write(f"\n----------------------------------------\n")
                    out.write(f"END FILE: {rel_path}\n")
                    out.write(f"^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n\n")
            else:
                 print(f"Warning: Source directory '{src_dir}' not found.")

        print(f"Success! Context packed to: {output_file.absolute()}")

    except Exception as e:
        print(f"Fatal error packing context: {e}")

if __name__ == "__main__":
    pack_context()