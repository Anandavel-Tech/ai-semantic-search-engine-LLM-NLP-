"""Utility to package the clean project codebase into a zip archive."""
import os
import zipfile
from pathlib import Path

def create_project_zip():
    base_dir = Path(__file__).resolve().parent.parent
    zip_path = base_dir / "nlp_semantic_search_project.zip"
    
    exclude_dirs = {".venv", "__pycache__", ".pytest_cache", ".git"}
    exclude_extensions = {".pyc", ".pyo"}
    
    count = 0
    with zipfile.ZipFile(str(zip_path), "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(str(base_dir)):
            dirs[:] = [d for d in dirs if d not in exclude_dirs]
            for file in files:
                if file == zip_path.name or Path(file).suffix in exclude_extensions:
                    continue
                file_full = Path(root) / file
                arc_name = file_full.relative_to(base_dir)
                zipf.write(file_full, arc_name)
                count += 1
                
    size_kb = os.path.getsize(str(zip_path)) / 1024.0
    print(f"ZIP archive created successfully at: {zip_path}")
    print(f"Total files archived: {count} ({size_kb:.1f} KB)")

if __name__ == "__main__":
    create_project_zip()
