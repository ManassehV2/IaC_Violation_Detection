#!/usr/bin/env python3
"""
sparse_clone_tf_repos.py

Functionality:
    - Iterates through all URL files (*.txt) in an input directory.
    - For each file (containing up to 1000 repo URLs), it:
        1. Performs a shallow clone without blobs (--depth 1, --filter=blob:none, --no-checkout).
        2. Enables sparse-checkout (cone mode if supported).
        3. Lists repository files; if no '*.tf' files are present, the repo is removed and skipped.
        4. Otherwise configures sparse-checkout to include only '*.tf' files.
        5. Checks out the working tree, pulling only the .tf blobs.
    - After finishing cloning all repos in one URL file, prompts the user whether to continue
      with the next URL file.
    - Repos with no Terraform files are skipped (marked SKIP).
    - Any clone or Git error is logged and the partial directory cleaned up (marked ERR).

Usage:
    python3 sparse_clone_tf_repos.py \
        --input-dir url_files/ \
        --output-dir tf_repos/ \
        --workers 4

Arguments:
    --input-dir  Directory containing URL files (*.txt), one per batch.
    --output-dir Directory to clone repos into.
    --workers    Number of parallel clone jobs.
"""

import argparse
import subprocess
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def clone_sparse_tf(url: str, output_dir: Path) -> str:
    """
    Clone a single repo sparsely, fetching only .tf files if present.

    Args:
        url: GitHub clone URL (e.g. https://github.com/user/repo.git)
        output_dir: Path to directory where repos are cloned

    Returns:
        Status string:
          - "OK   <name>" on success
          - "SKIP <name> (<reason>)" if skipped
          - "ERR  <name>: <error>" on failure
    """
    name = Path(url).stem
    repo_dir = output_dir / name

    if repo_dir.exists():
        return f"SKIP {name} (already exists)"

    try:
        # 1. Shallow clone without blobs, no checkout
        subprocess.run([
            "git", "clone",
            "--depth", "1",
            "--filter=blob:none",
            "--no-checkout",
            url,
            str(repo_dir)
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # 2. Enable sparse-checkout
        res = subprocess.run(
            ["git", "sparse-checkout", "init", "--cone"],
            cwd=repo_dir,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        if res.returncode != 0:
            subprocess.run(
                ["git", "config", "core.sparseCheckout", "true"],
                cwd=repo_dir,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )

        # 3. List all files in HEAD
        files = subprocess.check_output(
            ["git", "ls-tree", "-r", "--name-only", "HEAD"],
            cwd=repo_dir
        ).decode().splitlines()

        # 4. Skip if no .tf files
        if not any(f.endswith(".tf") for f in files):
            shutil.rmtree(repo_dir)
            return f"SKIP {name} (no .tf files)"

        # 5. Configure sparse-checkout to include only .tf files
        res = subprocess.run(
            ["git", "sparse-checkout", "set", "*.tf"],
            cwd=repo_dir,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        if res.returncode != 0:
            sparse_file = repo_dir / ".git" / "info" / "sparse-checkout"
            sparse_file.parent.mkdir(parents=True, exist_ok=True)
            sparse_file.write_text("*.tf\n")

        # 6. Checkout the sparse paths
        subprocess.run(
            ["git", "checkout", "HEAD"],
            cwd=repo_dir,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

        return f"OK   {name}"

    except Exception as e:
        if repo_dir.exists():
            shutil.rmtree(repo_dir)
        return f"ERR  {name}: {e}"

def main():
    parser = argparse.ArgumentParser(description="Sparse-clone Terraform repos from multiple URL files.")
    parser.add_argument(
        "--input-dir", "-i", type=Path, required=True,
        help="Directory containing URL files (*.txt), one per batch."
    )
    parser.add_argument(
        "--output-dir", "-o", type=Path, default=Path("tf_repos"),
        help="Directory to clone repos into."
    )
    parser.add_argument(
        "--workers", "-w", type=int, default=4,
        help="Number of parallel clone jobs."
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    url_files = sorted(args.input_dir.glob("*.txt"))
    total_files = len(url_files)

    for idx, url_file in enumerate(url_files, start=1):
        print(f"\n[{idx}/{total_files}] Cloning from {url_file.name}")
        urls = [line.strip() for line in url_file.read_text().splitlines() if line.strip()]

        results = []
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(clone_sparse_tf, url, args.output_dir): url for url in urls}
            for fut in tqdm(as_completed(futures),
                            total=len(futures),
                            desc=f"Repos in {url_file.name}", unit="repo"):
                results.append(fut.result())

        # write per-file clone log
        log_path = args.output_dir / f"clone_log_{url_file.stem}.txt"
        log_path.write_text("\n".join(results))
        print(f"Wrote log to {log_path}")

        '''#prompt user to continue
        if idx < total_files:
            cont = input("Continue with next URL file? (y/N): ")
            if cont.lower() != "y":
                print("Aborting further batches.")
                break
        '''

if __name__ == "__main__":
    main()
