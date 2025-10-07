#!/usr/bin/env python3
"""
scan_and_index_tf_repos.py

Automates scanning of sparse-cloned Terraform repos with tfsec, Terrascan,
skips each scanner independently if its output already exists, strips banners/ANSI,
always writes a JSON (empty if necessary), skips repos with no .tf files,
and builds a per-file CSV index of policy violations—including “no-violation” examples.

Functionality:
1. Find all git clone directories under repos_dir.
2. Skip any repo that has no “*.tf” files.
3. In parallel (ThreadPoolExecutor):
   a. If <tfsec_dir>/<repo>.json exists, skip tfsec; otherwise run tfsec.
   b. If <terrascan_dir>/<repo>.json exists, skip Terrascan; otherwise run Terrascan.
4. Write per-tool error logs.
5. Parse all JSON outputs and build a CSV index with columns:
     file_id, tf_path, tfsec_rules, terrascan_rules, checkov_rules
"""

import argparse
import csv
import json
import re
import subprocess
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# strip ANSI color codes
ANSI_RE = re.compile(r"\x1b\[[0-9;]*[mK]")
TIMEOUT = 300  # seconds per scanner

def clean_json_output(raw: str) -> str:
    no_ansi = ANSI_RE.sub("", raw)
    idx = no_ansi.find("{")
    return no_ansi[idx:] if idx >= 0 else no_ansi

def scan_one_repo(repo_path: Path, tfsec_dir: Path, terrascan_dir: Path,
                  timeout: int):
    """
    Run tfsec and Terrascan on a single repo, skipping
    if output already exists. Returns (name, tfsec_err, terrascan_err, did_scan).
    """
    name = repo_path.name
    # skip repos with no .tf
    if not any(repo_path.rglob("*.tf")):
        return name, None, None, False

    tfsec_out     = tfsec_dir / f"{name}.json"
    terrascan_out = terrascan_dir / f"{name}.json"

    tfsec_err = terrascan_err  = None

    # tfsec
    if not tfsec_out.exists():
        try:
            proc = subprocess.run(
                ["tfsec", str(repo_path), "--no-color", "--format", "json"],
                capture_output=True, text=True, timeout=timeout
            )
            out = proc.stdout or ""
            if out.strip():
                clean = clean_json_output(out)
                json.loads(clean)
                tfsec_out.write_text(clean)
            else:
                tfsec_out.write_text(json.dumps({"results": []}))
        except Exception as e:
            tfsec_out.write_text(json.dumps({"results": []}))
            tfsec_err = str(e)

    # Terrascan
    if not terrascan_out.exists():
        try:
            proc = subprocess.run(
                ["terrascan", "scan", "-d", str(repo_path), "-o", "json"],
                capture_output=True, text=True, timeout=timeout
            )
            out = proc.stdout or ""
            if out.strip():
                clean = clean_json_output(out)
                json.loads(clean)
                terrascan_out.write_text(clean)
            else:
                terrascan_out.write_text(json.dumps({"results": {"violations": []}}))
        except Exception as e:
            terrascan_out.write_text(json.dumps({"results": {"violations": []}}))
            terrascan_err = str(e)

    return name, tfsec_err, terrascan_err, True

def run_scanners(repos_dir: Path, tfsec_dir: Path, terrascan_dir: Path,
                  workers: int, timeout: int):
    tfsec_dir.mkdir(parents=True, exist_ok=True)
    terrascan_dir.mkdir(parents=True, exist_ok=True)
    

    tfsec_errors     = []
    terrascan_errors = []
    processed_repos  = []

    # find all repos (directories containing a .git folder)
    repos = [p.parent for p in repos_dir.rglob(".git")]

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(scan_one_repo, repo, tfsec_dir, terrascan_dir, timeout): repo
            for repo in repos
        }
        for fut in tqdm(as_completed(futures),
                        total=len(futures),
                        desc="Scanning repos",
                        unit="repo"):
            name, t_err, terr_err, did_scan = fut.result()
            if did_scan:
                processed_repos.append(name)
                if t_err:     tfsec_errors.append(f"{name}: {t_err}")
                if terr_err:  terrascan_errors.append(f"{name}: {terr_err}")

    # write error logs
    if tfsec_errors:
        (tfsec_dir / "errors.log").write_text("\n".join(tfsec_errors))
    if terrascan_errors:
        (terrascan_dir / "errors.log").write_text("\n".join(terrascan_errors))

    return processed_repos

def normalize_path(fn: str, repo_path: Path) -> str:
    p = Path(fn)
    try:
        return p.relative_to(repo_path).as_posix()
    except Exception:
        return p.name

def build_index(repos_dir: Path, tfsec_dir: Path, terrascan_dir: Path,
                output_csv: Path, processed_repos):
    repos_dir     = repos_dir.resolve()
    tfsec_dir     = tfsec_dir.resolve()
    terrascan_dir = terrascan_dir.resolve()
    

    tfsec_map     = defaultdict(lambda: {"rules": set(), "severities": set()})
    terrascan_map = defaultdict(lambda: {"rules": set(), "severities": set()})

    # parse tfsec
    for jf in tqdm(tfsec_dir.glob("*.json"), desc="Parsing tfsec", unit="repo"):
        repo = jf.stem
        if repo not in processed_repos: continue
        data = json.loads(jf.read_text())
        for r in data.get("results", []):
            fn = r.get("location", {}).get("filename")
            if not fn: continue
            rel = normalize_path(fn, repos_dir / repo)
            file_id = f"{repo}::{rel}"
            rule_id = r.get("rule_id")
            severity = r.get("severity", "UNKNOWN").upper()
            if rule_id:
                tfsec_map[file_id]["rules"].add(rule_id)
                tfsec_map[file_id]["severities"].add(severity)

    # parse terrascan
    for jf in tqdm(terrascan_dir.glob("*.json"), desc="Parsing terrascan", unit="repo"):
        repo = jf.stem
        if repo not in processed_repos: continue
        data = json.loads(jf.read_text())
        for v in data.get("results", {}).get("violations", []) or []:
            fn = v.get("file")
            if not fn: continue
            rel = normalize_path(fn, repos_dir / repo)
            file_id = f"{repo}::{rel}"
            rule_id = v.get("rule_id")
            severity = v.get("severity", "UNKNOWN").upper()
            if rule_id:
                terrascan_map[file_id]["rules"].add(rule_id)
                terrascan_map[file_id]["severities"].add(severity)


    # include every .tf file (even if no violations)
    for tf in repos_dir.rglob("*.tf"):
        rel_path = tf.relative_to(repos_dir).as_posix()
        parts = rel_path.split("/", 1)
        repo = parts[0]
        rel  = parts[1] if len(parts) > 1 else ""
        file_id = f"{repo}::{rel}"
        if file_id not in tfsec_map:
            tfsec_map[file_id] = {"rules": set(), "severities": set()}
        if file_id not in terrascan_map:
            terrascan_map[file_id] = {"rules": set(), "severities": set()}

    # write CSV
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "file_id", "tf_path", "tfsec_rules", "tfsec_severities", 
            "terrascan_rules", "terrascan_severities"
        ])
        all_files = sorted(tfsec_map.keys())
        for file_id in tqdm(all_files, desc="Writing CSV", unit="file"):
            repo, rel = file_id.split("::", 1)
            tf_path = (repos_dir / repo / rel).as_posix()
            writer.writerow([
                file_id,
                tf_path,
                "|".join(sorted(tfsec_map[file_id]["rules"])),
                "|".join(sorted(tfsec_map[file_id]["severities"])),
                "|".join(sorted(terrascan_map[file_id]["rules"])),
                "|".join(sorted(terrascan_map[file_id]["severities"])),
            ])

def main():
    p = argparse.ArgumentParser(
        description="Scan Terraform repos with tfsec, Terrascan & Checkov, then index"
    )
    p.add_argument(
        "-r", "--repos-dir",   type=Path, required=True,
        help="Dir of sparse-cloned repos"
    )
    p.add_argument(
        "--tfsec-dir",         type=Path, required=True,
        help="Where to write tfsec JSONs"
    )
    p.add_argument(
        "--terrascan-dir",     type=Path, required=True,
        help="Where to write Terrascan JSONs"
    )
    p.add_argument(
        "-o", "--output-index", type=Path, required=True,
        help="Output CSV index"
    )
    p.add_argument(
        "-w", "--workers",     type=int, default=4,
        help="Number of parallel scan jobs (default: 4)"
    )
    p.add_argument(
        "-t", "--timeout",     type=int, default=TIMEOUT,
        help="Seconds per scanner timeout"
    )
    args = p.parse_args()

    print("Running scanners on each repo…")
    processed = run_scanners(
        args.repos_dir,
        args.tfsec_dir,
        args.terrascan_dir,
        args.workers,
        args.timeout
    )

    print("Building dataset index…")
    build_index(
        args.repos_dir,
        args.tfsec_dir,
        args.terrascan_dir,
        args.output_index,
        processed
    )

    print(f"Done: index written to {args.output_index}")

if __name__ == "__main__":
    main()


'''
#!/usr/bin/env python3
"""
scan_and_index_tf_repos.py

Automates scanning of sparse-cloned Terraform repos with tfsec and Terrascan,
skips each scanner independently if its output already exists, strips banners/ANSI,
always writes a JSON (empty if necessary), skips repos with no .tf files,
and builds a per-file CSV index of policy violations—including “no-violation” examples.

Functionality:
1. Find all git clone directories under repos_dir.
2. Skip any repo that has no “*.tf” files.
3. In parallel (ThreadPoolExecutor):
   a. If <tfsec_dir>/<repo>.json exists, skip tfsec; otherwise run tfsec.
   b. If <terrascan_dir>/<repo>.json exists, skip Terrascan; otherwise run Terrascan.
4. Write per-tool error logs.
5. Parse all JSON outputs and build a CSV index with columns:
     file_id, tf_path, tfsec_rules, terrascan_rules
"""

import argparse
import csv
import json
import re
import subprocess
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# strip ANSI color codes
ANSI_RE = re.compile(r"\x1b\[[0-9;]*[mK]")
TIMEOUT = 300  # seconds per scanner

def clean_json_output(raw: str) -> str:
    no_ansi = ANSI_RE.sub("", raw)
    idx = no_ansi.find("{")
    return no_ansi[idx:] if idx >= 0 else no_ansi

def scan_one_repo(repo_path: Path,
                  tfsec_dir: Path,
                  terrascan_dir: Path,
                  timeout: int):
    """
    Run tfsec and Terrascan on a single repo, skipping if output already exists.
    Returns (name, tfsec_err, terrascan_err, did_scan).
    """
    name = repo_path.name
    # skip repos with no .tf
    if not any(repo_path.rglob("*.tf")):
        return name, None, None, False

    tfsec_out     = tfsec_dir / f"{name}.json"
    terrascan_out = terrascan_dir / f"{name}.json"

    tfsec_err = terrascan_err = None

    # tfsec
    if not tfsec_out.exists():
        try:
            proc = subprocess.run(
                ["tfsec", str(repo_path), "--no-color", "--format", "json"],
                capture_output=True, text=True, timeout=timeout
            )
            out = proc.stdout or ""
            if out.strip():
                clean = clean_json_output(out)
                json.loads(clean)
                tfsec_out.write_text(clean)
            else:
                tfsec_out.write_text(json.dumps({"results": []}))
        except Exception as e:
            tfsec_out.write_text(json.dumps({"results": []}))
            tfsec_err = str(e)

    # Terrascan
    if not terrascan_out.exists():
        try:
            proc = subprocess.run(
                ["terrascan", "scan", "-d", str(repo_path), "-o", "json"],
                capture_output=True, text=True, timeout=timeout
            )
            out = proc.stdout or ""
            if out.strip():
                clean = clean_json_output(out)
                json.loads(clean)
                terrascan_out.write_text(clean)
            else:
                terrascan_out.write_text(json.dumps({"results": {"violations": []}}))
        except Exception as e:
            terrascan_out.write_text(json.dumps({"results": {"violations": []}}))
            terrascan_err = str(e)

    return name, tfsec_err, terrascan_err, True

def run_scanners(repos_dir: Path,
                 tfsec_dir: Path,
                 terrascan_dir: Path,
                 workers: int,
                 timeout: int):
    tfsec_dir.mkdir(parents=True, exist_ok=True)
    terrascan_dir.mkdir(parents=True, exist_ok=True)

    tfsec_errors     = []
    terrascan_errors = []
    processed_repos  = []

    # find all repos (directories containing a .git folder)
    repos = [p.parent for p in repos_dir.rglob(".git")]

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(scan_one_repo, repo, tfsec_dir, terrascan_dir, timeout): repo
            for repo in repos
        }
        for fut in tqdm(as_completed(futures),
                        total=len(futures),
                        desc="Scanning repos",
                        unit="repo"):
            name, t_err, terr_err, did_scan = fut.result()
            if did_scan:
                processed_repos.append(name)
                if t_err:    tfsec_errors.append(f"{name}: {t_err}")
                if terr_err: terrascan_errors.append(f"{name}: {terr_err}")

    # write error logs
    if tfsec_errors:
        (tfsec_dir / "errors.log").write_text("\n".join(tfsec_errors))
    if terrascan_errors:
        (terrascan_dir / "errors.log").write_text("\n".join(terrascan_errors))

    return processed_repos

def normalize_path(fn: str, repo_path: Path) -> str:
    p = Path(fn)
    try:
        return p.relative_to(repo_path).as_posix()
    except Exception:
        return p.name

def build_index(repos_dir: Path,
                tfsec_dir: Path,
                terrascan_dir: Path,
                output_csv: Path,
                processed_repos):
    repos_dir     = repos_dir.resolve()
    tfsec_dir     = tfsec_dir.resolve()
    terrascan_dir = terrascan_dir.resolve()

    tfsec_map     = defaultdict(set)
    terrascan_map = defaultdict(set)

    # parse tfsec
    for jf in tqdm(tfsec_dir.glob("*.json"), desc="Parsing tfsec", unit="repo"):
        repo = jf.stem
        if repo not in processed_repos:
            continue
        data = json.loads(jf.read_text())
        for r in data.get("results", []):
            fn = r.get("location", {}).get("filename")
            if not fn:
                continue
            rel = normalize_path(fn, repos_dir / repo)
            tfsec_map[f"{repo}::{rel}"].add(r.get("rule_id"))

    # parse terrascan
    for jf in tqdm(terrascan_dir.glob("*.json"), desc="Parsing terrascan", unit="repo"):
        repo = jf.stem
        if repo not in processed_repos:
            continue
        data = json.loads(jf.read_text())
        for v in data.get("results", {}).get("violations", []) or []:
            fn = v.get("file")
            if not fn:
                continue
            rel = normalize_path(fn, repos_dir / repo)
            terrascan_map[f"{repo}::{rel}"].add(v.get("rule_id"))

    # include every .tf file (even if no violations)
    for tf in repos_dir.rglob("*.tf"):
        rel_path = tf.relative_to(repos_dir).as_posix()
        parts = rel_path.split("/", 1)
        repo = parts[0]
        rel  = parts[1] if len(parts) > 1 else ""
        file_id = f"{repo}::{rel}"
        tfsec_map.setdefault(file_id, set())
        terrascan_map.setdefault(file_id, set())

    # write CSV
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "file_id", "tf_path", "tfsec_rules", "terrascan_rules"
        ])
        all_files = sorted(tfsec_map.keys())
        for file_id in tqdm(all_files, desc="Writing CSV", unit="file"):
            repo, rel = file_id.split("::", 1)
            tf_path = (repos_dir / repo / rel).as_posix()
            writer.writerow([
                file_id,
                tf_path,
                "|".join(sorted(tfsec_map[file_id])),
                "|".join(sorted(terrascan_map[file_id])),
            ])

def main():
    p = argparse.ArgumentParser(
        description="Scan Terraform repos with tfsec & Terrascan, then index"
    )
    p.add_argument(
        "-r", "--repos-dir",   type=Path, required=True,
        help="Dir of sparse-cloned repos"
    )
    p.add_argument(
        "--tfsec-dir",         type=Path, required=True,
        help="Where to write tfsec JSONs"
    )
    p.add_argument(
        "--terrascan-dir",     type=Path, required=True,
        help="Where to write Terrascan JSONs"
    )
    p.add_argument(
        "-o", "--output-index", type=Path, required=True,
        help="Output CSV index"
    )
    p.add_argument(
        "-w", "--workers",     type=int, default=4,
        help="Number of parallel scan jobs (default: 4)"
    )
    p.add_argument(
        "-t", "--timeout",     type=int, default=TIMEOUT,
        help="Seconds per scanner timeout"
    )
    args = p.parse_args()

    print("Running scanners on each repo…")
    processed = run_scanners(
        args.repos_dir,
        args.tfsec_dir,
        args.terrascan_dir,
        args.workers,
        args.timeout
    )

    print("Building dataset index…")
    build_index(
        args.repos_dir,
        args.tfsec_dir,
        args.terrascan_dir,
        args.output_index,
        processed
    )

    print(f"Done: index written to {args.output_index}")

if __name__ == "__main__":
    main()
'''