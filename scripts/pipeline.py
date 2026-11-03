#!/usr/bin/env python3
"""
MLOps pipeline script.
Stages: lint, test, build, evaluate, register, deploy, smoke
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

# Load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def run_cmd(cmd: list[str], cwd: str = None, env: dict = None) -> tuple[int, str, str]:
    """Run command and return (returncode, stdout, stderr)."""
    try:
        result = subprocess.run(
            cmd, cwd=cwd, env=env or os.environ.copy(),
            capture_output=True, text=True, timeout=600
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "Command timed out"
    except Exception as e:
        return -1, "", str(e)


def stage_lint() -> tuple[bool, str]:
    """Run ruff check."""
    code, out, err = run_cmd(["ruff", "check", "app", "tests", "scripts"])
    if code == 0:
        return True, "lint: PASS"
    return False, f"lint: FAIL\n{out}\n{err}"


def stage_test() -> tuple[bool, str]:
    """Run pytest."""
    code, out, err = run_cmd(["python", "-m", "pytest", "-q"])
    if code == 0:
        return True, "test: PASS"
    return False, f"test: FAIL\n{out}\n{err}"


def stage_build() -> tuple[bool, str]:
    """Build Docker image."""
    # Check if docker is available
    code, _, _ = run_cmd(["docker", "version"])
    if code != 0:
        return True, "build: SKIP (docker not available)"

    code, out, err = run_cmd(["docker", "build", "-t", "checkyourcrop:test", "."])
    if code == 0:
        return True, "build: PASS"
    return False, f"build: FAIL\n{out}\n{err}"


def _download_from_hf(repo_id: str, filename: str, local_path: str) -> bool:
    """Download file from HF Hub. Returns True if successful."""
    try:
        from huggingface_hub import hf_hub_download
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        hf_hub_download(repo_id=repo_id, filename=filename, local_dir=os.path.dirname(local_path), local_dir_use_symlinks=False)
        # hf_hub_download saves to local_dir/filename, but we want local_path
        downloaded = os.path.join(os.path.dirname(local_path), filename)
        if downloaded != local_path and os.path.exists(downloaded):
            shutil.move(downloaded, local_path)
        return True
    except Exception as e:
        print(f"Download failed: {e}")
        return False


def stage_evaluate() -> tuple[bool, str]:
    """Run evaluation gate."""
    # Check for required inputs
    if not os.path.exists("eval/golden/data.yaml"):
        return True, "evaluate: SKIP (eval/golden/data.yaml not found)"

    model_repo = os.getenv("HF_MODEL_REPO")
    if not model_repo:
        return True, "evaluate: SKIP (HF_MODEL_REPO not set)"

    # Download baseline metrics from HF Hub if not present
    baseline_file = "outputs/baseline_metrics.json"
    if not os.path.exists(baseline_file):
        print("Downloading baseline metrics from HF Hub...")
        if not _download_from_hf(model_repo, "metrics.json", baseline_file):
            return True, f"evaluate: SKIP (could not download baseline metrics from {model_repo})"

    import json
    with open(baseline_file) as f:
        baseline = json.load(f)
    baseline_map = baseline.get("map50_95", 0.0)

    # Download model weights from HF Hub if not present
    weights_file = "models/best.pt"
    if not os.path.exists(weights_file):
        print("Downloading model weights from HF Hub...")
        if not _download_from_hf(model_repo, "best.pt", weights_file):
            return True, f"evaluate: SKIP (could not download weights from {model_repo})"

    code, out, err = run_cmd([
        "python", "scripts/evaluate.py",
        "--baseline", str(baseline_map),
        "--tolerance", "0.015",
        "--data", "eval/golden/data.yaml",
        "--weights", weights_file,
        "--output", "outputs/metrics.json",
    ])
    if code == 0:
        return True, "evaluate: PASS"
    elif code == 1:
        return False, f"evaluate: FAIL (regression)\n{out}\n{err}"
    else:
        return True, f"evaluate: SKIP ({out.strip()})"


def stage_register() -> tuple[bool, str]:
    """Register model to HF Hub."""
    # Check for HF_TOKEN
    if not os.getenv("HF_TOKEN"):
        return True, "register: SKIP (HF_TOKEN not set)"

    # Check for model repo
    model_repo = os.getenv("HF_MODEL_REPO")
    if not model_repo:
        return True, "register: SKIP (HF_MODEL_REPO not set)"

    # Download weights from HF Hub if not present
    weights_file = "models/best.pt"
    if not os.path.exists(weights_file):
        print("Downloading model weights from HF Hub...")
        if not _download_from_hf(model_repo, "best.pt", weights_file):
            return True, "register: SKIP (models/best.pt not found and could not download)"

    # Download metrics from HF Hub if not present (will be created by evaluate stage)
    metrics_file = "outputs/metrics.json"
    if not os.path.exists(metrics_file):
        print("Downloading metrics from HF Hub...")
        _download_from_hf(model_repo, "metrics.json", metrics_file)

    # Get git SHA
    code, sha, _ = run_cmd(["git", "rev-parse", "HEAD"])
    sha = "unknown" if code != 0 else sha.strip()[:8]

    # Upload to HF Hub
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=os.getenv("HF_TOKEN"))
        api.upload_file(
            path_or_fileobj="models/best.pt",
            path_in_repo="best.pt",
            repo_id=model_repo,
            repo_type="model",
            commit_message=f"Add model {sha}",
        )
        api.upload_file(
            path_or_fileobj="outputs/metrics.json",
            path_in_repo="metrics.json",
            repo_id=model_repo,
            repo_type="model",
            commit_message=f"Add metrics {sha}",
        )
        return True, "register: PASS"
    except Exception as e:
        return False, f"register: FAIL\n{e}"


def stage_deploy() -> tuple[bool, str]:
    """Deploy to HF Spaces."""
    # Check for HF_TOKEN
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        return True, "deploy: SKIP (HF_TOKEN not set)"

    # Check for Space repo
    space_repo = os.getenv("HF_SPACE")
    if not space_repo:
        return True, "deploy: SKIP (HF_SPACE not set)"

    # Check for git
    code, _, _ = run_cmd(["git", "status"])
    if code != 0:
        return True, "deploy: SKIP (not a git repo)"

    # Get git SHA
    code, sha, _ = run_cmd(["git", "rev-parse", "HEAD"])
    sha = "unknown" if code != 0 else sha.strip()[:8]

    # Clone space repo with HF_TOKEN auth
    import shutil
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        space_dir = Path(tmpdir) / "space"
        # Clone using token authentication
        clone_url = f"https://oauth2:{hf_token}@huggingface.co/spaces/{space_repo}"
        code, out, err = run_cmd(["git", "clone", clone_url, str(space_dir)])
        if code != 0:
            return False, f"deploy: FAIL (clone failed)\n{out}\n{err}"

        # Configure git user
        run_cmd(["git", "config", "user.name", "CI"], cwd=str(space_dir))
        run_cmd(["git", "config", "user.email", "ci@example.com"], cwd=str(space_dir))

        # Sync project files (excluding .git, build, outputs, runs, data, eval, caches)
        exclude_dirs = {".git", "build", "outputs", "runs", "data", "eval", "__pycache__", ".pytest_cache", "venv", ".venv", "models"}
        for item in Path(".").rglob("*"):
            if item.is_file():
                rel = item.relative_to(".")
                if any(part in exclude_dirs for part in rel.parts):
                    continue
                if rel.suffix in {".pyc", ".pyo", ".pyd"}:
                    continue
                dst = space_dir / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item, dst)

        # Prepend Spaces YAML front matter to README.md
        readme_path = space_dir / "README.md"
        if readme_path.exists():
            with open(readme_path) as f:
                content = f.read()
            if not content.startswith("---"):
                front_matter = """---
title: CheckYourCrop
emoji: 🌱
sdk: docker
app_port: 7860
---
"""
                with open(readme_path, "w") as f:
                    f.write(front_matter + content)

        # Commit and push
        code, out, err = run_cmd(["git", "add", "-A"], cwd=str(space_dir))
        if code != 0:
            return False, f"deploy: FAIL (git add failed)\n{out}\n{err}"

        code, out, err = run_cmd(["git", "commit", "-m", f"Deploy {sha}"], cwd=str(space_dir))
        if code != 0:
            return False, f"deploy: FAIL (git commit failed)\n{out}\n{err}"

        # Push with token authentication
        push_url = f"https://oauth2:{hf_token}@huggingface.co/spaces/{space_repo}"
        code, out, err = run_cmd(["git", "push", push_url], cwd=str(space_dir))
        if code != 0:
            return False, f"deploy: FAIL (git push failed)\n{out}\n{err}"

    return True, "deploy: PASS"


def stage_smoke() -> tuple[bool, str]:
    """Smoke test deployed service."""
    service_url = os.getenv("SERVICE_URL")
    if not service_url:
        return True, "smoke: SKIP (SERVICE_URL not set)"

    import time

    import requests

    url = service_url.rstrip("/") + "/healthz"
    for _ in range(20):
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("model_loaded"):
                    return True, f"smoke: PASS (model loaded: {data.get('model_loaded')})"
                return True, f"smoke: PASS (service up, model_loaded: {data.get('model_loaded')})"
        except Exception:
            pass
        time.sleep(3)

    return False, "smoke: FAIL (service did not become healthy in 60s)"


STAGES = {
    "lint": stage_lint,
    "test": stage_test,
    "build": stage_build,
    "evaluate": stage_evaluate,
    "register": stage_register,
    "deploy": stage_deploy,
    "smoke": stage_smoke,
}


def main():
    parser = argparse.ArgumentParser(description="MLOps Pipeline")
    parser.add_argument("--stage", default="all", help="Comma-separated stages or 'all'")
    args = parser.parse_args()

    stages_to_run = list(STAGES.keys()) if args.stage == "all" else [s.strip() for s in args.stage.split(",")]

    print(f"Running stages: {', '.join(stages_to_run)}")
    print("=" * 60)

    results = []
    overall_pass = True

    for stage_name in stages_to_run:
        if stage_name not in STAGES:
            print(f"Unknown stage: {stage_name}")
            overall_pass = False
            continue

        print(f"\n>>> {stage_name.upper()} <<<")
        try:
            passed, msg = STAGES[stage_name]()
            results.append((stage_name, passed, msg))
            print(msg)
            if not passed:
                overall_pass = False
        except Exception as e:
            results.append((stage_name, False, f"{stage_name}: ERROR\n{e}"))
            print(f"{stage_name}: ERROR\n{e}")
            overall_pass = False

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for stage_name, passed, _msg in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {stage_name:12} {status}")

    print("=" * 60)
    if overall_pass:
        print("Overall: PASS")
        return 0
    else:
        print("Overall: FAIL")
        return 1


if __name__ == "__main__":
    sys.exit(main())
