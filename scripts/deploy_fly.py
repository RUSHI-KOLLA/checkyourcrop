#!/usr/bin/env python3
"""
Deploy to Fly.io using flyctl.
"""
import json
import os
import subprocess
import sys


def run_cmd(cmd: list[str], cwd: str = None, env: dict = None) -> tuple[int, str, str]:
    """Run command and return (returncode, stdout, stderr)."""
    try:
        result = subprocess.run(
            cmd, cwd=cwd, env=env or os.environ.copy(),
            capture_output=True, text=True, timeout=300
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "Command timed out"
    except Exception as e:
        return -1, "", str(e)


def main():
    app_name = os.getenv("FLY_APP_NAME")
    fly_token = os.getenv("FLY_API_TOKEN")

    if not app_name:
        print("deploy_fly: SKIP (FLY_APP_NAME not set)")
        return 0
    if not fly_token:
        print("deploy_fly: SKIP (FLY_API_TOKEN not set)")
        return 0

    # Clean token (remove any whitespace/newlines)
    fly_token = fly_token.strip()

    # Check if flyctl is available
    code, _, _ = run_cmd(["flyctl", "version"])
    if code != 0:
        print("deploy_fly: SKIP (flyctl not installed)")
        return 0

    # Prepare environment with FLY_API_TOKEN
    env = {**os.environ, "FLY_API_TOKEN": fly_token}

    # Check if app exists
    print(f"Checking Fly app: {app_name}")
    code, out, err = run_cmd(["flyctl", "apps", "list"], env=env)
    app_exists = app_name in out

    if not app_exists:
        # Create app
        print(f"Creating Fly app: {app_name}")
        code, out, err = run_cmd(["flyctl", "apps", "create", app_name, "--generate-name"], env=env)
        if code != 0:
            # App might already exist, check again
            code, out, err = run_cmd(["flyctl", "apps", "list"], env=env)
            if app_name not in out:
                return False, f"deploy_fly: FAIL (app create failed)\n{out}\n{err}"

    # Set secrets
    secrets = [
        ("HF_TOKEN", os.getenv("HF_TOKEN")),
        ("MODEL_REPO", os.getenv("HF_MODEL_REPO", "kollarushi/checkyourcrop-model")),
        ("MODEL_FILENAME", os.getenv("MODEL_FILENAME", "best.pt")),
        ("MODEL_REVISION", os.getenv("MODEL_REVISION", "main")),
        ("PORT", "7860"),
    ]
    for key, value in secrets:
        if value:
            print(f"Setting secret: {key}")
            code, out, err = run_cmd(["flyctl", "secrets", "set", f"{key}={value}", "-a", app_name], env=env)
            if code != 0:
                print(f"Warning: failed to set secret {key}: {err}")

    # Deploy
    print(f"Deploying to Fly app: {app_name}")
    code, out, err = run_cmd(["flyctl", "deploy", "--dockerfile", "Dockerfile", "-a", app_name], env=env)
    if code != 0:
        return False, f"deploy_fly: FAIL (deploy failed)\n{out}\n{err}"

    # Get app URL
    code, out, err = run_cmd(["flyctl", "status", "-a", app_name, "--json"], env=env)
    if code == 0:
        try:
            status = json.loads(out)
            hostname = status.get("Hostname", f"{app_name}.fly.dev")
            print(f"deploy_fly: PASS (URL: https://{hostname})")
            return 0
        except Exception:
            pass

    print(f"deploy_fly: PASS (URL: https://{app_name}.fly.dev)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
