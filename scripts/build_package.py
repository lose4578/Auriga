"""Build a standalone wheel, including the existing Web build and pinned data.

Run `npm ci --prefix web` first. Staging never changes the source checkout.
"""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile


def build(output):
    root = Path(__file__).resolve().parents[1]
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="auriga-wheel-") as directory:
        stage = Path(directory)
        shutil.copy2(root / "pyproject.toml", stage / "pyproject.toml")
        shutil.copy2(root / "LICENSE", stage / "LICENSE")
        shutil.copytree(root / "auriga", stage / "auriga",
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "web_assets", "reference_sources"))
        shutil.copy2(root / "configs/auriga.json", stage / "auriga/core/default_config.json")
        shutil.copytree(root / "third_party", stage / "auriga/reference_sources/third_party",
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        subprocess.run(["node", str(root / "web/build.mjs"), "--outdir", str(stage / "auriga/web_assets")], check=True)
        wheel_directory = stage / "wheels"
        subprocess.run([sys.executable, "-m", "pip", "wheel", "--no-deps", "--no-build-isolation",
                        "--wheel-dir", str(wheel_directory), str(stage)], check=True)
        wheel, = wheel_directory.glob("*.whl")
        with zipfile.ZipFile(wheel) as archive:
            expected = {"auriga/web_assets/index.html", "auriga/web_assets/app.js",
                        "auriga/core/default_config.json", "auriga/cortex/session/codex_models.json",
                        "auriga/reference_sources/third_party/codex/LICENSE",
                        "auriga/reference_sources/third_party/codex/NOTICE",
                        "auriga/web_assets/library.html", "auriga/core/storage/read_models.py",
                        "auriga/apps/evaluation/queue.py", "auriga/apps/evaluation/report.py",
                        "auriga/runtime/policy/policy_workers/pi05_arx_client.py",
                        "auriga/runtime/policy/policy_workers/pi05_arx_manifest.json",
                        "auriga/runtime/policy/policy_workers/xpolicylab_ws_client.py",
                        "auriga/reference_sources/third_party/roboprobe/api-provenance.json"}
            missing = expected - set(archive.namelist())
            if missing:
                raise RuntimeError("wheel_resources_missing:" + ",".join(sorted(missing)))
        destination = output / wheel.name
        shutil.copy2(wheel, destination)
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("dist"))
    print(build(parser.parse_args().output))
