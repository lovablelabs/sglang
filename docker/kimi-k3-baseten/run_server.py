import argparse
import json
from pathlib import Path
import shlex
import subprocess


def main():
    parser = argparse.ArgumentParser(
        description="Run the pinned Kimi K3 / Baseten serving image on an eight-GPU Linux host."
    )
    parser.add_argument("--image", required=True)
    parser.add_argument("--checkpoints", required=True, type=Path)
    parser.add_argument("--print-command", action="store_true")
    args = parser.parse_args()
    if args.image.startswith("-"):
        parser.error("image must be a Docker image reference")
    root = args.checkpoints.resolve(strict=True)
    for model in ("Kimi-K3-NVFP4", "Kimi-K3-DSpark"):
        if not (root / model / "config.json").is_file():
            parser.error(f"missing {model}/config.json beneath checkpoint root")
    argv = json.loads((Path(__file__).parent / "server-argv.json").read_text())
    command = [
        "docker",
        "run",
        "--rm",
        "--gpus",
        "all",
        "--ipc=host",
        "--network=host",
        "--mount",
        f"type=bind,source={root},target=/work/checkpoints,readonly",
        "--entrypoint",
        "/opt/sglang/bin/python3",
        args.image,
        *argv[1:],
    ]
    if args.print_command:
        print(shlex.join(command))
    else:
        raise SystemExit(subprocess.call(command))


if __name__ == "__main__":
    main()
