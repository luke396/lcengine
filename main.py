"""Command-line entry point for launching the LCEngine Streamlit UI."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from app.config import config

if TYPE_CHECKING:
    from collections.abc import Sequence
    from logging import Logger

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_APP = PROJECT_ROOT / "app.py"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Build the CLI parser and read command-line arguments."""  # noqa: DOC201
    parser = argparse.ArgumentParser(
        description="Launch the LCEngine Streamlit web application.",
    )
    parser.add_argument(
        "--app",
        type=Path,
        default=DEFAULT_APP,
        help="Path to the Streamlit script (default: app.py).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8501,
        help="Port for the Streamlit server (default: 8501).",
    )
    parser.add_argument(
        "--address",
        default="localhost",
        help="Bind address for the Streamlit server (default: localhost).",
    )
    parser.add_argument(
        "--show",
        dest="headless",
        action="store_false",
        help="Open Streamlit in a browser window instead of headless mode.",
    )
    parser.set_defaults(headless=True)
    return parser.parse_args(argv)


def build_streamlit_command(
    script_path: Path,
    *,
    port: int,
    headless: bool,
    address: str,
) -> list[str]:
    """Construct the streamlit CLI invocation."""  # noqa: DOC201
    return [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(script_path),
        "--server.port",
        str(port),
        "--server.address",
        address,
        "--server.headless",
        "true" if headless else "false",
    ]


def run_streamlit(command: Sequence[str], logger: Logger) -> int:
    """Execute the configured streamlit command and return its exit code."""  # noqa: DOC201
    try:
        result = subprocess.run(
            command,
            check=False,
            cwd=PROJECT_ROOT,
        )
    except KeyboardInterrupt:
        logger.info("LCEngine stopped by user")
        return 0
    except OSError:
        logger.exception("Unable to launch Streamlit: %s")
        return 1
    return result.returncode


def main(argv: Sequence[str] | None = None) -> int:
    """Validate configuration and launch the Streamlit UI."""  # noqa: DOC201
    args = parse_args(argv)

    config.setup_logging()
    logger = config.get_logger(__name__)

    try:
        config.validate()
    except ValueError:
        logger.exception("Configuration invalid")
        return 1

    script_path = (
        args.app if args.app.is_absolute() else (PROJECT_ROOT / args.app)
    ).resolve()
    if not script_path.exists():
        logger.error("Streamlit script not found: %s", script_path)
        return 1

    logger.info(
        "Starting LCEngine Streamlit app at http://%s:%s (headless=%s)",
        args.address,
        args.port,
        args.headless,
    )

    command = build_streamlit_command(
        script_path,
        port=args.port,
        headless=args.headless,
        address=args.address,
    )

    return_code = run_streamlit(command, logger)
    if return_code != 0:
        logger.error("Streamlit exited with status %s", return_code)
    return return_code


if __name__ == "__main__":
    sys.exit(main())
