from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request


OPENROUTER_API_KEY = ""  # set via env: export OPENROUTER_API_KEY=...


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test whether an OpenRouter API key is valid.")
    parser.add_argument(
        "--api_key",
        default=None,
        help="OpenRouter API key. If omitted, use OPENROUTER_API_KEY from env or this file.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    api_key = args.api_key or os.getenv("OPENROUTER_API_KEY") or OPENROUTER_API_KEY
    if not api_key or api_key == "PASTE_YOUR_OPENROUTER_API_KEY_HERE":
        print(
            "Missing OPENROUTER_API_KEY. "
            "Paste it into scripts/test_openrouter_key.py, "
            "or pass --api_key, or set the environment variable."
        )
        return 2

    request = urllib.request.Request(
        "https://openrouter.ai/api/v1/key",
        headers={"Authorization": f"Bearer {api_key}"},
        method="GET",
    )

    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        print(f"HTTP {exc.code}")
        print(body)
        return 1
    except Exception as exc:
        print(f"Request failed: {exc}")
        return 1

    try:
        parsed = json.loads(body)
        print(json.dumps(parsed, ensure_ascii=False, indent=2))
    except json.JSONDecodeError:
        print(body)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
