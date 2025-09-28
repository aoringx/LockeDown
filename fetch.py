#!/usr/bin/env python3
"""
Simple client to fetch data from the FastAPI server in server.py

Usage:
  python fetch.py --url http://127.0.0.1:5000
"""

from __future__ import annotations

import argparse
import json
import sys
from urllib import request, error


def fetch_data(base_url: str, endpoint: str = "/data", timeout: float = 5.0):
	"""Fetch data from the server.

	Args:
		base_url: The base URL of the server (e.g., http://127.0.0.1:5000).
		endpoint: API endpoint path to fetch (default: /data).
		timeout: Socket timeout in seconds.

	Returns:
		A tuple (ok, payload_or_message). If ok is True, the second item is the
		parsed response (JSON if possible, otherwise text). If ok is False, the
		second item is an error message string.
	"""
	url = base_url.rstrip("/") + endpoint
	req = request.Request(url, method="GET")
	try:
		with request.urlopen(req, timeout=timeout) as resp:
			raw = resp.read()
			text = raw.decode("utf-8", errors="replace")
			# Try to parse JSON; if it fails, return the raw text
			try:
				data = json.loads(text)
			except json.JSONDecodeError:
				data = text
			return True, data
	except error.HTTPError as e:
		try:
			body = e.read().decode("utf-8", errors="replace")
		except Exception:
			body = ""
		return False, f"HTTP {e.code}: {e.reason} {body}"
	except error.URLError as e:
		return False, f"Connection error: {e.reason}"
	except Exception as e:
		return False, f"Unexpected error: {e}"


def main(argv: list[str] | None = None) -> int:
	parser = argparse.ArgumentParser(description="Fetch data from server.py FastAPI server")
	parser.add_argument(
		"--url",
		default="http://127.0.0.1:5000",
		help="Base URL of the server (default: http://127.0.0.1:5000)",
	)
	args = parser.parse_args(argv)

	ok, result = fetch_data(args.url)
	if ok:
		# Print the response as-is; if it's a string, it's likely a JSON string response
		print(result)
		return 0
	else:
		print(result, file=sys.stderr)
		return 1


if __name__ == "__main__":
	raise SystemExit(main())

