"""
Test configuration and utilities.

This module sets up the Python path for all tests to avoid conditional imports.
"""

import sys
from pathlib import Path

# Setup path once for all tests
_test_dir = Path(__file__).parent
_src_path = _test_dir.parent / "src"
if str(_src_path) not in sys.path:
    sys.path.insert(0, str(_src_path))


# Test utilities
class TestColors:
    """ANSI color codes for professional test output."""

    GREEN = "\033[92m"
    RED = "\033[91m"
    BLUE = "\033[94m"
    YELLOW = "\033[93m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


def print_test_result(
    test_name: str, success: bool, details: str = ""
) -> None:
    """Print formatted test results."""
    if success:
        print(
            f"{TestColors.GREEN}{TestColors.BOLD}[SUCCESS]{TestColors.RESET} {test_name}"
        )
        if details:
            print(f"          {details}")
    else:
        print(
            f"{TestColors.RED}{TestColors.BOLD}[FAIL]{TestColors.RESET} {test_name}"
        )
        if details:
            print(f"       {details}")
