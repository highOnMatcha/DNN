#!/usr/bin/env python3
"""
Vibe Coding Safeguard - Professional Code Quality Checker

Scans codebase for AI-generated language patterns, unprofessional tone,
and marketing-style comments that don't belong in production code.
"""

import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple


class VibeCodingSafeguard:
    """Detect unprofessional language patterns in code."""

    def __init__(self):
        # AI/educational language patterns
        self.ai_patterns = [
            r"\b(?:you|your|yours)\b",  # Personal pronouns
            r"\b(?:let's|let us)\b",  # Collaborative language
            r"\b(?:we'll|we will|we can|we should)\b",  # Inclusive pronouns
            r"\b(?:here's|here is)\b",  # Demonstrative language
            r"\b(?:simply|just|easily|quickly)\b",  # Minimizing adverbs
            r"\b(?:awesome|amazing|fantastic|great|cool)\b",  # Enthusiasm
            r"\b(?:tutorial|example|demo|walkthrough)\b",  # Educational terms
            r"\b(?:step by step|follow along)\b",  # Instructional language
            r"(?:don't worry|no worries|easy peasy)\b",  # Reassuring phrases
        ]

        # Marketing/sales language
        self.marketing_patterns = [
            r"\b(?:revolutionary|game-changing|cutting-edge)\b",
            r"\b(?:state-of-the-art|industry-leading|best-in-class)\b",
            r"\b(?:seamless|effortless|powerful|robust)\b",
            r"\b(?:innovative|groundbreaking|next-generation)\b",
            r"\b(?:enterprise-grade|production-ready|scalable)\b",
        ]

        # Emoji patterns
        self.emoji_patterns = [
            r"[\U0001F600-\U0001F64F]",  # Emoticons
            r"[\U0001F300-\U0001F5FF]",  # Misc symbols
            r"[\U0001F680-\U0001F6FF]",  # Transport symbols
            r"[\U0001F700-\U0001F77F]",  # Alchemical symbols
            r"[\U0001F780-\U0001F7FF]",  # Geometric shapes
            r"[\U0001F800-\U0001F8FF]",  # Supplemental arrows
            r"[\U0001F900-\U0001F9FF]",  # Supplemental symbols
            r"[\U0001FA00-\U0001FA6F]",  # Chess symbols
            r"[\U0001FA70-\U0001FAFF]",  # Symbols and pictographs
            r"[\U00002702-\U000027B0]",  # Dingbats
            r"[\U000024C2-\U0001F251]",  # Enclosed characters
            r"[✅❌⚠️🎯🔧🚀💯🏆🎉]",  # Common code emojis
        ]

        # Overly explanatory comments
        self.explanatory_patterns = [
            r"# This function does",
            r"# This method will",
            r"# Here we",
            r"# Now we",
            r"# First, we",
            r"# Next, we",
            r"# Finally, we",
            r"# The idea is",
            r"# What this does",
            r"# The purpose of this",
        ]

        # Acceptable technical exceptions
        self.exceptions = {
            "user",  # Valid technical term
            "users",
            "username",
            "userdata",
            "user_id",
            "easy_install",  # Package manager
            "simple_server",  # Valid module names
            "quick_sort",
            "fast_api",
        }

    def scan_file(self, file_path: Path) -> List[Tuple[int, str, str]]:
        """Scan a single file for vibe violations."""
        violations = []

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except (UnicodeDecodeError, PermissionError):
            return violations

        for line_num, line in enumerate(lines, 1):
            line_lower = line.lower()

            # Check AI patterns
            for pattern in self.ai_patterns:
                matches = re.finditer(pattern, line_lower, re.IGNORECASE)
                for match in matches:
                    word = match.group().lower()
                    if word not in self.exceptions:
                        violations.append(
                            (
                                line_num,
                                "AI_LANGUAGE",
                                f"Personal/educational language: "
                                f"'{match.group()}'",
                            )
                        )

            # Check marketing patterns
            for pattern in self.marketing_patterns:
                matches = re.finditer(pattern, line_lower, re.IGNORECASE)
                for match in matches:
                    violations.append(
                        (
                            line_num,
                            "MARKETING",
                            f"Marketing language: '{match.group()}'",
                        )
                    )

            # Check emoji patterns
            for pattern in self.emoji_patterns:
                matches = re.finditer(pattern, line)
                for match in matches:
                    violations.append(
                        (line_num, "EMOJI", f"Emoji found: '{match.group()}'")
                    )

            # Check explanatory patterns
            for pattern in self.explanatory_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    violations.append(
                        (
                            line_num,
                            "EXPLANATORY",
                            f"Overly explanatory comment: '{line.strip()}'",
                        )
                    )

        return violations

    def scan_directory(
        self, directory: Path
    ) -> Dict[str, List[Tuple[int, str, str]]]:
        """Scan entire directory for vibe violations."""
        results = {}

        # Python files only
        python_files = list(directory.rglob("*.py"))

        for file_path in python_files:
            # Skip certain directories
            if any(
                skip in str(file_path)
                for skip in [
                    "__pycache__",
                    ".git",
                    ".venv",
                    "venv",
                    "node_modules",
                ]
            ):
                continue

            violations = self.scan_file(file_path)
            if violations:
                relative_path = file_path.relative_to(directory)
                results[str(relative_path)] = violations

        return results

    def print_results(
        self, results: Dict[str, List[Tuple[int, str, str]]]
    ) -> int:
        """Print scan results in a professional format."""
        total_violations = 0

        if not results:
            print("✅ Vibe check passed - No unprofessional language detected")
            return 0

        print("❌ Vibe check failed - Unprofessional language detected:")
        print()

        for file_path, violations in results.items():
            print(f"File: {file_path}")
            for line_num, category, message in violations:
                print(f"  Line {line_num}: [{category}] {message}")
                total_violations += 1
            print()

        print(f"Total violations: {total_violations}")
        return total_violations


def main():
    """Main entry point."""
    if len(sys.argv) > 1:
        target_path = Path(sys.argv[1])
    else:
        target_path = Path(".")

    if not target_path.exists():
        print(f"Error: Path '{target_path}' does not exist")
        sys.exit(1)

    safeguard = VibeCodingSafeguard()

    if target_path.is_file():
        violations = safeguard.scan_file(target_path)
        results = {str(target_path): violations} if violations else {}
    else:
        results = safeguard.scan_directory(target_path)

    violation_count = safeguard.print_results(results)

    # Exit with error code if violations found
    sys.exit(1 if violation_count > 0 else 0)


if __name__ == "__main__":
    main()
