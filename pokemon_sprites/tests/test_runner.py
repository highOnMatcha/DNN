#!/usr/bin/env python3
"""
Comprehensive Test Runner for Pokemon Sprite Generation Pipeline
Runs all test suites: unit, integration, and performance tests with detailed reporting
"""

import argparse
import importlib.util
import sys
import time
import unittest
from pathlib import Path
from typing import Any, Dict, List

# Add src to path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))


# ANSI Color codes for professional terminal output
class Colors:
    GREEN = "\033[92m"
    RED = "\033[91m"
    BLUE = "\033[94m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


class PipelineTestRunner:
    """Professional test runner with comprehensive reporting."""

    def __init__(self):
        self.results = {
            "unit": {
                "passed": 0,
                "failed": 0,
                "errors": 0,
                "skipped": 0,
                "time": 0,
            },
            "integration": {
                "passed": 0,
                "failed": 0,
                "errors": 0,
                "skipped": 0,
                "time": 0,
            },
            "performance": {
                "passed": 0,
                "failed": 0,
                "errors": 0,
                "skipped": 0,
                "time": 0,
            },
        }
        self.total_start_time = time.time()

    def print_header(self, title: str):
        """Print professional test section header."""
        print(f"\n{Colors.CYAN}{Colors.BOLD}{'='*80}{Colors.RESET}")
        print(f"{Colors.CYAN}{Colors.BOLD}{title.center(80)}{Colors.RESET}")
        print(f"{Colors.CYAN}{Colors.BOLD}{'='*80}{Colors.RESET}")

    def print_success(self, message: str):
        """Print success message with green color."""
        print(f"{Colors.GREEN}{Colors.BOLD}[SUCCESS]{Colors.RESET} {message}")

    def print_fail(self, message: str):
        """Print failure message with red color."""
        print(f"{Colors.RED}{Colors.BOLD}[FAIL]{Colors.RESET} {message}")

    def print_info(self, message: str):
        """Print info message with blue color."""
        print(f"{Colors.BLUE}[INFO]{Colors.RESET} {message}")

    def print_warning(self, message: str):
        """Print warning message with yellow color."""
        print(f"{Colors.YELLOW}[WARNING]{Colors.RESET} {message}")

    def discover_tests(self, test_dir: Path) -> List[Path]:
        """Discover all test files in a directory."""
        test_files = []
        if test_dir.exists():
            for test_file in test_dir.glob("test_*.py"):
                test_files.append(test_file)
        return sorted(test_files)

    def run_test_file(self, test_file: Path, test_type: str) -> Dict[str, Any]:
        """Run a single test file and return results."""
        self.print_info(f"Running {test_file.name}")

        # Load the test module
        spec = importlib.util.spec_from_file_location(
            test_file.stem, test_file
        )
        if spec is None or spec.loader is None:
            self.print_fail(f"Could not load test module: {test_file.name}")
            return {
                "passed": 0,
                "failed": 1,
                "errors": 0,
                "skipped": 0,
                "time": 0,
            }

        try:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        except Exception as e:
            self.print_fail(f"Error loading {test_file.name}: {str(e)}")
            return {
                "passed": 0,
                "failed": 0,
                "errors": 1,
                "skipped": 0,
                "time": 0,
            }

        # Create test loader and suite
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromModule(module)

        # Run tests
        start_time = time.time()

        # Custom test result class to capture detailed results
        class TestResult(unittest.TextTestResult):
            def __init__(self, stream, descriptions, verbosity):
                super().__init__(stream, descriptions, verbosity)
                self.test_results = {
                    "passed": 0,
                    "failed": 0,
                    "errors": 0,
                    "skipped": 0,
                }

            def addSuccess(self, test):
                super().addSuccess(test)
                self.test_results["passed"] += 1

            def addError(self, test, err):
                super().addError(test, err)
                self.test_results["errors"] += 1

            def addFailure(self, test, err):
                super().addFailure(test, err)
                self.test_results["failed"] += 1

            def addSkip(self, test, reason):
                super().addSkip(test, reason)
                self.test_results["skipped"] += 1

        # Run the tests
        runner = unittest.TextTestRunner(
            stream=sys.stdout, verbosity=1, resultclass=TestResult
        )

        result = runner.run(suite)
        end_time = time.time()

        test_time = end_time - start_time

        # Extract results
        test_results = {
            "passed": result.test_results["passed"],
            "failed": result.test_results["failed"],
            "errors": result.test_results["errors"],
            "skipped": result.test_results["skipped"],
            "time": test_time,
        }

        # Update overall results
        for key in ["passed", "failed", "errors", "skipped"]:
            self.results[test_type][key] += test_results[key]
        self.results[test_type]["time"] += test_time

        # Print file results
        total_tests = sum(
            test_results[key]
            for key in ["passed", "failed", "errors", "skipped"]
        )
        if test_results["failed"] == 0 and test_results["errors"] == 0:
            self.print_success(
                f"{test_file.name}: {test_results['passed']}/{total_tests} tests passed ({test_time:.2f}s)"
            )
        else:
            self.print_fail(
                f"{test_file.name}: {test_results['passed']}/{total_tests} tests passed, "
                f"{test_results['failed']} failed, {test_results['errors']} errors ({test_time:.2f}s)"
            )

        return test_results

    def run_test_suite(self, test_type: str, test_dir: Path):
        """Run a complete test suite (unit, integration, or performance)."""
        self.print_header(f"{test_type.upper()} TESTS")

        if not test_dir.exists():
            self.print_warning(f"Test directory not found: {test_dir}")
            return

        test_files = self.discover_tests(test_dir)

        if not test_files:
            self.print_warning(f"No test files found in {test_dir}")
            return

        self.print_info(f"Found {len(test_files)} test files")

        # Run each test file
        for test_file in test_files:
            try:
                self.run_test_file(test_file, test_type)
            except Exception as e:
                self.print_fail(
                    f"Unexpected error running {test_file.name}: {str(e)}"
                )
                self.results[test_type]["errors"] += 1

        # Print suite summary
        suite_results = self.results[test_type]
        total_tests = sum(
            suite_results[key]
            for key in ["passed", "failed", "errors", "skipped"]
        )

        print(
            f"\n{Colors.BOLD}{test_type.upper()} TESTS SUMMARY:{Colors.RESET}"
        )
        print(f"  Total Tests: {total_tests}")
        print(
            f"  Passed: {Colors.GREEN}{suite_results['passed']}{Colors.RESET}"
        )
        print(f"  Failed: {Colors.RED}{suite_results['failed']}{Colors.RESET}")
        print(f"  Errors: {Colors.RED}{suite_results['errors']}{Colors.RESET}")
        print(
            f"  Skipped: {Colors.YELLOW}{suite_results['skipped']}{Colors.RESET}"
        )
        print(f"  Time: {suite_results['time']:.2f}s")

        if suite_results["failed"] == 0 and suite_results["errors"] == 0:
            self.print_success(
                f"{test_type.upper()} tests completed successfully"
            )
        else:
            self.print_fail(f"{test_type.upper()} tests completed with issues")

    def print_final_summary(self):
        """Print final test summary."""
        total_time = time.time() - self.total_start_time

        self.print_header("FINAL TEST SUMMARY")

        # Calculate totals
        total_passed = sum(
            self.results[suite]["passed"] for suite in self.results
        )
        total_failed = sum(
            self.results[suite]["failed"] for suite in self.results
        )
        total_errors = sum(
            self.results[suite]["errors"] for suite in self.results
        )
        total_skipped = sum(
            self.results[suite]["skipped"] for suite in self.results
        )
        total_tests = (
            total_passed + total_failed + total_errors + total_skipped
        )

        print(f"\n{Colors.BOLD}OVERALL RESULTS:{Colors.RESET}")
        print(f"  Total Test Suites: {len(self.results)}")
        print(f"  Total Tests: {total_tests}")
        print(f"  Passed: {Colors.GREEN}{total_passed}{Colors.RESET}")
        print(f"  Failed: {Colors.RED}{total_failed}{Colors.RESET}")
        print(f"  Errors: {Colors.RED}{total_errors}{Colors.RESET}")
        print(f"  Skipped: {Colors.YELLOW}{total_skipped}{Colors.RESET}")
        print(f"  Total Time: {total_time:.2f}s")

        # Success rate
        if total_tests > 0:
            success_rate = (total_passed / total_tests) * 100
            print(
                f"  Success Rate: {Colors.GREEN}{success_rate:.1f}%{Colors.RESET}"
            )

        # Per-suite breakdown
        print(f"\n{Colors.BOLD}PER-SUITE BREAKDOWN:{Colors.RESET}")
        for suite_name, suite_results in self.results.items():
            suite_total = sum(
                suite_results[key]
                for key in ["passed", "failed", "errors", "skipped"]
            )
            if suite_total > 0:
                suite_success_rate = (
                    suite_results["passed"] / suite_total
                ) * 100
                status_color = (
                    Colors.GREEN
                    if suite_results["failed"] == 0
                    and suite_results["errors"] == 0
                    else Colors.RED
                )
                print(
                    f"  {suite_name.capitalize()}: {status_color}{suite_success_rate:.1f}%{Colors.RESET} "
                    f"({suite_results['passed']}/{suite_total} passed, {suite_results['time']:.2f}s)"
                )

        # Final status
        print(f"\n{Colors.BOLD}{'='*80}{Colors.RESET}")
        if total_failed == 0 and total_errors == 0:
            self.print_success("ALL TESTS PASSED SUCCESSFULLY!")
        else:
            self.print_fail(
                f"TESTS COMPLETED WITH {total_failed} FAILURES AND {total_errors} ERRORS"
            )
        print(f"{Colors.BOLD}{'='*80}{Colors.RESET}")

        return total_failed == 0 and total_errors == 0


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(
        description="Run Pokemon Sprite Generation Pipeline Tests"
    )
    parser.add_argument(
        "--suite",
        "-s",
        choices=["unit", "integration", "performance", "all"],
        default="all",
        help="Test suite to run (default: all)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Verbose output"
    )
    parser.add_argument(
        "--failfast", "-f", action="store_true", help="Stop on first failure"
    )

    args = parser.parse_args()

    # Get test directories
    tests_dir = Path(__file__).parent
    unit_dir = tests_dir / "unit"
    integration_dir = tests_dir / "integration"
    performance_dir = tests_dir / "performance"

    # Create test runner
    runner = PipelineTestRunner()

    # Print welcome header
    runner.print_header("POKEMON SPRITE GENERATION PIPELINE - TEST SUITE")
    runner.print_info(
        "Professional testing framework for ensuring code robustness"
    )
    runner.print_info(f"Test directory: {tests_dir}")
    runner.print_info(f"Running suite: {args.suite}")

    # Run requested test suites
    try:
        if args.suite == "all" or args.suite == "unit":
            runner.run_test_suite("unit", unit_dir)

        if args.suite == "all" or args.suite == "integration":
            runner.run_test_suite("integration", integration_dir)

        if args.suite == "all" or args.suite == "performance":
            runner.run_test_suite("performance", performance_dir)

        # Print final summary
        success = runner.print_final_summary()

        # Exit with appropriate code
        sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        runner.print_warning("Test execution interrupted by user")
        sys.exit(130)
    except Exception as e:
        runner.print_fail(f"Unexpected error during test execution: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
