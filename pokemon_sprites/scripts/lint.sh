#!/bin/bash

# Lint and format all Python files in the project
# Usage: ./scripts/lint.sh [--fix]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    local status=$1
    local message=$2
    case $status in
        "success") echo -e "${GREEN}✅ $message${NC}" ;;
        "error") echo -e "${RED}❌ $message${NC}" ;;
        "warning") echo -e "${YELLOW}⚠️  $message${NC}" ;;
        "info") echo -e "${BLUE}ℹ️  $message${NC}" ;;
    esac
}

# Check if --fix flag is provided
FIX_MODE=false
if [[ "$1" == "--fix" ]]; then
    FIX_MODE=true
    print_status "info" "Running in fix mode - will auto-fix issues where possible"
fi

# Find all Python files
PYTHON_FILES=$(find . -name "*.py" -not -path "./.venv/*" -not -path "./venv/*" -not -path "./.git/*" -not -path "./__pycache__/*" -not -path "./build/*" -not -path "./dist/*")

if [ -z "$PYTHON_FILES" ]; then
    print_status "error" "No Python files found"
    exit 1
fi

print_status "info" "Found $(echo $PYTHON_FILES | wc -w) Python files to check"

# Track if any checks failed
CHECKS_FAILED=0

# Install missing tools if needed
install_if_missing() {
    local tool=$1
    local package=${2:-$1}
    if ! command -v $tool >/dev/null 2>&1; then
        print_status "warning" "$tool not found, installing..."
        pip install $package
    fi
}

# Ensure tools are installed
install_if_missing "autoflake"
install_if_missing "isort"
install_if_missing "black"
install_if_missing "flake8"

print_status "info" "Starting linting and formatting checks..."

# 1. Remove unused imports and variables with autoflake
print_status "info" "Step 1: Removing unused imports and variables..."
if $FIX_MODE; then
    autoflake --in-place --remove-all-unused-imports --remove-unused-variables --remove-duplicate-keys --expand-star-imports $PYTHON_FILES
    print_status "success" "autoflake: unused imports and variables removed"
else
    if ! autoflake --check --remove-all-unused-imports --remove-unused-variables --remove-duplicate-keys --expand-star-imports $PYTHON_FILES; then
        print_status "error" "autoflake: found unused imports or variables (run with --fix to auto-fix)"
        CHECKS_FAILED=1
    else
        print_status "success" "autoflake: no unused imports or variables found"
    fi
fi

# 2. Sort imports with isort
print_status "info" "Step 2: Sorting imports..."
if $FIX_MODE; then
    isort --profile=black --line-length=79 $PYTHON_FILES
    print_status "success" "isort: imports sorted"
else
    if ! isort --profile=black --line-length=79 --check-only --diff $PYTHON_FILES; then
        print_status "error" "isort: imports not properly sorted (run with --fix to auto-fix)"
        CHECKS_FAILED=1
    else
        print_status "success" "isort: imports are properly sorted"
    fi
fi

# 3. Format code with black
print_status "info" "Step 3: Formatting code..."
if $FIX_MODE; then
    black --line-length=79 $PYTHON_FILES
    print_status "success" "black: code formatted"
else
    if ! black --line-length=79 --check --diff $PYTHON_FILES; then
        print_status "error" "black: code not properly formatted (run with --fix to auto-fix)"
        CHECKS_FAILED=1
    else
        print_status "success" "black: code is properly formatted"
    fi
fi

# 4. Lint with flake8
print_status "info" "Step 4: Linting code..."
if ! flake8 $PYTHON_FILES; then
    print_status "error" "flake8: linting issues found"
    CHECKS_FAILED=1
else
    print_status "success" "flake8: no linting issues found"
fi

# 5. Check trailing whitespace and line endings
print_status "info" "Step 5: Checking whitespace and line endings..."
WHITESPACE_ISSUES=0
for file in $PYTHON_FILES; do
    # Check for trailing whitespace
    if grep -q '[[:space:]]$' "$file"; then
        if $FIX_MODE; then
            sed -i 's/[[:space:]]*$//' "$file"
        else
            print_status "warning" "Trailing whitespace found in $file"
            WHITESPACE_ISSUES=1
        fi
    fi
    
    # Ensure file ends with newline
    if [ -n "$(tail -c1 "$file")" ]; then
        if $FIX_MODE; then
            echo >> "$file"
        else
            print_status "warning" "File $file doesn't end with newline"
            WHITESPACE_ISSUES=1
        fi
    fi
done

if [ $WHITESPACE_ISSUES -eq 0 ] || $FIX_MODE; then
    print_status "success" "No whitespace issues found"
    if $FIX_MODE && [ $WHITESPACE_ISSUES -eq 1 ]; then
        print_status "success" "Whitespace issues fixed"
    fi
else
    print_status "error" "Whitespace issues found (run with --fix to auto-fix)"
    CHECKS_FAILED=1
fi

# Step 6: Professional language check
print_status "info" "Step 6: Checking professional language standards..."
if python scripts/vibe_coding_safeguard.py src/; then
    print_status "success" "vibe check: professional language standards met"
else
    print_status "error" "vibe check: unprofessional language detected"
    CHECKS_FAILED=$((CHECKS_FAILED + 1))
fi

# Summary
echo
print_status "info" "=== LINTING SUMMARY ==="
if [ $CHECKS_FAILED -eq 0 ]; then
    if $FIX_MODE; then
        print_status "success" "All files have been linted and formatted! ✨"
    else
        print_status "success" "All linting checks passed! 🚀"
    fi
    echo
    print_status "info" "Your code is ready for commit/push!"
else
    print_status "error" "Some linting checks failed."
    echo
    print_status "info" "To auto-fix most issues, run:"
    echo "  ./scripts/lint.sh --fix"
    echo
    print_status "info" "Or run individual tools:"
    echo "  autoflake --in-place --remove-all-unused-imports --remove-unused-variables \$(find . -name '*.py')"
    echo "  isort --profile=black --line-length=79 \$(find . -name '*.py')"
    echo "  black --line-length=79 \$(find . -name '*.py')"
    exit 1
fi
