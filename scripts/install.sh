#!/bin/bash

# Session-Rec Benchmark - Complete Installation Script
# This script installs all dependencies and sets up the session-rec framework

set -e  # Exit on error

echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║          Session-Based Recommendation Benchmark - Installation              ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check Python version
echo -e "${YELLOW}Checking Python version...${NC}"
python_version=$(python3 --version 2>&1 | grep -oP '\d+\.\d+')
required_version="3.9"

if (( $(echo "$python_version < $required_version" | bc -l) )); then
    echo -e "${RED}Error: Python $required_version or higher is required${NC}"
    echo "Current version: $python_version"
    exit 1
fi
echo -e "${GREEN}✓ Python $python_version detected${NC}"
echo ""

# Step 1: Install Python dependencies
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "${YELLOW}Step 1: Installing Python dependencies from requirements.txt${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Get project root (parent of scripts directory)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"
pip install -r requirements.txt
echo -e "${GREEN}✓ Core dependencies installed${NC}"
echo ""

# Step 2: Install session-rec specific dependencies
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "${YELLOW}Step 2: Installing session-rec specific dependencies${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
pip install scikit-optimize dill pympler
echo -e "${GREEN}✓ Session-rec dependencies installed${NC}"
echo ""

# Step 3: Clone session-rec framework
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "${YELLOW}Step 3: Setting up session-rec framework${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Check if session-rec-lib already exists
if [ -d "$PROJECT_ROOT/session-rec-lib" ]; then
    echo -e "${YELLOW}Warning: session-rec-lib directory already exists${NC}"
    read -p "Do you want to replace it? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing session-rec-lib..."
        rm -rf "$PROJECT_ROOT/session-rec-lib"
    else
        echo "Skipping session-rec installation"
        exit 0
    fi
fi

# Clone Python 3.9+ compatible fork
echo "Cloning session-rec framework (Python 3.9+ compatible fork)..."
cd /tmp
rm -rf session-rec-3-9  # Remove if exists
git clone --branch python39-compatibility https://github.com/hygo2025/session-rec-3-9.git session-rec-3-9

# Copy to project root
echo "Copying to project root..."
cp -r /tmp/session-rec-3-9 "$PROJECT_ROOT/session-rec-lib"

echo -e "${GREEN}✓ Session-rec framework cloned (Python 3.9+ compatible)${NC}"
echo ""

# Step 4: Verify compatibility fixes are present
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "${YELLOW}Step 4: Verifying Python 3.9+ compatibility${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

cd "$PROJECT_ROOT/session-rec-lib"

# Verify fixes are present
echo "Checking for time.perf_counter() (should find many)..."
grep -r "time.perf_counter()" . --include="*.py" | wc -l | xargs echo "  Found replacements:"

echo "Checking for old time.clock() (should find none)..."
clock_count=$(grep -r "time\.clock()" . --include="*.py" | wc -l)
if [ "$clock_count" -eq 0 ]; then
    echo -e "  ${GREEN}✓ No time.clock() found (good!)${NC}"
else
    echo -e "  ${RED}✗ Warning: Found $clock_count time.clock() instances${NC}"
fi

echo "Checking Pop.fit() signature..."
grep -A 1 "class Pop" algorithms/baselines/pop.py | tail -1
grep "def fit(self, data, test=None):" algorithms/baselines/pop.py > /dev/null && \
    echo -e "  ${GREEN}✓ Pop.fit() signature correct${NC}" || \
    echo -e "  ${RED}✗ Pop.fit() signature incorrect${NC}"

echo -e "${GREEN}✓ Using Python 3.9+ compatible fork${NC}"
echo -e "  Source: https://github.com/hygo2025/session-rec-3-9"
echo -e "  Branch: python39-compatibility"
echo ""

# Step 5: Verify installation
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "${YELLOW}Step 5: Verifying installation${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

cd "$PROJECT_ROOT"

# Check Python packages
echo "Checking Python packages..."
python3 -c "import numpy, pandas, scipy, sklearn, yaml, dill, skopt, pympler" && \
    echo -e "${GREEN}✓ All Python packages importable${NC}" || \
    echo -e "${RED}✗ Some Python packages missing${NC}"

# Check session-rec structure
echo "Checking session-rec structure..."
[ -d "session-rec-lib/algorithms" ] && \
    echo -e "${GREEN}✓ session-rec/algorithms found${NC}" || \
    echo -e "${RED}✗ session-rec/algorithms not found${NC}"

[ -d "session-rec-lib/evaluation" ] && \
    echo -e "${GREEN}✓ session-rec/evaluation found${NC}" || \
    echo -e "${RED}✗ session-rec/evaluation not found${NC}"

[ -f "session-rec-lib/run_config.py" ] && \
    echo -e "${GREEN}✓ session-rec/run_config.py found${NC}" || \
    echo -e "${RED}✗ session-rec/run_config.py not found${NC}"

echo ""

# Final summary
echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║                      INSTALLATION COMPLETED! ✅                              ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Installed:"
echo "  ✓ Core Python dependencies (numpy, pandas, scipy, etc.)"
echo "  ✓ Session-rec specific dependencies (scikit-optimize, dill, pympler)"
echo "  ✓ Session-rec framework (Python 3.9+ compatible fork)"
echo "    → https://github.com/hygo2025/session-rec-3-9"
echo ""
echo "Next steps:"
echo "  1. Prepare your data:"
echo "     cd src"
echo "     python data/prepare_dataset.py --start-date 2024-03-01 --end-date 2024-03-15 --output data/processed_sample"
echo ""
echo "  2. Convert to session-rec format:"
echo "     python data/convert_to_session_rec.py"
echo ""
echo "  3. Run benchmark:"
echo "     python run_session_rec.py --config configs/pop_only.yml"
echo ""
echo "Or use Makefile targets:"
echo "  make prepare-data"
echo "  make convert-data"
echo "  make test-pop"
echo ""
