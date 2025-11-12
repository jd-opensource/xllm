#!/bin/bash

# Color definitions
GREEN="\033[0;32m"
BLUE="\033[0;34m"
YELLOW="\033[0;33m"
RED="\033[0;31m"
NC="\033[0m" # No Color

# Configuration
REPO_ROOT=`pwd`

# Function to print section headers
print_section() {
    echo -e "\n${BLUE}=== $1 ===${NC}"
}

# Function to print success messages
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

# Function to print error messages and exit
print_error() {
    echo -e "${RED}✗ ERROR: $1${NC}"
    exit 1
}

# Function to check command success
check_success() {
    if [ $? -ne 0 ]; then
        print_error "$1"
    fi
}

if [ $(id -u) -ne 0 ]; then
	print_error "Require root permission, try sudo ./dependencies.sh"
fi


# Install yalantinglibs
print_section "Installing yalantinglibs"

# Check if thirdparties directory exists
if [ ! -d "${REPO_ROOT}/third_party/Mooncake/thirdparties" ]; then
    mkdir -p "${REPO_ROOT}/third_party/Mooncake/thirdparties"
    check_success "Failed to create Mooncake/thirdparties directory"
fi

# Change to thirdparties directory
cd "${REPO_ROOT}/third_party/Mooncake/thirdparties"
check_success "Failed to change to Mooncake/thirdparties directory"

# Check if yalantinglibs is already installed
if [ -d "yalantinglibs" ]; then
    echo -e "${YELLOW}yalantinglibs directory already exists. Removing for fresh install...${NC}"
    rm -rf yalantinglibs
    check_success "Failed to remove existing yalantinglibs directory"
fi

# Clone yalantinglibs
echo "Cloning yalantinglibs from https://gitcode.com/gh_mirrors/ya/yalantinglibs.git"
git clone https://gitcode.com/gh_mirrors/ya/yalantinglibs.git
check_success "Failed to clone yalantinglibs"

# Build and install yalantinglibs
cd yalantinglibs
check_success "Failed to change to yalantinglibs directory"

# Checkout version 0.5.5
echo "Checking out yalantinglibs version 0.5.5..."
git checkout 0.5.5
check_success "Failed to checkout yalantinglibs version 0.5.5"

mkdir -p build
check_success "Failed to create build directory"

cd build
check_success "Failed to change to build directory"

echo "Configuring yalantinglibs..."
cmake .. -DBUILD_EXAMPLES=OFF -DBUILD_BENCHMARK=OFF -DBUILD_UNIT_TESTS=OFF -DYLT_ENABLE_IBV=ON 
check_success "Failed to configure yalantinglibs"

echo "Building yalantinglibs (using $(nproc) cores)..."
cmake --build . -j$(nproc)
check_success "Failed to build yalantinglibs"

echo "Installing yalantinglibs..."
cmake --install .
check_success "Failed to install yalantinglibs"

sed -i '54s/target_link_libraries(${ylt_target_name} -libverbs)/target_link_libraries(${ylt_target_name} INTERFACE -libverbs)/' /usr/local/lib/cmake/yalantinglibs/config.cmake

print_success "yalantinglibs installed successfully"

