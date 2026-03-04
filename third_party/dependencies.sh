#!/bin/bash

set -euo pipefail

GREEN="\033[0;32m"
BLUE="\033[0;34m"
YELLOW="\033[0;33m"
RED="\033[0;31m"
NC="\033[0m"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEPS_ROOT_DIR="${HOME}/xllm_deps"
GOVER="1.23.8"

YALANTING_INSTALL_PREFIX="/usr/local/yalantinglibs"
YALANTING_CONFIG_FILE="${YALANTING_INSTALL_PREFIX}/lib/cmake/yalantinglibs/config.cmake"
YALANTING_REPO_URL="https://gitcode.com/gh_mirrors/ya/yalantinglibs.git"
YALANTING_SOURCE_DIR="${DEPS_ROOT_DIR}/yalantinglibs"
YALANTING_BUILD_DIR="${YALANTING_SOURCE_DIR}/build"

MOONCAKE_REPO_DIR="${DEPS_ROOT_DIR}/Mooncake"
MOONCAKE_CMAKE_FILE="${MOONCAKE_REPO_DIR}/CMakeLists.txt"
MOONCAKE_INSTALL_DIR="/usr/local/lib/python3.11/site-packages/mooncake"

MEMFABRIC_REPO_URL="https://gitcode.com/xLLM-AI/memfabric_hybrid.git"
MEMFABRIC_REPO_DIR="${DEPS_ROOT_DIR}/memfabric_hybrid"
MEMFABRIC_TARGET_COMMIT="87e6f12cc4eeb13c3733c6f698b927698e501cba"
MEMFABRIC_INSTALL_DIR="/usr/local/memfabric_hybrid"
MEMFABRIC_INSTALLER="output/memfabric_hybrid-1.0.0_linux_aarch64.run"

DEPENDENCY_NAME="${1:-}"
DEVICE="$(echo "${2:-}" | tr '[:upper:]' '[:lower:]')"
ENABLE_HA_RAW="${3:-false}"
ENABLE_HA="false"

print_section() {
    echo -e "\n${BLUE}=== $1 ===${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}$1${NC}"
}

print_error() {
    echo -e "${RED}✗ ERROR: $1${NC}"
    exit 1
}

run_or_die() {
    local error_message="$1"
    shift
    "$@" || print_error "${error_message}"
}

normalize_enable_ha() {
    local normalized
    normalized="$(echo "${ENABLE_HA_RAW}" | tr '[:upper:]' '[:lower:]')"

    case "${normalized}" in
        1|true|yes|y|on)
            ENABLE_HA="true"
            ;;
        0|false|no|n|off|"")
            ENABLE_HA="false"
            ;;
        *)
            print_error "Invalid enable_ha value: ${ENABLE_HA_RAW}. Use true/false."
            ;;
    esac
}

usage() {
    cat <<EOF
Usage: sh third_party/dependencies.sh <dependency_name> <device> [enable_ha]

dependency_name:
  yalantinglibs
  mooncake
  memfabric
  all

device:
  a2/a3/auto (optional when dependency_name != mooncake)

enable_ha:
  true/false (optional, default: false)
EOF
}

cleanup_dependency_workspace() {
    if [ -d "${DEPS_ROOT_DIR}" ]; then
        run_or_die "Failed to remove dependency workspace: ${DEPS_ROOT_DIR}" rm -rf "${DEPS_ROOT_DIR}"
        print_success "Removed dependency workspace: ${DEPS_ROOT_DIR}"
    fi
}

ensure_deps_root_dir() {
    run_or_die "Failed to create dependency workspace: ${DEPS_ROOT_DIR}" mkdir -p "${DEPS_ROOT_DIR}"
}

ensure_mooncake_yum_packages() {
    local mooncake_yum_packages=(
        yaml-cpp-devel
        glog-devel
        libcurl-devel
        jsoncpp-devel
        wget
    )

    command -v yum >/dev/null 2>&1 || print_error "yum not found, cannot install Mooncake system packages"

    print_section "Installing Mooncake required system packages"
    run_or_die \
        "Failed to install Mooncake required system packages" \
        yum install -y "${mooncake_yum_packages[@]}"
    print_success "Mooncake required system packages installed"
}

get_go_arch() {
    local arch
    arch="$(uname -m)"
    if [ "${arch}" = "aarch64" ]; then
        echo "arm64"
    elif [ "${arch}" = "x86_64" ]; then
        echo "amd64"
    else
        print_error "Unsupported architecture: ${arch}"
    fi
}

print_go_manual_install_guide() {
    local arch="$1"
    local go_archive="go${GOVER}.linux-${arch}.tar.gz"
    local go_url="https://go.dev/dl/${go_archive}"

    print_warning "Automatic Go download failed. If your network cannot access go.dev, install manually:"
    echo "Download URL: ${go_url}"
    echo "Manual install commands:"
    echo "  wget ${go_url}"
    echo "  rm -rf /usr/local/go"
    echo "  tar -C /usr/local -xzf ${go_archive}"
    echo "  rm -f ${go_archive}"
    echo "Env setup commands:"
    echo "  echo 'export PATH=\$PATH:/usr/local/go/bin' >> ~/.bashrc"
    echo "  source ~/.bashrc"
}

install_go() {
    local arch
    arch="$(get_go_arch)"
    local go_archive="go${GOVER}.linux-${arch}.tar.gz"
    local go_url="https://go.dev/dl/${go_archive}"

    print_section "Installing Go ${GOVER}"

    if ! wget -q --show-progress "${go_url}"; then
        print_go_manual_install_guide "${arch}"
        return 1
    fi

    if ! rm -rf /usr/local/go; then
        print_warning "Failed to remove existing /usr/local/go before install."
        rm -f "${go_archive}" || true
        return 1
    fi

    if ! tar -C /usr/local -xzf "${go_archive}"; then
        print_warning "Failed to extract ${go_archive} to /usr/local."
        rm -f "${go_archive}" || true
        return 1
    fi

    run_or_die "Failed to clean up Go archive" rm -f "${go_archive}"
    print_success "Go ${GOVER} installed successfully"
    return 0
}

ensure_go() {
    local expected_version="go${GOVER}"
    local go_in_path
    go_in_path="$(command -v go || true)"

    if [ -n "${go_in_path}" ]; then
        local go_version
        go_version="$(go version | awk '{print $3}')"
        if [ "${go_version}" = "${expected_version}" ]; then
            print_success "Go ${GOVER} found in PATH: ${go_in_path}"
            return
        fi
        print_warning "Found ${go_version} in PATH at ${go_in_path}. Will install Go ${GOVER}."
    else
        print_warning "Go is not found in current PATH."
        local local_go_bin="/usr/local/go/bin/go"
        if [ -x "${local_go_bin}" ]; then
            local local_go_version
            local_go_version="$("${local_go_bin}" version | awk '{print $3}')"
            if [ "${local_go_version}" = "${expected_version}" ]; then
                print_warning "Go ${GOVER} exists at ${local_go_bin}, but PATH is not active in this shell."
                print_warning "Run 'source ~/.bashrc' (or open a new terminal) and rerun this installation."
                exit 1
            fi
            print_warning "Found ${local_go_version} at ${local_go_bin}. Will install Go ${GOVER}."
        fi
    fi

    if ! install_go; then
        print_error "Failed to install Go ${GOVER} automatically."
    fi

    if ! grep -Fq 'export PATH=$PATH:/usr/local/go/bin' "${HOME}/.bashrc" 2>/dev/null; then
        if ! echo 'export PATH=$PATH:/usr/local/go/bin' >> "${HOME}/.bashrc"; then
            print_error "Unable to update ${HOME}/.bashrc. Add /usr/local/go/bin to PATH manually."
        fi
    fi

    print_warning "Go ${GOVER} was installed in this run."
    print_warning "This installation will now exit. Run 'source ~/.bashrc' (or open a new terminal), then rerun."
    exit 1
}

print_ha_go_hint() {
    print_warning "HA mode enables Mooncake high availability mode and depends on Go."
    print_warning "After Go install/update, run 'source ~/.bashrc' or open a new terminal before rebuilding."
}

find_first_existing_file() {
    local __result_var="$1"
    shift

    local candidate
    for candidate in "$@"; do
        if [ -f "${candidate}" ]; then
            printf -v "${__result_var}" "%s" "${candidate}"
            return 0
        fi
    done
    return 1
}

install_etcd_wrapper_artifacts() {
    local mooncake_etcd_dir="${MOONCAKE_REPO_DIR}/mooncake-common/etcd"
    local mooncake_etcd_build_dir="${MOONCAKE_REPO_DIR}/build/mooncake-common/etcd"
    local home_mooncake_etcd_dir="${HOME}/Mooncake/mooncake-common/etcd"

    local header_src=""
    local so_src=""

    find_first_existing_file header_src \
        "${mooncake_etcd_build_dir}/libetcd_wrapper.h" \
        "/usr/include/libetcd_wrapper.h" \
        || print_error "libetcd_wrapper.h not found after Mooncake HA build"

    find_first_existing_file so_src \
        "${mooncake_etcd_build_dir}/libetcd_wrapper.so" \
        "/usr/lib64/libetcd_wrapper.so" \
        || print_error "libetcd_wrapper.so not found after Mooncake HA build"

    run_or_die "Failed to prepare Mooncake etcd artifact dir: ${mooncake_etcd_dir}" mkdir -p "${mooncake_etcd_dir}"
    run_or_die "Failed to sync libetcd_wrapper.h to ${mooncake_etcd_dir}" install -m 0644 "${header_src}" "${mooncake_etcd_dir}/libetcd_wrapper.h"
    run_or_die "Failed to sync libetcd_wrapper.so to ${mooncake_etcd_dir}" install -m 0755 "${so_src}" "${mooncake_etcd_dir}/libetcd_wrapper.so"

    if [ -d "${home_mooncake_etcd_dir}" ]; then
        run_or_die "Failed to sync libetcd_wrapper.h to ${home_mooncake_etcd_dir}" install -m 0644 "${mooncake_etcd_dir}/libetcd_wrapper.h" "${home_mooncake_etcd_dir}/libetcd_wrapper.h"
        run_or_die "Failed to sync libetcd_wrapper.so to ${home_mooncake_etcd_dir}" install -m 0755 "${mooncake_etcd_dir}/libetcd_wrapper.so" "${home_mooncake_etcd_dir}/libetcd_wrapper.so"
    fi

    run_or_die "Failed to prepare /usr/lib64" mkdir -p /usr/lib64
    run_or_die "Failed to install libetcd_wrapper.h to /usr/include" install -m 0644 "${mooncake_etcd_dir}/libetcd_wrapper.h" /usr/include/libetcd_wrapper.h
    run_or_die "Failed to install libetcd_wrapper.so to /usr/lib64" install -m 0755 "${mooncake_etcd_dir}/libetcd_wrapper.so" /usr/lib64/libetcd_wrapper.so
    print_success "Installed libetcd_wrapper.h to /usr/include and libetcd_wrapper.so to /usr/lib64"
}

ensure_pybind11_for_memfabric() {
    local target_version="2.10.3"
    local pip_cmd
    local current_version

    if command -v pip >/dev/null 2>&1; then
        pip_cmd="pip"
    elif command -v pip3 >/dev/null 2>&1; then
        pip_cmd="pip3"
    else
        print_error "pip/pip3 not found, cannot check or install pybind11"
    fi

    current_version="$(${pip_cmd} show pybind11 2>/dev/null | awk -F': ' '$1=="Version"{print $2}' || true)"
    if [ "${current_version}" = "${target_version}" ]; then
        print_success "pybind11==${target_version} already installed"
        return
    fi

    print_section "Installing pybind11==${target_version} for memfabric build"
    if [ -n "${current_version}" ]; then
        print_warning "Current pybind11 version: ${current_version}"
    else
        print_warning "pybind11 not installed in current pip environment"
    fi
    run_or_die "Failed to install pybind11==${target_version}" "${pip_cmd}" install --upgrade "pybind11==${target_version}"
}

patch_yalantinglibs_config() {
    local config_file="$1"
    if [ -f "${config_file}" ]; then
        run_or_die \
            "Failed to patch yalantinglibs config.cmake" \
            sed -i \
            '54s/target_link_libraries(${ylt_target_name} -libverbs)/target_link_libraries(${ylt_target_name} INTERFACE -libverbs)/' \
            "${config_file}"
    fi
}

patch_yalantinglibs_easylog() {
    local easylog_header="$1"
    if [ -f "${easylog_header}" ]; then
        run_or_die \
            "Failed to patch yalantinglibs easylog severity default" \
            sed -i \
            '/std::atomic<Severity> min_severity_ =/,/#endif/c\  std::atomic<Severity> min_severity_ = Severity::WARN;' \
            "${easylog_header}"
    fi
}

install_yalantinglibs() {
    ensure_deps_root_dir

    if [ -f "${YALANTING_CONFIG_FILE}" ]; then
        print_success "yalantinglibs already installed."
        return
    fi

    print_section "Installing yalantinglibs"

    local thirdparties_dir
    thirdparties_dir="$(dirname "${YALANTING_SOURCE_DIR}")"
    local yalanting_easylog_header="${YALANTING_SOURCE_DIR}/include/ylt/easylog.hpp"

    run_or_die "Failed to create install dir ${YALANTING_INSTALL_PREFIX}" mkdir -p "${YALANTING_INSTALL_PREFIX}"
    run_or_die "Failed to create thirdparties dir ${thirdparties_dir}" mkdir -p "${thirdparties_dir}"
    run_or_die "Failed to remove old yalantinglibs source" rm -rf "${YALANTING_SOURCE_DIR}"

    run_or_die "Failed to clone yalantinglibs" git clone "${YALANTING_REPO_URL}" "${YALANTING_SOURCE_DIR}"
    run_or_die "Failed to checkout yalantinglibs version 0.5.5" git -C "${YALANTING_SOURCE_DIR}" checkout 0.5.5

    patch_yalantinglibs_easylog "${yalanting_easylog_header}"
    run_or_die "Failed to create yalantinglibs build dir" mkdir -p "${YALANTING_BUILD_DIR}"

    run_or_die \
        "Failed to configure yalantinglibs" \
        cmake \
        -S "${YALANTING_SOURCE_DIR}" \
        -B "${YALANTING_BUILD_DIR}" \
        -DCMAKE_INSTALL_PREFIX="${YALANTING_INSTALL_PREFIX}" \
        -DBUILD_EXAMPLES=OFF \
        -DBUILD_BENCHMARK=OFF \
        -DBUILD_UNIT_TESTS=OFF \
        -DYLT_ENABLE_IBV=ON

    run_or_die "Failed to build yalantinglibs" cmake --build "${YALANTING_BUILD_DIR}" -j"$(nproc)"
    run_or_die "Failed to install yalantinglibs" cmake --install "${YALANTING_BUILD_DIR}"

    patch_yalantinglibs_config "${YALANTING_CONFIG_FILE}"
    [ -f "${YALANTING_CONFIG_FILE}" ] || print_error "yalantinglibs config not found after install: ${YALANTING_CONFIG_FILE}"
    print_success "yalantinglibs installed successfully"
}

get_mooncake_commit_id() {
    local mooncake_commit_id
    mooncake_commit_id="$(git -C "${REPO_ROOT}" rev-parse HEAD:third_party/Mooncake)" \
        || print_error "Failed to read Mooncake submodule commit from xllm HEAD"
    [ -n "${mooncake_commit_id}" ] || print_error "Mooncake submodule commit is empty in xllm HEAD"
    echo "${mooncake_commit_id}"
}

get_mooncake_repo_url() {
    local gitmodules_path="${REPO_ROOT}/.gitmodules"
    [ -f "${gitmodules_path}" ] || print_error ".gitmodules not found: ${gitmodules_path}"

    local mooncake_repo_url
    mooncake_repo_url="$(git -C "${REPO_ROOT}" config --file "${gitmodules_path}" --get submodule.third_party/Mooncake.url)" \
        || print_error "Failed to read Mooncake submodule URL from ${gitmodules_path}"
    [ -n "${mooncake_repo_url}" ] || print_error "Failed to read Mooncake submodule URL from ${gitmodules_path}"
    echo "${mooncake_repo_url}"
}

install_mooncake() {
    ensure_deps_root_dir
    ensure_mooncake_yum_packages

    local mooncake_repo_url
    mooncake_repo_url="$(get_mooncake_repo_url)"

    local mooncake_commit_id
    mooncake_commit_id="$(get_mooncake_commit_id)"
    local mooncake_marker_path="${MOONCAKE_INSTALL_DIR}/${mooncake_commit_id}.ha-${ENABLE_HA}"

    if [ -e "${mooncake_marker_path}" ]; then
        print_success "Mooncake already installed."
        if [ "${ENABLE_HA}" = "true" ]; then
            install_etcd_wrapper_artifacts
            print_ha_go_hint
        fi
        return
    fi

    print_section "Installing Mooncake (${mooncake_commit_id}, device=${DEVICE:-unknown}, enable_ha=${ENABLE_HA})"

    if [ -d "${MOONCAKE_REPO_DIR}/.git" ]; then
        run_or_die "Failed to fetch Mooncake updates" git -C "${MOONCAKE_REPO_DIR}" fetch --all --tags --prune
    elif [ -e "${MOONCAKE_REPO_DIR}" ]; then
        print_error "Path exists but is not a git repository: ${MOONCAKE_REPO_DIR}"
    else
        run_or_die "Failed to clone Mooncake" git clone "${mooncake_repo_url}" "${MOONCAKE_REPO_DIR}"
    fi

    run_or_die "Failed to checkout Mooncake commit ${mooncake_commit_id}" git -C "${MOONCAKE_REPO_DIR}" checkout "${mooncake_commit_id}"
    run_or_die "Failed to initialize Mooncake submodules" git -C "${MOONCAKE_REPO_DIR}" submodule update --init --recursive

    [ -f "${MOONCAKE_CMAKE_FILE}" ] || print_error "Mooncake CMakeLists not found: ${MOONCAKE_CMAKE_FILE}"

    local mooncake_cmake_changed=0
    if grep -Fq '#add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/extern/pybind11)' "${MOONCAKE_CMAKE_FILE}"; then
        run_or_die \
            "Failed to un-comment pybind11 in ${MOONCAKE_CMAKE_FILE}" \
            sed -i \
            's|#add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/extern/pybind11)|add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/extern/pybind11)|' \
            "${MOONCAKE_CMAKE_FILE}"
        mooncake_cmake_changed=1
    fi

    local build_dir="${MOONCAKE_REPO_DIR}/build"
    run_or_die "Failed to create Mooncake build dir" mkdir -p "${build_dir}"

    local cmake_args=(
        -DBUILD_EXAMPLES=OFF
        -DBUILD_UNIT_TESTS=OFF
        -DYLT_ENABLE_IBV=ON
    )
    if [ "${ENABLE_HA}" = "true" ]; then
        ensure_go
        print_ha_go_hint
        cmake_args+=(
            -DUSE_ETCD=ON
            -DSTORE_USE_ETCD=ON
        )
    else
        cmake_args+=(
            -DUSE_ETCD=OFF
            -DSTORE_USE_ETCD=OFF
        )
    fi
    if [ "${DEVICE}" = "a2" ] || [ "${DEVICE}" = "a3" ]; then
        cmake_args+=(-DUSE_UB=ON)
    fi
    cmake_args+=("-Dyalantinglibs_DIR=${YALANTING_INSTALL_PREFIX}/lib/cmake/yalantinglibs")

    set +e
    (
        cd "${build_dir}" || exit 1
        cmake "${cmake_args[@]}" ..
        make -j"$(nproc)"
        make install
    )
    local build_status=$?
    set -e

    if [ "${mooncake_cmake_changed}" -eq 1 ]; then
        run_or_die \
            "Failed to restore ${MOONCAKE_CMAKE_FILE}" \
            sed -i \
            's|add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/extern/pybind11)|#add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/extern/pybind11)|' \
            "${MOONCAKE_CMAKE_FILE}"
    fi
    [ "${build_status}" -eq 0 ] || print_error "Failed to build/install Mooncake"

    if [ "${ENABLE_HA}" = "true" ]; then
        install_etcd_wrapper_artifacts
    fi

    [ -d "${MOONCAKE_INSTALL_DIR}" ] || print_error "Mooncake install dir not found: ${MOONCAKE_INSTALL_DIR}"
    run_or_die "Failed to create Mooncake commit marker: ${mooncake_marker_path}" touch "${mooncake_marker_path}"
    print_success "Mooncake installed for commit ${mooncake_commit_id} (enable_ha=${ENABLE_HA})"
}

install_memfabric() {
    ensure_deps_root_dir

    if [ "${DEVICE}" != "a2" ] && [ "${DEVICE}" != "a3" ]; then
        print_warning "Skip memfabric_hybrid install for device=${DEVICE:-unknown}; only a2/a3 require memfabric"
        return
    fi

    local memfabric_marker_path="${MEMFABRIC_INSTALL_DIR}/latest/${MEMFABRIC_TARGET_COMMIT}"
    if [ -e "${memfabric_marker_path}" ]; then
        print_success "memfabric_hybrid already installed"
        return
    fi

    print_section "Installing memfabric_hybrid (${MEMFABRIC_TARGET_COMMIT})"

    if [ -d "${MEMFABRIC_REPO_DIR}/.git" ]; then
        run_or_die "Failed to fetch memfabric_hybrid updates" git -C "${MEMFABRIC_REPO_DIR}" fetch --all --tags --prune
    elif [ -e "${MEMFABRIC_REPO_DIR}" ]; then
        print_error "Path exists but is not a git repository: ${MEMFABRIC_REPO_DIR}"
    else
        run_or_die "Failed to clone memfabric_hybrid" git clone "${MEMFABRIC_REPO_URL}" "${MEMFABRIC_REPO_DIR}"
    fi

    run_or_die "Failed to checkout memfabric_hybrid commit ${MEMFABRIC_TARGET_COMMIT}" git -C "${MEMFABRIC_REPO_DIR}" checkout "${MEMFABRIC_TARGET_COMMIT}"

    ensure_pybind11_for_memfabric

    run_or_die \
        "Failed to run memfabric build script" \
        bash -lc "cd '${MEMFABRIC_REPO_DIR}' && sh script/build_and_pack_run.sh"

    local installer_path="${MEMFABRIC_REPO_DIR}/${MEMFABRIC_INSTALLER}"
    [ -f "${installer_path}" ] || print_error "memfabric installer not found: ${installer_path}"

    run_or_die \
        "Failed to run memfabric installer" \
        bash -lc "cd '${MEMFABRIC_REPO_DIR}' && sh '${MEMFABRIC_INSTALLER}'"

    [ -d "${MEMFABRIC_INSTALL_DIR}" ] || print_error "memfabric install dir not found: ${MEMFABRIC_INSTALL_DIR}"
    run_or_die "Failed to create memfabric commit marker: ${memfabric_marker_path}" touch "${memfabric_marker_path}"
    print_success "memfabric_hybrid installed for commit ${MEMFABRIC_TARGET_COMMIT}"
}

main() {
    normalize_enable_ha

    if [ -z "${DEPENDENCY_NAME}" ]; then
        usage
        exit 1
    fi

    case "${DEPENDENCY_NAME}" in
        yalantinglibs)
            install_yalantinglibs
            ;;
        mooncake)
            install_mooncake
            ;;
        memfabric)
            install_memfabric
            ;;
        all)
            install_yalantinglibs
            install_mooncake
            install_memfabric
            cleanup_dependency_workspace
            ;;
        *)
            usage
            print_error "Unknown dependency_name: ${DEPENDENCY_NAME}"
            ;;
    esac
}

main "$@"
