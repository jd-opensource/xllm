#ifndef TEST_UTILS_H
#define TEST_UTILS_H

#include <filesystem>
#include <fstream>
#include <string>

namespace triton {
namespace test {
/**
 * @brief Get the binary directory path relative to the test executable
 *
 * The binary directory is expected to be at: <project_root>/binary/
 * This function tries to locate it by:
 * 1. Using compile-time defined BINARY_DIR if available
 * 2. Otherwise, inferring from current working directory or executable path
 *
 * @return std::string The path to the binary directory (with trailing slash)
 */
inline std::string GetBinaryDir() {
// Try compile-time defined path first
#ifdef TEST_BINARY_DIR
  std::string binary_dir = TEST_BINARY_DIR;
  if (!binary_dir.empty() && binary_dir.back() != '/') {
    binary_dir += "/";
  }
  return binary_dir;
#else
  return "";  // TODO process for undefined TEST_BINARY_DIR
#endif
}

/**
 * @brief Build full kernel binary path from filename
 *
 * @param filename Kernel binary filename (e.g., "add_kernel.npubin")
 * @return std::string Full path to the kernel binary
 */
inline std::string GetKernelBinaryPath(const std::string& filename) {
  std::string binary_dir = GetBinaryDir();
  return binary_dir + filename;
}

/**
 * @brief Check if a file exists
 *
 * @param filepath Full path to the file
 * @return true if file exists, false otherwise
 */
inline bool FileExists(const std::string& filepath) {
  return std::filesystem::exists(filepath) &&
         std::filesystem::is_regular_file(filepath);
}

/**
 * @brief Check if kernel binary exists
 *
 * @param filename Kernel binary filename
 * @return true if kernel binary exists, false otherwise
 */
inline bool KernelBinaryExists(const std::string& filename) {
  return FileExists(GetKernelBinaryPath(filename));
}
}  // namespace test
}  // namespace triton

#endif  // TEST_UTILS_H
