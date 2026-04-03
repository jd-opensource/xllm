# Repository Guidelines

## Project Structure & Module Organization
`xllm/` contains the product code. Core runtime, kernels, schedulers, and distributed code live in `xllm/core/`; service APIs are in `xllm/api_service/` and `xllm/server/`; Python bindings live in `xllm/pybind/`. Tests are usually colocated as `*_test.cpp` and registered in the nearest `CMakeLists.txt`. Use `examples/` for Python samples, `scripts/` for local helpers, `docs/en` and `docs/zh` for docs, `cibuild/` for backend CI helpers, and `third_party/` only for vendored dependency updates.

## Container Requirement
Run all builds, tests, and service processes inside the `kangmeng3-xllm` Docker container. Open a shell with `sudo -n docker exec -it kangmeng3-xllm /bin/bash`, or run one-off commands such as `sudo -n docker exec kangmeng3-xllm bash -lc 'python setup.py test --device cuda'`. Do not compile, test, or launch `xllm` services directly on the host.

## Build, Test, and Development Commands
Initialize submodules: `git submodule update --init`.
Install formatting hooks: `pip install pre-commit && pre-commit install`.
Inside the container, build with `python setup.py build --device cuda` (replace `cuda` with `npu`, `mlu`, `ilu`, or `musa` as needed). This drives CMake + Ninja and writes outputs under `build/`.
Inside the container, run the unit-test suite with `python setup.py test --device cuda`.
Inside the container, run one target while iterating with `python setup.py test --device cuda --test-name platform_vmm_test`.
Inside the container, build a wheel with `python setup.py bdist_wheel --device cuda`, and launch services with `xllm ...` only after entering `kangmeng3-xllm`.

## Coding Style & Naming Conventions
Follow `.clang-format`: Google-based style, spaces only, 2-space indentation, 80-column limit, and left-aligned pointers. Keep C++ filenames in `snake_case` (`chat_service_impl.cpp`, `rate_limiter.h`). Name tests `*_test.cpp`; GTest suites and fixtures use `PascalCase` such as `AnthropicProtocolTest`. Run `pre-commit run --all-files` before opening a PR.

## Testing Guidelines
Add or extend GTest coverage beside the module you change and wire new tests through the local `CMakeLists.txt` with existing `cc_test` patterns. Keep tests deterministic; some targets are hardware-specific. There is no published coverage threshold, so every PR should note the in-container test command and backend used for verification.

## Commit & Pull Request Guidelines
Match the existing commit style: `feat:`, `fix:`, `bugfix:`, `refactor:`, `docs:`, or `perf:` followed by a short imperative subject; issue references like `(#1105)` are optional but common. PRs should summarize the affected backend or hardware path, describe behavior changes, and list the verification commands you ran. Link the related issue when one exists, and attach logs or screenshots only when output or documentation rendering changed.
