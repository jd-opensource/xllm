# xLLM Coding Agent Instructions

## Quick Reference

| Task                                  | Command                                         |
| ------------------------------------- | ----------------------------------------------- |
| Initialize submodules                 | `git submodule update --init --recursive`                   |
| Build xLLM binary                     | `python setup.py build`                         |
| Build xLLM wheel                      | `python setup.py bdist_wheel`                   |
| Test xLLM                             | `python setup.py test`                          |
| Test specific unit test               | `python setup.py test --test-name <test_name>`  |
| Build xLLM binary for specific device | `python setup.py build --device <device>`       |
| Build xLLM wheel for specific device  | `python setup.py bdist_wheel --device <device>` |
| Install pre-commit hooks              | `pre-commit install`                            |

## Directory Structure

```
├── xllm/
|   : main source folder
│   ├── api_service/               # code for api services
│   ├── c_api/                     # code for c api
│   ├── cc_api/                    # code for cc api 
│   ├── core/  
│   │   : xllm core features folder
│   │   ├── common/                
│   │   ├── distributed_runtime/   # code for distributed and pd serving
│   │   ├── framework/             # code for execution orchestration
│   │   ├── kernels/               # adaption for npu kernels adaption
│   │   ├── layers/                # model layers impl
│   │   ├── platform/              # adaption for various platform
│   │   ├── runtime/               # code for worker and executor
│   │   ├── scheduler/             # code for batch and pd scheduler
│   │   └── util/
│   ├── function_call              # code for tool call parser
│   ├── models/                    # models impl
│   ├── parser/                    # parser reasoning
│   ├── processors/                # code for vlm pre-processing
│   ├── proto/                     # communication protocol
│   ├── pybind/                    # code for python bind
|   └── server/                    # xLLM server
├── examples/                      # examples of calling xLLM
├── tools/                         # code for npu time generations
└── xllm.cpp                       # entrypoint of xLLM
```

## Code Style Guide

* Follow the code style guide in [custom-code-style.md](.agent/skills/code-review/references/custom-code-style.md).
* Follow DDD (Domain Driven Design) principles, and keep the codebase clean and maintainable.
* Follow Google C++/Python Style Guide, if not specified in the code style guide.

## Review Instructions

* Review the code changes for quality, security, performance, and correctness following the project-specific standards.
* Review the code changes for DDD (Domain Driven Design) principles, and keep the codebase clean and maintainable.
* Review the code changes for Google C++/Python Style Guide, if not specified in the project-specific coding style.
* Review the code changes for the project-specific coding style in [custom-code-style.md](.agent/skills/code-review/references/custom-code-style.md).
