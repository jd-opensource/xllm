# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code), Codex (codex.com) or Cursor (cursor.com) when working with code in this repository.

## Project Overview

xLLM is an efficient LLM inference framework, specifically optimized for Chinese AI accelerators, enabling enterprise-grade deployment with enhanced efficiency and reduced cost.

## Quick Reference

| Task                                  | Command                                         |
| ------------------------------------- | ----------------------------------------------- |
| Initialize submodules                 | `git submodule update --init`                   |
| Build xLLM binary                     | `python setup.py build`                         |
| Build xLLM wheel                      | `python setup.py bdist_wheel`                   |
| Test xLLM                             | `python setup.py test`                          |
| Test specific unit test               | `python setup.py test --test-name <test_name>`  |
| Build xLLM binary for specific device | `python setup.py build --device <device>`       |
| Build xLLM wheel for specific device  | `python setup.py bdist_wheel --device <device>` |
| Install pre-commit hooks              | `pre-commit install`                            |




## Quick Start for Development

### Setup xLLM

```bash
git clone https://github.com/jd-opensource/xllm
cd xllm

# install pre-commit hooks for the first time
pip install pre-commit
pre-commit install

git submodule update --init
```

### Build xLLM

Build xLLM binary or wheel or so file.

```bash
# build bin
python setup.py build

# build wheel
python setup.py bdist_wheel

# build xllm so file
python setup.py build --generate-so true
```

### Unit Test


```bash
# test all unit tests
python setup.py test

# test specific unit test
# test_name is the name of the test case, for example:
#   python setup.py test --test-name common_test
python setup.py test --test-name <test_name>

```


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

* Follow the code style guide in [custom-code-style.md](.agent/skills/code-review/custom-code-style.md).
* Follow DDD (Domain Driven Design) principles, and keep the codebase clean and maintainable.
* Follow Google C++/Python Style Guide, if not specified in the code style guide.

## Review Instructions

* Review the code changes for quality, security, performance, and correctness following the project-specific standards.
* Review the code changes for DDD (Domain Driven Design) principles, and keep the codebase clean and maintainable.
* Review the code changes for Google C++/Python Style Guide, if not specified in the project-specific coding style.
* Review the code changes for the project-specific coding style in [custom-code-style.md](.agent/skills/code-review/custom-code-style.md).