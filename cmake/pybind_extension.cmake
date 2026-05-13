include(CMakeParseArguments)

# pybind_extension()
#
# Parameters:
# NAME: name of module
# HDRS: List of public header files for the library
# SRCS: List of source files for the library
# DEPS: List of other libraries to be linked in to the binary targets
# COPTS: List of private compile options
# DEFINES: List of public defines
# LINKOPTS: List of link options
#
# pybind_extension(
#   NAME
#     awesome
#   HDRS
#     "a.h"
#   SRCS
#     "a.cc"
# )
#

if(NOT DEFINED PYTHON_MODULE_EXTENSION OR NOT DEFINED PYTHON_MODULE_DEBUG_POSTFIX)
  execute_process(
    COMMAND
      "${Python_EXECUTABLE}" "-c"
      "import sys, importlib; s = importlib.import_module('distutils.sysconfig' if sys.version_info < (3, 10) else 'sysconfig'); print(s.get_config_var('EXT_SUFFIX') or s.get_config_var('SO'))"
    OUTPUT_VARIABLE _PYTHON_MODULE_EXT_SUFFIX
    ERROR_VARIABLE _PYTHON_MODULE_EXT_SUFFIX_ERR
    OUTPUT_STRIP_TRAILING_WHITESPACE)

  if(_PYTHON_MODULE_EXT_SUFFIX STREQUAL "")
    message(
      FATAL_ERROR "pybind11 could not query the module file extension, likely the 'distutils'"
                  "package is not installed. Full error message:\n${_PYTHON_MODULE_EXT_SUFFIX_ERR}"
    )
  endif()

  # This needs to be available for the pybind11_extension function
  if(NOT DEFINED PYTHON_MODULE_DEBUG_POSTFIX)
    get_filename_component(_PYTHON_MODULE_DEBUG_POSTFIX "${_PYTHON_MODULE_EXT_SUFFIX}" NAME_WE)
    set(PYTHON_MODULE_DEBUG_POSTFIX
        "${_PYTHON_MODULE_DEBUG_POSTFIX}"
        CACHE INTERNAL "")
  endif()

  if(NOT DEFINED PYTHON_MODULE_EXTENSION)
    get_filename_component(_PYTHON_MODULE_EXTENSION "${_PYTHON_MODULE_EXT_SUFFIX}" EXT)
    set(PYTHON_MODULE_EXTENSION
        "${_PYTHON_MODULE_EXTENSION}"
        CACHE INTERNAL "")
  endif()
endif()

function(pybind_extension)
  cmake_parse_arguments(
    PY # prefix
    "TESTONLY" # options
    "NAME" # one value args
    "HDRS;SRCS;COPTS;DEFINES;LINKOPTS;LINKDIRS;DEPS" # multi value args
    ${ARGN}
  )

  if(PY_TESTONLY AND (NOT BUILD_TESTING))
    return()
  endif()

  add_dependencies(export_module ${PY_NAME})

  add_library(${PY_NAME} SHARED)
  target_sources(${PY_NAME} 
    PRIVATE ${PY_SRCS} ${PY_HDRS}
  )
  target_link_libraries(${PY_NAME}
    PUBLIC ${PY_DEPS}
    PRIVATE ${PY_LINKOPTS}
  )
  # search directories for libraries
  target_link_directories(${PY_NAME}
    PUBLIC ${PY_LINKDIRS}
  )
  target_compile_options(${PY_NAME} PRIVATE ${PY_COPTS})
  target_compile_definitions(${PY_NAME} PUBLIC ${PY_DEFINES})

  # -fvisibility=hidden is required to allow multiple modules compiled against
  # different pybind versions to work properly, and for some features (e.g.
  # py::module_local). 
  if(NOT DEFINED CMAKE_CXX_VISIBILITY_PRESET)
    set_target_properties(${PY_NAME} PROPERTIES CXX_VISIBILITY_PRESET "hidden")
  endif()

  if(NOT DEFINED CMAKE_CUDA_VISIBILITY_PRESET)
    set_target_properties(${PY_NAME} PROPERTIES CUDA_VISIBILITY_PRESET "hidden")
  endif()

  set_target_properties(
    ${PY_NAME}
    PROPERTIES PREFIX ""
               DEBUG_POSTFIX "${PYTHON_MODULE_DEBUG_POSTFIX}"
               SUFFIX "${PYTHON_MODULE_EXTENSION}"
               BUILD_RPATH "$ORIGIN"
               INSTALL_RPATH "$ORIGIN")

endfunction()
