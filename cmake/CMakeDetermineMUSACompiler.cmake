# Try to find the compiler
find_program(CMAKE_MUSA_COMPILER
    NAMES mcc
    DOC "MUSA compiler"
)
set(CMAKE_MUSA_COMPILER_ENV_VAR "MUSA")

# Check if compiler was found
if(CMAKE_MUSA_COMPILER)
    set(CMAKE_MUSA_COMPILER_LOADED 1)
    message(STATUS "Found MUSA compiler: ${CMAKE_MUSA_COMPILER}")
else()
    message(FATAL_ERROR "MUSA compiler not found")
endif()

configure_file(${CMAKE_CURRENT_LIST_DIR}/CMakeMUSACompiler.cmake.in
	${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/${CMAKE_VERSION}/CMakeMUSACompiler.cmake IMMEDIATE @ONLY)