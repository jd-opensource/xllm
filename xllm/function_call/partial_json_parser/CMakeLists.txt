include(cc_library)

cc_library(
  NAME 
    partial_json_parser
  SRCS 
    src/parser.cpp
  INCLUDES 
    include
  DEPS
    nlohmann_json::nlohmann_json
)
