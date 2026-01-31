include(cc_library)

cc_library(
  NAME
    ilu_layers
  HDRS
    attention.h
    attention_metadata.h
  SRCS
    attention.cpp
  DEPS
    :common_layers
)
