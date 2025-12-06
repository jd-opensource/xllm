include(cc_library)

cc_library(
  NAME
    models
  HDRS
    model_registry.h
    models.h
  SRCS
    model_registry.cpp
  DEPS
    :model
)
