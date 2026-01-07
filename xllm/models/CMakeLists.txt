include(cc_library)

# Define the library
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
    :dit_cache
    :model_loader
)
