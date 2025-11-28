include(cc_library)

# Define the library
cc_library(
  NAME
    models
  HDRS
    model_registry.h
    models.h
    lazy_layer_loader.h
  SRCS
    model_registry.cpp
    lazy_layer_loader.cpp
  DEPS
    :model
)
