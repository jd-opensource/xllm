include(cc_library)

if(USE_MLU)
  # Modify dependencies for MLU
  list(APPEND BASE_DEPS :mlu_layers)
endif()

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
    ${BASE_DEPS}
)
