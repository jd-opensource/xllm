include(cc_library)

cc_library(
  NAME
    mlu_layers
  HDRS
    attention.h
    attention_metadata.h
    deepseek_v2_attention.h
    deepseek_v2_decoder_layer_impl.h
  SRCS
    attention.cpp
    deepseek_v2_attention.cpp
    deepseek_v2_decoder_layer_impl.cpp
  DEPS
    :common_layers
)
