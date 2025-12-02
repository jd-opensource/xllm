#include "siglip_encoder_loader.h"

namespace xllm {
namespace layer {

SiglipEncoderUpLoader::SiglipEncoderUpLoader(const ModelContext& context)
    : BaseLoader(0, context) {
  options_ = context.get_tensor_options();
}

void SiglipEncoderUpLoader::load_state_dict(const StateDict& state_dict) {
  const std::set<std::string> key_names = {"layer_norm1.weight",
                                           "layer_norm1.bias",
                                           "self_attn.q_proj.weight",
                                           "self_attn.q_proj.bias",
                                           "self_attn.k_proj.weight",
                                           "self_attn.k_proj.bias",
                                           "self_attn.v_proj.weight",
                                           "self_attn.v_proj.bias"};

  for (const auto& [name, tensor] : state_dict) {
    if (key_names.find(name) == key_names.end()) continue;

    auto weight_npu = tensor.to(options_);

    weights_map_[name] = weight_npu;
  }
}

SiglipEncoderDownLoader::SiglipEncoderDownLoader(const ModelContext& context)
    : BaseLoader(0, context) {
  options_ = context.get_tensor_options();
}

void SiglipEncoderDownLoader::load_state_dict(const StateDict& state_dict) {
  const std::set<std::string> key_names = {"self_attn.out_proj.weight",
                                           "self_attn.out_proj.bias",
                                           "layer_norm2.weight",
                                           "layer_norm2.bias",
                                           "mlp.fc1.weight",
                                           "mlp.fc1.bias",
                                           "mlp.fc2.weight",
                                           "mlp.fc2.bias"};

  for (const auto& [name, tensor] : state_dict) {
    if (key_names.find(name) == key_names.end()) continue;

    auto weight_npu = tensor.to(options_);

    weights_map_[name] = weight_npu;
  }
}

}  // namespace layer
}  // namespace xllm