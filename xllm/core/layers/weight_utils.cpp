#include "weight_utils.h"

namespace llm {

void WeightUtils::load_weight(const StateDict& state_dict,
                              const std::string& name,
                              torch::Tensor& weight,
                              bool& weight_is_loaded) {
  const auto tensor = state_dict.get_tensor(name);
  if (tensor.defined()) {
    CHECK(!weight_is_loaded)
        << "weight already loaded, name: " << state_dict.prefix() << name;
    CHECK_EQ(weight.sizes(), tensor.sizes())
        << "weight size mismatch for " << state_dict.prefix() << name;
    weight.copy_(tensor);
    weight_is_loaded = true;
  }
}

void WeightUtils::load_sharded_weight(const StateDict& state_dict,
                                      const std::string& name,
                                      int64_t dim,
                                      int32_t rank,
                                      int32_t world_size,
                                      torch::Tensor& weight,
                                      bool& weight_is_loaded) {
  const auto tensor =
      state_dict.get_sharded_tensor(name, dim, rank, world_size);
  if (tensor.defined()) {
    CHECK_EQ(weight.sizes(), tensor.sizes())
        << "weight size mismatch for " << state_dict.prefix() << name;
    weight.copy_(tensor);
    weight_is_loaded = true;
  }
}

void WeightUtils::load_sharded_weight(const StateDict& state_dict,
                                      const std::string& name,
                                      TensorTransform transform_func,
                                      int64_t dim,
                                      int32_t rank,
                                      int32_t world_size,
                                      torch::Tensor& weight,
                                      bool& weight_is_loaded) {
  auto tensor = state_dict.get_sharded_tensor(name, dim, rank, world_size);
  if (tensor.defined()) {
    tensor = transform_func(tensor);
    CHECK(!weight_is_loaded)
        << "weight already loaded, name: " << state_dict.prefix() << name;
    CHECK_EQ(weight.sizes(), tensor.sizes())
        << "weight size mismatch for " << state_dict.prefix() << name;
    weight.copy_(tensor);
    weight_is_loaded = true;
  }
}

void WeightUtils::load_fused_weight(
    const StateDict& state_dict,
    const std::vector<std::string>& prefixes,
    const std::string& name,
    int64_t dim,
    int32_t rank,
    int32_t world_size,
    std::vector<torch::Tensor>& accumulated_tensors,
    torch::Tensor& weight,
    bool& weight_is_loaded) {
  // return if the weight is already loaded
  if (weight_is_loaded) {
    return;
  }

  // resize the accumulated weight list if needed
  if (accumulated_tensors.size() < prefixes.size()) {
    accumulated_tensors.resize(prefixes.size());
  }

  // load the weights from the state_dict
  std::vector<torch::Tensor> tensors(prefixes.size());
  for (size_t i = 0; i < prefixes.size(); ++i) {
    const std::string tensor_name = prefixes[i] + name;
    const auto tensor =
        state_dict.get_sharded_tensor(tensor_name, dim, rank, world_size);
    if (tensor.defined()) {
      tensors[i] = tensor;
    } else if (accumulated_tensors[i].defined()) {
      // carry over the accumulated weight
      tensors[i] = accumulated_tensors[i];
    }
  }

  const bool all_loaded =
      std::all_of(tensors.begin(), tensors.end(), [](const torch::Tensor& t) {
        return t.defined();
      });
  if (!all_loaded) {
    // accumulate the weights for future merge
    for (size_t i = 0; i < tensors.size(); ++i) {
      if (!accumulated_tensors[i].defined() && tensors[i].defined()) {
        // make a clone for safety
        accumulated_tensors[i] = tensors[i].clone();
      }
    }
  } else {
    const auto merged_weight = torch::cat(tensors, /*dim=*/dim);
    CHECK_EQ(weight.sizes(), merged_weight.sizes())
        << "weight size mismatch for " << state_dict.prefix() << name;
    weight.copy_(merged_weight);
    // release the memory for weight_list
    accumulated_tensors.clear();
    weight_is_loaded = true;
  }
}

}  // namespace llm