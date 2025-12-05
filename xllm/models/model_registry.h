/* Copyright 2025 The xLLM Authors. All Rights Reserved.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>

#include "core/framework/dit_model_context.h"
#include "core/framework/model/causal_lm.h"
#include "core/framework/model/causal_vlm.h"
#include "core/framework/model/dit_model.h"
#include "core/framework/model/embedding_lm.h"
#include "core/framework/model/embedding_vlm.h"
#include "core/framework/model/mm_embedding_vlm.h"
#include "core/framework/model_context.h"
#include "core/framework/tokenizer/tokenizer_args.h"
#include "core/util/json_reader.h"
#include "core/util/type_traits.h"  // IWYU pragma: keep
#include "processors/image_processor.h"
#include "processors/input_processor.h"

namespace xllm {

using CausalLMFactory =
    std::function<std::unique_ptr<CausalLM>(const ModelContext& context)>;

using CausalVLMFactory =
    std::function<std::unique_ptr<CausalVLM>(const ModelContext& context)>;

using EmbeddingLMFactory =
    std::function<std::unique_ptr<EmbeddingLM>(const ModelContext& context)>;

using EmbeddingVLMFactory =
    std::function<std::unique_ptr<EmbeddingVLM>(const ModelContext& context)>;

using MMEmbeddingVLMFactory =
    std::function<std::unique_ptr<MMEmbeddingVLM>(const ModelContext& context)>;

using DiTModelFactory =
    std::function<std::unique_ptr<DiTModel>(const DiTModelContext& context)>;

using InputProcessorFactory =
    std::function<std::unique_ptr<InputProcessor>(const ModelArgs& args)>;

using ImageProcessorFactory =
    std::function<std::unique_ptr<ImageProcessor>(const ModelArgs& args)>;

using ModelArgsLoader =
    std::function<bool(const JsonReader& json, ModelArgs* args)>;

using QuantArgsLoader =
    std::function<bool(const JsonReader& json, QuantArgs* args)>;

using TokenizerArgsLoader =
    std::function<bool(const JsonReader& json, TokenizerArgs* args)>;

// TODO: add default args loader.
struct ModelMeta {
  CausalLMFactory causal_lm_factory;
  CausalVLMFactory causal_vlm_factory;
  EmbeddingLMFactory embedding_lm_factory;
  EmbeddingVLMFactory embedding_vlm_factory;
  MMEmbeddingVLMFactory mm_embedding_vlm_factory;
  DiTModelFactory dit_model_factory;
  InputProcessorFactory input_processor_factory;
  ImageProcessorFactory image_processor_factory;
  ModelArgsLoader model_args_loader;
  QuantArgsLoader quant_args_loader;
  TokenizerArgsLoader tokenizer_args_loader;
};

// Model registry is a singleton class that registers all models with the
// ModelFactory, ModelArgParser to facilitate model loading.
class ModelRegistry {
 public:
  static ModelRegistry* get_instance();

  static void register_causallm_factory(const std::string& name,
                                        CausalLMFactory factory);

  static void register_causalvlm_factory(const std::string& name,
                                         CausalVLMFactory factory);

  static void register_lm_embedding_factory(const std::string& name,
                                            EmbeddingLMFactory factory);

  static void register_vlm_embedding_factory(const std::string& name,
                                             EmbeddingVLMFactory factory);

  static void register_mm_embedding_vlm_factory(const std::string& name,
                                                MMEmbeddingVLMFactory factory);

  static void register_dit_model_factory(const std::string& name,
                                         DiTModelFactory factory);

  static void register_model_args_loader(const std::string& name,
                                         ModelArgsLoader loader);

  static void register_quant_args_loader(const std::string& name,
                                         QuantArgsLoader loader);

  static void register_tokenizer_args_loader(const std::string& name,
                                             TokenizerArgsLoader loader);

  static void register_input_processor_factory(const std::string& name,
                                               InputProcessorFactory factory);
  static void register_image_processor_factory(const std::string& name,
                                               ImageProcessorFactory factory);

  static CausalLMFactory get_causallm_factory(const std::string& name);

  static CausalVLMFactory get_causalvlm_factory(const std::string& name);

  static EmbeddingLMFactory get_embeddinglm_factory(const std::string& name);

  static EmbeddingVLMFactory get_embeddingvlm_factory(const std::string& name);

  static MMEmbeddingVLMFactory get_mm_embedding_vlm_factory(
      const std::string& name);

  static DiTModelFactory get_dit_model_factory(const std::string& name);

  static ModelArgsLoader get_model_args_loader(const std::string& name);

  static QuantArgsLoader get_quant_args_loader(const std::string& name);

  static TokenizerArgsLoader get_tokenizer_args_loader(const std::string& name);

  static InputProcessorFactory get_input_processor_factory(
      const std::string& name);

  static ImageProcessorFactory get_image_processor_factory(
      const std::string& name);

  static std::string get_model_backend(const std::string& name);

 private:
  std::unordered_map<std::string, ModelMeta> model_registry_;
  std::unordered_map<std::string, std::string> model_backend_;
};

std::unique_ptr<CausalLM> create_llm_model(const ModelContext& context);

std::unique_ptr<CausalVLM> create_vlm_model(const ModelContext& context);

std::unique_ptr<EmbeddingLM> create_lm_embedding_model(
    const ModelContext& context);

std::unique_ptr<EmbeddingVLM> create_vlm_embedding_model(
    const ModelContext& context);

std::unique_ptr<MMEmbeddingVLM> create_vlm_mm_embedding_model(
    const ModelContext& context);

std::unique_ptr<DiTModel> create_dit_model(const DiTModelContext& context);

// Macro to register a model with the ModelRegistry
#define REGISTER_CAUSAL_MODEL_WITH_VARNAME(VarName, ModelType, ModelClass) \
  const bool VarName##_registered = []() {                                 \
    ModelRegistry::register_causallm_factory(                              \
        #ModelType, [](const ModelContext& context) {                      \
          ModelClass model(context);                                       \
          model->eval();                                                   \
          return std::make_unique<xllm::CausalLMImpl<ModelClass>>(         \
              std::move(model), context.get_tensor_options());             \
        });                                                                \
    return true;                                                           \
  }()

#define REGISTER_CAUSAL_MODEL(ModelType, ModelClass) \
  REGISTER_CAUSAL_MODEL_WITH_VARNAME(ModelType, ModelType, ModelClass)

#define REGISTER_CAUSAL_VLM_MODEL_WITH_VARNAME(VarName, ModelType, ModelClass) \
  const bool VarName##_registered = []() {                                     \
    ModelRegistry::register_causalvlm_factory(                                 \
        #ModelType, [](const ModelContext& context) {                          \
          ModelClass model(context);                                           \
          model->eval();                                                       \
          return std::make_unique<xllm::CausalVLMImpl<ModelClass>>(            \
              std::move(model), context.get_tensor_options());                 \
        });                                                                    \
    return true;                                                               \
  }()

#define REGISTER_CAUSAL_VLM_MODEL(ModelType, ModelClass) \
  REGISTER_CAUSAL_VLM_MODEL_WITH_VARNAME(ModelType, ModelType, ModelClass)

// Macro to register a causal model with the ModelRegistry
#define REGISTER_EMBEDDING_MODEL_WITH_VARNAME(VarName, ModelType, ModelClass) \
  const bool VarName##_registered = []() {                                    \
    ModelRegistry::register_lm_embedding_factory(                             \
        #ModelType, [](const ModelContext& context) {                         \
          ModelClass model(context);                                          \
          model->eval();                                                      \
          return std::make_unique<xllm::EmbeddingLMImpl<ModelClass>>(         \
              std::move(model), context.get_tensor_options());                \
        });                                                                   \
    return true;                                                              \
  }()

#define REGISTER_EMBEDDING_MODEL(ModelType, ModelClass) \
  REGISTER_EMBEDDING_MODEL_WITH_VARNAME(ModelType, ModelType, ModelClass)

#define REGISTER_EMBEDDING_VLM_MODEL_WITH_VARNAME(                     \
    VarName, ModelType, ModelClass)                                    \
  const bool VarName##_registered = []() {                             \
    ModelRegistry::register_vlm_embedding_factory(                     \
        #ModelType, [](const ModelContext& context) {                  \
          ModelClass model(context);                                   \
          model->eval();                                               \
          return std::make_unique<xllm::EmbeddingVLMImpl<ModelClass>>( \
              std::move(model), context.get_tensor_options());         \
        });                                                            \
    return true;                                                       \
  }()

#define REGISTER_EMBEDDING_VLM_MODEL(ModelType, ModelClass) \
  REGISTER_EMBEDDING_VLM_MODEL_WITH_VARNAME(ModelType, ModelType, ModelClass)

#define REGISTER_MM_EMBEDDING_VLM_MODEL_WITH_VARNAME(                    \
    VarName, ModelType, ModelClass)                                      \
  const bool VarName##_registered = []() {                               \
    ModelRegistry::register_mm_embedding_vlm_factory(                    \
        #ModelType, [](const ModelContext& context) {                    \
          ModelClass model(context);                                     \
          model->eval();                                                 \
          return std::make_unique<xllm::MMEmbeddingVLMImpl<ModelClass>>( \
              std::move(model), context.get_tensor_options());           \
        });                                                              \
    return true;                                                         \
  }()

#define REGISTER_MM_EMBEDDING_VLM_MODEL(ModelType, ModelClass) \
  REGISTER_MM_EMBEDDING_VLM_MODEL_WITH_VARNAME(ModelType, ModelType, ModelClass)

#define REGISTER_DIT_MODEL_WITH_VARNAME(VarName, ModelType, ModelClass) \
  const bool VarName##_registered = []() {                              \
    ModelRegistry::register_dit_model_factory(                          \
        #ModelType, [](const DiTModelContext& context) {                \
          ModelClass model(context);                                    \
          model->eval();                                                \
          return std::make_unique<xllm::DiTModelImpl<ModelClass>>(      \
              std::move(model), context.get_tensor_options());          \
        });                                                             \
    return true;                                                        \
  }()

#define REGISTER_DIT_MODEL(ModelType, ModelClass) \
  REGISTER_DIT_MODEL_WITH_VARNAME(ModelType, ModelType, ModelClass)

#define REGISTER_INPUT_PROCESSOR_WITH_VARNAME(                \
    VarName, ModelType, InputProcessorClass)                  \
  const bool VarName##_input_processor_registered = []() {    \
    ModelRegistry::register_input_processor_factory(          \
        #ModelType, [](const ModelArgs& args) {               \
          return std::make_unique<InputProcessorClass>(args); \
        });                                                   \
    return true;                                              \
  }()

#define REGISTER_INPUT_PROCESSOR(ModelType, InputProcessorClass) \
  REGISTER_INPUT_PROCESSOR_WITH_VARNAME(                         \
      ModelType, ModelType, InputProcessorClass)

#define REGISTER_IMAGE_PROCESSOR_WITH_VARNAME(                \
    VarName, ModelType, ImageProcessorClass)                  \
  const bool VarName##_image_processor_registered = []() {    \
    ModelRegistry::register_image_processor_factory(          \
        #ModelType, [](const ModelArgs& args) {               \
          return std::make_unique<ImageProcessorClass>(args); \
        });                                                   \
    return true;                                              \
  }()

#define REGISTER_IMAGE_PROCESSOR(ModelType, ImageProcessorClass) \
  REGISTER_IMAGE_PROCESSOR_WITH_VARNAME(                         \
      ModelType, ModelType, ImageProcessorClass)

// Macro to register a model args loader with the ModelRegistry
#define REGISTER_MODEL_ARGS_LOADER_WITH_VARNAME(VarName, ModelType, Loader) \
  const bool VarName##_args_loader_registered = []() {                      \
    ModelRegistry::register_model_args_loader(#ModelType, Loader);          \
    return true;                                                            \
  }()

#define REGISTER_MODEL_ARGS_LOADER(ModelType, Loader) \
  REGISTER_MODEL_ARGS_LOADER_WITH_VARNAME(ModelType, ModelType, Loader)

#define REGISTER_MODEL_ARGS_WITH_VARNAME(VarName, ModelType, ...)       \
  REGISTER_MODEL_ARGS_LOADER_WITH_VARNAME(                              \
      VarName, ModelType, [](const JsonReader& json, ModelArgs* args) { \
        UNUSED_PARAMETER(json);                                         \
        UNUSED_PARAMETER(args);                                         \
        __VA_ARGS__();                                                  \
        return true;                                                    \
      })

#define REGISTER_MODEL_ARGS(ModelType, ...) \
  REGISTER_MODEL_ARGS_WITH_VARNAME(ModelType, ModelType, __VA_ARGS__)

// Macro to register a quantization args loader with the ModelRegistry
#define REGISTER_QUANT_ARGS_LOADER_WITH_VARNAME(VarName, ModelType, Loader) \
  const bool VarName##_quant_args_loader_registered = []() {                \
    ModelRegistry::register_quant_args_loader(#ModelType, Loader);          \
    return true;                                                            \
  }()

#define REGISTER_QUANT_ARGS_LOADER(ModelType, Loader) \
  REGISTER_QUANT_ARGS_LOADER_WITH_VARNAME(ModelType, ModelType, Loader)

// Macro to register a tokenizer args loader with the ModelRegistry
#define REGISTER_TOKENIZER_ARGS_LOADER_WITH_VARNAME(                   \
    VarName, ModelType, Loader)                                        \
  const bool VarName##_tokenizer_args_loader_registered = []() {       \
    ModelRegistry::register_tokenizer_args_loader(#ModelType, Loader); \
    return true;                                                       \
  }()

#define REGISTER_TOKENIZER_ARGS_LOADER(ModelType, Loader) \
  REGISTER_TOKENIZER_ARGS_LOADER_WITH_VARNAME(ModelType, ModelType, Loader)

#define REGISTER_TOKENIZER_ARGS_WITH_VARNAME(VarName, ModelType, ...)       \
  REGISTER_TOKENIZER_ARGS_LOADER_WITH_VARNAME(                              \
      VarName, ModelType, [](const JsonReader& json, TokenizerArgs* args) { \
        UNUSED_PARAMETER(json);                                             \
        UNUSED_PARAMETER(args);                                             \
        __VA_ARGS__();                                                      \
        return true;                                                        \
      })

#define REGISTER_TOKENIZER_ARGS(ModelType, ...) \
  REGISTER_TOKENIZER_ARGS_WITH_VARNAME(ModelType, ModelType, __VA_ARGS__)

#define LOAD_ARG(arg_name, json_name)                          \
  [&] {                                                        \
    auto value = args->arg_name();                             \
    using value_type = remove_optional_t<decltype(value)>;     \
    if (auto data_value = json.value<value_type>(json_name)) { \
      args->arg_name() = data_value.value();                   \
    }                                                          \
  }()

#define LOAD_ARG_OR(arg_name, json_name, default_value)                     \
  [&] {                                                                     \
    auto value = args->arg_name();                                          \
    using value_type = remove_optional_t<decltype(value)>;                  \
    args->arg_name() = json.value_or<value_type>(json_name, default_value); \
  }()

#define LOAD_ARG_OR_FUNC(arg_name, json_name, ...)             \
  [&] {                                                        \
    auto value = args->arg_name();                             \
    using value_type = remove_optional_t<decltype(value)>;     \
    if (auto data_value = json.value<value_type>(json_name)) { \
      args->arg_name() = data_value.value();                   \
    } else {                                                   \
      args->arg_name() = __VA_ARGS__();                        \
    }                                                          \
  }()

#define SET_ARG(arg_name, value) [&] { args->arg_name() = value; }()

}  // namespace xllm
