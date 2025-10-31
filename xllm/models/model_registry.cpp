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

#include "model_registry.h"

#include <glog/logging.h>

#include <iostream>

#include "models.h"
#include "rec/onerec.h"

namespace {

// Safe logging macro to avoid crashes during static initialization
#define SAFE_LOG_WARNING(message)                       \
  do {                                                  \
    if (google::IsGoogleLoggingInitialized()) {         \
      LOG(WARNING) << message;                          \
    } else {                                            \
      std::cerr << "WARNING: " << message << std::endl; \
    }                                                   \
  } while (0)

#define SAFE_LOG_ERROR(message)                       \
  do {                                                \
    if (google::IsGoogleLoggingInitialized()) {       \
      LOG(ERROR) << message;                          \
    } else {                                          \
      std::cerr << "ERROR: " << message << std::endl; \
    }                                                 \
  } while (0)

#define SAFE_LOG_INFO(message)                       \
  do {                                               \
    if (google::IsGoogleLoggingInitialized()) {      \
      LOG(INFO) << message;                          \
    } else {                                         \
      std::cerr << "INFO: " << message << std::endl; \
    }                                                \
  } while (0)

}  // anonymous namespace

namespace xllm {

ModelRegistry* ModelRegistry::get_instance() {
  static ModelRegistry registry;

  return &registry;
}

void ModelRegistry::register_causallm_factory(const std::string& name,
                                              CausalLMFactory factory) {
  ModelRegistry* instance = get_instance();

  if (instance->model_registry_[name].causal_lm_factory != nullptr) {
    SAFE_LOG_WARNING("causal lm factory for " << name
                                              << " already registered.");
  } else {
    instance->model_registry_[name].causal_lm_factory = factory;
  }
}

void ModelRegistry::register_causalvlm_factory(const std::string& name,
                                               CausalVLMFactory factory) {
  ModelRegistry* instance = get_instance();

  if (instance->model_registry_[name].causal_vlm_factory != nullptr) {
    SAFE_LOG_WARNING("causal vlm factory for " << name
                                               << " already registered.");
  } else {
    instance->model_registry_[name].causal_vlm_factory = factory;
  }
}

void ModelRegistry::register_embeddinglm_factory(const std::string& name,
                                                 EmbeddingLMFactory factory) {
  ModelRegistry* instance = get_instance();

  if (instance->model_registry_[name].embedding_lm_factory != nullptr) {
    SAFE_LOG_WARNING("embedding lm factory for " << name
                                                 << " already registered.");
  } else {
    instance->model_registry_[name].embedding_lm_factory = factory;
  }
}

void ModelRegistry::register_dit_model_factory(const std::string& name,
                                               DiTModelFactory factory) {
  ModelRegistry* instance = get_instance();

  if (instance->model_registry_[name].dit_model_factory != nullptr) {
    SAFE_LOG_WARNING("DiT model factory for " << name
                                              << " already registered.");
  } else {
    instance->model_registry_[name].dit_model_factory = factory;
  }
}

void ModelRegistry::register_input_processor_factory(
    const std::string& name,
    InputProcessorFactory factory) {
  ModelRegistry* instance = get_instance();

  if (instance->model_registry_[name].input_processor_factory != nullptr) {
    SAFE_LOG_WARNING("input processor factory for " << name
                                                    << " already registered.");
  } else {
    instance->model_registry_[name].input_processor_factory = factory;
  }
}

void ModelRegistry::register_image_processor_factory(
    const std::string& name,
    ImageProcessorFactory factory) {
  ModelRegistry* instance = get_instance();

  if (instance->model_registry_[name].image_processor_factory != nullptr) {
    SAFE_LOG_WARNING("image processor factory for " << name
                                                    << " already registered.");
  } else {
    instance->model_registry_[name].image_processor_factory = factory;
  }
}

void ModelRegistry::register_model_args_loader(const std::string& name,
                                               ModelArgsLoader loader) {
  ModelRegistry* instance = get_instance();

  if (instance->model_registry_[name].model_args_loader != nullptr) {
    SAFE_LOG_WARNING("model args loader for " << name
                                              << " already registered.");
  } else {
    instance->model_registry_[name].model_args_loader = loader;
  }
}

void ModelRegistry::register_quant_args_loader(const std::string& name,
                                               QuantArgsLoader loader) {
  ModelRegistry* instance = get_instance();

  if (instance->model_registry_[name].quant_args_loader != nullptr) {
    SAFE_LOG_WARNING("quant args loader for " << name
                                              << " already registered.");
  } else {
    instance->model_registry_[name].quant_args_loader = loader;
  }
}

void ModelRegistry::register_tokenizer_args_loader(const std::string& name,
                                                   TokenizerArgsLoader loader) {
  ModelRegistry* instance = get_instance();

  if (instance->model_registry_[name].tokenizer_args_loader != nullptr) {
    SAFE_LOG_WARNING("tokenizer args loader for " << name
                                                  << " already registered.");
  } else {
    instance->model_registry_[name].tokenizer_args_loader = loader;
  }
}

CausalLMFactory ModelRegistry::get_causallm_factory(const std::string& name) {
  ModelRegistry* instance = get_instance();

  return instance->model_registry_[name].causal_lm_factory;
}

CausalVLMFactory ModelRegistry::get_causalvlm_factory(const std::string& name) {
  ModelRegistry* instance = get_instance();

  return instance->model_registry_[name].causal_vlm_factory;
}

EmbeddingLMFactory ModelRegistry::get_embeddinglm_factory(
    const std::string& name) {
  ModelRegistry* instance = get_instance();

  return instance->model_registry_[name].embedding_lm_factory;
}

DiTModelFactory ModelRegistry::get_dit_model_factory(const std::string& name) {
  ModelRegistry* instance = get_instance();
  return instance->model_registry_[name].dit_model_factory;
}

InputProcessorFactory ModelRegistry::get_input_processor_factory(
    const std::string& name) {
  ModelRegistry* instance = get_instance();

  return instance->model_registry_[name].input_processor_factory;
}

ImageProcessorFactory ModelRegistry::get_image_processor_factory(
    const std::string& name) {
  ModelRegistry* instance = get_instance();

  return instance->model_registry_[name].image_processor_factory;
}

ModelArgsLoader ModelRegistry::get_model_args_loader(const std::string& name) {
  ModelRegistry* instance = get_instance();

  return instance->model_registry_[name].model_args_loader;
}

QuantArgsLoader ModelRegistry::get_quant_args_loader(const std::string& name) {
  ModelRegistry* instance = get_instance();

  return instance->model_registry_[name].quant_args_loader;
}

TokenizerArgsLoader ModelRegistry::get_tokenizer_args_loader(
    const std::string& name) {
  ModelRegistry* instance = get_instance();

  return instance->model_registry_[name].tokenizer_args_loader;
}

std::unique_ptr<CausalLM> create_llm_model(const ModelContext& context) {
  // get the factory function for the model type from model registry
  auto factory = ModelRegistry::get_causallm_factory(
      context.get_model_args().model_type());
  if (factory) {
    return factory(context);
  }

  LOG(ERROR) << "Unsupported model type: "
             << context.get_model_args().model_type();

  return nullptr;
}

std::unique_ptr<CausalVLM> create_vlm_model(const ModelContext& context) {
  // get the factory function for the model type from model registry
  auto factory = ModelRegistry::get_causalvlm_factory(
      context.get_model_args().model_type());
  if (factory) {
    return factory(context);
  }

  LOG(ERROR) << "Unsupported model type: "
             << context.get_model_args().model_type();

  return nullptr;
}

std::unique_ptr<EmbeddingLM> create_embeddinglm_model(
    const ModelContext& context) {
  // get the factory function for the model type from model registry
  auto factory = ModelRegistry::get_embeddinglm_factory(
      context.get_model_args().model_type());
  if (factory) {
    return factory(context);
  }

  LOG(ERROR) << "Unsupported model type: "
             << context.get_model_args().model_type();

  return nullptr;
}

std::unique_ptr<DiTModel> create_dit_model(const DiTModelContext& context) {
  // get the factory function for the model type from model registry
  auto factory = ModelRegistry::get_dit_model_factory(context.model_type());
  if (factory) {
    return factory(context);
  }
  LOG(INFO) << "DiT Model type: " << context.model_type();
  LOG(ERROR) << "Unsupported model type: " << context.model_type();

  return nullptr;
}

}  // namespace xllm
