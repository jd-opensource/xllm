/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "operations/fusion/linear/linear.h"
#include "operations/fusion/linear/linear_parallel.h"
#include "operations/fusion/norm/norm_linear.h"
#include "operations/fusion/attention/fusion_attention.h"
#include "operations/fusion/mlp/mlp.h"
#include "models/qwen2/layer/decoder_layer.h"

namespace atb_speed {
namespace qwen {

std::map<std::string, std::vector<std::string>> GetQwenLayerInTensorCandidates()
{
    std::map<std::string, std::vector<std::string>> qwenLayerInTensorCandiadates = {
        {"default_weight", {
            // shape: [hiddenSize]
            "in_input_norm_weight", "in_input_norm_bias", "in_input_norm_new_weight", "in_input_norm_new_bias",
            // Pack:
            // MHA [3 * numAttentionHeadsPerRank * hiddenSizePerAttentionHead, hiddenSize]
            // GQA [(numAttentionHeadsPerRank + 2 * numKeyValueHeadsPerRank) * hiddenSizePerAttentionHead, hiddenSize]
            // No pack:
            // (Q) shape: [numAttentionHeadsPerRank * hiddenSizePerAttentionHead, hiddenSize]
            "in_qkv_weight_0", "in_qkv_bias_0", "in_qkv_descale_0", "in_qkv_offset_0", "in_qkv_scale_0",
            "in_qkv_compress_idx_0",
            // Pack: no usage; No pack: (K) shape: [numKeyValueHeadsPerRank * hiddenSizePerAttentionHead, hiddenSize]
            "in_qkv_weight_1", "in_qkv_bias_1", "in_qkv_descale_1", "in_qkv_offset_1", "in_qkv_scale_1",
            "in_qkv_compress_idx_1",
            // Pack: no usage; No pack: (V) shape: [numKeyValueHeadsPerRank * hiddenSizePerAttentionHead, hiddenSize]
            "in_qkv_weight_2", "in_qkv_bias_2", "in_qkv_descale_2", "in_qkv_offset_2", "in_qkv_scale_2",
            "in_qkv_compress_idx_2",
            // shape: [hiddenSize, numAttentionHeadsPerRank * hiddenSizePerAttentionHead]
            "in_qkv_dense_weight", "in_qkv_dense_bias", "in_qkv_dense_descale", "in_qkv_dense_offset",
            "in_qkv_dense_scale", "in_qkv_dense_compress_idx",
            // shape: [hiddenSize]
            "in_post_attn_norm_weight", "in_post_attn_norm_bias", "in_post_attn_norm_new_weight",
            "in_post_attn_norm_new_bias",
            // Pack: shape: [2 * intermediateSizePerRank, hiddenSize]
            // No pack: (Gate) shape: [intermediateSizePerRank, hiddenSize]
            "in_mlp_weight_0", "in_mlp_bias_0", "in_mlp_descale_0", "in_mlp_offset_0", "in_mlp_scale_0",
            "in_mlp_compress_idx_0",
            // Pack: no usage; No pack: (Up) shape: [intermediateSizePerRank, hiddenSize]
            "in_mlp_weight_1", "in_mlp_bias_1", "in_mlp_descale_1", "in_mlp_offset_1", "in_mlp_scale_1",
            "in_mlp_compress_idx_1",
            // shape: [hiddenSize, intermediateSizePerRank]
            "in_mlp_down_weight", "in_mlp_down_bias", "in_mlp_down_descale", "in_mlp_down_offset",
            "in_mlp_down_scale", "in_mlp_down_compress_idx"}},
        {"kv_quant", {
            "in_k_quant_scale", "in_k_dequant_scale", "in_v_quant_scale", "in_v_dequant_scale",
            "in_k_quant_offset", "in_k_dequant_offset", "in_v_quant_offset", "in_v_dequant_offset"}},
        {"default", {
            "in_hidden_states",  // shape: FA: [batchSize, seqLen, hiddenSize] PA: [seqLen, hiddenSize]
            "in_cos_embedding", "in_sin_embedding", "in_attention_mask", "in_k_cache", "in_v_cache", "in_seq_len",
            "in_token_offset", "in_layer_id", "in_block_tables", "in_slots"}},
        {"q_len", {"in_q_len"}},
        {"logn_enable", {"kv_cache_idx"}},
        {"lora_common", {"in_seq_len_cum_sum"}},
        {"lora_attn", {
            "in_qkv_lora_a_0", "in_qkv_lora_b_0", "in_qkv_lora_a_1", "in_qkv_lora_b_1",
            "in_qkv_lora_a_2", "in_qkv_lora_b_2", "in_qkv_dense_lora_a", "in_qkv_dense_lora_b"}
        },
        {"lora_mlp", {
            "in_mlp_lora_a_0", "in_mlp_lora_b_0", "in_mlp_lora_a_1", "in_mlp_lora_b_1",
            "in_mlp_down_lora_a", "in_mlp_down_lora_b"}
        }
    };
    return qwenLayerInTensorCandiadates;
}

std::map<std::string, std::vector<std::string>> GetQwenLayerIntermediateTensorCandidates()
{
    std::map<std::string, std::vector<std::string>> qwenLayerIntermediateTensorCandiadates = {
        {"default", {"intermediate_attn_out", "intermediate_mlp_out"}},
    };
    return qwenLayerIntermediateTensorCandiadates;
}

std::map<std::string, uint32_t> ConstructTensorMap(
    const DecoderLayerParam &param,
    uint32_t &inTensorNum, uint32_t &outTensorNum, uint32_t &internalTensorNum)
{
    auto qwenLayerInTensorCandiadates = GetQwenLayerInTensorCandidates();
    auto qwenLayerIntermediateTensorCandiadates = GetQwenLayerIntermediateTensorCandidates();

    std::vector<std::string> inTensorList = {};
    std::vector<std::string> intermediateTensorList = {};
    std::vector<std::string> outTensorList = {"out"};

    // 添加默认的Tensor
    atb_speed::common::AddTensorToList(qwenLayerInTensorCandiadates, "default_weight", inTensorList);

    // 添加KV cache int8特性的Tensor
    if (param.kvQuant) {
        atb_speed::common::AddTensorToList(qwenLayerInTensorCandiadates, "kv_quant", inTensorList);
    }

    atb_speed::common::AddTensorToList(qwenLayerInTensorCandiadates, "default", inTensorList);
    atb_speed::common::AddTensorToList(
        qwenLayerIntermediateTensorCandiadates, "default", intermediateTensorList);

    // 添加并行解码特性或SplitFuse的Tensor
    if (param.supportSpeculate || param.enableSplitFuse) {
        atb_speed::common::AddTensorToList(qwenLayerInTensorCandiadates, "q_len", inTensorList);
    }

    // 添加lora特性的Tensor
    if (param.supportLora) {
        atb_speed::common::AddTensorToList(qwenLayerInTensorCandiadates, "lora_common", inTensorList);
        atb_speed::common::AddTensorToList(qwenLayerInTensorCandiadates, "lora_attn", inTensorList);
        atb_speed::common::AddTensorToList(qwenLayerInTensorCandiadates, "lora_mlp", inTensorList);
    }

    // 添加logn特性的Tensor
    if (param.enableLogN) {
        atb_speed::common::AddTensorToList(qwenLayerInTensorCandiadates, "logn_enable", inTensorList);
    }

    inTensorNum = inTensorList.size();
    outTensorNum = outTensorList.size();
    internalTensorNum = intermediateTensorList.size();

    return atb_speed::common::GetTensorMap(inTensorList, outTensorList, intermediateTensorList);
}

void SetFusionAttentionParamPart(
    atb_speed::common::FusionAttentionParam<atb::infer::RmsNormParam> &fusionAttentionParam,
    const DecoderLayerParam &param
)
{
    fusionAttentionParam.isGroupedQueryAttention = param.numAttentionHeadsPerRank != param.numKeyValueHeadsPerRank;
    fusionAttentionParam.isBF16 = param.isBF16;
    fusionAttentionParam.qkvHasBias = true;
    fusionAttentionParam.layerLinearQuantType = param.linearQuantType;
    fusionAttentionParam.layerLinearTransposeType = param.linearTransposeType;
    fusionAttentionParam.packQuantType = param.packQuantType.at(0);
    fusionAttentionParam.quantGroupSize = param.quantGroupSize;
    // TODO:lcoc fail to run when enble tp
    fusionAttentionParam.supportLcoc = false;//param.supportLcoc;
    fusionAttentionParam.supportLora = param.supportLora;
    fusionAttentionParam.loraEnableGMM = param.loraEnableGMM;
    atb::infer::RmsNormParam attenRmsNormParam;
    attenRmsNormParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
    attenRmsNormParam.normParam.epsilon = param.rmsNormEps;
    fusionAttentionParam.normParamType = attenRmsNormParam;
    atb::infer::RmsNormParam attenRmsNormQuantParam;
    attenRmsNormQuantParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
    attenRmsNormQuantParam.normParam.epsilon = param.rmsNormEps;
    attenRmsNormQuantParam.normParam.quantType = atb::infer::QUANT_INT8;
    fusionAttentionParam.normQuantParamType = attenRmsNormQuantParam;
    // rope param
    fusionAttentionParam.rotaryType = atb_speed::common::RotaryType::ALL_ROTARY;
    fusionAttentionParam.ropeParam.rotaryCoeff = 2;  // 2:旋转系数
    if (param.enableLogN) {
        fusionAttentionParam.pageAttentionParam.scaleType = atb::infer::PagedAttentionParam::SCALE_TYPE_LOGN;
    }
}

void SetFusionAttentionParam(
    atb_speed::common::FusionAttentionParam<atb::infer::RmsNormParam> &fusionAttentionParam,
    const DecoderLayerParam &param
)
{
    SetFusionAttentionParamPart(fusionAttentionParam, param);
    fusionAttentionParam.isFA = param.isFA;
    fusionAttentionParam.isPrefill = param.isPrefill;
    fusionAttentionParam.enableSplitFuse = param.enableSplitFuse;
    fusionAttentionParam.headDim = param.hiddenSizePerAttentionHead;
    fusionAttentionParam.selfAttentionParam.headNum = param.numAttentionHeadsPerRank;
    fusionAttentionParam.selfAttentionParam.kvHeadNum = param.numKeyValueHeadsPerRank;
    if (param.hiddenSizePerAttentionHead == 0) {
        std::stringstream ss;
        ss << "Cannot be devided by zero. Param hiddenSizePerAttentionHead is zero!" << std::endl;
        throw std::runtime_error(ss.str());
    }
    fusionAttentionParam.selfAttentionParam.qkScale = 1.0 / sqrt(param.hiddenSizePerAttentionHead);
    fusionAttentionParam.selfAttentionParam.isTriuMask = param.isPrefill ? 1 : 0;
    if (param.isFA) {
        fusionAttentionParam.selfAttentionParam.calcType = param.isPrefill ?
            atb::infer::SelfAttentionParam::CalcType::ENCODER :
            atb::infer::SelfAttentionParam::CalcType::DECODER;
    } else {
        fusionAttentionParam.selfAttentionParam.calcType = atb::infer::SelfAttentionParam::CalcType::PA_ENCODER;
    }
    fusionAttentionParam.selfAttentionParam.maskType = atb::infer::SelfAttentionParam::MaskType::MASK_TYPE_NORM;
    fusionAttentionParam.pageAttentionParam.headNum = param.numAttentionHeadsPerRank;
    fusionAttentionParam.pageAttentionParam.kvHeadNum = param.numKeyValueHeadsPerRank;
    fusionAttentionParam.pageAttentionParam.qkScale = 1.0 / sqrt(param.hiddenSizePerAttentionHead);
    if (param.supportSpeculate) {
        fusionAttentionParam.pageAttentionParam.calcType = atb::infer::PagedAttentionParam::CalcType::CALC_TYPE_SPEC;
        fusionAttentionParam.pageAttentionParam.maskType = atb::infer::PagedAttentionParam::MaskType::MASK_TYPE_SPEC;
    }
    if (param.enableSplitFuse) {
        fusionAttentionParam.pageAttentionParam.calcType = \
            atb::infer::PagedAttentionParam::CalcType::CALC_TYPE_SPEC;
        fusionAttentionParam.pageAttentionParam.maskType = \
            atb::infer::PagedAttentionParam::MaskType::MASK_TYPE_NORM;
    }
    fusionAttentionParam.selfOutLinearTensorParallelInfo = { param.rank, param.worldSize, param.backend };
    if (param.kvQuant) {
        fusionAttentionParam.pageAttentionParam.quantType = atb::infer::PagedAttentionParam::TYPE_DEQUANT_FUSION;
        fusionAttentionParam.pageAttentionParam.maskType = atb::infer::PagedAttentionParam::UNDEFINED;
        fusionAttentionParam.pageAttentionParam.hasQuantOffset  = true;
    }
}

int64_t AddFusionAttention(atb::Node &attentionNode, const DecoderLayerParam &param,
    std::map<std::string, uint32_t> &tensorMap)
{
    // attention
    atb_speed::common::FusionAttentionParam<atb::infer::RmsNormParam> fusionAttentionParam;
    SetFusionAttentionParam(fusionAttentionParam, param);
    CHECK_OPERATION_STATUS_RETURN(Attention(fusionAttentionParam, &attentionNode.operation));
    std::vector<std::string> attnInTensorNames = {
        "in_hidden_states", "in_input_norm_weight", "in_input_norm_bias", "in_input_norm_new_weight",
        "in_input_norm_new_bias",
        "in_qkv_weight_0", "in_qkv_scale_0", "in_qkv_offset_0", "in_qkv_descale_0", "in_qkv_bias_0",
        "in_qkv_compress_idx_0",
        "in_qkv_weight_1", "in_qkv_scale_1", "in_qkv_offset_1", "in_qkv_descale_1", "in_qkv_bias_1",
        "in_qkv_compress_idx_1",
        "in_qkv_weight_2", "in_qkv_scale_2", "in_qkv_offset_2", "in_qkv_descale_2", "in_qkv_bias_2",
        "in_qkv_compress_idx_2",
        "in_cos_embedding", "in_sin_embedding", "in_seq_len", "in_k_cache", "in_v_cache",
        "in_attention_mask",
        "in_token_offset", "in_layer_id", "in_block_tables", "in_slots",
        "in_qkv_dense_weight", "in_qkv_dense_scale", "in_qkv_dense_offset", "in_qkv_dense_descale",
        "in_qkv_dense_bias", "in_qkv_dense_compress_idx"
    };
    if (param.supportSpeculate || param.enableSplitFuse) {
        attnInTensorNames.push_back("in_q_len");
    }
    auto qwenLayerInTensorCandiadates = GetQwenLayerInTensorCandidates();
    if (param.kvQuant) {
        for (std::string tensor : qwenLayerInTensorCandiadates.at("kv_quant")) {
            attnInTensorNames.push_back(tensor);
        }
    }
    if (param.supportLora) {
        attnInTensorNames.push_back("in_seq_len_cum_sum");
        for (std::string tensor : qwenLayerInTensorCandiadates.at("lora_attn")) {
            attnInTensorNames.push_back(tensor);
        }
    }

    if (param.enableLogN) {
        attnInTensorNames.push_back("kv_cache_idx");
    }
    attentionNode.inTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, attnInTensorNames);
    attentionNode.outTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_attn_out"});

    return atb::NO_ERROR;
}

void SetMlpParam(atb_speed::common::MlpParam<atb::infer::RmsNormParam> &mlpParam, const DecoderLayerParam &param)
{
    mlpParam.isBF16 = param.isBF16;
    mlpParam.layerLinearQuantType = param.linearQuantType;
    mlpParam.layerLinearTransposeType = param.linearTransposeType;
    mlpParam.packQuantType = param.packQuantType.at(1);
    mlpParam.quantGroupSize = param.quantGroupSize;
    // w2_w1(gate_up)
    mlpParam.mlpPackType = atb_speed::common::GATE_UP_WEIGHT_PACK;
    atb::infer::RmsNormParam mlpRmsNormParam;
    mlpRmsNormParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
    mlpRmsNormParam.normParam.epsilon = param.rmsNormEps;
    mlpParam.normParamType = mlpRmsNormParam;
    atb::infer::RmsNormParam mlpRmsNormQuantParam;
    mlpRmsNormQuantParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
    mlpRmsNormQuantParam.normParam.epsilon = param.rmsNormEps;
    mlpRmsNormQuantParam.normParam.quantType = atb::infer::QUANT_INT8;
    mlpParam.normQuantParamType = mlpRmsNormQuantParam;
    mlpParam.supportLora = param.supportLora;
    mlpParam.loraEnableGMM = param.loraEnableGMM;
    // c_proj(down)
    mlpParam.downLinearTensorParallelInfo = {param.rank, param.worldSize, param.backend};
    mlpParam.supportLcoc = param.supportLcoc;
    if (param.supportSwiGLU) {
        mlpParam.activationParam.activationType = atb::infer::ActivationType::ACTIVATION_SWIGLU_FORWARD;
        mlpParam.activationParam.dim = -1;
    } else {
        mlpParam.activationParam.activationType = atb::infer::ActivationType::ACTIVATION_SWISH;
    }
}

int64_t AddMlp(atb::Node &mlpParallelNode, const DecoderLayerParam &param,
    std::map<std::string, uint32_t> &tensorMap)
{
    atb_speed::common::MlpParam<atb::infer::RmsNormParam> mlpParam;
    SetMlpParam(mlpParam, param);
    if (param.supportSwiGLU) {
        CHECK_OPERATION_STATUS_RETURN(MlpSwiGLU(mlpParam, &mlpParallelNode.operation));
    } else {
        CHECK_OPERATION_STATUS_RETURN(Mlp(mlpParam, &mlpParallelNode.operation));
    }
    std::vector<std::string> mlpInTensorNames = {
        "in_hidden_states", "in_post_attn_norm_weight", "in_post_attn_norm_bias", "in_post_attn_norm_new_weight",
        "in_post_attn_norm_new_bias", "in_mlp_weight_0", "in_mlp_scale_0", "in_mlp_offset_0", "in_mlp_descale_0",
        "in_mlp_bias_0", "in_mlp_compress_idx_0", "in_mlp_weight_1", "in_mlp_scale_1", "in_mlp_offset_1",
        "in_mlp_descale_1", "in_mlp_bias_1", "in_mlp_compress_idx_1", "in_mlp_down_weight", "in_mlp_down_scale",
        "in_mlp_down_offset", "in_mlp_down_descale", "in_mlp_down_bias", "in_mlp_down_compress_idx"
    };
    auto qwenLayerInTensorCandiadates = GetQwenLayerInTensorCandidates();
    if (param.supportLora) {
        mlpInTensorNames.push_back("in_seq_len_cum_sum");
        for (std::string tensor : qwenLayerInTensorCandiadates.at("lora_mlp")) {
            mlpInTensorNames.push_back(tensor);
        }
    }

    std::vector<std::string> mlpOutTensorName = {"intermediate_mlp_out"};

    mlpParallelNode.inTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, mlpInTensorNames);
    mlpParallelNode.outTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, mlpOutTensorName);
    return atb::NO_ERROR;
}

atb::Status DecoderLayer(const DecoderLayerParam &param, atb::Operation **operation)
{
    atb::GraphParam opGraph;
    opGraph.name = param.isPrefill ? "Prefill_layer" : "Decoder_layer";
    std::map<std::string, uint32_t> tensorMap = ConstructTensorMap(
        param, opGraph.inTensorNum, opGraph.outTensorNum, opGraph.internalTensorNum);

    atb::Node attentionNode;
    atb::Node selfResidualAddNode;
    atb::Node mlpParallelNode;
    atb::Node mlpResidualAddNode;

    CHECK_OPERATION_STATUS_RETURN(AddFusionAttention(attentionNode, param, tensorMap));
    opGraph.nodes.push_back(attentionNode);
    // residual
    atb::infer::ElewiseParam addParam;
    addParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(addParam, &selfResidualAddNode.operation));
    selfResidualAddNode.inTensorIds = \
        atb_speed::common::GetTensorIdxList(tensorMap, {"in_hidden_states", "intermediate_attn_out"});
    selfResidualAddNode.outTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, {"in_hidden_states"});
    opGraph.nodes.push_back(selfResidualAddNode);
    CHECK_OPERATION_STATUS_RETURN(AddMlp(mlpParallelNode, param, tensorMap));
    opGraph.nodes.push_back(mlpParallelNode);
    // residual
    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(addParam, &mlpResidualAddNode.operation));
    mlpResidualAddNode.inTensorIds = \
        atb_speed::common::GetTensorIdxList(tensorMap, {"in_hidden_states", "intermediate_mlp_out"});
    mlpResidualAddNode.outTensorIds = atb_speed::common::GetTensorIdxList(tensorMap, {"out"});
    opGraph.nodes.push_back(mlpResidualAddNode);

    opGraph.inferShapeFunc = [=](const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                 atb::SVector<atb::TensorDesc> &outTensorDescs) {
        outTensorDescs.at(0) = inTensorDescs.at(0);
        return atb::NO_ERROR;
    };

    CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(opGraph, operation));
    return atb::NO_ERROR;
}

}  // namespace qwen
}  // namespace atb_speed