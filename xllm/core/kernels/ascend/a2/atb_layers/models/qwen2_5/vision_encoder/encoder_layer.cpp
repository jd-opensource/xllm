
#include "models/qwen2_5/operation/base_operations.h"

namespace atb_speed {
namespace qwen {

std::map<std::string, std::vector<std::string>>
GetQwen2_5LayerTensorCandidates() {
  std::map<std::string, std::vector<std::string>> qwenLayerTensorCandiadates = {
      {"default_weight",
       {"in_input_norm_weight",
        "in_post_attn_norm_weight",
        "in_qkv_weight",
        "in_qkv_bias",
        "in_attn_proj_weight",
        "in_attn_proj_bias",
        "in_mlp_gate_weight",
        "in_mlp_gate_bias",
        "in_mlp_up_weight",
        "in_mlp_up_bias",
        "in_mlp_down_weight",
        "in_mlp_down_bias",
        "in_q_weight",
        "in_q_bias",
        "in_k_weight",
        "in_k_bias",
        "in_v_weight",
        "in_v_bias"}},
      {"default_input",
       {
           "in_hidden_states",
           "in_cos_embedding",
           "in_sin_embedding",
           "in_seq_len",
       }},
      {"out", {"out"}},
      {"parallel_intermediate_out", {"proj_add_bias_out", "down_add_bias_out"}},
      {"intermediate_out",
       {"input_norm_out",
        "qkv_linear_out",
        "intermediate_q",
        "intermediate_k",
        "intermediate_v",
        "intermediate_rope_k",
        "intermediate_rope_q",
        "intermediate_atten_out",
        "intermediate_proj_out",
        "intermediate_add_out",
        "intermediate_post_atten_norm_out",
        "intermediate_gateup_out",
        "intermediate_silu_out",
        "intermediate_down_out"}},
      {"q_len", {"in_q_len"}}};
  return qwenLayerTensorCandiadates;
}

std::map<std::string, uint32_t> ConstructTensorMap(
    const VisionEncoderLayerParam& param,
    uint32_t& inTensorNum,
    uint32_t& outTensorNum,
    uint32_t& internalTensorNum) {
  auto qwenLayerTensorCandiadates = GetQwen2_5LayerTensorCandidates();
  std::vector<std::string> inTensorList = {};
  std::vector<std::string> intermediateTensorList = {};
  std::vector<std::string> outTensorList = {};
  // weight
  atb_speed::common::AddTensorToList(
      qwenLayerTensorCandiadates, "default_weight", inTensorList);
  // input
  atb_speed::common::AddTensorToList(
      qwenLayerTensorCandiadates, "default_input", inTensorList);
  // out
  atb_speed::common::AddTensorToList(
      qwenLayerTensorCandiadates, "out", outTensorList);
  // intermediate
  atb_speed::common::AddTensorToList(
      qwenLayerTensorCandiadates, "intermediate_out", intermediateTensorList);
  if (param.worldSize > 1) {
    atb_speed::common::AddTensorToList(qwenLayerTensorCandiadates,
                                       "parallel_intermediate_out",
                                       intermediateTensorList);
  }
  inTensorNum = inTensorList.size();
  outTensorNum = outTensorList.size();
  internalTensorNum = intermediateTensorList.size();

  return atb_speed::common::GetTensorMap(
      inTensorList, outTensorList, intermediateTensorList);
}

atb::Status BuildAttentionBlock(
    const VisionEncoderLayerParam& param,
    const std::map<std::string, uint32_t>& tensorMap,
    atb::GraphParam& opGraph,
    bool isTp) {
  atb::Node inputNormNode;
  atb::Node qkvlinearNode;
  atb::Node splitNode;
  atb::Node ropeNode;
  atb::Node selfAttentionNode;
  atb::Node outProjNode;
  atb::Node selfResidualAddNode;
  atb::Node proj_add_biasNode;

  // Input Norm
  CreateNormOperation(param,&inputNormNode.operation);
  inputNormNode.inTensorIds = atb_speed::common::GetTensorIdxList(
      tensorMap, {"in_hidden_states", "in_input_norm_weight"});
  inputNormNode.outTensorIds =
      atb_speed::common::GetTensorIdxList(tensorMap, {"input_norm_out"});
  opGraph.nodes.push_back(inputNormNode);

  // QKV Linear
  CreateLinearOperation(&qkvlinearNode.operation);
  qkvlinearNode.inTensorIds = atb_speed::common::GetTensorIdxList(
      tensorMap, {"input_norm_out", "in_qkv_weight", "in_qkv_bias"});
  qkvlinearNode.outTensorIds =
      atb_speed::common::GetTensorIdxList(tensorMap, {"qkv_linear_out"});
  opGraph.nodes.push_back(qkvlinearNode);

  // Split
  CreateSplitOperation(&splitNode.operation);
  splitNode.inTensorIds =
      atb_speed::common::GetTensorIdxList(tensorMap, {"qkv_linear_out"});
  splitNode.outTensorIds = atb_speed::common::GetTensorIdxList(
      tensorMap, {"intermediate_q", "intermediate_k", "intermediate_v"});
  opGraph.nodes.push_back(splitNode);

  // Rope
  CreateRopeOperation(&ropeNode.operation);
  ropeNode.inTensorIds =
      atb_speed::common::GetTensorIdxList(tensorMap,
                                          {"intermediate_q",
                                           "intermediate_k",
                                           "in_cos_embedding",
                                           "in_sin_embedding",
                                           "in_seq_len"});
  ropeNode.outTensorIds = atb_speed::common::GetTensorIdxList(
      tensorMap, {"intermediate_rope_q", "intermediate_rope_k"});
  opGraph.nodes.push_back(ropeNode);

  // Self Attention
  CreateSelfAttentionOperation(param, &selfAttentionNode.operation);
  selfAttentionNode.inTensorIds =
      atb_speed::common::GetTensorIdxList(tensorMap,
                                          {"intermediate_rope_q",
                                           "intermediate_rope_k",
                                           "intermediate_v",
                                           "in_seq_len"});
  selfAttentionNode.outTensorIds = atb_speed::common::GetTensorIdxList(
      tensorMap, {"intermediate_atten_out"});
  opGraph.nodes.push_back(selfAttentionNode);

  // Output Projection
  if (isTp) {
    CreateLinearParallelOperation(param, &outProjNode.operation,false);
    outProjNode.inTensorIds = atb_speed::common::GetTensorIdxList(
        tensorMap, {"intermediate_atten_out", "in_attn_proj_weight"});
  } else {
    CreateLinearOperation(&outProjNode.operation);
    outProjNode.inTensorIds = atb_speed::common::GetTensorIdxList(
        tensorMap,
        {"intermediate_atten_out", "in_attn_proj_weight", "in_attn_proj_bias"});
  }
  outProjNode.outTensorIds =
      atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_proj_out"});
  opGraph.nodes.push_back(outProjNode);

  // Tp Bias Add
  if (isTp) {
    CreateResidualAddOperation(&proj_add_biasNode.operation);
    proj_add_biasNode.inTensorIds = atb_speed::common::GetTensorIdxList(
        tensorMap, {"intermediate_proj_out", "in_attn_proj_bias"});
    proj_add_biasNode.outTensorIds =
        atb_speed::common::GetTensorIdxList(tensorMap, {"proj_add_bias_out"});
    opGraph.nodes.push_back(proj_add_biasNode);
  }

  // Residual Add
  CreateResidualAddOperation(&selfResidualAddNode.operation);
  std::string residualInput =
      isTp ? "proj_add_bias_out" : "intermediate_proj_out";
  selfResidualAddNode.inTensorIds = atb_speed::common::GetTensorIdxList(
      tensorMap, {"in_hidden_states", residualInput});
  selfResidualAddNode.outTensorIds =
      atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_add_out"});
  opGraph.nodes.push_back(selfResidualAddNode);

  return atb::NO_ERROR;
}

atb::Status BuildMLPBlock(const VisionEncoderLayerParam& param,
                          const std::map<std::string, uint32_t>& tensorMap,
                          atb::GraphParam& opGraph,
                          bool isTp) {
  atb::Node postNormNode;
  atb::Node gateUpNode;
  atb::Node siluNode;
  atb::Node downNode;
  atb::Node down_add_biasNode;
  atb::Node mlpResidualAddNode;

  // Post Norm
  CreateNormOperation(param,&postNormNode.operation);
  postNormNode.inTensorIds = atb_speed::common::GetTensorIdxList(
      tensorMap, {"intermediate_add_out", "in_post_attn_norm_weight"});
  postNormNode.outTensorIds = atb_speed::common::GetTensorIdxList(
      tensorMap, {"intermediate_post_atten_norm_out"});
  opGraph.nodes.push_back(postNormNode);

  // GateUp Linear
  CreateLinearOperation(&gateUpNode.operation);
  gateUpNode.inTensorIds =
      atb_speed::common::GetTensorIdxList(tensorMap,
                                          {"intermediate_post_atten_norm_out",
                                           "in_mlp_gate_weight",
                                           "in_mlp_gate_bias"});
  gateUpNode.outTensorIds = atb_speed::common::GetTensorIdxList(
      tensorMap, {"intermediate_gateup_out"});
  opGraph.nodes.push_back(gateUpNode);

  // silu Activation
  CreateActivateOperation(&siluNode.operation);
  siluNode.inTensorIds = atb_speed::common::GetTensorIdxList(
      tensorMap, {"intermediate_gateup_out"});
  siluNode.outTensorIds =
      atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_silu_out"});
  opGraph.nodes.push_back(siluNode);

  // Down Linear
  if (isTp) {
    CreateLinearParallelOperation(param, &downNode.operation);
    downNode.inTensorIds = atb_speed::common::GetTensorIdxList(
        tensorMap, {"intermediate_silu_out", "in_mlp_down_weight"});
  } else {
    CreateLinearOperation(&downNode.operation);
    downNode.inTensorIds = atb_speed::common::GetTensorIdxList(
        tensorMap,
        {"intermediate_silu_out", "in_mlp_down_weight", "in_mlp_down_bias"});
  }
  downNode.outTensorIds =
      atb_speed::common::GetTensorIdxList(tensorMap, {"intermediate_down_out"});
  opGraph.nodes.push_back(downNode);

  // Tp Bias Add
  if (isTp) {
    CreateResidualAddOperation(&down_add_biasNode.operation);
    down_add_biasNode.inTensorIds = atb_speed::common::GetTensorIdxList(
        tensorMap, {"intermediate_down_out", "in_mlp_down_bias"});
    down_add_biasNode.outTensorIds =
        atb_speed::common::GetTensorIdxList(tensorMap, {"down_add_bias_out"});
    opGraph.nodes.push_back(down_add_biasNode);
  }

  // MLP Residual Add
  std::string downInput = isTp ? "down_add_bias_out" : "intermediate_down_out";
  CreateResidualAddOperation(&mlpResidualAddNode.operation);
  mlpResidualAddNode.inTensorIds = atb_speed::common::GetTensorIdxList(
      tensorMap, {downInput, "intermediate_add_out"});
  mlpResidualAddNode.outTensorIds =
      atb_speed::common::GetTensorIdxList(tensorMap, {"out"});
  opGraph.nodes.push_back(mlpResidualAddNode);

  return atb::NO_ERROR;
}

atb::Status EncoderLayer(const VisionEncoderLayerParam& param,
                         atb::Operation** operation) {
  atb::GraphParam opGraph;
  opGraph.name = "Vision_Encoder_layer";
  bool isTp = (param.worldSize > 1);

  // tensor names
  std::map<std::string, uint32_t> tensorMap =
      ConstructTensorMap(param,
                         opGraph.inTensorNum,
                         opGraph.outTensorNum,
                         opGraph.internalTensorNum);

  // Attention
  BuildAttentionBlock(param, tensorMap, opGraph, isTp);

  // MLP
  BuildMLPBlock(param, tensorMap, opGraph, isTp);

  opGraph.inferShapeFunc =
      [=](const atb::SVector<atb::TensorDesc>& inTensorDescs,
          atb::SVector<atb::TensorDesc>& outTensorDescs) {
        if (!inTensorDescs.empty() && !outTensorDescs.empty()) {
          outTensorDescs.at(0) = inTensorDescs.at(0);
        }
        return atb::NO_ERROR;
      };

  CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(opGraph, operation));
  return atb::NO_ERROR;
}

}  // namespace qwen
}  // namespace atb_speed