#pragma once
#include "atb/atb_infer.h"
#include "atb_speed/utils/operation_util.h"
#include "models/qwen2_5/vision_encoder/encoder_layer.h"
#include "operations/fusion/utils.h"

namespace atb_speed {
namespace qwen {

atb::Status CreateNormOperation(const VisionEncoderLayerParam& vision_encoder_param,
  atb::Operation** op) {
  atb::infer::RmsNormParam param;
  param.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
  param.normParam.epsilon = vision_encoder_param.rmsNormEps;
  CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(param, op));
  return atb::NO_ERROR;
}

atb::Status CreateLinearOperation(atb::Operation** op) {
  atb::infer::LinearParam param;
  param.transposeA = false;
  param.transposeB = true;
  param.hasBias = true;
  param.outDataType = aclDataType::ACL_DT_UNDEFINED;
  param.enAccum = false;
  CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(param, op));
  return atb::NO_ERROR;
}
atb::Status CreateLinearParallelOperation(
    const VisionEncoderLayerParam& vision_encoder_param,
    atb::Operation** op,bool trans_weight=true) {
  atb::infer::LinearParallelParam param;
  param.rank = vision_encoder_param.rank;
  param.type = atb::infer::LinearParallelParam::ParallelType::LINEAR_ALL_REDUCE;
  param.rankSize = vision_encoder_param.worldSize;
  param.backend = vision_encoder_param.backend;
  param.transWeight = trans_weight;
  CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(param, op));
  return atb::NO_ERROR;
}

atb::Status CreateActivateOperation(atb::Operation** op) {
  atb::infer::ActivationParam opParam;
  opParam.activationType =
      atb::infer::ActivationType::ACTIVATION_SWIGLU_FORWARD;
  CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(opParam, op));
  return atb::NO_ERROR;
}

atb::Status CreateMulOperation(atb::Operation** op) {
  atb::infer::ElewiseParam mulParam;
  mulParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_MUL;
  CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(mulParam, op));
  return atb::NO_ERROR;
}

atb::Status CreateSplitOperation(atb::Operation** op) {
  atb::infer::SplitParam param;
  param.splitDim = -1;  
  param.splitNum = 3;  
  CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(param, op));
  return atb::NO_ERROR;
}

atb::Status CreateRopeOperation(atb::Operation** op) {
  atb::infer::RopeParam ropeParam;
  ropeParam.rotaryCoeff = 2; 
  CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(ropeParam, op));
  return atb::NO_ERROR;
}

atb::Status CreateSelfAttentionOperation(const VisionEncoderLayerParam& param,
                                         atb::Operation** op) {
  atb::infer::SelfAttentionParam attentionParam;
  attentionParam.headNum = param.numAttentionHeadsPerRank;
  attentionParam.kvHeadNum = param.numAttentionHeadsPerRank;
  attentionParam.calcType =
      atb::infer::SelfAttentionParam::CalcType::PA_ENCODER;
  attentionParam.maskType =
      atb::infer::SelfAttentionParam::MaskType::MASK_TYPE_UNDEFINED;
  attentionParam.qkScale = 1.0 / sqrt(param.hiddenSizePerAttentionHead);
  CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(attentionParam, op));
  return atb::NO_ERROR;
}

atb::Status CreateResidualAddOperation(atb::Operation** op) {
  atb::infer::ElewiseParam addParam;
  addParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
  CHECK_OPERATION_STATUS_RETURN(atb::CreateOperation(addParam, op));
  return atb::NO_ERROR;
}
}  // namespace qwen
}  // namespace atb_speed
