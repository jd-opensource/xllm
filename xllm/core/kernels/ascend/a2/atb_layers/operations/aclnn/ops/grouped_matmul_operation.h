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
#ifndef ATB_SPEED_PLUGIN_ACLNN_GROUPED_MATMUL_OPERATION_H
#define ATB_SPEED_PLUGIN_ACLNN_GROUPED_MATMUL_OPERATION_H
#include "operations/aclnn/core/acl_nn_operation.h"
#include "operations/aclnn/core/acl_nn_operation_cache.h"
#include "aclnnop/aclnn_grouped_matmul_v4.h"
#include "atb_speed/utils/operation_util.h"

namespace atb_speed {
namespace common {
enum GmmQuantType : int {
    NONE = 0,
    W8A8_CHANNEL,
    W8A16_CHANNEL,
    W4A16_CHANNEL,
    W8A8_TOKEN,
    W4A8_GROUP
};

struct AclNNGroupedMatmulParam {
    /// A flag indicating whether the second input matrix needs to be transposed
    bool transposeB = false;
    /// The quantization type of the operation
    int quantType = NONE;
    /// A flag indicating whether the matmul operation includes a bias tensor
    bool hasBias = false;
    /// The data type of the output of the operation
    aclDataType outDataType = ACL_FLOAT16;
};

/// This class defines an operator that consists of a group of matrix multiplications.
/// Meanwhile, this operator supports different quantization types.
///
/// This class makes uses of `aclnnGroupedMatmulV4GetWorkspaceSize` and `aclnnGroupedMatmulV4`
/// from the AscendCL API.
///
/// Inputs to the operator:
/// Name                    | Dtype | Shape |
/// ------------------------|-------|-------|
/// input                   | *     | [m,k] |
/// weight                  | *     | [e,n,k] if `transposeB` is true; otherwise, [e,k,n] |
/// biasOptional            | *     | [e,n] |
/// scaleOptional           | *     | [e,n] |
/// offsetOptional          | *     | [e,n] |
/// antiquantScaleOptional  | *     | [e,n] |
/// antiquantOffsetOptional | *     | [e,n] |
/// groupList               | int64 | [e]   |
/// * Note: the data type of inputs are specific to the quantization type/technique chosen for the model
///
/// Outputs of the operator:
/// Name   | Dtype               | Shape |
/// -------|---------------------|-------|
/// output | float16 or bfloat16 | [m,n] |
///
/// Example:
/// \code
/// enum TensorIdx : uint32_t {
///     IN_INPUT = 0,
///     IN_WEIGHT,
///     IN_GROUP_LIST,
///     OUT,
/// };
///
/// atb::Node &gmmNode = opGraph.nodes.at(nodeId++);
/// atb_speed::common::AclNNGroupedMatmulParam gmmParam;
/// gmmParam.quantType = gmmQuantType;
/// gmmParam.outDataType = param.outDataType;
/// gmmParam.transposeB = param.transposeB;
/// gmmNode.operation = new atb_speed::common::GroupedMatmulOperation("gmmNode", gmmParam);
/// gmmNode.inTensorIds = {IN_INPUT, IN_WEIGHT, IN_GROUP_LIST};
/// gmmNode.outTensorIds = {OUT};
/// \endcode

class GroupedMatmulOperation : public AclNNOperation {
public:
   explicit GroupedMatmulOperation(const std::string &name, AclNNGroupedMatmulParam param);
    ~GroupedMatmulOperation() override;
    atb::Status InferShape(const atb::SVector<atb::TensorDesc> &inTensorDescs,
        atb::SVector<atb::TensorDesc> &outTensorDescs) const override;
    uint32_t GetInputNum() const override;
    uint32_t GetOutputNum() const override;

private:
    int SetAclNNWorkspaceExecutor() override;
    int ExecuteAclNNOp(uint8_t *workspace, aclrtStream &stream) override;

    atb::Status CreateAclNNInTensorVariantPack(const atb::VariantPack &variantPack) override;
    atb::Status CreateAclNNOutTensorVariantPack(const atb::VariantPack &variantPack) override;

    int CreateW8A8(AclNNVariantPack &aclnnVariantPack);
    int CreateA16(AclNNVariantPack &aclnnVariantPack);
    int CreateW8A8Token(AclNNVariantPack &aclnnVariantPack);
    int CreateW4A8(AclNNVariantPack &aclnnVariantPack);
    int IndexGmmQuant(AclNNVariantPack &aclnnVariantPack);

    std::vector<aclTensor *> yTensorVector;
    std::vector<std::vector<aclTensor *>> inputVectorOfTensor;
    std::vector<aclTensor *> weightTensorVector;
    int64_t splitItem = 2;
    int64_t groupType = 0;
    int64_t groupListType = 0; // 0 : GMMActType::GMM_ACT_TYPE_NONE
    int64_t actType = 0;
    AclNNGroupedMatmulParam param_;
};
}  // namespace common
}  // namespace atb_speed
#endif  // ATB_SPEED_PLUGIN_ACLNN_GROUPED_MATMUL_OPERATION_H