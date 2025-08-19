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
#include "atb_speed/utils/singleton.h"
#include "atb_speed/base/external_comm_manager.h"
#include "models/base/param/mapping.h"

namespace atb_speed {
namespace base {

void Mapping::ParseParam(const nlohmann::json &paramJson)
{
    this->worldSize_ = FetchJsonParam<uint32_t>(paramJson, "worldSize");
    this->rank_ = FetchJsonParam<uint32_t>(paramJson, "rank");
    this->rankTableFile_ = FetchJsonParam<std::string>(paramJson, "rankTableFile");
    this->localWorldSize_ = FetchJsonParam<uint32_t>(paramJson, "localWorldSize");
    GetSingleton<ExternalCommManager>().SetLcclCommDomainRange(
        FetchJsonParam<uint32_t>(paramJson, "lcclCommDomainLowerBound"),
        FetchJsonParam<uint32_t>(paramJson, "lcclCommDomainUpperBound")
    );
    std::map<ParallelType, std::string> strategyKeyMap = {
        {WORD_EMBED_TP, "wordEmbedTp"},
        {WORD_EMBED_DP, "wordEmbedDp"},
        {ATTN_TP, "attnTp"},
        {ATTN_DP, "attnDp"},
        {ATTN_INNER_SP, "attnInnerSp"},
        {ATTN_O_PROJ_TP, "attnOProjTp"},
        {ATTN_O_PROJ_DP, "attnOProjDp"},
        {MLP_TP, "mlpTp"},
        {MLP_DP, "mlpDp"},
        {MOE_TP, "moeTp"},
        {MOE_EP, "moeEp"},
        {LM_HEAD_TP, "lmHeadTp"},
        {LM_HEAD_DP, "lmHeadDp"},
        {LCOC_ATTN_TP, "lcocAttnTp"},
    };
    for (auto it = strategyKeyMap.begin(); it != strategyKeyMap.end(); it++) {
        atb_speed::common::ParallelInfo parallelInfo = atb_speed::common::ParallelInfo();
        const nlohmann::json &curParamJson = paramJson[it->second];
        parallelInfo.rank =  FetchJsonParam<uint32_t>(curParamJson, "rank");
        parallelInfo.rankIds =  FetchJsonParam<std::vector<uint32_t>>(curParamJson["rankIds"], "rankIds", true);
        parallelInfo.bufferSize =  FetchJsonParam<uint32_t>(curParamJson, "bufferSize");
        parallelInfo.groupId = FetchJsonParam<uint32_t>(curParamJson, "groupId");
        parallelInfo.backend = FetchJsonParam<std::string>(curParamJson, "backend");
        this->Register(it->first, parallelInfo);
    }
}

void Mapping::InitCommDomain(std::string defaultBackend)
{
    this->defaultBackend_ = defaultBackend;

    // Create global comm
    GetSingleton<ExternalCommManager>().Init(this->rankTableFile_,
        static_cast<uint32_t>(this->worldSize_), static_cast<uint32_t>(this->rank_));

    // Create sub comm
    for (auto it = this->parallelStrategies_.begin(); it != this->parallelStrategies_.end(); it++) {
        std::string backend = it->second.backend != "" ? it->second.backend : this->defaultBackend_;

        // change to hccl if the communication channel across nodes
        int32_t currentDevice = -1;
        for (uint32_t item : it->second.rankIds) {
            if (currentDevice != -1 && static_cast<int32_t>(ceil(item / this->localWorldSize_)) != currentDevice) {
                backend = "hccl";
                break;
            }
            currentDevice = static_cast<int32_t>(ceil(item / this->localWorldSize_));
        }

        // The hccl backend is utilized in the single node scenario
        // when a rankTableFile is supplied and the communication channel spans the entire world size.
        if (this->worldSize_ <= this->localWorldSize_ && this->rankTableFile_ != "" && \
            it->second.rankIds.size() == this->worldSize_) {
            backend = "hccl";
        }
        it->second.InitCommDomain(backend);
    }
    this->isInitialized_ = true;
}

uint32_t Mapping::localWorldSize() const{
   return localWorldSize_;
}

void Mapping::Register(ParallelType parallelType, atb_speed::common::ParallelInfo parallelInfo)
{
    this->parallelStrategies_[parallelType] = parallelInfo;
}

const atb_speed::common::ParallelInfo& Mapping::Get(ParallelType parallelType) const
{
    std::stringstream ss;
    auto it = this->parallelStrategies_.find(parallelType);
    if (it == this->parallelStrategies_.end()) {
        ss << "Mapping: Parallel type [" << parallelType << "] is not found. "
           << "Existing strategies are " << this->ToString();
        throw std::out_of_range(ss.str());
    }
    return it->second;
}

std::string Mapping::ToString() const
{
    std::stringstream ss;
    ss << "Mapping Info: worldSize: " << this->worldSize_
        << ", rank: " << this->rank_
        << ", defaultBackend: " << this->defaultBackend_
        << ", localWorldSize: " << this->localWorldSize_
        << ",[" << WORD_EMBED_TP << "] wordEmbeddingTensorParallel: "
        << this->parallelStrategies_.at(WORD_EMBED_TP).ToString()
        << ",[" << WORD_EMBED_DP << "] wordEmbeddingDataParallel: "
        << this->parallelStrategies_.at(WORD_EMBED_DP).ToString()
        << ",[" << LM_HEAD_TP << "] lmHeadTensorParallel: "
        << this->parallelStrategies_.at(LM_HEAD_TP).ToString()
        << ",[" << LM_HEAD_DP << "] lmHeadDataParallel: "
        << this->parallelStrategies_.at(LM_HEAD_DP).ToString()
        << ",[" << ATTN_TP << "] attnTensorParallel: "
        << this->parallelStrategies_.at(ATTN_TP).ToString()
        << ",[" << ATTN_DP << "] attnDataParallel: "
        << this->parallelStrategies_.at(ATTN_DP).ToString()
        << ",[" << ATTN_INNER_SP << "] attnInnerSequenceParallel: "
        << this->parallelStrategies_.at(ATTN_INNER_SP).ToString()
        << ",[" << MLP_TP << "] mlpTensorParallel: "
        << this->parallelStrategies_.at(MLP_TP).ToString()
        << ",[" << MLP_DP << "] mlpDataParallel: "
        << this->parallelStrategies_.at(MLP_DP).ToString()
        << ",[" << MOE_TP << "] sharedMlpTensorParallel: "
        << this->parallelStrategies_.at(MOE_TP).ToString()
        << ",[" << MOE_EP << "] moeExpertParallel: "
        << this->parallelStrategies_.at(MOE_EP).ToString()
        << ",[" << LCOC_ATTN_TP << "] lcocAttnTP: "
        << this->parallelStrategies_.at(LCOC_ATTN_TP).ToString();
    return ss.str();
}

} // namespace base
} // namesapce atb_speed