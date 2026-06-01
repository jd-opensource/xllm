// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, , Inc. All rights reserved.

#pragma once

// #include "ck_tile/ops/gemm/block/block_gemm_areg_bgmem_creg_v1.hpp"
// #include "ck_tile/ops/gemm/block/block_gemm_areg_bgmem_creg_v1_default_policy.hpp"
// #include "ck_tile/ops/gemm/block/block_gemm_areg_breg_creg_v1.hpp"
// #include "ck_tile/ops/gemm/block/block_gemm_areg_breg_creg_v1_custom_policy.hpp"
// #include "ck_tile/ops/gemm/block/block_gemm_areg_breg_creg_v1_default_policy.hpp"
// #include "ck_tile/ops/gemm/block/block_gemm_areg_bsmem_creg_one_warp_v1.hpp"
// #include "ck_tile/ops/gemm/block/block_gemm_areg_bsmem_creg_v1.hpp"
// #include "ck_tile/ops/gemm/block/block_gemm_areg_bsmem_creg_v1_custom_policy.hpp"
// #include "ck_tile/ops/gemm/block/block_gemm_areg_bsmem_creg_v1_default_policy.hpp"
// #include "ck_tile/ops/gemm/block/block_gemm_areg_bsmem_creg_v2.hpp"
// #include "ck_tile/ops/gemm/block/block_gemm_areg_bsmem_creg_v2_custom_policy.hpp"
// #include "ck_tile/ops/gemm/block/block_gemm_areg_bsmem_creg_v2_default_policy.hpp"
// #include "ck_tile/ops/gemm/block/block_gemm_asmem_breg_creg_v1.hpp"
// #include "ck_tile/ops/gemm/block/block_gemm_asmem_breg_creg_v1_custom_policy.hpp"
// #include "ck_tile/ops/gemm/block/block_gemm_asmem_breg_creg_v1_default_policy.hpp"
// #include "ck_tile/ops/gemm/block/block_gemm_asmem_bsmem_creg_v1_default_policy.hpp"
// #include "ck_tile/ops/gemm/block/block_gemm_problem.hpp"
// #include "ck_tile/ops/gemm/kernel/gemm_kernel.hpp"
// #include "ck_tile/ops/gemm/kernel/gemm_tile_partitioner.hpp"
// #include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_mem.hpp"
// #include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_scheduler.hpp"
// #include "ck_tile/ops/gemm/pipeline/gemm_pipeline_agmem_bgmem_creg_v1.hpp"
// #include "ck_tile/ops/gemm/pipeline/gemm_pipeline_agmem_bgmem_creg_v1_default_policy.hpp"
// #include "ck_tile/ops/gemm/pipeline/gemm_pipeline_agmem_bgmem_creg_v2.hpp"
// #include "ck_tile/ops/gemm/pipeline/gemm_pipeline_agmem_bgmem_creg_v2_default_policy.hpp"
// #include "ck_tile/ops/gemm/pipeline/gemm_pipeline_problem.hpp"
// #include "ck_tile/ops/gemm/pipeline/gemm_universal_pipeline_ag_bg_cr_policy.hpp"
// #include "ck_tile/ops/gemm/pipeline/tile_gemm_shape.hpp"
// #include "ck_tile/ops/gemm/pipeline/tile_gemm_traits.hpp"
// #include "ck_tile/ops/gemm/warp/warp_gemm.hpp"
// #include "ck_tile/ops/gemm/warp/warp_gemm_attribute_mfma.hpp"
// #include "ck_tile/ops/gemm/warp/warp_gemm_attribute_mfma_impl.hpp"
// #include "ck_tile/ops/gemm/warp/warp_gemm_dispatcher.hpp"
// #include "ck_tile/ops/gemm/warp/warp_gemm_impl.hpp"
// #include "ck_tile/ops/common/generic_2d_block_shape.hpp"
// #include "ck_tile/ops/common/tensor_layout.hpp"

//mmac include
//block
#include "ck_tile/ops/gemm/block/block_gemm_problem.hpp"
#include "ck_tile/ops/gemm/block/block_gemm_asmem_bsmem_creg_v1_custom_policy.hpp"
#include "ck_tile/ops/gemm/block/mmac_block_gemm_asmem_bsmem_creg_v1.hpp"

//kernel
#include "ck_tile/ops/gemm/kernel/gemm_kernel.hpp"
#include "ck_tile/ops/gemm/kernel/gemm_tile_partitioner.hpp"

//pipeline
#include "ck_tile/ops/gemm/pipeline/mmac_gemm_universal_pipeline_ag_bg_cr_policy.hpp"
#include "ck_tile/ops/gemm/pipeline/mmac_gemm_pipeline_agmem_bgmem_creg_v1.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_scheduler.hpp"
#include "ck_tile/ops/gemm/pipeline/tile_gemm_shape.hpp"
#include "ck_tile/ops/gemm/pipeline/tile_gemm_traits.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_problem.hpp"


//warp
#include "ck_tile/ops/gemm/warp/warp_mmac_gemm.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_attribute_mmac.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_attribute_mmac_impl.hpp"
#include "ck_tile/ops/gemm/warp/warp_mmac_gemm_dispatcher.hpp"
#include "ck_tile/ops/gemm/warp/warp_mmac_gemm_impl.hpp"
#include "ck_tile/ops/gemm/warp/warp_dsreadm_attribute.hpp"
#include "ck_tile/ops/gemm/warp/warp_dsreadm_attribute_impl.hpp"
#include "ck_tile/ops/gemm/warp/warp_dsreadm_dispatcher.hpp"
#include "ck_tile/ops/gemm/warp/warp_dsreadm_format_attribute.hpp"
#include "ck_tile/ops/gemm/warp/warp_dsreadm_format_attribute_impl.hpp"
#include "ck_tile/ops/gemm/warp/warp_dsreadm_format_dispatcher.hpp"
#include "ck_tile/ops/gemm/warp/warp_dsreadm_format_dispatcher_v2.hpp"

//common
#include "ck_tile/ops/common/generic_2d_block_shape.hpp"
#include "ck_tile/ops/common/tensor_layout.hpp"
