// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, , Inc. All rights reserved.

#pragma once

#include "ck_tile/core/arch/amd_buffer_addressing.hpp"
#include "ck_tile/core/container/multi_index.hpp"
#include "ck_tile/ops/common/tensor_layout.hpp"

namespace ck_tile {

struct __attribute__((packed)) tls_resource
{
    union
    {
        const void* ptr;
        struct
        {
            uint32_t DW_0_DATA;

            union
            {
                struct
                {
                    uint32_t base_address_hi : 16;
                    uint32_t reserved : 9;
                    uint32_t cache_swizzle : 1;
                    uint32_t rank : 2;
                    uint32_t data_type : 3;
                    uint32_t is_filter : 1;
                };

                uint32_t DW_1_DATA;
            } DW_1_UNION;
        };
    };

    union
    {
        struct
        {
            uint32_t C : 16;
            uint32_t W : 16;
        } GLOBAL_DIM_LO_4D;

        uint32_t DW_2_DATA;
    } DW_2_UNION;

    union
    {
        struct
        {
            uint32_t H : 16;
            uint32_t N : 16;
        } GLOBAL_DIM_HI_4D;

        uint32_t DW_3_DATA;
    } DW_3_UNION;
};

namespace detail {
namespace tls_resource {

// clang-format off
template <typename DataType> struct data_type_traits;
template <> struct data_type_traits<fp16_t> { static constexpr auto value = number<2>{}; };
template <> struct data_type_traits<int8_t> { static constexpr auto value = number<3>{}; };

template <typename Layout> struct is_filter_traits;
template <> struct is_filter_traits<tensor_layout::convolution::NHWGC>      { static constexpr auto value = number<0>{}; };
template <> struct is_filter_traits<tensor_layout::convolution::NGCHWc<32>> { static constexpr auto value = number<0>{}; };
template <> struct is_filter_traits<tensor_layout::convolution::NGCHWc<64>> { static constexpr auto value = number<0>{}; };
template <> struct is_filter_traits<tensor_layout::convolution::NGCHWc<128>> { static constexpr auto value = number<0>{}; };

template <> struct is_filter_traits<tensor_layout::convolution::GKYXC>      { static constexpr auto value = number<1>{}; };
template <> struct is_filter_traits<tensor_layout::convolution::GKCYXc<32>> { static constexpr auto value = number<1>{}; };
template <> struct is_filter_traits<tensor_layout::convolution::GKCYXc<64>> { static constexpr auto value = number<1>{}; };
template <> struct is_filter_traits<tensor_layout::convolution::GKCYXc<128>> { static constexpr auto value = number<1>{}; };
// clang-format on
} // namespace tls_resource
} // namespace detail

template <index_t rank = 4, typename TensorLengths, index_t data_type, index_t is_filter>
CK_TILE_DEVICE int32x4_t make_tls_resource(const void* ptr,
                                           const TensorLengths& tensor_lens,
                                           number<data_type>,
                                           number<is_filter>)
{
    tls_resource res{{ptr}, {{0, 0}}, {{0, 0}}};

    if constexpr((rank == 4) && (TensorLengths::size() == 4))
    {
        res.DW_1_UNION.cache_swizzle = 1;
        res.DW_1_UNION.rank          = 1;
        res.DW_1_UNION.data_type     = data_type;
        res.DW_1_UNION.is_filter     = is_filter;

        // lens must be in {N, H, W, C} order
        res.DW_2_UNION.GLOBAL_DIM_LO_4D.C = tensor_lens[number<3>{}];
        res.DW_2_UNION.GLOBAL_DIM_LO_4D.W = tensor_lens[number<2>{}];
        res.DW_3_UNION.GLOBAL_DIM_HI_4D.H = tensor_lens[number<1>{}];
        res.DW_3_UNION.GLOBAL_DIM_HI_4D.N = tensor_lens[number<0>{}];
    }
    else
    {
        static_assert(false, "Unsupported tensor rank");
    }

    int32x4_t r = __builtin_bit_cast(int32x4_t, res);
    r.x         = __builtin_amdgcn_readfirstlane(r.x);
    r.y         = __builtin_amdgcn_readfirstlane(r.y);
    r.z         = __builtin_amdgcn_readfirstlane(r.z);
    r.w         = __builtin_amdgcn_readfirstlane(r.w);

    return r;
}

struct __attribute__((packed)) tls_sampler
{
    union
    {
        struct
        {
            uint32_t ox : 16;
            uint32_t oy : 16;
        };

        uint32_t DW_0_DATA;
    } DW_0_UNION;

    union
    {
        struct
        {
            uint32_t on : 16;
            uint32_t coffset : 11;
            uint32_t channels_per_pixel : 5;
        };

        uint32_t DW_1_DATA;
    } DW_1_UNION;

    union
    {
        struct
        {
            uint32_t pixel_per_column : 3;
            uint32_t element_stride : 3;
            uint32_t dilation_rate : 2;
            uint32_t padding_num : 2;
            uint32_t load_mode : 3;
            uint32_t filter_size : 2;
            uint32_t coord_type : 1;
            uint32_t reserved : 16;
        };

        uint32_t DW_2_DATA;
    } DW_2_UNION;

    union
    {
        struct
        {
            uint32_t interleave : 3;
            uint32_t oob_fill : 2;
            uint32_t swizzle : 3;
            uint32_t l2policy : 3;
            uint32_t reserved : 21;
        };

        uint32_t DW_3_DATA;
    } DW_3_UNION;
};

namespace detail {
namespace tls_sampler {

// clang-format off
template <index_t ElemSizeBytes, index_t ChannelsPerPixel> struct channels_per_pixel_traits;
template <> struct channels_per_pixel_traits<4, 16> { static constexpr auto value = number<0>{}; };
template <> struct channels_per_pixel_traits<2, 32> { static constexpr auto value = number<1>{}; };
template <> struct channels_per_pixel_traits<1, 64> { static constexpr auto value = number<2>{}; };

template <index_t PixelPerColumn> struct pixel_per_column_traits;
template <> struct pixel_per_column_traits<16> { static constexpr auto value = number<0>{}; };
template <> struct pixel_per_column_traits<32> { static constexpr auto value = number<1>{}; };

template <typename Strides> struct element_stride_traits;
template <> struct element_stride_traits<sequence<1, 1>> { static constexpr auto value = number<0>{}; };
template <> struct element_stride_traits<sequence<2, 2>> { static constexpr auto value = number<1>{}; };

template <typename Dilations> struct dilation_rate_traits;
template <> struct dilation_rate_traits<sequence<1, 1>> { static constexpr auto value = number<0>{}; };

template <typename Pads> struct padding_num_traits;
template <> struct padding_num_traits<sequence<0, 0, 0, 0>> { static constexpr auto value = number<0>{}; };
template <> struct padding_num_traits<sequence<1, 1, 1, 1>> { static constexpr auto value = number<1>{}; };
template <> struct padding_num_traits<sequence<2, 2, 2, 2>> { static constexpr auto value = number<2>{}; };
template <> struct padding_num_traits<sequence<3, 3, 3, 3>> { static constexpr auto value = number<3>{}; };

template <typename FilterLengths> struct filter_size_traits;
template <> struct filter_size_traits<sequence<1, 1>> { static constexpr auto value = number<0>{}; };
template <> struct filter_size_traits<sequence<3, 3>> { static constexpr auto value = number<1>{}; };
template <> struct filter_size_traits<sequence<5, 5>> { static constexpr auto value = number<2>{}; };
template <> struct filter_size_traits<sequence<7, 7>> { static constexpr auto value = number<3>{}; };

template <typename layout> struct interleave_traits;
template <> struct interleave_traits<tensor_layout::convolution::NHWGC>       { static constexpr auto value = number<0>{}; };
template <> struct interleave_traits<tensor_layout::convolution::NGCHWc<32>>  { static constexpr auto value = number<1>{}; };
template <> struct interleave_traits<tensor_layout::convolution::NGCHWc<64>>  { static constexpr auto value = number<2>{}; };
template <> struct interleave_traits<tensor_layout::convolution::NGCHWc<128>> { static constexpr auto value = number<3>{}; };

template <> struct interleave_traits<tensor_layout::convolution::GKYXC>       { static constexpr auto value = number<0>{}; };
template <> struct interleave_traits<tensor_layout::convolution::GKCYXc<32>>  { static constexpr auto value = number<1>{}; };
template <> struct interleave_traits<tensor_layout::convolution::GKCYXc<64>>  { static constexpr auto value = number<2>{}; };
template <> struct interleave_traits<tensor_layout::convolution::GKCYXc<128>> { static constexpr auto value = number<3>{}; };

template <typename Pads> struct oob_fill_traits;
template <> struct oob_fill_traits<sequence<0, 0, 0, 0>> { static constexpr auto value = number<0>{}; };
template <> struct oob_fill_traits<sequence<1, 1, 1, 1>> { static constexpr auto value = number<1>{}; };
template <> struct oob_fill_traits<sequence<2, 2, 2, 2>> { static constexpr auto value = number<1>{}; };
template <> struct oob_fill_traits<sequence<3, 3, 3, 3>> { static constexpr auto value = number<1>{}; };
// clang-format on

} // namespace tls_sampler
} // namespace detail

template <index_t channels_per_pixel,
          index_t pixel_per_column,
          index_t element_stride,
          index_t dilation_rate,
          index_t padding_num,
          index_t filter_size,
          index_t coord_type,
          index_t interleave,
          index_t oob_fill>
CK_TILE_DEVICE int32x4_t make_tls_sampler(index_t on,
                                          index_t oy,
                                          index_t ox,
                                          number<channels_per_pixel>,
                                          number<pixel_per_column>,
                                          number<element_stride>,
                                          number<dilation_rate>,
                                          number<padding_num>,
                                          number<filter_size>,
                                          number<coord_type>,
                                          number<interleave>,
                                          number<oob_fill>)
{
    tls_sampler sampler{{{0, 0}}, {{0, 0, 0}}, {{0, 0, 0, 0, 0, 0, 0, 0}}, {{0, 0, 0, 0, 0}}};

    sampler.DW_0_UNION.ox = ox;
    sampler.DW_0_UNION.oy = oy;
    sampler.DW_1_UNION.on = on;

    sampler.DW_1_UNION.coffset            = 0;
    sampler.DW_1_UNION.channels_per_pixel = channels_per_pixel;

    sampler.DW_2_UNION.pixel_per_column = pixel_per_column;
    sampler.DW_2_UNION.element_stride   = element_stride;
    sampler.DW_2_UNION.dilation_rate    = dilation_rate;
    sampler.DW_2_UNION.padding_num      = padding_num;
    sampler.DW_2_UNION.load_mode        = 1;
    sampler.DW_2_UNION.filter_size      = filter_size;
    sampler.DW_2_UNION.coord_type       = coord_type;

    sampler.DW_3_UNION.interleave = interleave;
    sampler.DW_3_UNION.oob_fill   = oob_fill;

    int32x4_t r = __builtin_bit_cast(int32x4_t, sampler);
    r.x         = __builtin_amdgcn_readfirstlane(r.x);
    r.y         = __builtin_amdgcn_readfirstlane(r.y);
    r.z         = __builtin_amdgcn_readfirstlane(r.z);
    r.w         = __builtin_amdgcn_readfirstlane(r.w);

    return r;
}

template <typename T,
          index_t offset0,
          index_t offset1,
          index_t samp_num,
          index_t samp_idx,
          bool use_m0 = false>
CK_TILE_DEVICE void hcu_async_tls_asm(CK_TILE_LDS_ADDR T* smem,
                                      const int32x4_t& resource,
                                      const int32x4_t& sampler,
                                      number<offset0>,
                                      number<offset1>,
                                      number<samp_num>,
                                      number<samp_idx>,
                                      bool_constant<use_m0> = {})
{
#if defined(__gfx92a__) || defined(__gfx946__)
    const auto soffset = __builtin_amdgcn_readfirstlane(reinterpret_cast<uintptr_t>(smem));

    if constexpr(CK_TILE_TENSOR_LOAD_FORCE_M0 || use_m0)
    {
        asm volatile(
            "s_mov_b32 m0, %0\n\t"
            "tensor_load %1, %2, m0 offset0:%3 offset1:%4 samp_num:%5 samp_idx:%6 lds\n\t" ::"s"(
                soffset),
            "s"(resource),
            "s"(sampler),
            "n"(offset0),
            "n"(offset1),
            "n"(samp_num),
            "n"(samp_idx)
            : "memory");
    }
    else
    {
        asm volatile(
            "tensor_load %0, %1, %2 offset0:%3 offset1:%4 samp_num:%5 samp_idx:%6 lds" ::"s"(
                resource),
            "s"(sampler),
            "s"(soffset),
            "n"(offset0),
            "n"(offset1),
            "n"(samp_num),
            "n"(samp_idx)
            : "memory");
    }
#else
    detail::swallow{smem,
                    resource,
                    sampler,
                    number<offset0>{},
                    number<offset1>{},
                    number<samp_num>{},
                    number<samp_idx>{},
                    bool_constant<use_m0>{}};
#endif
}

} // namespace ck_tile
