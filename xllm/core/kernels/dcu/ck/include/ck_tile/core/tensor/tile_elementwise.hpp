// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, , Inc. All rights reserved.

#pragma once

#include "ck_tile/core/config.hpp"
#include "ck_tile/core/container/array.hpp"
#include "ck_tile/core/container/sequence.hpp"
#include "ck_tile/core/container/tuple.hpp"
#include "ck_tile/core/container/container_helper.hpp"
#include "ck_tile/core/tensor/tensor_adaptor.hpp"
#include "ck_tile/core/tensor/null_tensor.hpp"
#include "ck_tile/core/utility/functional.hpp"
#include "ck_tile/core/utility/type_traits.hpp"

namespace ck_tile {

// TODO: support tensors with different distribution
template <typename InOutElementFunc,
          typename... InOutDstrTensors,
          typename = std::enable_if_t<std::conjunction_v<
              std::negation<std::is_same<std::remove_const_t<InOutDstrTensors>, null_tensor>>...>>>
CK_TILE_DEVICE void tile_elementwise_inout(const InOutElementFunc& inout_element_func,
                                           InOutDstrTensors&... inout_dstr_tensors)
{
    // TODO: make sure all distributed tensors have same lengths and distribution
    // static_assert(xxx);

    constexpr index_t thread_buffer_size =
        __type_pack_element<0, InOutDstrTensors...>::get_thread_buffer_size();

    using value_type = typename __type_pack_element<0, InOutDstrTensors...>::DataType;

    if constexpr(std::is_same_v<value_type, fp16_t>)
    {
        static_assert(thread_buffer_size % 2 == 0);

        // pk f16 impls
        static_for<0, thread_buffer_size / 2, 1>{}([&](auto i) {
            inout_element_func(
                inout_dstr_tensors.get_thread_buffer().template get_as<fp16x2_t>().at(i)...);
        });
    }
    else
    {
        static_for<0, thread_buffer_size, 1>{}(
            [&](auto i) { inout_element_func(inout_dstr_tensors.get_thread_buffer().at(i)...); });
    }
}

template <typename InElementFunc,
          typename... InTensor,
          typename = std::enable_if_t<
              std::conjunction_v<std::negation<std::is_same<InTensor, null_tensor>>...>>>
CK_TILE_DEVICE auto tile_elementwise_in(const InElementFunc& in_element_func,
                                        const InTensor&... in_dstr_tensors)
{
    using OutDataType = decltype(in_element_func(typename InTensor::DataType{}...));

    // TODO: make sure all distributed tensors have same lengths and distribution
    // static_assert(xxx);
    constexpr auto in_tile_dstr = __type_pack_element<0, InTensor...>::get_tile_distribution();

    constexpr index_t thread_buffer_size =
        __type_pack_element<0, InTensor...>::get_thread_buffer_size();

    auto out_dstr_tensor = make_static_distributed_tensor<OutDataType>(in_tile_dstr);

    static_for<0, thread_buffer_size, 1>{}([&](auto i) {
        out_dstr_tensor.get_thread_buffer()(i) =
            in_element_func(in_dstr_tensors.get_thread_buffer()[i]...);
    });

    return out_dstr_tensor;
}

/**
 * @brief  Template function that "unpacks" a tuple and applies an element-wise operation.
 *
 * @param in_element_func    Function to apply element-wise.
 * @param t                  Any container containing elements to process, with known size and
 * tuple-like semantic.
 * @return Calls tile_elementwise_inout with unpacked tuple elements.
 */
template <typename InElementFunc, typename Tuple, size_t... I>
CK_TILE_DEVICE auto tile_elementwise_inout_unpack(const InElementFunc& in_element_func,
                                                  const Tuple& t,
                                                  std::index_sequence<I...>)
{
    return tile_elementwise_inout(in_element_func, t[number<I>{}]...);
}

/**
 * @brief  Template function that "unpacks" a tuple and applies an element-wise operation.
 *
 * @param in_element_func   Function to apply element-wise.
 * @param t                 Any container containing elements to process, with known size and
 * tuple-like semantic.
 * @return Calls the overloaded function, passing an index sequence.
 */
template <typename InElementFunc, typename Tuple>
CK_TILE_DEVICE auto tile_elementwise_inout_unpack(const InElementFunc& in_element_func,
                                                  const Tuple& t)
{
    static constexpr auto size = Tuple::size();
    return tile_elementwise_inout_unpack(in_element_func, t, std::make_index_sequence<size>{});
}

template <typename DstrTensors, typename T>
CK_TILE_DEVICE void set_tile(DstrTensors& dstr_tensor, const T& value)
{
    tile_elementwise_inout(
        [&value](auto& x) {
            x = type_convert<typename DstrTensors::DataType, remove_cvref_t<T>>(value);
        },
        dstr_tensor);
}

template <typename T>
CK_TILE_DEVICE void set_tile(null_tensor&, const T&)
{
}

// TODO: prefer to use per-dword value to set a tensor, in case compiler not doing well with
// sub-dword tensor...
template <typename DstrTensors, index_t v, bool skip_subdword_opt = false>
CK_TILE_DEVICE void
set_tile(DstrTensors& dstr_tensor, number<v>, bool_constant<skip_subdword_opt> = {})
{
    using elem_type             = typename DstrTensors::DataType;
    constexpr index_t elem_size = sizeof(elem_type);

    constexpr index_t tensor_bytes = DstrTensors::get_thread_buffer_size() * elem_size;

    // # bytes per write = 4
    if constexpr(v == 0 && tensor_bytes % 4 == 0 && !skip_subdword_opt)
    {
#if CK_TILE_WORKAROUND_ROCM_6_1_SCRATCH_MEMORY_ISSUE
        auto& buffer = dstr_tensor.get_thread_buffer();

        static_for<0, tensor_bytes / 4, 1>{}([&](auto i_write) {
            if constexpr(elem_size == 1)
            {
                // # elements per write = 4
                constexpr auto values = ext_vector_t<elem_type, 4>{0, 0, 0, 0};

                buffer[i_write * 4 + 0] = values.x;
                buffer[i_write * 4 + 1] = values.y;
                buffer[i_write * 4 + 2] = values.z;
                buffer[i_write * 4 + 3] = values.w;
            }
            else if constexpr(elem_size == 2)
            {
                // # elements per write = 2
                constexpr auto values = ext_vector_t<elem_type, 2>{0, 0};

                buffer[i_write * 2 + 0] = values.x;
                buffer[i_write * 2 + 1] = values.y;
            }
            else if constexpr(elem_size == 4)
            {
                // # elements per write = 1
                constexpr elem_type value = 0;

                buffer[i_write] = value;
            }
            else
            {
                static_assert(false, "type not supported");
            }
        });
#else
        using dvec_t = array<index_t, tensor_bytes / 4>;
        auto& tensor = reinterpret_cast<dvec_t&>(dstr_tensor.get_thread_buffer());
        for(auto i = 0; i < tensor.size(); i++)
            tensor.get(i) = v;
#endif
    }
    else
    {
        tile_elementwise_inout([](auto& x) { x = type_convert<elem_type, index_t>(v); },
                               dstr_tensor);
    }
}

template <index_t v>
CK_TILE_DEVICE void set_tile(null_tensor&, number<v>)
{
}

template <typename DstrTensors>
CK_TILE_DEVICE void clear_tile(DstrTensors& dstr_tensor)
{
    set_tile(dstr_tensor, 0);
}

namespace impl {
// TODO: this is ugly
template <typename OutDataType, typename InTensor>
CK_TILE_DEVICE auto cast_tile_pk_fp8_fp32(const InTensor& in_dstr_tensors)
{
#if defined(__gfx94__)
    // This API is designed to use the _pk_ serious of function
    constexpr auto in_tile_dstr = InTensor::get_tile_distribution();

    constexpr index_t thread_buffer_size = InTensor::get_thread_buffer_size();
    static_assert(thread_buffer_size % 4 == 0);
    constexpr index_t thread_buffer_size_pk = thread_buffer_size / 4;

    auto out_dstr_tensor = make_static_distributed_tensor<OutDataType>(in_tile_dstr);
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wuninitialized"
    // __builtin_amdgcn_cvt_pk_fp8_f32() this builtin require the old value, and
    // will generate a v_mov_b32 vxxx [old] before cvt, which result in unwanted ISA
    // so we prepare an uninitialized variable purposely, and turn off the warning
    int dummy_old;
    static_for<0, thread_buffer_size_pk, 1>{}([&](auto i) {
        uint32_t x = __builtin_amdgcn_cvt_pk_fp8_f32(
            in_dstr_tensors.get_thread_buffer()[number<4 * i + 0>{}],
            in_dstr_tensors.get_thread_buffer()[number<4 * i + 1>{}],
            dummy_old,
            false); // false -> WORD0

        uint32_t y = __builtin_amdgcn_cvt_pk_fp8_f32(
            in_dstr_tensors.get_thread_buffer()[number<4 * i + 2>{}],
            in_dstr_tensors.get_thread_buffer()[number<4 * i + 3>{}],
            dummy_old,
            false); // false -> WORD0

        constexpr int32_t m0 = 0x05040100;
        using vec_t          = array<OutDataType, 4>;

        vec_t d = bit_cast<vec_t>(__builtin_amdgcn_perm(y, x, m0));
        out_dstr_tensor.get_thread_buffer().template set_as<vec_t>(number<i>{}, d);
    });
#pragma clang diagnostic pop

    return out_dstr_tensor;
#else
    // fallback
    return tile_elementwise_in(type_convert<OutDataType, typename InTensor::DataType>,
                               in_dstr_tensors);
#endif
}

template <typename OutDataType, typename InTensor>
CK_TILE_DEVICE auto cast_tile_pk_fp16_fp32(const InTensor& in_dstr_tensors)
{
#if defined(__gfx928__) || defined(__gfx936__) || defined(__gfx938__) || defined(__gfx92a__) || \
    defined(__gfx946__)
    // This API is designed to use the _pk_ serious of function
    constexpr auto in_tile_dstr = InTensor::get_tile_distribution();

    constexpr index_t thread_buffer_size = InTensor::get_thread_buffer_size();
    static_assert(thread_buffer_size % 2 == 0);
    constexpr index_t thread_buffer_size_pk = thread_buffer_size / 2;

    auto out_dstr_tensor = make_static_distributed_tensor<OutDataType>(in_tile_dstr);

    // TODO: this is rtz cvt, need be very careful
    for(index_t i = 0; i < thread_buffer_size_pk; i++)
    {
        auto o = __builtin_amdgcn_cvt_pkrtz(in_dstr_tensors.get_thread_buffer()[2 * i + 0],
                                            in_dstr_tensors.get_thread_buffer()[2 * i + 1]);

        out_dstr_tensor.get_thread_buffer().at(2 * i + 0) = o.x;
        out_dstr_tensor.get_thread_buffer().at(2 * i + 1) = o.y;
    }

    return out_dstr_tensor;
#else
    // fallback
    return tile_elementwise_in(type_convert<OutDataType, typename InTensor::DataType>,
                               in_dstr_tensors);
#endif
}

#if CK_TILE_USE_SUBDWORD_TILE_CAST
// this function assume either src or dst (or both) date type is under 1 dword
// we pack subdword value into 1 dword to avoid compiler's default subdword behavior(which is buggy)
template <typename OutDataType, typename InTensor>
CK_TILE_DEVICE auto cast_tile_opt_subdword(const InTensor& in_dstr_tensors)
{
    constexpr auto in_tile_dstr = InTensor::get_tile_distribution();

    auto out_dstr_tensor = make_static_distributed_tensor<OutDataType>(in_tile_dstr);

    using i_type                   = remove_cvref_t<typename InTensor::DataType>;
    using o_type                   = remove_cvref_t<OutDataType>;
    constexpr index_t i_elem_bytes = sizeof(i_type);
    constexpr index_t o_elem_bytes = sizeof(o_type);
    static_assert(i_elem_bytes < 4 || o_elem_bytes < 4);

    constexpr index_t bulk_size =
        (i_elem_bytes >= o_elem_bytes) ? (4 / o_elem_bytes) : (4 / i_elem_bytes);
    static_assert(bulk_size != 0);

    using o_bulk_type =
        std::conditional_t<i_elem_bytes >= o_elem_bytes, float, array<o_type, bulk_size>>;

    constexpr index_t thread_buffer_size = InTensor::get_thread_buffer_size();

    constexpr index_t iters = thread_buffer_size / bulk_size;
    constexpr index_t rems  = thread_buffer_size % bulk_size;

    // cast the sequence per-bulk
    static_for<0, iters, 1>{}([&](auto i) {
        union bulk_wrapper
        {
            o_bulk_type bulk{};
            o_type data[bulk_size];
        } o_bulk;

        // TODO: should use below function, but somehow will result in spill (same as c-forloop)
        static_for<0, bulk_size, 1>{}([&o_bulk, &in_dstr_tensors, &i](auto ib) {
            o_bulk.data[ib.value] = static_cast<o_type>(
                in_dstr_tensors.get_thread_buffer()
                    .template get_as<i_type>()[number<bulk_size * i.value + ib.value>{}]);
        });

        // TODO: fixme, should use above!
        // static_assert(sizeof(i_type) / sizeof(o_type) == 2);
        // o_bulk.data[0] = static_cast<o_type>(
        //     in_dstr_tensors.get_thread_buffer().template get_as<i_type>()[number<2 * i + 0>{}]);
        // o_bulk.data[1] = static_cast<o_type>(
        //     in_dstr_tensors.get_thread_buffer().template get_as<i_type>()[number<2 * i + 1>{}]);

        out_dstr_tensor.get_thread_buffer().template set_as<o_bulk_type>(i, o_bulk.bulk);
    });

    static_for<0, rems, 1>{}([&](auto r) {
        // TODO: introducing local scratch pad?
        auto idx = number<iters * bulk_size + r>{};
        out_dstr_tensor.get_thread_buffer().at(idx) =
            static_cast<o_type>(in_dstr_tensors.get_thread_buffer().at(idx));
    });

    return out_dstr_tensor;
}
#endif
} // namespace impl

template <typename DstType, bool UsePKCVTCast = true, typename SrcTensor>
CK_TILE_DEVICE auto cast_tile(const SrcTensor& src_tensor, bool_constant<UsePKCVTCast> = {})
{
    if constexpr((std::is_same_v<DstType, fp8_t> ||
                  std::is_same_v<DstType, bf8_t>)&&std::is_same_v<typename SrcTensor::DataType,
                                                                  float> &&
                 (SrcTensor::get_thread_buffer_size() % 4 == 0))
    {
        return impl::cast_tile_pk_fp8_fp32<DstType, SrcTensor>(src_tensor);
    }
#if CK_TILE_USE_PK_FP16_TILE_CAST
    else if constexpr(std::is_same_v<DstType, fp16_t> &&
                      std::is_same_v<typename SrcTensor::DataType, float> &&
                      UsePKCVTCast &&
                      (SrcTensor::get_thread_buffer_size() % 2 == 0))
    {
        return impl::cast_tile_pk_fp16_fp32<DstType, SrcTensor>(src_tensor);
    }
#endif
#if CK_TILE_USE_SUBDWORD_TILE_CAST
    else if constexpr(sizeof(DstType) < 4 || sizeof(typename SrcTensor::DataType) < 4)
    {
        return impl::cast_tile_opt_subdword<DstType, SrcTensor>(src_tensor);
    }
#endif
    else
        return tile_elementwise_in(type_convert<DstType, typename SrcTensor::DataType>, src_tensor);
}

// no-op function for null_tensor arguments
template <typename InOutElementFunc,
          typename... MaybeNullTensor,
          typename = std::enable_if_t<
              std::disjunction_v<std::is_same<remove_cvref_t<MaybeNullTensor>, null_tensor>...>>>
CK_TILE_DEVICE void tile_elementwise_inout(const InOutElementFunc&, MaybeNullTensor&&...)
{
}

// no-op function for null_tensor arguments
template <typename InElementFunc,
          typename... MaybeNullTensor,
          typename = std::enable_if_t<
              std::disjunction_v<std::is_same<remove_cvref_t<MaybeNullTensor>, null_tensor>...>>>
CK_TILE_DEVICE auto tile_elementwise_in(const InElementFunc&, MaybeNullTensor&&...)
{
    return null_tensor{};
}

} // namespace ck_tile
