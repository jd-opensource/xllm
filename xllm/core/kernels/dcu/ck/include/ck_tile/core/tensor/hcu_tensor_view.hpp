// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, , Inc. All rights reserved.

#pragma once

#include "ck_tile/core/arch/arch.hpp"
#include "ck_tile/core/config.hpp"
#include "ck_tile/core/numeric/integer.hpp"
#include "ck_tile/core/numeric/integral_constant.hpp"
#include "ck_tile/core/algorithm/coordinate_transform.hpp"
#include "ck_tile/core/container/container_helper.hpp"
#include "ck_tile/core/numeric/math.hpp"
#include "ck_tile/core/tensor/hcu_buffer_view.hpp"
#include "ck_tile/core/tensor/tensor_descriptor.hpp"
#include "ck_tile/core/utility/functional.hpp"
#include "ck_tile/core/utility/type_traits.hpp"

namespace ck_tile {
template <typename BufferView_,
          typename TensorDesc_,
          memory_operation_enum DstInMemOp_ = memory_operation_enum::set>
struct hcu_tensor_view
{
    using buffer_view = remove_reference_t<BufferView_>;
    using DataType    = typename buffer_view::type;
    using TensorDesc  = remove_cvref_t<TensorDesc_>;
    using TensorIndex = array<index_t, TensorDesc::get_num_of_top_dimension()>;
    using TensorCoord = decltype(make_tensor_coordinate(TensorDesc{}, TensorIndex{}));
    static constexpr auto DstInMemOp = DstInMemOp_;

    CK_TILE_HOST_DEVICE constexpr hcu_tensor_view() = default;

    CK_TILE_HOST_DEVICE constexpr hcu_tensor_view(const buffer_view& buffer_view,
                                                  const TensorDesc& desc)
        : buf_{buffer_view}, desc_{desc}
    {
    }

    CK_TILE_HOST_DEVICE constexpr auto& get_tensor_descriptor() const { return desc_; }

    CK_TILE_HOST_DEVICE static constexpr index_t get_num_of_dimension()
    {
        return TensorDesc::get_num_of_top_dimension();
    }

    CK_TILE_HOST_DEVICE constexpr const auto& get_buffer_view() const { return buf_; }

    CK_TILE_HOST_DEVICE constexpr auto& get_buffer_view() { return buf_; }

    template <typename X,
              bool oob_conditional_check = true,
              typename std::enable_if<
                  std::is_same_v<typename vector_traits<remove_cvref_t<X>>::scalar_type,
                                 typename vector_traits<remove_cvref_t<DataType>>::scalar_type>,
                  bool>::type = false>
    CK_TILE_DEVICE constexpr void
    async_get_vectorized_elements_asm(CK_TILE_LDS_ADDR remove_cvref_t<DataType>* smem,
                                      index_t element_offset,
                                      index_t linear_offset,
                                      bool is_valid,
                                      bool_constant<oob_conditional_check> = {}) const
    {
        if constexpr(buffer_view::is_struct_buffer())
        {
            index_t byte_offset = element_offset * sizeof(DataType);
            index_t vindex      = byte_offset >> int(__builtin_log2f(buf_.stride_));
            index_t voffset     = byte_offset & (buf_.stride_ - 1);
            index_t ioffset     = linear_offset * sizeof(DataType);

            buf_.template async_get_asm<X>(
                smem, vindex, voffset, ioffset, is_valid, bool_constant<oob_conditional_check>{});
        }
        else
        {
            index_t voffset = element_offset * sizeof(DataType);
            index_t ioffset = linear_offset * sizeof(DataType);

            buf_.template async_get_asm<X>(
                smem, voffset, ioffset, is_valid, bool_constant<oob_conditional_check>{});
        }
    }

    template <typename X,
              bool oob_conditional_check = true,
              typename std::enable_if<
                  std::is_same_v<typename vector_traits<remove_cvref_t<X>>::scalar_type,
                                 typename vector_traits<remove_cvref_t<DataType>>::scalar_type>,
                  bool>::type = false>
    CK_TILE_DEVICE constexpr void
    async_get_vectorized_elements_asm(CK_TILE_LDS_ADDR remove_cvref_t<DataType>* smem,
                                      const TensorCoord& coord,
                                      index_t linear_offset,
                                      bool is_valid,
                                      bool_constant<oob_conditional_check> = {}) const
    {
        async_get_vectorized_elements_asm(smem,
                                          coord.get_offset(),
                                          linear_offset,
                                          is_valid,
                                          bool_constant<oob_conditional_check>{});
    }

    template <typename X,
              bool oob_conditional_check = true,
              typename std::enable_if<
                  std::is_same_v<typename vector_traits<remove_cvref_t<X>>::scalar_type,
                                 typename vector_traits<remove_cvref_t<DataType>>::scalar_type>,
                  bool>::type = false>
    CK_TILE_DEVICE constexpr void
    async_get_vectorized_elements_asm(CK_TILE_LDS_ADDR remove_cvref_t<DataType>* smem,
                                      const TensorCoord& coord,
                                      index_t linear_offset,
                                      bool_constant<oob_conditional_check> = {}) const
    {
        async_get_vectorized_elements_asm(
            smem,
            coord,
            linear_offset,
            coordinate_has_valid_offset_assuming_top_index_is_valid(desc_, coord),
            bool_constant<oob_conditional_check>{});
    }

    template <typename X,
              bool oob_conditional_check = true,
              typename std::enable_if<
                  std::is_same_v<typename vector_traits<remove_cvref_t<X>>::scalar_type,
                                 typename vector_traits<remove_cvref_t<DataType>>::scalar_type>,
                  bool>::type = false>
    CK_TILE_DEVICE constexpr void
    async_get_vectorized_elements_wrapped_asm(CK_TILE_LDS_ADDR remove_cvref_t<DataType>* smem,
                                              index_t element_offset,
                                              index_t linear_offset,
                                              index_t lds_wrap_offset,
                                              bool is_valid,
                                              bool_constant<oob_conditional_check> = {}) const
    {
        if constexpr(buffer_view::is_struct_buffer())
        {
            index_t byte_offset = element_offset * sizeof(DataType);
            index_t vindex      = byte_offset >> int(__builtin_log2f(buf_.stride_));
            index_t voffset     = byte_offset & (buf_.stride_ - 1);
            index_t ioffset     = linear_offset * sizeof(DataType);

            buf_.template async_get_wrapped_asm<X>(smem,
                                                   vindex,
                                                   voffset,
                                                   ioffset,
                                                   lds_wrap_offset,
                                                   is_valid,
                                                   bool_constant<oob_conditional_check>{});
        }
        else
        {
            index_t voffset = element_offset * sizeof(DataType);
            index_t ioffset = linear_offset * sizeof(DataType);

            buf_.template async_get_wrapped_asm<X>(smem,
                                                   voffset,
                                                   ioffset,
                                                   lds_wrap_offset,
                                                   is_valid,
                                                   bool_constant<oob_conditional_check>{});
        }
    }

    template <typename X,
              bool oob_conditional_check = true,
              typename std::enable_if<
                  std::is_same_v<typename vector_traits<remove_cvref_t<X>>::scalar_type,
                                 typename vector_traits<remove_cvref_t<DataType>>::scalar_type>,
                  bool>::type = false>
    CK_TILE_DEVICE constexpr void
    async_get_vectorized_elements_wrapped_asm(CK_TILE_LDS_ADDR remove_cvref_t<DataType>* smem,
                                              const TensorCoord& coord,
                                              index_t linear_offset,
                                              index_t lds_wrap_offset,
                                              bool is_valid,
                                              bool_constant<oob_conditional_check> = {}) const
    {
        async_get_vectorized_elements_wrapped_asm(smem,
                                                  coord.get_offset(),
                                                  linear_offset,
                                                  lds_wrap_offset,
                                                  is_valid,
                                                  bool_constant<oob_conditional_check>{});
    }

    template <typename X,
              bool oob_conditional_check = true,
              typename std::enable_if<
                  std::is_same_v<typename vector_traits<remove_cvref_t<X>>::scalar_type,
                                 typename vector_traits<remove_cvref_t<DataType>>::scalar_type>,
                  bool>::type = false>
    CK_TILE_DEVICE constexpr void
    async_get_vectorized_elements_wrapped_asm(CK_TILE_LDS_ADDR remove_cvref_t<DataType>* smem,
                                              const TensorCoord& coord,
                                              index_t linear_offset,
                                              index_t lds_wrap_offset,
                                              bool_constant<oob_conditional_check> = {}) const
    {
        async_get_vectorized_elements_wrapped_asm(
            smem,
            coord,
            linear_offset,
            lds_wrap_offset,
            coordinate_has_valid_offset_assuming_top_index_is_valid(desc_, coord),
            bool_constant<oob_conditional_check>{});
    }

    template <typename X,
              bool oob_conditional_check = true,
              typename std::enable_if<
                  std::is_same_v<typename vector_traits<remove_cvref_t<X>>::scalar_type,
                                 typename vector_traits<remove_cvref_t<DataType>>::scalar_type>,
                  bool>::type = false>
    CK_TILE_DEVICE constexpr remove_cvref_t<X>
    get_vectorized_elements_asm(index_t element_offset,
                                index_t linear_offset,
                                bool is_valid,
                                bool_constant<oob_conditional_check> = {}) const
    {
        if constexpr(buffer_view::is_struct_buffer())
        {
            index_t byte_offset = element_offset * sizeof(DataType);
            index_t vindex      = byte_offset >> int(__builtin_log2f(buf_.stride_));
            index_t voffset     = byte_offset & (buf_.stride_ - 1);
            index_t ioffset     = linear_offset * sizeof(DataType);

            return buf_.template get_asm<X>(
                vindex, voffset, ioffset, is_valid, bool_constant<oob_conditional_check>{});
        }
        else
        {
            index_t voffset = element_offset * sizeof(DataType);
            index_t ioffset = linear_offset * sizeof(DataType);

            return buf_.template get_asm<X>(
                voffset, ioffset, is_valid, bool_constant<oob_conditional_check>{});
        }
    }

    template <typename X,
              bool oob_conditional_check = true,
              typename std::enable_if<
                  std::is_same_v<typename vector_traits<remove_cvref_t<X>>::scalar_type,
                                 typename vector_traits<remove_cvref_t<DataType>>::scalar_type>,
                  bool>::type = false>
    CK_TILE_DEVICE constexpr remove_cvref_t<X>
    get_vectorized_elements_asm(const TensorCoord& coord,
                                index_t linear_offset,
                                bool is_valid,
                                bool_constant<oob_conditional_check> = {}) const
    {
        return get_vectorized_elements_asm<X>(
            coord.get_offset(), linear_offset, is_valid, bool_constant<oob_conditional_check>{});
    }

    template <typename X,
              bool oob_conditional_check = true,
              typename std::enable_if<
                  std::is_same_v<typename vector_traits<remove_cvref_t<X>>::scalar_type,
                                 typename vector_traits<remove_cvref_t<DataType>>::scalar_type>,
                  bool>::type = false>
    CK_TILE_DEVICE constexpr remove_cvref_t<X>
    get_vectorized_elements_asm(const TensorCoord& coord,
                                index_t linear_offset,
                                bool_constant<oob_conditional_check> = {}) const
    {
        return get_vectorized_elements_asm<X>(
            coord,
            linear_offset,
            coordinate_has_valid_offset_assuming_top_index_is_valid(desc_, coord),
            bool_constant<oob_conditional_check>{});
    }

    CK_TILE_HOST_DEVICE void print() const
    {
        printf("hcu_tensor_view{");

        // buf_
        printf("buf_: ");
        print(buf_);
        printf(", ");

        // desc_
        printf("desc_: ");
        print(desc_);

        printf("}");
    }

    // member
    buffer_view buf_;
    TensorDesc desc_;
};

template <address_space_enum BufferAddressSpace = address_space_enum::global,
          typename DataType,
          typename... Ts>
CK_TILE_HOST_DEVICE constexpr auto make_hcu_tensor_view(DataType* p, tensor_descriptor<Ts...>& desc)
{
    auto buffer_view = make_hcu_buffer_view<BufferAddressSpace>(p, desc.get_element_space_size());

    return hcu_tensor_view<decltype(buffer_view), decltype(desc)>{buffer_view, desc};
}

template <address_space_enum BufferAddressSpace = address_space_enum::global,
          typename DataType,
          typename... Ts>
CK_TILE_HOST_DEVICE constexpr auto
make_hcu_tensor_view(DataType* p, tensor_descriptor<Ts...>& desc, index_t stride)
{
    static_assert(BufferAddressSpace == address_space_enum::global);

    auto buffer_view =
        make_hcu_buffer_view<BufferAddressSpace>(p, desc.get_element_space_size(), stride);

    return hcu_tensor_view<decltype(buffer_view), decltype(desc)>{buffer_view, desc};
}

} // namespace ck_tile
