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
#include "ck_tile/core/tensor/tensor_descriptor.hpp"
#include "ck_tile/core/utility/functional.hpp"
#include "ck_tile/core/utility/type_traits.hpp"

namespace ck_tile {

/*
 * tensor_view
 * abstract the underneath memory buffer(global, LDS, etc...)
 * and provide a unified get/set function for access
 *
 * For addressing into the buffer we use 2 variable to control:
 * coord : ND tensor coordinate, will calculate the actual offset inside
 * linear_offset : 1D offset, will be used in the immediate field of
 *   the buffer instruction to help reduce register usage
 *
 * User can use either of the field, or both to indexing into the tensor
 *
 * We usually provide 2 set of API for buffer get/set, e.g.
 * get_vectorized_elements()/get_vectorized_elements_raw()
 * the former usually will call intrinsic or normal C function, the later
 * usually will call inline-asm function
 *
 */
template <typename BufferView_,
          typename TensorDesc_,
          memory_operation_enum DstInMemOp_ = memory_operation_enum::set>
struct tensor_view
{
    using buffer_view = remove_reference_t<BufferView_>;
    using DataType    = typename buffer_view::type;
    using TensorDesc  = remove_cvref_t<TensorDesc_>;
    using TensorIndex = array<index_t, TensorDesc::get_num_of_top_dimension()>;
    using TensorCoord = decltype(make_tensor_coordinate(TensorDesc{}, TensorIndex{}));
    static constexpr auto DstInMemOp = DstInMemOp_;

    CK_TILE_HOST_DEVICE constexpr tensor_view() = default;

    CK_TILE_HOST_DEVICE constexpr tensor_view(const buffer_view& buffer_view,
                                              const TensorDesc& desc)
        : buf_{buffer_view}, desc_{desc}
    {
    }

    CK_TILE_HOST_DEVICE void init_raw() { buf_.init_raw(); }

    CK_TILE_HOST_DEVICE constexpr auto& get_tensor_descriptor() const { return desc_; }

    CK_TILE_HOST_DEVICE static constexpr index_t get_num_of_dimension()
    {
        return TensorDesc::get_num_of_top_dimension();
    }

    CK_TILE_HOST_DEVICE constexpr const auto& get_buffer_view() const { return buf_; }

    CK_TILE_HOST_DEVICE constexpr auto& get_buffer_view() { return buf_; }

    // X is vector of DataType.
    // "coord" is coordinate of DataType, not X. "coord" should be aligned to X
    template <typename X,
              bool oob_conditional_check = true,
              bool swizzle               = false,
              typename std::enable_if<
                  std::is_same_v<typename vector_traits<remove_cvref_t<X>>::scalar_type,
                                 typename vector_traits<remove_cvref_t<DataType>>::scalar_type>,
                  bool>::type = false>
    CK_TILE_HOST_DEVICE constexpr remove_cvref_t<X>
    get_vectorized_elements(const TensorCoord& coord,
                            index_t linear_offset,
                            bool_constant<oob_conditional_check> = {},
                            bool_constant<swizzle>               = {}) const
    {
        return get_vectorized_elements<X>(
            coord,
            linear_offset,
            coordinate_has_valid_offset_assuming_top_index_is_valid(desc_, coord),
            bool_constant<oob_conditional_check>{},
            bool_constant<swizzle>{});
    }

    template <typename X,
              bool oob_conditional_check = true,
              bool swizzle               = false,
              typename std::enable_if<
                  std::is_same_v<typename vector_traits<remove_cvref_t<X>>::scalar_type,
                                 typename vector_traits<remove_cvref_t<DataType>>::scalar_type>,
                  bool>::type = false>
    CK_TILE_HOST_DEVICE constexpr remove_cvref_t<X>
    get_vectorized_elements(const TensorCoord& coord,
                            index_t linear_offset,
                            bool is_valid_element, // flag
                            bool_constant<oob_conditional_check> = {},
                            bool_constant<swizzle>               = {}) const
    {
        if constexpr(swizzle)
        {
            constexpr auto swizzle_functor = make_swizzle(number<sizeof(X)>{});
            return buf_.template get<X>(swizzle_functor(coord.get_offset()),
                                        linear_offset,
                                        is_valid_element,
                                        bool_constant<oob_conditional_check>{});
        }
        else
        {
            return buf_.template get<X>(coord.get_offset(),
                                        linear_offset,
                                        is_valid_element,
                                        bool_constant<oob_conditional_check>{});
        }
    }

    template <typename X,
              bool oob_conditional_check = true,
              typename std::enable_if<
                  std::is_same_v<typename vector_traits<remove_cvref_t<X>>::scalar_type,
                                 typename vector_traits<remove_cvref_t<DataType>>::scalar_type>,
                  bool>::type = false>
    CK_TILE_HOST_DEVICE constexpr remove_cvref_t<X>
    get_vectorized_elements_by_inline_asm(const TensorCoord& coord,
                                          index_t linear_offset,
                                          bool_constant<oob_conditional_check> = {}) const
    {
        return buf_.template get_by_inline_asm<X>(
            coord.get_offset(),
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
    CK_TILE_HOST_DEVICE constexpr remove_cvref_t<X>
    get_vectorized_elements_by_inline_asm(const TensorCoord& coord,
                                          index_t linear_offset,
                                          bool is_valid_element, // flag
                                          bool_constant<oob_conditional_check> = {}) const
    {
        return buf_.template get_by_inline_asm<X>(coord.get_offset(),
                                                  linear_offset,
                                                  is_valid_element,
                                                  bool_constant<oob_conditional_check>{});
    }

    template <typename X,
              bool oob_conditional_check = true,
              bool cache_swizzle         = false,
              typename std::enable_if<
                  std::is_same_v<typename vector_traits<remove_cvref_t<X>>::scalar_type,
                                 typename vector_traits<remove_cvref_t<DataType>>::scalar_type>,
                  bool>::type = false>
    CK_TILE_HOST_DEVICE constexpr remove_cvref_t<X>
    get_vectorized_elements_by_inline_asm(const index_t element_offset,
                                          index_t linear_offset,
                                          bool is_valid_element, // flag
                                          bool_constant<oob_conditional_check> = {},
                                          bool_constant<cache_swizzle>         = {}) const
    {
        return buf_.template get_by_inline_asm<X>(element_offset,
                                                  linear_offset,
                                                  is_valid_element,
                                                  bool_constant<oob_conditional_check>{},
                                                  bool_constant<cache_swizzle>{});
    }

    template <typename X,
              typename DsreadmInstr,
              typename std::enable_if<
                  std::is_same_v<typename vector_traits<remove_cvref_t<X>>::scalar_type,
                                 typename vector_traits<remove_cvref_t<DataType>>::scalar_type>,
                  bool>::type = false>
    CK_TILE_HOST_DEVICE constexpr remove_cvref_t<X> get_vectorized_elements_by_dsreadm(
        const DsreadmInstr& instr, const TensorCoord& coord, index_t linear_offset) const
    {
        return buf_.template get_by_dsreadm<X>(instr, coord.get_offset(), linear_offset);
    }

    // X is vector of DataType.
    // "coord" is coordinate of DataType, not X. "coord" should be aligned to X
    template <typename X,
              bool oob_conditional_check = true,
              bool pre_nop               = false,
              bool swizzle               = false,
              typename std::enable_if<
                  std::is_same_v<typename vector_traits<remove_cvref_t<X>>::scalar_type,
                                 typename vector_traits<remove_cvref_t<DataType>>::scalar_type>,
                  bool>::type = false>
    CK_TILE_HOST_DEVICE void get_vectorized_elements_raw(remove_cvref_t<X>& dst,
                                                         const TensorCoord& coord,
                                                         index_t linear_offset,
                                                         bool_constant<oob_conditional_check> = {},
                                                         bool_constant<pre_nop>               = {},
                                                         bool_constant<swizzle> = {}) const
    {
        if constexpr(swizzle)
        {
            constexpr auto swizzle_functor = make_swizzle(number<sizeof(X)>{});

            return buf_.template get_raw<X, oob_conditional_check, pre_nop>(
                dst,
                swizzle_functor(coord.get_offset()),
                linear_offset,
                coordinate_has_valid_offset_assuming_top_index_is_valid(desc_, coord),
                bool_constant<pre_nop>{});
        }
        else
        {
            return buf_.template get_raw<X, oob_conditional_check, pre_nop>(
                dst,
                coord.get_offset(),
                linear_offset,
                coordinate_has_valid_offset_assuming_top_index_is_valid(desc_, coord),
                bool_constant<pre_nop>{});
        }
    }

    template <typename X,
              bool oob_conditional_check = true,
              bool pre_nop               = false,
              bool swizzle               = false,
              typename std::enable_if<
                  std::is_same_v<typename vector_traits<remove_cvref_t<X>>::scalar_type,
                                 typename vector_traits<remove_cvref_t<DataType>>::scalar_type>,
                  bool>::type = false>
    CK_TILE_HOST_DEVICE void get_vectorized_elements_raw(remove_cvref_t<X>& dst,
                                                         const TensorCoord& coord,
                                                         index_t linear_offset,
                                                         bool is_valid_element,
                                                         bool_constant<oob_conditional_check> = {},
                                                         bool_constant<pre_nop>               = {},
                                                         bool_constant<swizzle> = {}) const
    {
        if constexpr(swizzle)
        {
            constexpr auto swizzle_functor = make_swizzle(number<sizeof(X)>{});

            return buf_.template get_raw<X, oob_conditional_check, pre_nop>(
                dst,
                swizzle_functor(coord.get_offset()),
                linear_offset,
                is_valid_element,
                bool_constant<pre_nop>{});
        }
        else
        {
            return buf_.template get_raw<X, oob_conditional_check, pre_nop>(
                dst, coord.get_offset(), linear_offset, is_valid_element, bool_constant<pre_nop>{});
        }
    }

    // X is vector of DataType.
    // "coord" is coordinate of DataType, not X. "coord" should be aligned to X
    template <typename X,
              typename std::enable_if<
                  std::is_same_v<typename vector_traits<remove_cvref_t<X>>::scalar_type,
                                 typename vector_traits<remove_cvref_t<DataType>>::scalar_type>,
                  bool>::type = false>
    CK_TILE_HOST_DEVICE void get_vectorized_elements_raw(remove_cvref_t<X>& dst,
                                                         index_t offset) const
    {

        return buf_.template get_raw<X, false, false>(dst, offset, 0, true, bool_constant<false>{});
    }

    template <typename X,
              bool oob_conditional_check = true,
              typename std::enable_if<
                  std::is_same_v<typename vector_traits<remove_cvref_t<X>>::scalar_type,
                                 typename vector_traits<remove_cvref_t<DataType>>::scalar_type>,
                  bool>::type = false>
    CK_TILE_HOST_DEVICE constexpr void
    async_get_vectorized_elements(CK_TILE_LDS_ADDR remove_cvref_t<DataType>* smem,
                                  const TensorCoord& coord,
                                  index_t linear_offset,
                                  bool_constant<oob_conditional_check> = {}) const
    {
        return buf_.template async_get<X>(
            smem,
            coord.get_offset(),
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
    CK_TILE_HOST_DEVICE constexpr void
    async_get_vectorized_elements(CK_TILE_LDS_ADDR remove_cvref_t<DataType>* smem,
                                  const TensorCoord& coord,
                                  index_t linear_offset,
                                  bool is_valid_element,
                                  bool_constant<oob_conditional_check> = {}) const
    {
        return buf_.template async_get<X>(smem,
                                          coord.get_offset(),
                                          linear_offset,
                                          is_valid_element,
                                          bool_constant<oob_conditional_check>{});
    }

    template <typename X,
              bool oob_conditional_check = true,
              typename std::enable_if<
                  std::is_same_v<typename vector_traits<remove_cvref_t<X>>::scalar_type,
                                 typename vector_traits<remove_cvref_t<DataType>>::scalar_type>,
                  bool>::type = false>
    CK_TILE_HOST_DEVICE constexpr void
    async_get_vectorized_elements_by_inline_asm(CK_TILE_LDS_ADDR remove_cvref_t<DataType>* smem,
                                                const TensorCoord& coord,
                                                index_t linear_offset,
                                                bool_constant<oob_conditional_check> = {}) const
    {
        return buf_.template async_get_by_inline_asm<X>(
            smem,
            coord.get_offset(),
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
    CK_TILE_HOST_DEVICE constexpr void
    async_get_vectorized_elements_by_inline_asm(CK_TILE_LDS_ADDR remove_cvref_t<DataType>* smem,
                                                const TensorCoord& coord,
                                                index_t linear_offset,
                                                bool is_valid_element,
                                                bool_constant<oob_conditional_check> = {}) const
    {
        return buf_.template async_get_by_inline_asm<X>(smem,
                                                        coord.get_offset(),
                                                        linear_offset,
                                                        is_valid_element,
                                                        bool_constant<oob_conditional_check>{});
    }

    template <typename X,
              bool pre_nop = false,
              typename std::enable_if<
                  std::is_same_v<typename vector_traits<remove_cvref_t<X>>::scalar_type,
                                 typename vector_traits<remove_cvref_t<DataType>>::scalar_type>,
                  bool>::type = false>
    CK_TILE_HOST_DEVICE constexpr void
    async_get_vectorized_elements_raw(remove_cvref_t<DataType>* smem,
                                      const TensorCoord& coord,
                                      index_t linear_offset,
                                      bool_constant<pre_nop> = {}) const
    {
        return buf_.template async_get_raw<X>(
            smem,
            coord.get_offset(),
            linear_offset,
            coordinate_has_valid_offset_assuming_top_index_is_valid(desc_, coord),
            bool_constant<pre_nop>{});
    }

    template <typename X,
              bool pre_nop = false,
              typename std::enable_if<
                  std::is_same_v<typename vector_traits<remove_cvref_t<X>>::scalar_type,
                                 typename vector_traits<remove_cvref_t<DataType>>::scalar_type>,
                  bool>::type = false>
    CK_TILE_HOST_DEVICE constexpr void
    async_get_vectorized_elements_raw(remove_cvref_t<DataType>* smem,
                                      const TensorCoord& coord,
                                      index_t linear_offset,
                                      bool is_valid_element,
                                      bool_constant<pre_nop> = {}) const
    {
        return buf_.template async_get_raw<X>(
            smem, coord.get_offset(), linear_offset, is_valid_element, bool_constant<pre_nop>{});
    }

    // X is vector of DataType.
    // "coord" is coordinate of DataType, not X. "coord" should be aligned to X
    template <typename X,
              bool oob_conditional_check = true,
              bool swizzle               = false,
              typename std::enable_if<
                  std::is_same_v<typename vector_traits<remove_cvref_t<X>>::scalar_type,
                                 typename vector_traits<remove_cvref_t<DataType>>::scalar_type>,
                  bool>::type = false>
    CK_TILE_HOST_DEVICE constexpr void
    set_vectorized_elements(const TensorCoord& coord,
                            index_t linear_offset,
                            const X& x,
                            bool_constant<oob_conditional_check> = {},
                            bool_constant<swizzle>               = {})
    {
        set_vectorized_elements(
            coord,
            linear_offset,
            coordinate_has_valid_offset_assuming_top_index_is_valid(desc_, coord),
            x,
            bool_constant<oob_conditional_check>{},
            bool_constant<swizzle>{});
    }

    template <typename X,
              bool oob_conditional_check = true,
              bool swizzle               = false,
              typename std::enable_if<
                  std::is_same_v<typename vector_traits<remove_cvref_t<X>>::scalar_type,
                                 typename vector_traits<remove_cvref_t<DataType>>::scalar_type>,
                  bool>::type = false>
    CK_TILE_HOST_DEVICE constexpr void
    set_vectorized_elements(const TensorCoord& coord,
                            index_t linear_offset,
                            bool is_valid_element,
                            const X& x,
                            bool_constant<oob_conditional_check> = {},
                            bool_constant<swizzle>               = {})
    {
        if constexpr(swizzle)
        {
            constexpr auto swizzle_functor = make_swizzle(number<sizeof(X)>{});
            buf_.template set<X, oob_conditional_check>(
                swizzle_functor(coord.get_offset()), linear_offset, is_valid_element, x);
        }
        else
        {
            buf_.template set<X, oob_conditional_check>(
                coord.get_offset(), linear_offset, is_valid_element, x);
        }
    }

    template <typename X,
              bool oob_conditional_check = true,
              typename std::enable_if<
                  std::is_same_v<typename vector_traits<remove_cvref_t<X>>::scalar_type,
                                 typename vector_traits<remove_cvref_t<DataType>>::scalar_type>,
                  bool>::type = false>
    CK_TILE_HOST_DEVICE constexpr void
    set_vectorized_elements_raw(const TensorCoord& coord,
                                index_t linear_offset,
                                const X& x,
                                bool_constant<oob_conditional_check> = {})
    {
        buf_.template set_raw<X, oob_conditional_check>(
            coord.get_offset(),
            linear_offset,
            coordinate_has_valid_offset_assuming_top_index_is_valid(desc_, coord),
            x);
    }

    template <typename X,
              bool oob_conditional_check = true,
              typename std::enable_if<
                  std::is_same_v<typename vector_traits<remove_cvref_t<X>>::scalar_type,
                                 typename vector_traits<remove_cvref_t<DataType>>::scalar_type>,
                  bool>::type = false>
    CK_TILE_HOST_DEVICE constexpr void
    set_vectorized_elements_raw(const TensorCoord& coord,
                                index_t linear_offset,
                                bool is_valid_element,
                                const X& x,
                                bool_constant<oob_conditional_check> = {})
    {
        buf_.template set_raw<X, oob_conditional_check>(
            coord.get_offset(), linear_offset, is_valid_element, x);
    }

    // X is vector of DataType.
    // "coord" is coordinate of DataType, not X. "coord" should be aligned to X
    template <typename X,
              bool oob_conditional_check = true,
              typename std::enable_if<
                  std::is_same_v<typename vector_traits<remove_cvref_t<X>>::scalar_type,
                                 typename vector_traits<remove_cvref_t<DataType>>::scalar_type>,
                  bool>::type = false>
    CK_TILE_HOST_DEVICE constexpr void
    update_vectorized_elements(const TensorCoord& coord,
                               index_t linear_offset,
                               const X& x,
                               bool_constant<oob_conditional_check> = {})
    {
        buf_.template update<DstInMemOp, X, oob_conditional_check>(
            coord.get_offset(),
            linear_offset,
            coordinate_has_valid_offset_assuming_top_index_is_valid(desc_, coord),
            x);
    }

    template <typename X,
              bool oob_conditional_check = true,
              typename std::enable_if<
                  std::is_same_v<typename vector_traits<remove_cvref_t<X>>::scalar_type,
                                 typename vector_traits<remove_cvref_t<DataType>>::scalar_type>,
                  bool>::type = false>
    CK_TILE_HOST_DEVICE constexpr void
    update_vectorized_elements(const TensorCoord& coord,
                               index_t linear_offset,
                               bool is_valid_element,
                               const X& x,
                               bool_constant<oob_conditional_check> = {})
    {
        buf_.template update<DstInMemOp, X, oob_conditional_check>(
            coord.get_offset(), linear_offset, is_valid_element, x);
    }

    CK_TILE_HOST_DEVICE void print() const
    {
        printf("tensor_view{");

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

// placeholder type if we want to opt-out a tile view parameter
struct null_tensor_view
{
};

template <address_space_enum BufferAddressSpace = address_space_enum::generic,
          typename DataType,
          typename... Ts>
CK_TILE_HOST_DEVICE constexpr auto make_tensor_view(DataType* p,
                                                    const tensor_descriptor<Ts...>& desc)
{
    auto buffer_view = make_buffer_view<BufferAddressSpace>(p, desc.get_element_space_size());

    return tensor_view<decltype(buffer_view), decltype(desc)>{buffer_view, desc};
}

template <address_space_enum BufferAddressSpace = address_space_enum::global,
          typename DataType,
          typename... Ts>
CK_TILE_HOST_DEVICE constexpr auto
make_tensor_view(DataType* p, const tensor_descriptor<Ts...>& desc, const index_t const_stride)
{
    auto buffer_view = make_buffer_view<BufferAddressSpace>(
        p, desc.get_element_space_size(), DataType(0), const_stride);

    return tensor_view<decltype(buffer_view), decltype(desc)>{buffer_view, desc};
}

template <address_space_enum BufferAddressSpace = address_space_enum::generic,
          memory_operation_enum DstInMemOp      = memory_operation_enum::set,
          typename DataType,
          typename... Lengths,
          typename... Strides,
          index_t GuaranteedLastDimensionVectorLength                                   = -1,
          index_t GuaranteedLastDimensionVectorStride                                   = -1,
          typename std::enable_if<sizeof...(Lengths) == sizeof...(Strides), bool>::type = false>
CK_TILE_HOST_DEVICE constexpr auto
make_naive_tensor_view(DataType* p,
                       const tuple<Lengths...>& lengths,
                       const tuple<Strides...>& strides,
                       number<GuaranteedLastDimensionVectorLength> = number<-1>{},
                       number<GuaranteedLastDimensionVectorStride> = number<-1>{})
{
    auto desc = make_naive_tensor_descriptor(lengths,
                                             strides,
                                             number<GuaranteedLastDimensionVectorLength>{},
                                             number<GuaranteedLastDimensionVectorStride>{});

    auto buffer_view = make_buffer_view<BufferAddressSpace>(p, desc.get_element_space_size());

    return tensor_view<decltype(buffer_view), decltype(desc), DstInMemOp>{buffer_view, desc};
}

template <address_space_enum BufferAddressSpace = address_space_enum::generic,
          typename DataType,
          typename... Lengths,
          index_t GuaranteedLastDimensionVectorLength = -1>
CK_TILE_HOST_DEVICE constexpr auto
make_naive_tensor_view_packed(DataType* p,
                              const tuple<Lengths...>& lengths,
                              number<GuaranteedLastDimensionVectorLength> = number<-1>{})
{
    auto desc =
        make_naive_tensor_descriptor_packed(lengths, number<GuaranteedLastDimensionVectorLength>{});

    auto buffer_view = make_buffer_view<BufferAddressSpace>(p, desc.get_element_space_size());

    return tensor_view<decltype(buffer_view), decltype(desc)>{buffer_view, desc};
}

template <typename OldTensorView,
          typename NewTransforms,
          typename NewLowerDimensionOldVisibleIdss,
          typename NewUpperDimensionNewVisibleIdss>
CK_TILE_HOST_DEVICE constexpr auto transform_tensor_view(const OldTensorView& old_tensor_view,
                                                         const NewTransforms& new_transforms,
                                                         NewLowerDimensionOldVisibleIdss,
                                                         NewUpperDimensionNewVisibleIdss)
{
    auto new_desc = transform_tensor_descriptor(old_tensor_view.desc_,
                                                new_transforms,
                                                NewLowerDimensionOldVisibleIdss{},
                                                NewUpperDimensionNewVisibleIdss{});

    return tensor_view<typename OldTensorView::buffer_view,
                       remove_cvref_t<decltype(new_desc)>,
                       remove_cvref_t<OldTensorView>::DstInMemOp>{old_tensor_view.buf_, new_desc};
}

template <typename TensorView,
          typename TileLengths, // tuple<...>
          typename DoPads>      // sequence<bool, bool, ...>
CK_TILE_HOST_DEVICE constexpr auto
pad_tensor_view(const TensorView& tensor_view, const TileLengths& tile_lengths, DoPads)
{
    constexpr index_t num_dim = DoPads::size();

    static_assert(num_dim == TileLengths::size() && num_dim == TensorView::get_num_of_dimension(),
                  "wrong! inconsistent # of dimensions");

    // transforms
    const auto transforms = generate_tuple(
        [&](auto idim) {
            const auto old_length = tensor_view.get_tensor_descriptor().get_length(idim);

            const auto tile_length = tile_lengths[idim];

            const auto new_length = integer_divide_ceil(old_length, tile_length) * tile_length;

            const auto pad_length = new_length - old_length;

            constexpr bool DoPad = DoPads::at(idim);

            const auto transform =
                conditional_expr<DoPad>(make_right_pad_transform(old_length, pad_length),
                                        make_pass_through_transform(old_length));

            return transform;
        },
        number<num_dim>{});

    // lower dimension Id
    const auto lower_dimss =
        generate_tuple([&](auto idim) { return sequence<idim.value>{}; }, number<num_dim>{});

    // upper dimension Id
    const auto upper_dimss = lower_dimss;

    return transform_tensor_view(tensor_view, transforms, lower_dimss, upper_dimss);
}

} // namespace ck_tile
