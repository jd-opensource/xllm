// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, , Inc. All rights reserved.

#pragma once

#include "ck_tile/core/config.hpp"
#include "ck_tile/core/arch/arch.hpp"
#include "ck_tile/core/arch/generic_memory_space_atomic.hpp"
#include "ck_tile/core/arch/hcu_buffer_addressing.hpp"
#include "ck_tile/core/container/array.hpp"
#include "ck_tile/core/numeric/integer.hpp"
#include "ck_tile/core/numeric/integral_constant.hpp"
#include "ck_tile/core/numeric/float8.hpp"
#include "ck_tile/core/numeric/half.hpp"
#include "ck_tile/core/numeric/bfloat16.hpp"
#include "ck_tile/core/utility/type_traits.hpp"

namespace ck_tile {

template <address_space_enum BufferAddressSpace,
          typename T,
          typename BufferSizeType,
          bool StructBuffer>
struct hcu_buffer_view;

template <typename T, typename BufferSizeType>
struct hcu_buffer_view<address_space_enum::global, T, BufferSizeType, false>
{
    using type = T;

    T* p_data_ = nullptr;
    BufferSizeType buffer_size_;
    int32x4_t cached_buf_res_;
    remove_cvref_t<T> invalid_element_value_ = T{0};

    CK_TILE_HOST_DEVICE constexpr hcu_buffer_view(T* p_data, BufferSizeType buffer_size)
        : p_data_{p_data}, buffer_size_{buffer_size}, cached_buf_res_{0}
    {
        cached_buf_res_ = make_wave_buffer_resource(p_data_, buffer_size * sizeof(type));
    }

    CK_TILE_DEVICE static constexpr address_space_enum get_address_space()
    {
        return address_space_enum::global;
    }

    CK_TILE_DEVICE static constexpr bool is_struct_buffer() { return false; }

    template <typename X,
              bool oob_conditional_check = true,
              typename std::enable_if<
                  std::is_same<typename vector_traits<remove_cvref_t<X>>::scalar_type,
                               typename vector_traits<remove_cvref_t<T>>::scalar_type>::value,
                  bool>::type = false>
    CK_TILE_DEVICE constexpr void async_get_asm(CK_TILE_LDS_ADDR remove_cvref_t<T>* smem,
                                                index_t voffset,
                                                index_t ioffset,
                                                bool is_valid,
                                                bool_constant<oob_conditional_check> = {}) const
    {
        // X is vector of T
        constexpr index_t scalar_per_t_vector = vector_traits<remove_cvref_t<T>>::vector_size;
        constexpr index_t scalar_per_x_vector = vector_traits<remove_cvref_t<X>>::vector_size;

        static_assert(scalar_per_x_vector % scalar_per_t_vector == 0,
                      "wrong! X should contain multiple T");

        constexpr index_t t_per_x = scalar_per_x_vector / scalar_per_t_vector;

        hcu_async_buffer_load_asm<remove_cvref_t<T>, t_per_x>(
            smem,
            cached_buf_res_,
            voffset,
            ioffset,
            is_valid,
            bool_constant<oob_conditional_check>{});
    }

    template <typename X,
              bool oob_conditional_check = true,
              typename std::enable_if<
                  std::is_same<typename vector_traits<remove_cvref_t<X>>::scalar_type,
                               typename vector_traits<remove_cvref_t<T>>::scalar_type>::value,
                  bool>::type = false>
    CK_TILE_DEVICE constexpr void
    async_get_wrapped_asm(CK_TILE_LDS_ADDR remove_cvref_t<T>* smem,
                          index_t voffset,
                          index_t ioffset,
                          index_t lds_wrap_offset,
                          bool is_valid,
                          bool_constant<oob_conditional_check> = {}) const
    {
        // X is vector of T
        constexpr index_t scalar_per_t_vector = vector_traits<remove_cvref_t<T>>::vector_size;
        constexpr index_t scalar_per_x_vector = vector_traits<remove_cvref_t<X>>::vector_size;

        static_assert(scalar_per_x_vector % scalar_per_t_vector == 0,
                      "wrong! X should contain multiple T");

        constexpr index_t t_per_x = scalar_per_x_vector / scalar_per_t_vector;

        hcu_async_buffer_load_asm<remove_cvref_t<T>, t_per_x>(
            smem,
            cached_buf_res_,
            voffset,
            ioffset,
            lds_wrap_offset,
            is_valid,
            bool_constant<oob_conditional_check>{});
    }

    template <typename X,
              bool oob_conditional_check = true,
              typename std::enable_if<
                  std::is_same<typename vector_traits<remove_cvref_t<X>>::scalar_type,
                               typename vector_traits<remove_cvref_t<T>>::scalar_type>::value,
                  bool>::type = false>
    CK_TILE_DEVICE constexpr auto get_asm(index_t voffset,
                                          index_t ioffset,
                                          bool is_valid,
                                          bool_constant<oob_conditional_check> = {}) const
    {
        // X is vector of T
        constexpr index_t scalar_per_t_vector = vector_traits<remove_cvref_t<T>>::vector_size;
        constexpr index_t scalar_per_x_vector = vector_traits<remove_cvref_t<X>>::vector_size;

        static_assert(scalar_per_x_vector % scalar_per_t_vector == 0,
                      "wrong! X should contain multiple T");

        constexpr index_t t_per_x = scalar_per_x_vector / scalar_per_t_vector;

        return hcu_buffer_load_asm<remove_cvref_t<T>, t_per_x>(
            cached_buf_res_, voffset, ioffset, is_valid, bool_constant<oob_conditional_check>{});
    }

    CK_TILE_HOST_DEVICE void print() const
    {
        printf("hcu_buffer_view{");

        // AddressSpace
        printf("AddressSpace: Global, ");

        // p_data_
        printf("p_data_: %p, ", static_cast<void*>(const_cast<remove_cvref_t<T>*>(p_data_)));

        // buffer_size_
        printf("buffer_size_: ");
        print(buffer_size_);
        printf(", ");

        // invalid_element_value_
        printf("invalid_element_value_: ");
        print(invalid_element_value_);

        printf("}");
    }
};

template <typename T, typename BufferSizeType>
struct hcu_buffer_view<address_space_enum::global, T, BufferSizeType, true>
{
    using type = T;

    T* p_data_ = nullptr;
    BufferSizeType buffer_size_;
    int32x4_t cached_buf_res_;
    remove_cvref_t<T> invalid_element_value_ = T{0};
    index_t stride_;

    CK_TILE_HOST_DEVICE constexpr hcu_buffer_view(T* p_data,
                                                  BufferSizeType buffer_size,
                                                  index_t stride)
        : p_data_{p_data}, buffer_size_{buffer_size}, cached_buf_res_{0}, stride_{stride}
    {
        cached_buf_res_ = make_wave_buffer_resource(p_data, buffer_size * sizeof(type), stride);
    }

    CK_TILE_DEVICE static constexpr address_space_enum get_address_space()
    {
        return address_space_enum::global;
    }

    CK_TILE_DEVICE static constexpr bool is_struct_buffer() { return true; }

    template <typename X,
              bool oob_conditional_check = true,
              typename std::enable_if<
                  std::is_same<typename vector_traits<remove_cvref_t<X>>::scalar_type,
                               typename vector_traits<remove_cvref_t<T>>::scalar_type>::value,
                  bool>::type = false>
    CK_TILE_DEVICE constexpr void async_get_asm(CK_TILE_LDS_ADDR remove_cvref_t<T>* smem,
                                                index_t vindex,
                                                index_t voffset,
                                                index_t ioffset,
                                                bool is_valid,
                                                bool_constant<oob_conditional_check> = {}) const
    {
        // X is vector of T
        constexpr index_t scalar_per_t_vector = vector_traits<remove_cvref_t<T>>::vector_size;
        constexpr index_t scalar_per_x_vector = vector_traits<remove_cvref_t<X>>::vector_size;

        static_assert(scalar_per_x_vector % scalar_per_t_vector == 0,
                      "wrong! X should contain multiple T");

        constexpr index_t t_per_x = scalar_per_x_vector / scalar_per_t_vector;

        hcu_async_struct_buffer_load_asm<remove_cvref_t<T>, t_per_x>(
            smem,
            cached_buf_res_,
            vindex,
            voffset,
            ioffset,
            is_valid,
            bool_constant<oob_conditional_check>{});
    }

    template <typename X,
              bool oob_conditional_check = true,
              typename std::enable_if<
                  std::is_same<typename vector_traits<remove_cvref_t<X>>::scalar_type,
                               typename vector_traits<remove_cvref_t<T>>::scalar_type>::value,
                  bool>::type = false>
    CK_TILE_DEVICE constexpr void
    async_get_wrapped_asm(CK_TILE_LDS_ADDR remove_cvref_t<T>* smem,
                          index_t vindex,
                          index_t voffset,
                          index_t ioffset,
                          index_t lds_wrap_offset,
                          bool is_valid,
                          bool_constant<oob_conditional_check> = {}) const
    {
        // X is vector of T
        constexpr index_t scalar_per_t_vector = vector_traits<remove_cvref_t<T>>::vector_size;
        constexpr index_t scalar_per_x_vector = vector_traits<remove_cvref_t<X>>::vector_size;

        static_assert(scalar_per_x_vector % scalar_per_t_vector == 0,
                      "wrong! X should contain multiple T");

        constexpr index_t t_per_x = scalar_per_x_vector / scalar_per_t_vector;

        hcu_async_struct_buffer_load_asm<remove_cvref_t<T>, t_per_x>(
            smem,
            cached_buf_res_,
            vindex,
            voffset,
            ioffset,
            lds_wrap_offset,
            is_valid,
            bool_constant<oob_conditional_check>{});
    }

    template <typename X,
              bool oob_conditional_check = true,
              typename std::enable_if<
                  std::is_same<typename vector_traits<remove_cvref_t<X>>::scalar_type,
                               typename vector_traits<remove_cvref_t<T>>::scalar_type>::value,
                  bool>::type = false>
    CK_TILE_DEVICE constexpr auto get_asm(index_t vindex,
                                          index_t voffset,
                                          index_t ioffset,
                                          bool is_valid,
                                          bool_constant<oob_conditional_check> = {}) const
    {
        // X is vector of T
        constexpr index_t scalar_per_t_vector = vector_traits<remove_cvref_t<T>>::vector_size;
        constexpr index_t scalar_per_x_vector = vector_traits<remove_cvref_t<X>>::vector_size;

        static_assert(scalar_per_x_vector % scalar_per_t_vector == 0,
                      "wrong! X should contain multiple T");

        constexpr index_t t_per_x = scalar_per_x_vector / scalar_per_t_vector;

        return hcu_struct_buffer_load_asm<remove_cvref_t<T>, t_per_x>(
            cached_buf_res_,
            vindex,
            voffset,
            ioffset,
            is_valid,
            bool_constant<oob_conditional_check>{});
    }

    CK_TILE_HOST_DEVICE void print() const
    {
        printf("hcu_buffer_view{");

        // AddressSpace
        printf("AddressSpace: Global, ");

        // p_data_
        printf("p_data_: %p, ", static_cast<void*>(const_cast<remove_cvref_t<T>*>(p_data_)));

        // buffer_size_
        printf("buffer_size_: ");
        print(buffer_size_);
        printf(", ");

        // stride_
        printf("stride_: ");
        print(stride_);
        printf(", ");

        // invalid_element_value_
        printf("invalid_element_value_: ");
        print(invalid_element_value_);

        printf("}");
    }
};

template <address_space_enum BufferAddressSpace, typename T, typename BufferSizeType>
CK_TILE_HOST_DEVICE constexpr auto make_hcu_buffer_view(T* p, BufferSizeType buffer_size)
{
    return hcu_buffer_view<BufferAddressSpace, T, BufferSizeType, false>{p, buffer_size};
}

template <address_space_enum BufferAddressSpace, typename T, typename BufferSizeType>
CK_TILE_HOST_DEVICE constexpr auto
make_hcu_buffer_view(T* p, BufferSizeType buffer_size, index_t stride)
{
    static_assert(BufferAddressSpace == address_space_enum::global);

    return hcu_buffer_view<BufferAddressSpace, T, BufferSizeType, true>{p, buffer_size, stride};
}

} // namespace ck_tile
