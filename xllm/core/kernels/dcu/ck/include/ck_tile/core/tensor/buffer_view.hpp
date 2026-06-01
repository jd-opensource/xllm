// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, , Inc. All rights reserved.

#pragma once

#include "ck_tile/core/config.hpp"
#include "ck_tile/core/arch/arch.hpp"
#include "ck_tile/core/arch/amd_buffer_addressing.hpp"
#include "ck_tile/core/arch/generic_memory_space_atomic.hpp"
#include "ck_tile/core/container/array.hpp"
#include "ck_tile/core/numeric/integer.hpp"
#include "ck_tile/core/numeric/integral_constant.hpp"
#include "ck_tile/core/numeric/float8.hpp"
#include "ck_tile/core/numeric/half.hpp"
#include "ck_tile/core/numeric/bfloat16.hpp"
#include "ck_tile/core/utility/type_traits.hpp"

namespace ck_tile {

// T may be scalar or vector
// X may be scalar or vector
// T and X have same scalar type
// X contains multiple T
// FIXME: InvalidElementUseNumericalZeroValue and invalid_element_value_ should be a property of
//        transforms of tensor_view/Tensor
// FIXME: amd_buffer_coherence_enum is only meaningful for buffer addressing. Need to split
// buffer_view definition for different memory address space (Global/GenericLds/Vgpr)
template <address_space_enum BufferAddressSpace,
          typename T,
          typename BufferSizeType,
          bool InvalidElementUseNumericalZeroValue,
          amd_buffer_coherence_enum Coherence = amd_buffer_coherence_enum::coherence_default>
struct buffer_view;

// Address Space: generic
// T may be scalar or vector
// X may be scalar or vector
// T and X have same scalar type
// X contains multiple T
// FIXME: InvalidElementUseNumericalZeroValue and invalid_element_value_ should be a property of
//        transforms of tensor_view/Tensor
template <typename T, typename BufferSizeType, bool InvalidElementUseNumericalZeroValue>
struct buffer_view<address_space_enum::generic,
                   T,
                   BufferSizeType,
                   InvalidElementUseNumericalZeroValue,
                   amd_buffer_coherence_enum::coherence_default>
{
    using type = T;

    T* p_data_ = nullptr;
    BufferSizeType buffer_size_;
    remove_cvref_t<T> invalid_element_value_ = T{0};

    CK_TILE_HOST_DEVICE constexpr buffer_view()
        : p_data_{}, buffer_size_{}, invalid_element_value_{}
    {
    }

    CK_TILE_HOST_DEVICE constexpr buffer_view(T* p_data, BufferSizeType buffer_size)
        : p_data_{p_data}, buffer_size_{buffer_size}, invalid_element_value_{0}
    {
    }

    CK_TILE_HOST_DEVICE constexpr buffer_view(T* p_data,
                                              BufferSizeType buffer_size,
                                              T invalid_element_value)
        : p_data_{p_data}, buffer_size_{buffer_size}, invalid_element_value_{invalid_element_value}
    {
    }

    CK_TILE_HOST_DEVICE void init_raw() {}

    CK_TILE_DEVICE static constexpr address_space_enum get_address_space()
    {
        return address_space_enum::generic;
    }

    // i is offset of T
    // FIXME: doesn't do is_valid check
    CK_TILE_DEVICE constexpr const T& operator[](index_t i) const { return p_data_[i]; }

    // i is offset of T
    // FIXME: doesn't do is_valid check
    CK_TILE_DEVICE constexpr T& operator()(index_t i) { return p_data_[i]; }

    // i is offset of T, not X. i should be aligned to X
    template <typename X,
              bool oob_conditional_check = true,
              typename std::enable_if<
                  std::is_same<typename vector_traits<remove_cvref_t<X>>::scalar_type,
                               typename vector_traits<remove_cvref_t<T>>::scalar_type>::value,
                  bool>::type = false>
    CK_TILE_DEVICE constexpr auto get(index_t i,
                                      index_t linear_offset,
                                      bool is_valid_element,
                                      bool_constant<oob_conditional_check> = {}) const
    {
        // X contains multiple T
        constexpr index_t scalar_per_t_vector = vector_traits<remove_cvref_t<T>>::vector_size;

        constexpr index_t scalar_per_x_vector = vector_traits<remove_cvref_t<X>>::vector_size;

        static_assert(scalar_per_x_vector % scalar_per_t_vector == 0,
                      "wrong! X should contain multiple T");

        if(is_valid_element)
        {
#if CK_TILE_EXPERIMENTAL_USE_MEMCPY_FOR_VECTOR_ACCESS
            X tmp;

            __builtin_memcpy(&tmp, &(p_data_[i + linear_offset]), sizeof(X));

            return tmp;
#else
            return *c_style_pointer_cast<const X*>(&p_data_[i + linear_offset]);
#endif
        }
        else
        {
            if constexpr(InvalidElementUseNumericalZeroValue)
            {
                return X{numeric<remove_cvref_t<T>>::zero()};
            }
            else
            {
                return X{invalid_element_value_};
            }
        }
    }

    // i is offset of T, not X. i should be aligned to X
    template <memory_operation_enum Op,
              typename X,
              typename std::enable_if<
                  std::is_same<typename vector_traits<remove_cvref_t<X>>::scalar_type,
                               typename vector_traits<remove_cvref_t<T>>::scalar_type>::value,
                  bool>::type = false>
    CK_TILE_DEVICE void update(index_t i, index_t linear_offset, bool is_valid_element, const X& x)
    {
        if constexpr(Op == memory_operation_enum::set)
        {
            this->template set<X>(i, linear_offset, is_valid_element, x);
        }
        // FIXME: remove memory_operation_enum::add
        else if constexpr(Op == memory_operation_enum::add)
        {
            auto tmp = this->template get<X>(i, linear_offset, is_valid_element);
            this->template set<X>(i, linear_offset, is_valid_element, x + tmp);
        }
    }

    // i is offset of T, not X. i should be aligned to X
    template <typename X,
              typename std::enable_if<
                  std::is_same<typename vector_traits<remove_cvref_t<X>>::scalar_type,
                               typename vector_traits<remove_cvref_t<T>>::scalar_type>::value,
                  bool>::type = false>
    CK_TILE_DEVICE void set(index_t i, index_t linear_offset, bool is_valid_element, const X& x)
    {
        // X contains multiple T
        constexpr index_t scalar_per_t_vector = vector_traits<remove_cvref_t<T>>::vector_size;

        constexpr index_t scalar_per_x_vector = vector_traits<remove_cvref_t<X>>::vector_size;

        static_assert(scalar_per_x_vector % scalar_per_t_vector == 0,
                      "wrong! X should contain multiple T");

        if(is_valid_element)
        {
#if CK_TILE_EXPERIMENTAL_USE_MEMCPY_FOR_VECTOR_ACCESS
            X tmp = x;

            __builtin_memcpy(&(p_data_[i + linear_offset]), &tmp, sizeof(X));
#else
            *c_style_pointer_cast<X*>(&p_data_[i + linear_offset]) = x;
#endif
        }
    }

    // FIXME: remove
    CK_TILE_DEVICE static constexpr bool is_static_buffer() { return false; }

    // FIXME: remove
    CK_TILE_DEVICE static constexpr bool is_dynamic_buffer() { return true; }

    CK_TILE_HOST_DEVICE void print() const
    {
        printf("buffer_view{");

        // AddressSpace
        printf("AddressSpace: generic, ");

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

// Address Space: Global
// T may be scalar or vector
// X may be scalar or vector
// T and X have same scalar type
// X contains multiple T
// FIXME: InvalidElementUseNumericalZeroValue and invalid_element_value_ should be a property of
//        transforms of tensor_view/Tensor
template <typename T,
          typename BufferSizeType,
          bool InvalidElementUseNumericalZeroValue,
          amd_buffer_coherence_enum Coherence>
struct buffer_view<address_space_enum::global,
                   T,
                   BufferSizeType,
                   InvalidElementUseNumericalZeroValue,
                   Coherence>
{
    using type = T;

    T* p_data_ = nullptr;
    BufferSizeType buffer_size_;
    int32x4_t cached_buf_res_;
    remove_cvref_t<T> invalid_element_value_ = T{0};
    index_t const_stride_;

    CK_TILE_HOST_DEVICE constexpr buffer_view()
        : p_data_{}, buffer_size_{}, cached_buf_res_{0}, invalid_element_value_{}, const_stride_{0}
    {
    }

    CK_TILE_HOST_DEVICE constexpr buffer_view(T* p_data, BufferSizeType buffer_size)
        : p_data_{p_data},
          buffer_size_{buffer_size},
          cached_buf_res_{0},
          invalid_element_value_{0},
          const_stride_{0}
    {
    }

    CK_TILE_HOST_DEVICE constexpr buffer_view(T* p_data,
                                              BufferSizeType buffer_size,
                                              T invalid_element_value)
        : p_data_{p_data},
          buffer_size_{buffer_size},
          cached_buf_res_{0},
          invalid_element_value_{invalid_element_value},
          const_stride_{0}
    {
    }

    CK_TILE_HOST_DEVICE constexpr buffer_view(T* p_data,
                                              BufferSizeType buffer_size,
                                              T invalid_element_value,
                                              index_t const_stride)
        : p_data_{p_data},
          buffer_size_{buffer_size},
          cached_buf_res_{0},
          invalid_element_value_{invalid_element_value},
          const_stride_{const_stride}
    {
    }

    // this is non constexpr intentially (will call some intrinsic internally)
    // Must call for buffers that need *_raw load/store
    CK_TILE_HOST_DEVICE void init_raw()
    {
        cached_buf_res_ = make_wave_buffer_resource(p_data_, buffer_size_ * sizeof(type));
    }

    CK_TILE_DEVICE static constexpr address_space_enum get_address_space()
    {
        return address_space_enum::global;
    }

    // i is offset of T
    // FIXME: doesn't do is_valid check
    CK_TILE_DEVICE constexpr const T& operator[](index_t i) const { return p_data_[i]; }

    // i is offset of T
    // FIXME: doesn't do is_valid check
    CK_TILE_DEVICE constexpr T& operator()(index_t i) { return p_data_[i]; }

    // i is offset of T, not X. i should be aligned to X
    template <typename X,
              bool oob_conditional_check = true,
              typename std::enable_if<
                  std::is_same<typename vector_traits<remove_cvref_t<X>>::scalar_type,
                               typename vector_traits<remove_cvref_t<T>>::scalar_type>::value,
                  bool>::type = false>
    CK_TILE_DEVICE constexpr auto get(index_t i,
                                      index_t linear_offset,
                                      bool is_valid_element,
                                      bool_constant<oob_conditional_check> = {}) const
    {
        // X contains multiple T
        constexpr index_t scalar_per_t_vector = vector_traits<remove_cvref_t<T>>::vector_size;

        constexpr index_t scalar_per_x_vector = vector_traits<remove_cvref_t<X>>::vector_size;

        static_assert(scalar_per_x_vector % scalar_per_t_vector == 0,
                      "wrong! X should contain multiple T");

#if CK_TILE_USE_AMD_BUFFER_LOAD
        bool constexpr use_amd_buffer_addressing = true;
#else
        bool constexpr use_amd_buffer_addressing = false;
#endif

        if constexpr(use_amd_buffer_addressing)
        {
            constexpr index_t t_per_x = scalar_per_x_vector / scalar_per_t_vector;

            if constexpr(InvalidElementUseNumericalZeroValue)
            {
                return amd_buffer_load_invalid_element_return_zero<remove_cvref_t<T>,
                                                                   t_per_x,
                                                                   Coherence,
                                                                   oob_conditional_check>(
                    p_data_, i + linear_offset, is_valid_element, buffer_size_);
            }
            else
            {
                return amd_buffer_load_invalid_element_return_customized_value<
                    remove_cvref_t<T>,
                    t_per_x,
                    Coherence,
                    oob_conditional_check>(p_data_,
                                           i + linear_offset,
                                           is_valid_element,
                                           buffer_size_,
                                           invalid_element_value_);
            }
        }
        else
        {
            if(is_valid_element)
            {
#if CK_TILE_EXPERIMENTAL_USE_MEMCPY_FOR_VECTOR_ACCESS
                X tmp;

                __builtin_memcpy(&tmp, &(p_data_[i + linear_offset]), sizeof(X));

                return tmp;
#else
                return *c_style_pointer_cast<const X*>(&p_data_[i + linear_offset]);
#endif
            }
            else
            {
                if constexpr(InvalidElementUseNumericalZeroValue)
                {
                    return X{numeric<remove_cvref_t<T>>::zero()};
                }
                else
                {
                    return X{invalid_element_value_};
                }
            }
        }
    }

    // i is offset of T, not X. i should be aligned to X
    template <typename X,
              bool oob_conditional_check = true,
              bool cache_swizzle         = false,
              typename std::enable_if<
                  std::is_same<typename vector_traits<remove_cvref_t<X>>::scalar_type,
                               typename vector_traits<remove_cvref_t<T>>::scalar_type>::value,
                  bool>::type = false>
    CK_TILE_DEVICE constexpr auto get_by_inline_asm(index_t i,
                                                    index_t linear_offset,
                                                    bool is_valid_element,
                                                    bool_constant<oob_conditional_check> = {},
                                                    bool_constant<cache_swizzle>         = {}) const
    {
        // X contains multiple T
        constexpr index_t scalar_per_t_vector = vector_traits<remove_cvref_t<T>>::vector_size;

        constexpr index_t scalar_per_x_vector = vector_traits<remove_cvref_t<X>>::vector_size;

        static_assert(scalar_per_x_vector % scalar_per_t_vector == 0,
                      "wrong! X should contain multiple T");

        constexpr index_t t_per_x = scalar_per_x_vector / scalar_per_t_vector;

        if constexpr(cache_swizzle)
        {
            // TODO: how to ensure const_stride is pow2?
            return amd_buffer_load_inline_asm_invalid_element_return_zero<remove_cvref_t<T>,
                                                                          t_per_x,
                                                                          Coherence,
                                                                          oob_conditional_check>(
                p_data_, i + linear_offset, is_valid_element, buffer_size_, const_stride_);
        }
        else
        {
            return amd_buffer_load_inline_asm_invalid_element_return_zero<remove_cvref_t<T>,
                                                                          t_per_x,
                                                                          Coherence,
                                                                          oob_conditional_check>(
                p_data_, i + linear_offset, is_valid_element, buffer_size_);
        }
    }

    // i is offset of T, not X. i should be aligned to X
    template <typename X,
              bool oob_conditional_check = true,
              bool pre_nop               = false,
              typename std::enable_if<
                  std::is_same<typename vector_traits<remove_cvref_t<X>>::scalar_type,
                               typename vector_traits<remove_cvref_t<T>>::scalar_type>::value,
                  bool>::type = false>
    CK_TILE_DEVICE constexpr auto get_raw(remove_cvref_t<X>& dst,
                                          index_t v_offset,
                                          index_t i_offset,
                                          bool is_valid_element,
                                          bool_constant<pre_nop> = {}) const
    {
        constexpr index_t scalar_per_t_vector = vector_traits<remove_cvref_t<T>>::vector_size;

        constexpr index_t scalar_per_x_vector = vector_traits<remove_cvref_t<X>>::vector_size;

        static_assert(scalar_per_x_vector % scalar_per_t_vector == 0,
                      "wrong! X should contain multiple T");

        constexpr index_t t_per_x = scalar_per_x_vector / scalar_per_t_vector;

        amd_buffer_load_raw<remove_cvref_t<T>, t_per_x, Coherence, oob_conditional_check, pre_nop>(
            dst, cached_buf_res_, v_offset, i_offset, is_valid_element, bool_constant<pre_nop>{});
    }

    // i is offset of T, not X. i should be aligned to X
    template <typename X,
              bool oob_conditional_check = true,
              typename std::enable_if<
                  std::is_same<typename vector_traits<remove_cvref_t<X>>::scalar_type,
                               typename vector_traits<remove_cvref_t<T>>::scalar_type>::value,
                  bool>::type = false>
    CK_TILE_DEVICE constexpr auto async_get(CK_TILE_LDS_ADDR remove_cvref_t<T>* smem,
                                            index_t i,
                                            index_t linear_offset,
                                            bool is_valid_element,
                                            bool_constant<oob_conditional_check> = {}) const
    {
        // X is vector of T
        constexpr index_t scalar_per_t_vector = vector_traits<remove_cvref_t<T>>::vector_size;
        constexpr index_t scalar_per_x_vector = vector_traits<remove_cvref_t<X>>::vector_size;

        static_assert(scalar_per_x_vector % scalar_per_t_vector == 0,
                      "wrong! X should contain multiple T");

        constexpr index_t t_per_x = scalar_per_x_vector / scalar_per_t_vector;

        amd_async_buffer_load_with_oob<remove_cvref_t<T>, t_per_x, Coherence>(
            smem,
            cached_buf_res_,
            i,
            linear_offset,
            is_valid_element,
            bool_constant<oob_conditional_check>{});
    }

    // i is offset of T, not X. i should be aligned to X
    template <typename X,
              bool oob_conditional_check = true,
              typename std::enable_if<
                  std::is_same<typename vector_traits<remove_cvref_t<X>>::scalar_type,
                               typename vector_traits<remove_cvref_t<T>>::scalar_type>::value,
                  bool>::type = false>
    CK_TILE_DEVICE constexpr auto
    async_get_by_inline_asm(CK_TILE_LDS_ADDR remove_cvref_t<T>* smem,
                            index_t i,
                            index_t linear_offset,
                            bool is_valid_element,
                            bool_constant<oob_conditional_check> = {}) const
    {
        // X is vector of T
        constexpr index_t scalar_per_t_vector = vector_traits<remove_cvref_t<T>>::vector_size;
        constexpr index_t scalar_per_x_vector = vector_traits<remove_cvref_t<X>>::vector_size;

        static_assert(scalar_per_x_vector % scalar_per_t_vector == 0,
                      "wrong! X should contain multiple T");

        constexpr index_t t_per_x = scalar_per_x_vector / scalar_per_t_vector;

        amd_async_buffer_load_inline_asm_with_oob<remove_cvref_t<T>, t_per_x, Coherence>(
            smem,
            cached_buf_res_,
            i,
            linear_offset,
            is_valid_element,
            bool_constant<oob_conditional_check>{});
    }

    // i is offset of T, not X. i should be aligned to X
    template <typename X,
              bool pre_nop = false,
              typename std::enable_if<
                  std::is_same<typename vector_traits<remove_cvref_t<X>>::scalar_type,
                               typename vector_traits<remove_cvref_t<T>>::scalar_type>::value,
                  bool>::type = false>
    CK_TILE_DEVICE constexpr auto async_get_raw(remove_cvref_t<T>* smem,
                                                index_t i,
                                                index_t linear_offset,
                                                bool /*is_valid_element*/,
                                                bool_constant<pre_nop> = {}) const
    {
        // X is vector of T
        constexpr index_t scalar_per_t_vector = vector_traits<remove_cvref_t<T>>::vector_size;
        constexpr index_t scalar_per_x_vector = vector_traits<remove_cvref_t<X>>::vector_size;

        static_assert(scalar_per_x_vector % scalar_per_t_vector == 0,
                      "wrong! X should contain multiple T");

        constexpr index_t t_per_x = scalar_per_x_vector / scalar_per_t_vector;

        amd_async_buffer_load_with_oob_raw<remove_cvref_t<T>, t_per_x, Coherence>(
            smem, cached_buf_res_, i, linear_offset, bool_constant<pre_nop>{});
    }

    // i is offset of T, not X. i should be aligned to X
    template <memory_operation_enum Op,
              typename X,
              typename std::enable_if<
                  std::is_same<typename vector_traits<remove_cvref_t<X>>::scalar_type,
                               typename vector_traits<remove_cvref_t<T>>::scalar_type>::value,
                  bool>::type = false>
    CK_TILE_DEVICE void update(index_t i, index_t linear_offset, bool is_valid_element, const X& x)
    {
        if constexpr(Op == memory_operation_enum::set)
        {
            this->template set<X>(i, linear_offset, is_valid_element, x);
        }
        else if constexpr(Op == memory_operation_enum::atomic_add)
        {
            this->template atomic_add<X>(i, linear_offset, is_valid_element, x);
        }
        else if constexpr(Op == memory_operation_enum::atomic_max)
        {
            this->template atomic_max<X>(i, linear_offset, is_valid_element, x);
        }
        // FIXME: remove memory_operation_enum::add
        else if constexpr(Op == memory_operation_enum::add)
        {
            auto tmp = this->template get<X>(i, linear_offset, is_valid_element);
            this->template set<X>(i, linear_offset, is_valid_element, x + tmp);
            // tmp += x;
            // this->template set<X>(i, is_valid_element, tmp);
        }
    }

    // i is offset of T, not X. i should be aligned to X
    template <typename X,
              bool oob_conditional_check = true,
              typename std::enable_if<
                  std::is_same<typename vector_traits<remove_cvref_t<X>>::scalar_type,
                               typename vector_traits<remove_cvref_t<T>>::scalar_type>::value,
                  bool>::type = false>
    CK_TILE_DEVICE void set(index_t i, index_t linear_offset, bool is_valid_element, const X& x)
    {
        // X contains multiple T
        constexpr index_t scalar_per_t_vector = vector_traits<remove_cvref_t<T>>::vector_size;

        constexpr index_t scalar_per_x_vector = vector_traits<remove_cvref_t<X>>::vector_size;

        static_assert(scalar_per_x_vector % scalar_per_t_vector == 0,
                      "wrong! X should contain multiple T");

#if CK_TILE_USE_AMD_BUFFER_STORE
        bool constexpr use_amd_buffer_addressing = true;
#else
        bool constexpr use_amd_buffer_addressing = false;
#endif

        if constexpr(use_amd_buffer_addressing)
        {
            constexpr index_t t_per_x = scalar_per_x_vector / scalar_per_t_vector;

            amd_buffer_store<remove_cvref_t<T>, t_per_x, Coherence>(
                x, p_data_, i + linear_offset, is_valid_element, buffer_size_);
        }
        else
        {
            if(is_valid_element)
            {
#if CK_TILE_EXPERIMENTAL_USE_MEMCPY_FOR_VECTOR_ACCESS
                X tmp = x;

                __builtin_memcpy(&(p_data_[i + linear_offset]), &tmp, sizeof(X));
#else
                *c_style_pointer_cast<X*>(&p_data_[i + linear_offset]) = x;
#endif
            }
        }
    }

    // i is offset of T, not X. i should be aligned to X
    template <typename X,
              bool oob_conditional_check = true,
              typename std::enable_if<
                  std::is_same<typename vector_traits<remove_cvref_t<X>>::scalar_type,
                               typename vector_traits<remove_cvref_t<T>>::scalar_type>::value,
                  bool>::type = false>
    CK_TILE_DEVICE void set_raw(index_t i, index_t linear_offset, bool is_valid_element, const X& x)
    {
        // X contains multiple T
        constexpr index_t scalar_per_t_vector = vector_traits<remove_cvref_t<T>>::vector_size;

        constexpr index_t scalar_per_x_vector = vector_traits<remove_cvref_t<X>>::vector_size;

        static_assert(scalar_per_x_vector % scalar_per_t_vector == 0,
                      "wrong! X should contain multiple T");

        constexpr index_t t_per_x = scalar_per_x_vector / scalar_per_t_vector;
        amd_buffer_store_raw<remove_cvref_t<T>, t_per_x, Coherence, oob_conditional_check>(
            x, p_data_, i, linear_offset, is_valid_element, buffer_size_);
    }

    // TODO: support buffer_atomic_pk_add in future
    template <typename X,
              typename std::enable_if<
                  std::is_same<typename vector_traits<remove_cvref_t<X>>::scalar_type,
                               typename vector_traits<remove_cvref_t<T>>::scalar_type>::value,
                  bool>::type = false>
    CK_TILE_DEVICE void
    atomic_add(index_t i, index_t linear_offset, bool is_valid_element, const X& x)
    {
        using scalar_t = typename vector_traits<remove_cvref_t<T>>::scalar_type;

        // X contains multiple T
        constexpr index_t scalar_per_t_vector = vector_traits<remove_cvref_t<T>>::vector_size;

        constexpr index_t scalar_per_x_vector = vector_traits<remove_cvref_t<X>>::vector_size;

        static_assert(scalar_per_x_vector % scalar_per_t_vector == 0,
                      "wrong! X should contain multiple T");

        static_assert(get_address_space() == address_space_enum::global, "only support global mem");

#if CK_TILE_USE_AMD_BUFFER_ATOMIC_ADD_INTEGER && CK_TILE_USE_AMD_BUFFER_ATOMIC_ADD_FLOAT
        bool constexpr use_amd_buffer_addressing =
            std::is_same_v<remove_cvref_t<scalar_t>, int32_t> ||
            std::is_same_v<remove_cvref_t<scalar_t>, float> ||
            (std::is_same_v<remove_cvref_t<scalar_t>, half_t> && scalar_per_x_vector % 2 == 0);
#elif CK_TILE_USE_AMD_BUFFER_ATOMIC_ADD_INTEGER && (!CK_TILE_USE_AMD_BUFFER_ATOMIC_ADD_FLOAT)
        bool constexpr use_amd_buffer_addressing =
            std::is_same_v<remove_cvref_t<scalar_t>, int32_t>;
#elif (!CK_TILE_USE_AMD_BUFFER_ATOMIC_ADD_INTEGER) && CK_TILE_USE_AMD_BUFFER_ATOMIC_ADD_FLOAT
        bool constexpr use_amd_buffer_addressing =
            std::is_same_v<remove_cvref_t<scalar_t>, float> ||
            (std::is_same_v<remove_cvref_t<scalar_t>, half_t> && scalar_per_x_vector % 2 == 0);
#else
        bool constexpr use_amd_buffer_addressing = false;
#endif

        constexpr index_t t_per_x = scalar_per_x_vector / scalar_per_t_vector;

        if constexpr(use_amd_buffer_addressing)
        {
            amd_buffer_atomic_add<remove_cvref_t<T>, t_per_x>(
                x, p_data_, i + linear_offset, is_valid_element, buffer_size_);
        }
        else
        {
            if(is_valid_element)
            {
                atomic_add_g<remove_cvref_t<T>, t_per_x>(&p_data_[i + linear_offset], x);
            }
        }
    }

    template <typename X,
              typename std::enable_if<
                  std::is_same<typename vector_traits<remove_cvref_t<X>>::scalar_type,
                               typename vector_traits<remove_cvref_t<T>>::scalar_type>::value,
                  bool>::type = false>
    CK_TILE_DEVICE void
    atomic_max(index_t i, index_t linear_offset, bool is_valid_element, const X& x)
    {
        // X contains multiple T
        constexpr index_t scalar_per_t_vector = vector_traits<remove_cvref_t<T>>::vector_size;

        constexpr index_t scalar_per_x_vector = vector_traits<remove_cvref_t<X>>::vector_size;

        static_assert(scalar_per_x_vector % scalar_per_t_vector == 0,
                      "wrong! X should contain multiple T");

        static_assert(get_address_space() == address_space_enum::global, "only support global mem");

#if CK_TILE_USE_AMD_BUFFER_ATOMIC_MAX_FLOAT64
        using scalar_t = typename vector_traits<remove_cvref_t<T>>::scalar_type;
        bool constexpr use_amd_buffer_addressing = std::is_same_v<remove_cvref_t<scalar_t>, double>;
#else
        bool constexpr use_amd_buffer_addressing = false;
#endif

        constexpr index_t t_per_x = scalar_per_x_vector / scalar_per_t_vector;

        if constexpr(use_amd_buffer_addressing)
        {
            amd_buffer_atomic_max<remove_cvref_t<T>, t_per_x>(
                x, p_data_, i + linear_offset, is_valid_element, buffer_size_);
        }
        else if(is_valid_element)
        {
            atomic_max_g<remove_cvref_t<T>, t_per_x>(&p_data_[i + linear_offset], x);
        }
    }

    // FIXME: remove
    CK_TILE_DEVICE static constexpr bool is_static_buffer() { return false; }

    // FIXME: remove
    CK_TILE_DEVICE static constexpr bool is_dynamic_buffer() { return true; }

    CK_TILE_HOST_DEVICE void print() const
    {
        printf("buffer_view{");

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

// Address Space: LDS
// T may be scalar or vector
// X may be scalar or vector
// T and X have same scalar type
// X contains multiple T
// FIXME: InvalidElementUseNumericalZeroValue and invalid_element_value_ should be a property of
//        transforms of tensor_view/Tensor
template <typename T, typename BufferSizeType, bool InvalidElementUseNumericalZeroValue>
struct buffer_view<address_space_enum::lds,
                   T,
                   BufferSizeType,
                   InvalidElementUseNumericalZeroValue,
                   amd_buffer_coherence_enum::coherence_default>
{
    using type = T;

    T* p_data_ = nullptr;
    BufferSizeType buffer_size_;
    remove_cvref_t<T> invalid_element_value_ = T{0};

    CK_TILE_HOST_DEVICE constexpr buffer_view()
        : p_data_{}, buffer_size_{}, invalid_element_value_{}
    {
    }

    CK_TILE_HOST_DEVICE constexpr buffer_view(T* p_data, BufferSizeType buffer_size)
        : p_data_{p_data}, buffer_size_{buffer_size}, invalid_element_value_{0}
    {
    }

    CK_TILE_HOST_DEVICE constexpr buffer_view(T* p_data,
                                              BufferSizeType buffer_size,
                                              T invalid_element_value)
        : p_data_{p_data}, buffer_size_{buffer_size}, invalid_element_value_{invalid_element_value}
    {
    }

    CK_TILE_HOST_DEVICE void init_raw() {}

    CK_TILE_DEVICE static constexpr address_space_enum get_address_space()
    {
        return address_space_enum::lds;
    }

    // i is offset of T
    // FIXME: doesn't do is_valid check
    CK_TILE_DEVICE constexpr const T& operator[](index_t i) const { return p_data_[i]; }

    // i is offset of T
    // FIXME: doesn't do is_valid check
    CK_TILE_DEVICE constexpr T& operator()(index_t i) { return p_data_[i]; }

    // i is offset of T, not X. i should be aligned to X
    // NOTE: get_raw is preferred.
    template <typename X,
              bool oob_conditional_check = true,
              typename std::enable_if<
                  std::is_same<typename vector_traits<remove_cvref_t<X>>::scalar_type,
                               typename vector_traits<remove_cvref_t<T>>::scalar_type>::value,
                  bool>::type = false>
    CK_TILE_DEVICE constexpr auto get(index_t i,
                                      index_t linear_offset,
                                      bool is_valid_element,
                                      bool_constant<oob_conditional_check> = {}) const
    {
        // X contains multiple T
        constexpr index_t scalar_per_t_vector = vector_traits<remove_cvref_t<T>>::vector_size;

        constexpr index_t scalar_per_x_vector = vector_traits<remove_cvref_t<X>>::vector_size;

        static_assert(scalar_per_x_vector % scalar_per_t_vector == 0,
                      "wrong! X should contain multiple T");

        if constexpr(oob_conditional_check)
        {
            if(is_valid_element)
            {
#if CK_TILE_EXPERIMENTAL_USE_MEMCPY_FOR_VECTOR_ACCESS
                X tmp;

                __builtin_memcpy(&tmp, &(p_data_[i + linear_offset]), sizeof(X));

                return tmp;
#else
                using buf_t = ext_vector_t<typename vector_traits<remove_cvref_t<T>>::scalar_type,
                                           scalar_per_t_vector * scalar_per_x_vector>;
                // using buf_t = ushort __attribute__((ext_vector_type(8)));
                auto rtn = *c_style_pointer_cast<const buf_t*>(&p_data_[i + linear_offset]);
                return bit_cast<X>(rtn);
#endif
            }
            else
            {
                if constexpr(InvalidElementUseNumericalZeroValue)
                {
                    return X{numeric<remove_cvref_t<T>>::zero()};
                }
                else
                {
                    return X{invalid_element_value_};
                }
            }
        }
        else
        {
            auto rtn = *c_style_pointer_cast<const X*>(&p_data_[i + linear_offset]);
            return bit_cast<X>(rtn);
        }
    }

    // i is offset of T, not X. i should be aligned to X
    // NOTE: get_raw is preferred.
    template <typename X,
              typename DsreadmInstr,
              typename std::enable_if<
                  std::is_same<typename vector_traits<remove_cvref_t<X>>::scalar_type,
                               typename vector_traits<remove_cvref_t<T>>::scalar_type>::value,
                  bool>::type = false>
    CK_TILE_DEVICE constexpr auto
    get_by_dsreadm(const DsreadmInstr& instr, index_t i, index_t linear_offset) const
    {
        // X contains multiple T
        constexpr index_t scalar_per_t_vector = vector_traits<remove_cvref_t<T>>::vector_size;

        constexpr index_t scalar_per_x_vector = vector_traits<remove_cvref_t<X>>::vector_size;

        static_assert(scalar_per_x_vector % scalar_per_t_vector == 0,
                      "wrong! X should contain multiple T");

        return bit_cast<X>(instr(&p_data_[i + linear_offset]));
    }

    // i is offset of T, not X. i should be aligned to X
    template <typename X,
              bool oob_conditional_check = true,
              bool pre_nop               = false,
              typename std::enable_if<
                  std::is_same<typename vector_traits<remove_cvref_t<X>>::scalar_type,
                               typename vector_traits<remove_cvref_t<T>>::scalar_type>::value,
                  bool>::type = false>
    CK_TILE_DEVICE constexpr auto get_raw(remove_cvref_t<X>& dst,
                                          index_t v_offset,
                                          index_t i_offset,
                                          bool /*is_valid_element*/,
                                          bool_constant<pre_nop> = {}) const
    {
        smem_load<sizeof(X)>{}(
            dst, reinterpret_cast<uintptr_t>(p_data_) + v_offset * sizeof(T), i_offset * sizeof(T));
    }

    // i is offset of T, not X. i should be aligned to X
    template <memory_operation_enum Op,
              typename X,
              typename std::enable_if<
                  std::is_same<typename vector_traits<remove_cvref_t<X>>::scalar_type,
                               typename vector_traits<remove_cvref_t<T>>::scalar_type>::value,
                  bool>::type = false>
    CK_TILE_DEVICE void update(index_t i, index_t linear_offset, bool is_valid_element, const X& x)
    {
        if constexpr(Op == memory_operation_enum::set)
        {
            this->template set<X>(i, linear_offset, is_valid_element, x);
        }
        // FIXME: remove memory_operation_enum::add
        else if constexpr(Op == memory_operation_enum::add)
        {
            auto tmp = this->template get<X>(i, linear_offset, is_valid_element);
            this->template set<X>(i, linear_offset, is_valid_element, x + tmp);
        }
    }

    // i is offset of T, not X. i should be aligned to X
    template <typename X,
              typename std::enable_if<
                  std::is_same<typename vector_traits<remove_cvref_t<X>>::scalar_type,
                               typename vector_traits<remove_cvref_t<T>>::scalar_type>::value,
                  bool>::type = false>
    CK_TILE_DEVICE void set(index_t i, index_t linear_offset, bool is_valid_element, const X& x)
    {
        // X contains multiple T
        constexpr index_t scalar_per_t_vector = vector_traits<remove_cvref_t<T>>::vector_size;

        constexpr index_t scalar_per_x_vector = vector_traits<remove_cvref_t<X>>::vector_size;

        static_assert(scalar_per_x_vector % scalar_per_t_vector == 0,
                      "wrong! X should contain multiple T");

        i += linear_offset; // simplicity

        if(is_valid_element)
        {
#if CK_TILE_EXPERIMENTAL_USE_MEMCPY_FOR_VECTOR_ACCESS
            X tmp = x;

            __builtin_memcpy(&(p_data_[i]), &tmp, sizeof(X));
#else
            using buf_t = ext_vector_t<typename vector_traits<remove_cvref_t<T>>::scalar_type,
                                        scalar_per_t_vector * scalar_per_x_vector>;

            *c_style_pointer_cast<buf_t*>(&p_data_[i]) = reinterpret_cast<const buf_t&>(x);
#endif
        }
        
    }

    // FIXME: remove
    CK_TILE_DEVICE static constexpr bool is_static_buffer() { return false; }

    // FIXME: remove
    CK_TILE_DEVICE static constexpr bool is_dynamic_buffer() { return true; }

    CK_TILE_HOST_DEVICE void print() const
    {
        printf("buffer_view{");

        // AddressSpace
        printf("AddressSpace: Lds, ");

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

// Address Space: Vgpr
// T may be scalar or vector
// X may be scalar or vector
// T and X have same scalar type
// X contains multiple T
// FIXME: InvalidElementUseNumericalZeroValue and invalid_element_value_ should be a property of
//        transforms of tensor_view/Tensor
template <typename T, typename BufferSizeType, bool InvalidElementUseNumericalZeroValue>
struct buffer_view<address_space_enum::vgpr,
                   T,
                   BufferSizeType,
                   InvalidElementUseNumericalZeroValue,
                   amd_buffer_coherence_enum::coherence_default>
{
    using type = T;

    T* p_data_ = nullptr;
    BufferSizeType buffer_size_;
    remove_cvref_t<T> invalid_element_value_ = T{0};

    CK_TILE_HOST_DEVICE constexpr buffer_view()
        : p_data_{}, buffer_size_{}, invalid_element_value_{}
    {
    }

    CK_TILE_HOST_DEVICE constexpr buffer_view(T* p_data, BufferSizeType buffer_size)
        : p_data_{p_data}, buffer_size_{buffer_size}, invalid_element_value_{0}
    {
    }

    CK_TILE_HOST_DEVICE constexpr buffer_view(T* p_data,
                                              BufferSizeType buffer_size,
                                              T invalid_element_value)
        : p_data_{p_data}, buffer_size_{buffer_size}, invalid_element_value_{invalid_element_value}
    {
    }

    CK_TILE_HOST_DEVICE void init_raw() {}

    CK_TILE_DEVICE static constexpr address_space_enum get_address_space()
    {
        return address_space_enum::vgpr;
    }

    // i is offset of T
    // FIXME: doesn't do is_valid check
    CK_TILE_DEVICE constexpr const T& operator[](index_t i) const { return p_data_[i]; }

    // i is offset of T
    // FIXME: doesn't do is_valid check
    CK_TILE_DEVICE constexpr T& operator()(index_t i) { return p_data_[i]; }

    // i is offset of T, not X. i should be aligned to X
    template <typename X,
              bool oob_conditional_check = true,
              typename std::enable_if<
                  std::is_same<typename vector_traits<remove_cvref_t<X>>::scalar_type,
                               typename vector_traits<remove_cvref_t<T>>::scalar_type>::value,
                  bool>::type = false>
    CK_TILE_DEVICE constexpr auto get(index_t i,
                                      index_t /*linear_offset*/,
                                      bool is_valid_element,
                                      bool_constant<oob_conditional_check> = {}) const
    {
        // X contains multiple T
        constexpr index_t scalar_per_t_vector = vector_traits<remove_cvref_t<T>>::vector_size;

        constexpr index_t scalar_per_x_vector = vector_traits<remove_cvref_t<X>>::vector_size;

        static_assert(scalar_per_x_vector % scalar_per_t_vector == 0,
                      "wrong! X should contain multiple T");

        if(is_valid_element)
        {
#if CK_TILE_EXPERIMENTAL_USE_MEMCPY_FOR_VECTOR_ACCESS
            X tmp;

            __builtin_memcpy(&tmp, &(p_data_[i]), sizeof(X));

            return tmp;
#else
            return *c_style_pointer_cast<const X*>(&p_data_[i]);
#endif
        }
        else
        {
            if constexpr(InvalidElementUseNumericalZeroValue)
            {
                return X{numeric<remove_cvref_t<T>>::zero()};
            }
            else
            {
                return X{invalid_element_value_};
            }
        }
    }

    // i is offset of T, not X. i should be aligned to X
    template <memory_operation_enum Op,
              typename X,
              typename std::enable_if<
                  std::is_same<typename vector_traits<remove_cvref_t<X>>::scalar_type,
                               typename vector_traits<remove_cvref_t<T>>::scalar_type>::value,
                  bool>::type = false>
    CK_TILE_DEVICE void update(index_t i, index_t linear_offset, bool is_valid_element, const X& x)
    {
        if constexpr(Op == memory_operation_enum::set)
        {
            this->template set<X>(i, linear_offset, is_valid_element, x);
        }
        // FIXME: remove memory_operation_enum::add
        else if constexpr(Op == memory_operation_enum::add)
        {
            auto tmp = this->template get<X>(i, linear_offset, is_valid_element);
            this->template set<X>(i, linear_offset, is_valid_element, x + tmp);
        }
    }

    // i is offset of T, not X. i should be aligned to X
    template <typename X,
              typename std::enable_if<
                  std::is_same<typename vector_traits<remove_cvref_t<X>>::scalar_type,
                               typename vector_traits<remove_cvref_t<T>>::scalar_type>::value,
                  bool>::type = false>
    CK_TILE_DEVICE void set(index_t i, index_t linear_offset, bool is_valid_element, const X& x)
    {
        // X contains multiple T
        constexpr index_t scalar_per_t_vector = vector_traits<remove_cvref_t<T>>::vector_size;

        constexpr index_t scalar_per_x_vector = vector_traits<remove_cvref_t<X>>::vector_size;

        static_assert(scalar_per_x_vector % scalar_per_t_vector == 0,
                      "wrong! X should contain multiple T");

        if(is_valid_element)
        {
#if CK_TILE_EXPERIMENTAL_USE_MEMCPY_FOR_VECTOR_ACCESS
            X tmp = x;

            __builtin_memcpy(&(p_data_[i + linear_offset]), &tmp, sizeof(X));
#else
            *c_style_pointer_cast<X*>(&p_data_[i + linear_offset]) = x;
#endif
        }
    }

    // FIXME: remove
    CK_TILE_DEVICE static constexpr bool is_static_buffer() { return false; }

    // FIXME: remove
    CK_TILE_DEVICE static constexpr bool is_dynamic_buffer() { return true; }

    CK_TILE_HOST_DEVICE void print() const
    {
        printf("buffer_view{");

        // AddressSpace
        printf("AddressSpace: Vgpr, ");

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

template <address_space_enum BufferAddressSpace,
          amd_buffer_coherence_enum Coherence = amd_buffer_coherence_enum::coherence_default,
          typename T,
          typename BufferSizeType>
CK_TILE_HOST_DEVICE constexpr auto make_buffer_view(T* p, BufferSizeType buffer_size)
{
    return buffer_view<BufferAddressSpace, T, BufferSizeType, true, Coherence>{p, buffer_size};
}

template <address_space_enum BufferAddressSpace,
          amd_buffer_coherence_enum Coherence = amd_buffer_coherence_enum::coherence_default,
          typename T,
          typename BufferSizeType,
          typename X,
          typename std::enable_if<std::is_same<remove_cvref_t<T>, remove_cvref_t<X>>::value,
                                  bool>::type = false>
CK_TILE_HOST_DEVICE constexpr auto
make_buffer_view(T* p, BufferSizeType buffer_size, X invalid_element_value)
{
    return buffer_view<BufferAddressSpace, T, BufferSizeType, false, Coherence>{
        p, buffer_size, invalid_element_value};
}

template <address_space_enum BufferAddressSpace,
          amd_buffer_coherence_enum Coherence = amd_buffer_coherence_enum::coherence_default,
          typename T,
          typename BufferSizeType,
          typename X,
          typename std::enable_if<std::is_same<remove_cvref_t<T>, remove_cvref_t<X>>::value,
                                  bool>::type = false>
CK_TILE_HOST_DEVICE constexpr auto
make_buffer_view(T* p, BufferSizeType buffer_size, X invalid_element_value, index_t const_stride)
{
    static_assert(BufferAddressSpace == address_space_enum::global);

    return buffer_view<BufferAddressSpace, T, BufferSizeType, false, Coherence>{
        p, buffer_size, invalid_element_value, const_stride};
}

} // namespace ck_tile
