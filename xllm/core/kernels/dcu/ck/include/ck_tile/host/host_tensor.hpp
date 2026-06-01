// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, , Inc. All rights reserved.

#pragma once

#include <algorithm>
#include <cassert>
#include <iostream>
#include <iomanip>
#include <numeric>
#include <utility>
#include <vector>
#include <functional>
#include <fstream>

#include "ck_tile/core.hpp"
#include "ck_tile/host/joinable_thread.hpp"
#include "ck_tile/host/ranges.hpp"

namespace ck_tile {

template <typename Range>
CK_TILE_HOST std::ostream& LogRange(std::ostream& os,
                                    Range&& range,
                                    std::string delim,
                                    int precision = std::cout.precision(),
                                    int width     = 0)
{
    bool first = true;
    for(auto&& v : range)
    {
        if(first)
            first = false;
        else
            os << delim;
        os << std::setw(width) << std::setprecision(precision) << v;
    }
    return os;
}

template <typename T, typename Range>
CK_TILE_HOST std::ostream& LogRangeAsType(std::ostream& os,
                                          Range&& range,
                                          std::string delim,
                                          int precision = std::cout.precision(),
                                          int width     = 0)
{
    bool first = true;
    for(auto&& v : range)
    {
        if(first)
            first = false;
        else
            os << delim;
        os << std::setw(width) << std::setprecision(precision) << static_cast<T>(v);
    }
    return os;
}

template <typename F, typename T, std::size_t... Is>
CK_TILE_HOST auto call_f_unpack_args_impl(F f, T args, std::index_sequence<Is...>)
{
    return f(std::get<Is>(args)...);
}

template <typename F, typename T>
CK_TILE_HOST auto call_f_unpack_args(F f, T args)
{
    constexpr std::size_t N = std::tuple_size<T>{};

    return call_f_unpack_args_impl(f, args, std::make_index_sequence<N>{});
}

template <typename F, typename T, std::size_t... Is>
CK_TILE_HOST auto construct_f_unpack_args_impl(T args, std::index_sequence<Is...>)
{
    return F(std::get<Is>(args)...);
}

template <typename F, typename T>
CK_TILE_HOST auto construct_f_unpack_args(F, T args)
{
    constexpr std::size_t N = std::tuple_size<T>{};

    return construct_f_unpack_args_impl<F>(args, std::make_index_sequence<N>{});
}

/**
 * @brief Descriptor for tensors in host memory.
 *
 * HostTensorDescriptor manages the shape (dimensions) and memory layout (strides)
 * of a tensor in host memory. It provides functionality to:
 * - Store tensor dimensions and strides
 * - Calculate default strides for contiguous memory layout
 * - Convert multi-dimensional indices to linear memory offsets
 * - Query tensor metadata (dimensions, element counts, etc.)
 *
 * The class supports both automatic stride calculation for contiguous memory layout
 * and custom strides for more complex memory patterns.
 */
struct HostTensorDescriptor
{
    HostTensorDescriptor() = default;

    void CalculateStrides()
    {
        mStrides.clear();
        mStrides.resize(mLens.size(), 0);
        if(mStrides.empty())
            return;

        mStrides.back() = 1;
        std::partial_sum(mLens.rbegin(),
                         mLens.rend() - 1,
                         mStrides.rbegin() + 1,
                         std::multiplies<std::size_t>());
    }

    template <typename X, typename = std::enable_if_t<std::is_convertible_v<X, std::size_t>>>
    HostTensorDescriptor(const std::initializer_list<X>& lens) : mLens(lens.begin(), lens.end())
    {
        this->CalculateStrides();
    }

    template <typename Lengths,
              typename = std::enable_if_t<
                  std::is_convertible_v<ck_tile::ranges::range_value_t<Lengths>, std::size_t>>>
    HostTensorDescriptor(const Lengths& lens) : mLens(lens.begin(), lens.end())
    {
        this->CalculateStrides();
    }

    template <typename X,
              typename Y,
              typename = std::enable_if_t<std::is_convertible_v<X, std::size_t> &&
                                          std::is_convertible_v<Y, std::size_t>>>
    HostTensorDescriptor(const std::initializer_list<X>& lens,
                         const std::initializer_list<Y>& strides)
        : mLens(lens.begin(), lens.end()), mStrides(strides.begin(), strides.end())
    {
    }

    template <typename Lengths,
              typename Strides,
              typename = std::enable_if_t<
                  std::is_convertible_v<ck_tile::ranges::range_value_t<Lengths>, std::size_t> &&
                  std::is_convertible_v<ck_tile::ranges::range_value_t<Strides>, std::size_t>>>
    HostTensorDescriptor(const Lengths& lens, const Strides& strides)
        : mLens(lens.begin(), lens.end()), mStrides(strides.begin(), strides.end())
    {
    }

    std::size_t get_num_of_dimension() const { return mLens.size(); }
    /**
     * @brief Calculates the total number of elements in the tensor.
     *
     * Computes the product of all dimension lengths to determine the
     * total element count in the tensor.
     *
     * @pre The lengths array (mLens) and strides array (mStrides) must have
     *      the same size.
     *
     * @return The total number of elements in the tensor.
     */
    std::size_t get_element_size() const
    {
        assert(mLens.size() == mStrides.size());
        return std::accumulate(
            mLens.begin(), mLens.end(), std::size_t{1}, std::multiplies<std::size_t>());
    }
    /**
     * @brief Calculates the total element space required for the tensor in memory.
     *
     * This method computes the minimum size of contiguous memory needed to store
     * all elements of the tensor, taking into account the tensor's dimensions and
     * strides. The calculation is based on the formula: 1 + max((length_i - 1) * stride_i)
     * across all dimensions.
     *
     * Dimensions with length 0 are skipped in this calculation.
     *
     * @return The size of the tensor's element space (number of elements).
     */
    std::size_t get_element_space_size() const
    {
        std::size_t space = 1;
        for(std::size_t i = 0; i < mLens.size(); ++i)
        {
            if(mLens[i] == 0)
                continue;

            space += (mLens[i] - 1) * mStrides[i];
        }
        return space;
    }

    std::size_t get_length(std::size_t dim) const { return mLens[dim]; }

    const std::vector<std::size_t>& get_lengths() const { return mLens; }

    std::size_t get_stride(std::size_t dim) const { return mStrides[dim]; }

    const std::vector<std::size_t>& get_strides() const { return mStrides; }

    /**
     * @brief Calculates the linear offset from multi-dimensional indices.
     *
     * Converts a set of N-dimensional indices into a single linear offset by computing
     * the inner product of the indices with the tensor's strides.
     *
     * @tparam Is Parameter pack of index types (should be convertible to std::size_t)
     * @param is Variable number of indices, one for each dimension of the tensor
     * @return std::size_t Linear offset corresponding to the given multi-dimensional indices
     *
     * @pre The number of indices must match the number of dimensions in the tensor
     */
    template <typename... Is>
    std::size_t GetOffsetFromMultiIndex(Is... is) const
    {
        assert(sizeof...(Is) == this->get_num_of_dimension());
        std::initializer_list<std::size_t> iss{static_cast<std::size_t>(is)...};
        return std::inner_product(iss.begin(), iss.end(), mStrides.begin(), std::size_t{0});
    }

    /**
     * @brief Calculates the linear memory offset from a multi-dimensional index
     *
     * Computes the linear offset by performing an inner product between the provided
     * multi-dimensional indices and the tensor's strides.
     *
     * @param iss Vector containing the multi-dimensional indices
     * @return The calculated linear offset as a size_t
     */
    std::size_t GetOffsetFromMultiIndex(const std::vector<std::size_t>& iss) const
    {
        return std::inner_product(iss.begin(), iss.end(), mStrides.begin(), std::size_t{0});
    }

    friend std::ostream& operator<<(std::ostream& os, const HostTensorDescriptor& desc)
    {
        os << "dim " << desc.get_num_of_dimension() << ", ";

        os << "lengths {";
        LogRange(os, desc.get_lengths(), ", ");
        os << "}, ";

        os << "strides {";
        LogRange(os, desc.get_strides(), ", ");
        os << "}";

        return os;
    }

    private:
    std::vector<std::size_t> mLens;    ///< Lengths of each dimension
    std::vector<std::size_t> mStrides; ///< Strides for each dimension
};

template <typename New2Old>
CK_TILE_HOST HostTensorDescriptor transpose_host_tensor_descriptor_given_new2old(
    const HostTensorDescriptor& a, const New2Old& new2old)
{
    std::vector<std::size_t> new_lengths(a.get_num_of_dimension());
    std::vector<std::size_t> new_strides(a.get_num_of_dimension());

    for(std::size_t i = 0; i < a.get_num_of_dimension(); i++)
    {
        new_lengths[i] = a.get_lengths()[new2old[i]];
        new_strides[i] = a.get_strides()[new2old[i]];
    }

    return HostTensorDescriptor(new_lengths, new_strides);
}

template <typename F, typename... Xs>
struct ParallelTensorFunctor
{
    F mF;
    static constexpr std::size_t NDIM = sizeof...(Xs);
    std::array<std::size_t, NDIM> mLens;
    std::array<std::size_t, NDIM> mStrides;
    std::size_t mN1d;

    ParallelTensorFunctor(F f, Xs... xs) : mF(f), mLens({static_cast<std::size_t>(xs)...})
    {
        mStrides.back() = 1;
        std::partial_sum(mLens.rbegin(),
                         mLens.rend() - 1,
                         mStrides.rbegin() + 1,
                         std::multiplies<std::size_t>());
        mN1d = mStrides[0] * mLens[0];
    }

    std::array<std::size_t, NDIM> GetNdIndices(std::size_t i) const
    {
        std::array<std::size_t, NDIM> indices;

        for(std::size_t idim = 0; idim < NDIM; ++idim)
        {
            indices[idim] = i / mStrides[idim];
            i -= indices[idim] * mStrides[idim];
        }

        return indices;
    }

    void operator()(std::size_t num_thread = 1) const
    {
        std::size_t work_per_thread = (mN1d + num_thread - 1) / num_thread;

        std::vector<joinable_thread> threads(num_thread);

        for(std::size_t it = 0; it < num_thread; ++it)
        {
            std::size_t iw_begin = it * work_per_thread;
            std::size_t iw_end   = std::min((it + 1) * work_per_thread, mN1d);

            auto f = [this, iw_begin, iw_end] {
                for(std::size_t iw = iw_begin; iw < iw_end; ++iw)
                {
                    call_f_unpack_args(this->mF, this->GetNdIndices(iw));
                }
            };
            threads[it] = joinable_thread(f);
        }
    }
};

template <typename F, typename... Xs>
CK_TILE_HOST auto make_ParallelTensorFunctor(F f, Xs... xs)
{
    return ParallelTensorFunctor<F, Xs...>(f, xs...);
}

template <typename T>
struct HostTensor
{
    using Descriptor = HostTensorDescriptor;
    using Data       = std::vector<T>;

    template <typename X>
    HostTensor(std::initializer_list<X> lens) : mDesc(lens), mData(get_element_space_size())
    {
    }

    template <typename X, typename Y>
    HostTensor(std::initializer_list<X> lens, std::initializer_list<Y> strides)
        : mDesc(lens, strides), mData(get_element_space_size())
    {
    }

    template <typename Lengths>
    HostTensor(const Lengths& lens) : mDesc(lens), mData(get_element_space_size())
    {
    }

    template <typename Lengths, typename Strides>
    HostTensor(const Lengths& lens, const Strides& strides)
        : mDesc(lens, strides), mData(get_element_space_size())
    {
    }

    HostTensor(const Descriptor& desc) : mDesc(desc), mData(get_element_space_size()) {}

    template <typename OutT>
    HostTensor<OutT> CopyAsType() const
    {
        HostTensor<OutT> ret(mDesc);
        std::transform(mData.cbegin(), mData.cend(), ret.mData.begin(), [](auto value) {
            return ck_tile::type_convert<OutT>(value);
        });
        return ret;
    }

    HostTensor()                  = delete;
    HostTensor(const HostTensor&) = default;
    HostTensor(HostTensor&&)      = default;

    ~HostTensor() = default;

    HostTensor& operator=(const HostTensor&) = default;
    HostTensor& operator=(HostTensor&&)      = default;

    template <typename FromT>
    explicit HostTensor(const HostTensor<FromT>& other) : HostTensor(other.template CopyAsType<T>())
    {
    }

    std::size_t get_length(std::size_t dim) const { return mDesc.get_length(dim); }

    decltype(auto) get_lengths() const { return mDesc.get_lengths(); }

    std::size_t get_stride(std::size_t dim) const { return mDesc.get_stride(dim); }

    decltype(auto) get_strides() const { return mDesc.get_strides(); }

    std::size_t get_num_of_dimension() const { return mDesc.get_num_of_dimension(); }

    std::size_t get_element_size() const { return mDesc.get_element_size(); }

    std::size_t get_element_space_size() const
    {
        constexpr index_t PackedSize = ck_tile::numeric_traits<remove_cvref_t<T>>::PackedSize;
        return mDesc.get_element_space_size() / PackedSize;
    }

    std::size_t get_element_space_size_in_bytes() const
    {
        return sizeof(T) * get_element_space_size();
    }

    // void SetZero() { ck_tile::ranges::fill<T>(mData, 0); }
    void SetZero()
    {
        // if constexpr(std::is_same_v<T, e8m0_t>)
        //     std::fill(mData.begin(), mData.end(), e8m0_t{1.f});
        // else
            std::fill(mData.begin(), mData.end(), 0);
    }

    template <typename F>
    void ForEach_impl(F&& f, std::vector<size_t>& idx, size_t rank)
    {
        if(rank == mDesc.get_num_of_dimension())
        {
            f(*this, idx);
            return;
        }
        // else
        for(size_t i = 0; i < mDesc.get_lengths()[rank]; i++)
        {
            idx[rank] = i;
            ForEach_impl(std::forward<F>(f), idx, rank + 1);
        }
    }

    template <typename F>
    void ForEach(F&& f)
    {
        std::vector<size_t> idx(mDesc.get_num_of_dimension(), 0);
        ForEach_impl(std::forward<F>(f), idx, size_t(0));
    }

    template <typename F>
    void ForEach_impl(const F&& f, std::vector<size_t>& idx, size_t rank) const
    {
        if(rank == mDesc.get_num_of_dimension())
        {
            f(*this, idx);
            return;
        }
        // else
        for(size_t i = 0; i < mDesc.get_lengths()[rank]; i++)
        {
            idx[rank] = i;
            ForEach_impl(std::forward<const F>(f), idx, rank + 1);
        }
    }

    template <typename F>
    void ForEach(const F&& f) const
    {
        std::vector<size_t> idx(mDesc.get_num_of_dimension(), 0);
        ForEach_impl(std::forward<const F>(f), idx, size_t(0));
    }

    template <typename G>
    void GenerateTensorValue(G g, std::size_t num_thread = 1)
    {
        switch(mDesc.get_num_of_dimension())
        {
        case 1: {
            auto f = [&](auto i) { (*this)(i) = g(i); };
            make_ParallelTensorFunctor(f, mDesc.get_lengths()[0])(num_thread);
            break;
        }
        case 2: {
            auto f = [&](auto i0, auto i1) { (*this)(i0, i1) = g(i0, i1); };
            make_ParallelTensorFunctor(f, mDesc.get_lengths()[0], mDesc.get_lengths()[1])(
                num_thread);
            break;
        }
        case 3: {
            auto f = [&](auto i0, auto i1, auto i2) { (*this)(i0, i1, i2) = g(i0, i1, i2); };
            make_ParallelTensorFunctor(f,
                                       mDesc.get_lengths()[0],
                                       mDesc.get_lengths()[1],
                                       mDesc.get_lengths()[2])(num_thread);
            break;
        }
        case 4: {
            auto f = [&](auto i0, auto i1, auto i2, auto i3) {
                (*this)(i0, i1, i2, i3) = g(i0, i1, i2, i3);
            };
            make_ParallelTensorFunctor(f,
                                       mDesc.get_lengths()[0],
                                       mDesc.get_lengths()[1],
                                       mDesc.get_lengths()[2],
                                       mDesc.get_lengths()[3])(num_thread);
            break;
        }
        case 5: {
            auto f = [&](auto i0, auto i1, auto i2, auto i3, auto i4) {
                (*this)(i0, i1, i2, i3, i4) = g(i0, i1, i2, i3, i4);
            };
            make_ParallelTensorFunctor(f,
                                       mDesc.get_lengths()[0],
                                       mDesc.get_lengths()[1],
                                       mDesc.get_lengths()[2],
                                       mDesc.get_lengths()[3],
                                       mDesc.get_lengths()[4])(num_thread);
            break;
        }
        case 6: {
            auto f = [&](auto i0, auto i1, auto i2, auto i3, auto i4, auto i5) {
                (*this)(i0, i1, i2, i3, i4, i5) = g(i0, i1, i2, i3, i4, i5);
            };
            make_ParallelTensorFunctor(f,
                                       mDesc.get_lengths()[0],
                                       mDesc.get_lengths()[1],
                                       mDesc.get_lengths()[2],
                                       mDesc.get_lengths()[3],
                                       mDesc.get_lengths()[4],
                                       mDesc.get_lengths()[5])(num_thread);
            break;
        }
        default: throw std::runtime_error("unspported dimension");
        }
    }

    template <typename... Is>
    std::size_t GetOffsetFromMultiIndex(Is... is) const
    {
        constexpr index_t PackedSize = ck_tile::numeric_traits<remove_cvref_t<T>>::PackedSize;
        return mDesc.GetOffsetFromMultiIndex(is...) / PackedSize;
    }

    template <typename... Is>
    T& operator()(Is... is)
    {
        return mData[GetOffsetFromMultiIndex(is...)];
    }

    template <typename... Is>
    const T& operator()(Is... is) const
    {
        return mData[GetOffsetFromMultiIndex(is...)];
    }

    T& operator()(const std::vector<std::size_t>& idx)
    {
        return mData[GetOffsetFromMultiIndex(idx)];
    }

    const T& operator()(const std::vector<std::size_t>& idx) const
    {
        return mData[GetOffsetFromMultiIndex(idx)];
    }

    HostTensor<T> transpose(std::vector<size_t> axes = {}) const
    {
        if(axes.empty())
        {
            axes.resize(this->get_num_of_dimension());
            std::iota(axes.rbegin(), axes.rend(), 0);
        }
        if(axes.size() != mDesc.get_num_of_dimension())
        {
            throw std::runtime_error(
                "HostTensor::transpose(): size of axes must match tensor dimension");
        }
        std::vector<size_t> tlengths, tstrides;
        for(const auto& axis : axes)
        {
            tlengths.push_back(get_lengths()[axis]);
            tstrides.push_back(get_strides()[axis]);
        }
        HostTensor<T> ret(*this);
        ret.mDesc = HostTensorDescriptor(tlengths, tstrides);
        return ret;
    }

    HostTensor<T> transpose(std::vector<size_t> axes = {})
    {
        return const_cast<HostTensor<T> const*>(this)->transpose(axes);
    }

    typename Data::iterator begin() { return mData.begin(); }

    typename Data::iterator end() { return mData.end(); }

    typename Data::pointer data() { return mData.data(); }

    typename Data::const_iterator begin() const { return mData.begin(); }

    typename Data::const_iterator end() const { return mData.end(); }

    typename Data::const_pointer data() const { return mData.data(); }

    typename Data::size_type size() const { return mData.size(); }

    // return a slice of this tensor
    // for simplicity we just copy the data and return a new tensor
    auto slice(std::vector<size_t> s_begin, std::vector<size_t> s_end) const
    {
        assert(s_begin.size() == s_end.size());
        assert(s_begin.size() == get_num_of_dimension());

        std::vector<size_t> s_len(s_begin.size());
        std::transform(
            s_end.begin(), s_end.end(), s_begin.begin(), s_len.begin(), std::minus<size_t>{});
        HostTensor<T> sliced_tensor(s_len);

        sliced_tensor.ForEach([&](auto& self, auto idx) {
            std::vector<size_t> src_idx(idx.size());
            std::transform(
                idx.begin(), idx.end(), s_begin.begin(), src_idx.begin(), std::plus<size_t>{});
            self(idx) = operator()(src_idx);
        });

        return sliced_tensor;
    }

    template <typename U = T>
    auto AsSpan() const
    {
        constexpr std::size_t FromSize = sizeof(T);
        constexpr std::size_t ToSize   = sizeof(U);

        using Element = std::add_const_t<std::remove_reference_t<U>>;
        return ck_tile::span<Element>{reinterpret_cast<Element*>(data()),
                                      size() * FromSize / ToSize};
    }

    template <typename U = T>
    auto AsSpan()
    {
        constexpr std::size_t FromSize = sizeof(T);
        constexpr std::size_t ToSize   = sizeof(U);

        using Element = std::remove_reference_t<U>;
        return ck_tile::span<Element>{reinterpret_cast<Element*>(data()),
                                      size() * FromSize / ToSize};
    }

    /**
     * @brief Print only the first N elements of the tensor
     *
     * @param os Output stream to write to
     * @param n Number of elements to print (default: 5)
     * @return std::ostream& Reference to the output stream
     */
    std::ostream& print_first_n(std::ostream& os, std::size_t n = 5) const
    {
        os << mDesc;
        os << "[";
        for(typename Data::size_type idx = 0; idx < std::min(n, mData.size()); ++idx)
        {
            if(0 < idx)
            {
                os << ", ";
            }
            if constexpr(std::is_same_v<T, bf16_t> || std::is_same_v<T, fp16_t> ||
                         std::is_same_v<T, fp8_t> || std::is_same_v<T, bf8_t>)
            {
                os << type_convert<float>(mData[idx]) << " #### ";
            }
            else if constexpr(std::is_same_v<T, ck_tile::pk_int4_t>)
            {
                auto unpacked = pk_int4_t_to_int8x2_t(mData[idx]);
                os << "pk(" << static_cast<int>(unpacked[0]) << ", "
                   << static_cast<int>(unpacked[1]) << ") #### ";
            }
            else if constexpr(std::is_same_v<T, int8_t>)
            {
                os << static_cast<int>(mData[idx]);
            }
            else
            {
                os << mData[idx];
            }
        }
        if(mData.size() > n)
        {
            os << ", ...";
        }
        os << "]";
        return os;
    }

    friend std::ostream& operator<<(std::ostream& os, const HostTensor<T>& t)
    {
        os << t.mDesc;
        os << "[";
        for(typename Data::size_type idx = 0; idx < t.mData.size(); ++idx)
        {
            if(0 < idx)
            {
                os << ", ";
            }
            if constexpr(std::is_same_v<T, bf16_t> || std::is_same_v<T, fp16_t> ||
                         std::is_same_v<T, fp8_t> || std::is_same_v<T, bf8_t>)
            {
                os << type_convert<float>(t.mData[idx]) << " #### ";
            }
            else if constexpr(std::is_same_v<T, ck_tile::pk_int4_t>)
            {
                auto unpacked = pk_int4_t_to_int8x2_t(t.mData[idx]);
                os << "pk(" << static_cast<int>(unpacked[0]) << ", "
                   << static_cast<int>(unpacked[1]) << ") #### ";
            }
            else
            {
                os << t.mData[idx];
            }
        }
        os << "]";
        return os;
    }

    // read data from a file, as dtype
    // the file could dumped from torch as (targeting tensor is t here)
    // numpy.savetxt("f.txt", t.view(-1).numpy())
    // numpy.savetxt("f.txt", t.cpu().view(-1).numpy()) # from cuda to cpu to save
    // numpy.savetxt("f.txt", t.cpu().view(-1).numpy(), fmt="%d")   # save as int
    // will output f.txt, each line is a value
    // dtype=float or int, internally will cast to real type
    void loadtxt(std::string file_name, std::string dtype = "float")
    {
        std::ifstream file(file_name);

        if(file.is_open())
        {
            std::string line;

            index_t cnt = 0;
            while(std::getline(file, line))
            {
                if(cnt >= static_cast<index_t>(mData.size()))
                {
                    throw std::runtime_error(std::string("data read from file:") + file_name +
                                             " is too big");
                }

                if(dtype == "float")
                {
                    mData[cnt] = type_convert<T>(std::stof(line));
                }
                else if(dtype == "int" || dtype == "int32")
                {
                    mData[cnt] = type_convert<T>(std::stoi(line));
                }
                cnt++;
            }
            file.close();
            if(cnt < static_cast<index_t>(mData.size()))
            {
                std::cerr << "Warning! reading from file:" << file_name
                          << ", does not match the size of this tensor" << std::endl;
            }
        }
        else
        {
            // Print an error message to the standard error
            // stream if the file cannot be opened.
            throw std::runtime_error(std::string("unable to open file:") + file_name);
        }
    }

    // can save to a txt file and read from torch as:
    // torch.from_numpy(np.loadtxt('f.txt', dtype=np.int32/np.float32...)).view([...]).contiguous()
    void savetxt(std::string file_name, std::string dtype = "float")
    {
        std::ofstream file(file_name);

        if(file.is_open())
        {
            for(auto& itm : mData)
            {
                if(dtype == "float")
                    file << type_convert<float>(itm) << std::endl;
                else if(dtype == "int")
                    file << type_convert<int>(itm) << std::endl;
                else if(dtype == "int8_t")
                    file << static_cast<int>(type_convert<ck_tile::int8_t>(itm)) << std::endl;
                else
                    // TODO: we didn't implement operator<< for all custom
                    // data types, here fall back to float in case compile error
                    file << type_convert<float>(itm) << std::endl;
            }
            file.close();
        }
        else
        {
            // Print an error message to the standard error
            // stream if the file cannot be opened.
            throw std::runtime_error(std::string("unable to open file:") + file_name);
        }
    }

    Descriptor mDesc;
    Data mData;
};

/**
 * @brief Creates a host tensor descriptor with specified dimensions and layout
 *
 * Constructs a HostTensorDescriptor with appropriate strides based on whether the tensor
 * layout is row-major or column-major. This is determined via the compile-time template
 * parameter `is_row_major`.
 *
 * @tparam is_row_major Compile-time flag indicating if the layout is row-major (true) or
 * column-major (false)
 *
 * @param row Number of rows in the tensor
 * @param col Number of columns in the tensor
 * @param stride Stride between adjacent rows (for row-major) or columns (for column-major)
 *
 * @return HostTensorDescriptor with shape {row, col} and strides:
 *         - For row-major: {stride, 1}
 *         - For column-major: {1, stride}
 */
template <bool is_row_major>
auto host_tensor_descriptor(std::size_t row,
                            std::size_t col,
                            std::size_t stride,
                            bool_constant<is_row_major>)
{
    using namespace ck_tile::literals;

    if constexpr(is_row_major)
    {
        return HostTensorDescriptor({row, col}, {stride, 1_uz});
    }
    else
    {
        return HostTensorDescriptor({row, col}, {1_uz, stride});
    }
}

template <bool is_row_major>
auto get_default_stride(std::size_t row,
                        std::size_t col,
                        std::size_t stride,
                        bool_constant<is_row_major>)
{
    if(stride == 0)
    {
        if constexpr(is_row_major)
        {
            return col;
        }
        else
        {
            return row;
        }
    }
    else
        return stride;
}
} // namespace ck_tile
