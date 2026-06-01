// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, , Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_attribute_mmac_impl.hpp"

namespace ck_tile {

template <typename WarpGemmAttributeMmacImpl_,
          index_t MRepeat,
          index_t NRepeat,
          index_t MInterleave,
          index_t NInterleave,
          index_t kKIter>
struct WarpGemmAttributeMmacIterateK
{
    static_assert(kKIter > 0, "wrong!");

    using Impl = remove_cvref_t<WarpGemmAttributeMmacImpl_>;

    using ADataType = typename Impl::ADataType;
    using BDataType = typename Impl::BDataType;
    using CDataType = typename Impl::CDataType;

    // ext_vector_t<datatype, ext_vector_t>, 2d array 2 x vec_a
    using ABufType = thread_buffer<typename Impl::AVecType, kKIter * MRepeat * MInterleave>;
    using BBufType = thread_buffer<typename Impl::BVecType, kKIter * NRepeat * NInterleave>;
    using CBufType =
        thread_buffer<typename Impl::CVecType, MRepeat * NRepeat * MInterleave * NInterleave>;

    static constexpr index_t kM = Impl::kM * MRepeat * MInterleave;
    static constexpr index_t kN = Impl::kN * NRepeat * NInterleave;
    static constexpr index_t kK = Impl::kK * kKIter;
    static constexpr index_t kKPerThread = Impl::kABKPerLane * kKIter;

    using AWarpDstrEncoding =
        tile_distribution_encoding<sequence<>,
                                   tuple<sequence<MRepeat, Impl::kAMLane, MInterleave>,
                                         sequence<Impl::kABKLane, Impl::kABKPerLane * kKIter>>,
                                   tuple<sequence<2, 1>>,
                                   tuple<sequence<0, 1>>,
                                   sequence<1, 1, 2>,
                                   sequence<0, 2, 1>>;

    using BWarpDstrEncoding =
        tile_distribution_encoding<sequence<>,
                                   tuple<sequence<NRepeat, Impl::kBNLane, NInterleave>,
                                         sequence<Impl::kABKLane, Impl::kABKPerLane * kKIter>>,
                                   tuple<sequence<2, 1>>,
                                   tuple<sequence<0, 1>>,
                                   sequence<1, 1, 2>,
                                   sequence<0, 2, 1>>;

    using CWarpDstrEncoding = tile_distribution_encoding<
        sequence<>,
        tuple<sequence<MRepeat, Impl::kCMLane, MInterleave, Impl::kCMPerLane>,
              sequence<NRepeat, Impl::kCNLane, NInterleave, Impl::kCN0PerLane, Impl::kCN1PerLane>>,
        tuple<sequence<2, 1>>,
        tuple<sequence<1, 1>>,
        sequence<1, 2, 1, 1, 2, 2, 2>, // <MRepeat, NRepeat, MInterleave, MPerLane, N0PerLane,
                                       // NmmacInterleave, N1PerLane>
        sequence<0, 0, 2, 3, 3, 2, 4>>;

    using CWarpOutputDstrEncoding = tile_distribution_encoding<
        sequence<>,
        tuple<sequence<MRepeat, Impl::kCMLane, MInterleave, Impl::kCMPerLane>,
              sequence<NRepeat, Impl::kCN0PerLane, Impl::kCNLane, NInterleave * Impl::kCN1PerLane>>,
        tuple<sequence<2, 1>>,
        tuple<sequence<2, 1>>,
        sequence<1, 2, 1, 1, 2, 2>, // <MRepeat, NRepeat, MInterleave, MPerLane, N0PerLane,
                                    // NmmacInterleave, N1PerLane>
        sequence<0, 0, 2, 3, 1, 3>>;

    // c_vec += a_vec * b_vec
    CK_TILE_DEVICE void
    operator()(CBufType& c_buf, const ABufType& a_buf, const BBufType& b_buf) const
    {
        static_for<0, MRepeat, 1>{}([&](auto iMRepeat) {
            static_for<0, NRepeat, 1>{}([&](auto iNRepeat) {
                static_for<0, MInterleave, 1>{}([&](auto iMInterleave) {
                    static_for<0, NInterleave, 1>{}([&](auto iNInterleave) {
                        static_for<0, kKIter, 1>{}([&](auto iKIter) {
                            // 1 mmac issue
                            Impl{}(c_buf(iMRepeat * NRepeat * MInterleave * NInterleave +
                                         iNRepeat * MInterleave * NInterleave +
                                         iMInterleave * NInterleave + iNInterleave),
                                   a_buf[iMRepeat * MInterleave * kKIter + iMInterleave * kKIter +
                                         iKIter],
                                   b_buf[iNRepeat * NInterleave * kKIter + iNInterleave * kKIter +
                                         iKIter]);
                        });
                    });
                });
            });
        });
    }

    // c_vec = a_vec * b_vec
    // AVecType = ext_vector_t<ADataType, AWarpTensor::get_thread_buffer_size()>;
    CK_TILE_DEVICE CBufType operator()(const ABufType& a_buf, const BBufType& b_buf) const
    {
        CBufType c_buf;

        // c += a * b
        static_for<0, MRepeat, 1>{}([&](auto iMRepeat) {
            static_for<0, NRepeat, 1>{}([&](auto iNRepeat) {
                static_for<0, MInterleave, 1>{}([&](auto iMInterleave) {
                    static_for<0, NInterleave, 1>{}([&](auto iNInterleave) {
                        static_for<0, kKIter, 1>{}([&](auto iKIter) {
                            // 1 mmac issue
                            Impl{}(c_buf(iMRepeat * NRepeat * MInterleave * NInterleave +
                                         iNRepeat * MInterleave * NInterleave +
                                         iMInterleave * NInterleave + iNInterleave),
                                   a_buf[iMRepeat * MInterleave * kKIter + iMInterleave * kKIter +
                                         iKIter],
                                   b_buf[iNRepeat * NInterleave * kKIter + iNInterleave * kKIter +
                                         iKIter]);
                        });
                    });
                });
            });
        });

        return c_buf;
    }

    template <typename CWarpTensor>
    CK_TILE_DEVICE auto MakeCOutputLayout(const CWarpTensor& c_warp_tensor) const
    {
        constexpr auto c_warp_output_distribution =
            make_static_tile_distribution(CWarpOutputDstrEncoding{});
        auto c_warp_output_tensor =
            make_static_distributed_tensor<CDataType>(c_warp_output_distribution);

        static_for<0, MRepeat, 1>{}([&](auto iMR) {
            static_for<0, NRepeat, 1>{}([&](auto iNR) {
                static_for<0, MInterleave, 1>{}([&](auto iMI) {
                    static_for<0, NInterleave, 1>{}([&](auto iNI) {
                        static_for<0, Impl::kCN0PerLane, 1>{}([&](auto iCN0) {
                            c_warp_output_tensor.set_y_sliced_thread_data(
                                sequence<iMR, iNR, iMI, 0, iCN0, iNI>{},      // output_orgin
                                sequence<1, 1, 1, 1, 1, Impl::kCN1PerLane>{}, // output_set_length
                                c_warp_tensor.get_y_sliced_thread_data( // get calculate tensor and
                                                                        // change to output layout
                                    sequence<iMR, iNR, iMI, 0, iNI, iCN0, 0>{},
                                    sequence<1, 1, 1, 1, 1, 1, Impl::kCN1PerLane>{}));
                        });
                    });
                });
            });
        });

        return c_warp_output_tensor;
    }
};

template <typename WarpGemmAttributeMmacImpl_,
          index_t MRepeat_,
          index_t NRepeat_,
          index_t MInterleave_,
          index_t NInterleave_,
          index_t kKIter>
struct WarpGemmAttributeMmacIterateKTransC
    : public WarpGemmAttributeMmacIterateK<WarpGemmAttributeMmacImpl_,
                                           MRepeat_,
                                           NRepeat_,
                                           MInterleave_,
                                           NInterleave_,
                                           kKIter>
{
    using Base = WarpGemmAttributeMmacIterateK<WarpGemmAttributeMmacImpl_,
                                               MRepeat_,
                                               NRepeat_,
                                               MInterleave_,
                                               NInterleave_,
                                               kKIter>;

    using Impl = typename Base::Impl;

    using ADataType = typename Base::ADataType;
    using BDataType = typename Base::BDataType;
    using CDataType = typename Base::CDataType;

    using ABufType = typename Base::ABufType;
    using BBufType = typename Base::BBufType;
    using CBufType = typename Base::CBufType;

    static constexpr index_t kM          = Base::kM;
    static constexpr index_t kN          = Base::kN;
    static constexpr index_t kK          = Base::kK;
    static constexpr index_t MRepeat     = MRepeat_;
    static constexpr index_t NRepeat     = NRepeat_;
    static constexpr index_t MInterleave = MInterleave_;
    static constexpr index_t NInterleave = NInterleave_;

    using AWarpDstrEncoding = typename Base::AWarpDstrEncoding;

    using BWarpDstrEncoding = typename Base::BWarpDstrEncoding;

    // clang-format off
    // lane cluster: {Impl::kCMLane, Impl::kCNLane}
    // lane slice: {MRepeat,NRepeat, MInterleave, NInterleave, Impl::kCNPerLane, Impl::kCM0PerLane, Impl::kCM1PerLane}
    // clang-format on
    using CWarpDstrEncoding = tile_distribution_encoding<
        sequence<>,
        tuple<sequence<MRepeat, Impl::kCM0PerLane, Impl::kCMLane, Impl::kCM1PerLane, MInterleave>,
              sequence<NRepeat, Impl::kCNLane, Impl::kCNPerLane, NInterleave>>,
        tuple<sequence<1, 2>>,
        tuple<sequence<2, 1>>,
        sequence<1, 2, 1, 2, 2, 1, 1>,
        sequence<0, 0, 4, 3, 2, 1, 3>>;

    // clang-format off
    // lane cluster: {Impl::kCMLane, Impl::kCNLane}
    // lane slice: {MRepeat, NRepeat, Impl::kCM0PerLane, Impl::kCM1PerLane, MInterleave, Impl::kCNPerLane, NInterleave}
    // clang-format on
    using CWarpPermuteDstrEncoding = tile_distribution_encoding<
        sequence<>,
        tuple<sequence<MRepeat, Impl::kCM0PerLane, Impl::kCMLane, Impl::kCM1PerLane, MInterleave>,
              sequence<NRepeat, Impl::kCNLane, Impl::kCNPerLane, NInterleave>>,
        tuple<sequence<1, 2>>,
        tuple<sequence<2, 1>>,
        sequence<1, 2, 1, 1, 1, 2, 2>,
        sequence<0, 0, 1, 3, 4, 2, 3>>;

    CK_TILE_DEVICE void
    operator()(CBufType& c_buf, const ABufType& a_buf, const BBufType& b_buf) const
    {
        static_for<0, MRepeat, 1>{}([&](auto iMRepeat) {
            static_for<0, NRepeat, 1>{}([&](auto iNRepeat) {
                static_for<0, MInterleave, 1>{}([&](auto iMInterleave) {
                    static_for<0, NInterleave, 1>{}([&](auto iNInterleave) {
                        static_for<0, kKIter, 1>{}([&](auto iKIter) {
                            // 1 mmac issue
                            Impl{}(c_buf(iMRepeat * NRepeat * MInterleave * NInterleave +
                                         iNRepeat * MInterleave * NInterleave +
                                         iMInterleave * NInterleave + iNInterleave),
                                   a_buf[iMRepeat * MInterleave * kKIter + iMInterleave * kKIter +
                                         iKIter],
                                   b_buf[iNRepeat * NInterleave * kKIter + iNInterleave * kKIter +
                                         iKIter]);
                        });
                    });
                });
            });
        });
    }

    template <typename CWarpTensor>
    CK_TILE_DEVICE static auto GetCWarpPermuteTensor(const CWarpTensor& c_warp_tensor)
    {
        constexpr auto c_warp_permute_dstr =
            make_static_tile_distribution(CWarpPermuteDstrEncoding{});

        auto c_warp_permute_tensor = make_static_distributed_tensor<CDataType>(c_warp_permute_dstr);

        using SFC = space_filling_curve<sequence<MRepeat,
                                                 NRepeat,
                                                 MInterleave,
                                                 NInterleave,
                                                 Impl::kCNPerLane,
                                                 Impl::kCM0PerLane,
                                                 Impl::kCM1PerLane>,
                                        sequence<0, 1, 2, 3, 4, 5, 6>,
                                        sequence<1, 1, 1, 1, 1, 1, 1>>;

        constexpr auto num_access = SFC::get_num_of_access();

        static_for<0, num_access, 1>{}([&](auto access_id) {
            constexpr auto idx = SFC::get_index(access_id);

            constexpr auto iMR  = idx.at(number<0>{});
            constexpr auto iNR  = idx.at(number<1>{});
            constexpr auto iMI  = idx.at(number<2>{});
            constexpr auto iNI  = idx.at(number<3>{});
            constexpr auto iCN  = idx.at(number<4>{});
            constexpr auto iCM0 = idx.at(number<5>{});
            constexpr auto iCM1 = idx.at(number<6>{});

            c_warp_permute_tensor.set_y_sliced_thread_data(
                sequence<iMR, iNR, iCM0, iCM1, iMI, iCN, iNI>{},
                sequence<1, 1, 1, 1, 1, 1, 1>{},
                c_warp_tensor.get_y_sliced_thread_data(
                    sequence<iMR, iNR, iMI, iNI, iCN, iCM0, iCM1>{},
                    sequence<1, 1, 1, 1, 1, 1, 1>{}));
        });

        return c_warp_permute_tensor;
    }
};

// V2 version refers to KIterate stride =  warp_size * vec_a/b， not continuous vec_a/b
template <typename WarpGemmAttributeMmacImpl_,
          index_t MRepeat_,
          index_t NRepeat_,
          index_t MInterleave_,
          index_t NInterleave_,
          index_t kKIter>
struct WarpGemmAttributeMmacIterateKTransC_v2
    : WarpGemmAttributeMmacIterateKTransC<WarpGemmAttributeMmacImpl_,
                                          MRepeat_,
                                          NRepeat_,
                                          MInterleave_,
                                          NInterleave_,
                                          kKIter>
{
    using Base = WarpGemmAttributeMmacIterateKTransC<WarpGemmAttributeMmacImpl_,
                                                     MRepeat_,
                                                     NRepeat_,
                                                     MInterleave_,
                                                     NInterleave_,
                                                     kKIter>;
    using Impl = typename Base::Impl;
    static constexpr index_t MRepeat     = MRepeat_;
    static constexpr index_t NRepeat     = NRepeat_;
    static constexpr index_t MInterleave = MInterleave_;
    static constexpr index_t NInterleave = NInterleave_;

    using ADataType = typename Base::ADataType;
    using BDataType = typename Base::BDataType;
    using CDataType = typename Base::CDataType;

    using ABufType = typename Base::ABufType;
    using BBufType = typename Base::BBufType;
    using CBufType = typename Base::CBufType;

    using AWarpDstrEncoding =
        tile_distribution_encoding<sequence<>,
                                   tuple<sequence<MRepeat_, Impl::kAMLane, MInterleave_>,
                                         sequence<kKIter, Impl::kABKLane, Impl::kABKPerLane>>,
                                   tuple<sequence<2, 1>>,
                                   tuple<sequence<1, 1>>,
                                   sequence<1, 1, 2, 2>,
                                   sequence<0, 2, 0, 2>>;
    using BWarpDstrEncoding =
        tile_distribution_encoding<sequence<>,
                                   tuple<sequence<NRepeat_, Impl::kBNLane, NInterleave_>,
                                         sequence<kKIter, Impl::kABKLane, Impl::kABKPerLane>>,
                                   tuple<sequence<2, 1>>,
                                   tuple<sequence<1, 1>>,
                                   sequence<1, 1, 2, 2>,
                                   sequence<0, 2, 0, 2>>;
    using CWarpDstrEncoding = tile_distribution_encoding<
        sequence<>,
        tuple<sequence<MRepeat_, MInterleave_, Impl::kCM0PerLane, Impl::kCMLane, Impl::kCM1PerLane>,
              sequence<NRepeat_, NInterleave_, Impl::kCNLane, Impl::kCNPerLane>>,
        tuple<sequence<1, 2>>,
        tuple<sequence<3, 2>>,
        sequence<1, 2, 1, 2, 2, 1, 1>,
        sequence<0, 0, 1, 1, 3, 2, 4>>;

    CK_TILE_DEVICE void
    operator()(CBufType& c_buf, const ABufType& a_buf, const BBufType& b_buf) const
    {
        static_for<0, MRepeat, 1>{}([&](auto iMRepeat) {
            static_for<0, NRepeat, 1>{}([&](auto iNRepeat) {
                static_for<0, MInterleave, 1>{}([&](auto iMInterleave) {
                    static_for<0, NInterleave, 1>{}([&](auto iNInterleave) {
                        static_for<0, kKIter, 1>{}([&](auto iKIter) {
                            // 1 mmac issue
                            Impl{}(c_buf(iMRepeat * NRepeat * MInterleave * NInterleave +
                                         iNRepeat * MInterleave * NInterleave +
                                         iMInterleave * NInterleave + iNInterleave),
                                   a_buf[iMRepeat * MInterleave * kKIter + iMInterleave * kKIter +
                                         iKIter],
                                   b_buf[iNRepeat * NInterleave * kKIter + iNInterleave * kKIter +
                                         iKIter]);
                        });
                    });
                });
            });
        });
    }
};

template <typename WarpGemmAttributeMmacImpl_,
          index_t MRepeat,
          index_t NRepeat,
          index_t MInterleave,
          index_t NInterleave,
          index_t kKIter>
struct WarpGemmAttributeMmacIterateKTransC_ds128
    : public WarpGemmAttributeMmacIterateK<WarpGemmAttributeMmacImpl_,
                                           MRepeat,
                                           NRepeat,
                                           MInterleave,
                                           NInterleave,
                                           kKIter>
{
    using Base = WarpGemmAttributeMmacIterateK<WarpGemmAttributeMmacImpl_,
                                               MRepeat,
                                               NRepeat,
                                               MInterleave,
                                               NInterleave,
                                               kKIter>;

    using Impl = typename Base::Impl;

    using ADataType = typename Base::ADataType;
    using BDataType = typename Base::BDataType;
    using CDataType = typename Base::CDataType;

    using ABufType = typename Base::ABufType;
    using BBufType = typename Base::BBufType;
    using CBufType = typename Base::CBufType;

    static constexpr index_t kM = Base::kM;
    static constexpr index_t kN = Base::kN;
    static constexpr index_t kK = Base::kK;

    // ds_128
    static constexpr index_t kABKPerLane  = Impl::kABKPerLane * kKIter;
    static constexpr index_t kABKPerLane1 = 16 / sizeof(ADataType);
    static constexpr index_t kABKPerLane0 = kABKPerLane / kABKPerLane1;

    // static_assert(kABKPerLane % kABKPerLane1 == 0);

    using AWarpDstrEncoding =
        tile_distribution_encoding<sequence<>,
                                   tuple<sequence<MRepeat, Impl::kAMLane, MInterleave>,
                                         sequence<kABKPerLane0, Impl::kABKLane, kABKPerLane1>>,
                                   tuple<sequence<2, 1>>,
                                   tuple<sequence<1, 1>>,
                                   sequence<1, 1, 2, 2>,
                                   sequence<0, 2, 0, 2>>;

    using BWarpDstrEncoding =
        tile_distribution_encoding<sequence<>,
                                   tuple<sequence<NRepeat, Impl::kBNLane, NInterleave>,
                                         sequence<kABKPerLane0, Impl::kABKLane, kABKPerLane1>>,
                                   tuple<sequence<2, 1>>,
                                   tuple<sequence<1, 1>>,
                                   sequence<1, 1, 2, 2>,
                                   sequence<0, 2, 0, 2>>;

    // clang-format off
    // lane cluster: {Impl::kCMLane, Impl::kCNLane}
    // lane slice: {MRepeat,NRepeat, MInterleave, NInterleave, Impl::kCNPerLane, Impl::kCM0PerLane, Impl::kCM1PerLane}
    // clang-format on
    using CWarpDstrEncoding = tile_distribution_encoding<
        sequence<>,
        tuple<sequence<MRepeat, Impl::kCM0PerLane, Impl::kCMLane, Impl::kCM1PerLane, MInterleave>,
              sequence<NRepeat, Impl::kCNLane, Impl::kCNPerLane, NInterleave>>,
        tuple<sequence<1, 2>>,
        tuple<sequence<2, 1>>,
        sequence<1, 2, 1, 2, 2, 1, 1>,
        sequence<0, 0, 4, 3, 2, 1, 3>>;

    // FIXME: compatible
    using CWarpPermuteDstrEncoding = CWarpDstrEncoding;
        
    CK_TILE_DEVICE void
    operator()(CBufType& c_buf, const ABufType& a_buf, const BBufType& b_buf) const
    {
        static_for<0, MRepeat, 1>{}([&](auto iMRepeat) {
            static_for<0, NRepeat, 1>{}([&](auto iNRepeat) {
                static_for<0, MInterleave, 1>{}([&](auto iMInterleave) {
                    static_for<0, NInterleave, 1>{}([&](auto iNInterleave) {
                        static_for<0, kKIter, 1>{}([&](auto iKIter) {
                            // 1 mmac issue
                            Impl{}(c_buf(iMRepeat * NRepeat * MInterleave * NInterleave +
                                         iNRepeat * MInterleave * NInterleave +
                                         iMInterleave * NInterleave + iNInterleave),
                                   a_buf[iMRepeat * MInterleave * kKIter + iMInterleave * kKIter +
                                         iKIter],
                                   b_buf[iNRepeat * NInterleave * kKIter + iNInterleave * kKIter +
                                         iKIter]);
                        });
                    });
                });
            });
        });
    }
};

// MoE, for loading Matix B into swizzled LDS in the 2nd GEMM of MOE. K larger than 64
template <typename WarpGemmAttributeMmacImpl_,
          index_t MRepeat_,
          index_t NRepeat_,
          index_t MInterleave_,
          index_t NInterleave_,
          index_t kKIter>
struct WarpGemmAttributeMmacIterateKTransC_v3
    : WarpGemmAttributeMmacIterateKTransC<WarpGemmAttributeMmacImpl_,
                                          MRepeat_,
                                          NRepeat_,
                                          MInterleave_,
                                          NInterleave_,
                                          kKIter>
{
    static_assert((kKIter > 0) && (kKIter % 2 == 0), "kKIter wrong!");

    // V2 version refers to KIterate stride =  warp_size * vec_a/b， not continuous vec_a/b
    using Base = WarpGemmAttributeMmacIterateKTransC<WarpGemmAttributeMmacImpl_,
                                                     MRepeat_,
                                                     NRepeat_,
                                                     MInterleave_,
                                                     NInterleave_,
                                                     kKIter>;
    using Impl = typename Base::Impl;
    static constexpr index_t MRepeat     = MRepeat_;
    static constexpr index_t NRepeat     = NRepeat_;
    static constexpr index_t MInterleave = MInterleave_;
    static constexpr index_t NInterleave = NInterleave_;

    using ADataType = typename Base::ADataType;
    using BDataType = typename Base::BDataType;
    using CDataType = typename Base::CDataType;

    using ABufType = typename Base::ABufType;
    using BBufType = typename Base::BBufType;
    using CBufType = typename Base::CBufType;

    using AWarpDstrEncoding = tile_distribution_encoding<
        sequence<>,
        tuple<sequence<MRepeat_, Impl::kAMLane, MInterleave_>,
              sequence<kKIter / 2, Impl::kABKLane, Impl::kABKPerLane * 2>>, // multiply by 2:  make
                                                                            // ds_read_b64 to
                                                                            // ds_read_b128
        tuple<sequence<2, 1>>,
        tuple<sequence<1, 1>>,
        sequence<1, 1, 2, 2>,
        sequence<0, 2, 0, 2>>;
    using BWarpDstrEncoding = tile_distribution_encoding<
        sequence<>,
        tuple<sequence<NRepeat_, Impl::kBNLane, NInterleave_>,
              sequence<kKIter / 2, Impl::kABKLane, Impl::kABKPerLane * 2>>,
        tuple<sequence<2, 1>>,
        tuple<sequence<1, 1>>,
        sequence<1, 1, 2, 2>,
        sequence<0, 2, 0, 2>>;
    using CWarpDstrEncoding = tile_distribution_encoding<
        sequence<>,
        tuple<sequence<MRepeat_, MInterleave_, Impl::kCM0PerLane, Impl::kCMLane, Impl::kCM1PerLane>,
              sequence<NRepeat_, NInterleave_, Impl::kCNLane, Impl::kCNPerLane>>,
        tuple<sequence<1, 2>>,
        tuple<sequence<3, 2>>,
        sequence<1, 2, 1, 2, 2, 1, 1>,
        sequence<0, 0, 1, 1, 3, 2, 4>>;

    CK_TILE_DEVICE void
    operator()(CBufType& c_buf, const ABufType& a_buf, const BBufType& b_buf) const
    {
        static_for<0, MRepeat, 1>{}([&](auto iMRepeat) {
            static_for<0, NRepeat, 1>{}([&](auto iNRepeat) {
                static_for<0, MInterleave, 1>{}([&](auto iMInterleave) {
                    static_for<0, NInterleave, 1>{}([&](auto iNInterleave) {
                        static_for<0, kKIter, 1>{}([&](auto iKIter) {
                            // 1 mmac issue
                            Impl{}(c_buf(iMRepeat * NRepeat * MInterleave * NInterleave +
                                         iNRepeat * MInterleave * NInterleave +
                                         iMInterleave * NInterleave + iNInterleave),
                                   a_buf[iMRepeat * MInterleave * kKIter + iMInterleave * kKIter +
                                         iKIter],
                                   b_buf[iNRepeat * NInterleave * kKIter + iNInterleave * kKIter +
                                         iKIter]);
                        });
                    });
                });
            });
        });
    }
};

template <typename WarpGemmAttributeMmacImpl_,
          index_t MRepeat_,
          index_t NRepeat_,
          index_t MInterleave_,
          index_t NInterleave_,
          index_t kKIter>
struct WarpGemmAttributeMmacIterateKShuffle
    : public WarpGemmAttributeMmacIterateK<WarpGemmAttributeMmacImpl_,
                                           MRepeat_,
                                           NRepeat_,
                                           MInterleave_,
                                           NInterleave_,
                                           kKIter>
{
    static_assert(kKIter > 0, "wrong!");

    using Impl = remove_cvref_t<WarpGemmAttributeMmacImpl_>;
    using Base = WarpGemmAttributeMmacIterateK<WarpGemmAttributeMmacImpl_,
                                               MRepeat_,
                                               NRepeat_,
                                               MInterleave_,
                                               NInterleave_,
                                               kKIter>;

    using ADataType = typename Impl::ADataType;
    using BDataType = typename Impl::BDataType;
    using CDataType = typename Impl::CDataType;

    // ext_vector_t<datatype, ext_vector_t>, 2d array 2 x vec_a
    using ABufType = thread_buffer<typename Impl::AVecType, kKIter * MRepeat_ * MInterleave_>;
    using BBufType = thread_buffer<typename Impl::BVecType, kKIter * NRepeat_ * NInterleave_>;
    using CBufType =
        thread_buffer<typename Impl::CVecType, MRepeat_ * NRepeat_ * MInterleave_ * NInterleave_>;

    static constexpr index_t kM = Impl::kM * MRepeat_ * MInterleave_;
    static constexpr index_t kN = Impl::kN * NRepeat_ * NInterleave_;
    static constexpr index_t kK = Impl::kK * kKIter;

    using AWarpDstrEncoding =
        tile_distribution_encoding<sequence<>,
                                   tuple<sequence<MRepeat_, Impl::kAMLane, MInterleave_>,
                                         sequence<kKIter/2, Impl::kABKLane, Impl::kABKPerLane*2>>,
                                   tuple<sequence<2, 1>>,
                                   tuple<sequence<1, 1>>,
                                   sequence<1, 1, 2, 2>,
                                   sequence<0, 2, 0, 2>>;

    using BWarpDstrEncoding =
        tile_distribution_encoding<sequence<>,
                                   tuple<sequence<NRepeat_, Impl::kBNLane, NInterleave_>,
                                         sequence<kKIter/2,Impl::kABKLane, Impl::kABKPerLane * 2>>,
                                   tuple<sequence<2, 1>>,
                                   tuple<sequence<1, 1>>,
                                   sequence<1, 1, 2, 2>,
                                   sequence<0, 2, 0, 2>>;

    using CWarpDstrEncoding = typename Base::CWarpDstrEncoding;

    // c_vec += a_vec * b_vec
    CK_TILE_DEVICE void
    operator()(CBufType& c_buf, const ABufType& a_buf, const BBufType& b_buf) const
    {
        static_for<0, MRepeat_, 1>{}([&](auto iMRepeat) {
            static_for<0, NRepeat_, 1>{}([&](auto iNRepeat) {
                static_for<0, MInterleave_, 1>{}([&](auto iMInterleave) {
                    static_for<0, NInterleave_, 1>{}([&](auto iNInterleave) {
                        static_for<0, kKIter/2, 1>{}([&](auto iKIter) {
                            // 1 mmac issue
                            Impl{}(c_buf(iMRepeat * NRepeat_ * MInterleave_ * NInterleave_ +
                                         iNRepeat * MInterleave_ * NInterleave_ +
                                         iMInterleave * NInterleave_ + iNInterleave),
                                   a_buf[iMRepeat * MInterleave_ * kKIter + iMInterleave * kKIter +
                                         iKIter * 2],
                                   b_buf[iNRepeat * NInterleave_ * kKIter + iNInterleave * kKIter +
                                         iKIter * 2]);
                            Impl{}(c_buf(iMRepeat * NRepeat_ * MInterleave_ * NInterleave_ +
                                         iNRepeat * MInterleave_ * NInterleave_ +
                                         iMInterleave * NInterleave_ + iNInterleave),
                                   a_buf[iMRepeat * MInterleave_ * kKIter + iMInterleave * kKIter +
                                         iKIter * 2 + 1],
                                   b_buf[iNRepeat * NInterleave_ * kKIter + iNInterleave * kKIter +
                                         iKIter * 2 + 1]);
                        });
                    });
                });
            });
        });
    }

    // c_vec = a_vec * b_vec
    // AVecType = ext_vector_t<ADataType, AWarpTensor::get_thread_buffer_size()>;
    CK_TILE_DEVICE CBufType operator()(const ABufType& a_buf, const BBufType& b_buf) const
    {
        CBufType c_buf;

        // c += a * b
        static_for<0, MRepeat_, 1>{}([&](auto iMRepeat) {
            static_for<0, NRepeat_, 1>{}([&](auto iNRepeat) {
                static_for<0, MInterleave_, 1>{}([&](auto iMInterleave) {
                    static_for<0, NInterleave_, 1>{}([&](auto iNInterleave) {
                        static_for<0, kKIter/2, 1>{}([&](auto iKIter) {
                            // 1 mmac issue
                            Impl{}(c_buf(iMRepeat * NRepeat_ * MInterleave_ * NInterleave_ +
                                         iNRepeat * MInterleave_ * NInterleave_ +
                                         iMInterleave * NInterleave_ + iNInterleave),
                                   a_buf[iMRepeat * MInterleave_ * kKIter + iMInterleave * kKIter +
                                         iKIter * 2],
                                   b_buf[iNRepeat * NInterleave_ * kKIter + iNInterleave * kKIter +
                                         iKIter * 2]);
                            Impl{}(c_buf(iMRepeat * NRepeat_ * MInterleave_ * NInterleave_ +
                                         iNRepeat * MInterleave_ * NInterleave_ +
                                         iMInterleave * NInterleave_ + iNInterleave),
                                   a_buf[iMRepeat * MInterleave_ * kKIter + iMInterleave * kKIter +
                                         iKIter * 2 + 1],
                                   b_buf[iNRepeat * NInterleave_ * kKIter + iNInterleave * kKIter +
                                         iKIter * 2 + 1]);
                        });
                    });
                });
            });
        });

        return c_buf;
    }
};


template <typename WarpGemmAttributeMmacImpl_,
          index_t MRepeat_,
          index_t NRepeat_,
          index_t MInterleave_,
          index_t NInterleave_,
          index_t kKIter>
struct WarpGemmAttributeMmacIterateKTransC_Shuffle
    : public WarpGemmAttributeMmacIterateK<WarpGemmAttributeMmacImpl_,
                                           MRepeat_,
                                           NRepeat_,
                                           MInterleave_,
                                           NInterleave_,
                                           kKIter>
{
    using Base = WarpGemmAttributeMmacIterateKTransC<WarpGemmAttributeMmacImpl_,
                                                    MRepeat_,
                                                    NRepeat_,
                                                    MInterleave_,
                                                    NInterleave_,
                                                    kKIter>;

    using Impl = typename Base::Impl;

    using ADataType = typename Base::ADataType;
    using BDataType = typename Base::BDataType;
    using CDataType = typename Base::CDataType;

    using ABufType = typename Base::ABufType;
    using BBufType = typename Base::BBufType;
    using CBufType = typename Base::CBufType;

    static constexpr index_t kM          = Base::kM;
    static constexpr index_t kN          = Base::kN;
    static constexpr index_t kK          = Base::kK;
    static constexpr index_t MRepeat     = MRepeat_;
    static constexpr index_t NRepeat     = NRepeat_;
    static constexpr index_t MInterleave = MInterleave_;
    static constexpr index_t NInterleave = NInterleave_;

    using AWarpDstrEncoding =
        tile_distribution_encoding<sequence<>,
                                   tuple<sequence<MRepeat_, Impl::kAMLane, MInterleave_>,   // <1, 16, 1>
                                         sequence<kKIter/2, Impl::kABKLane, Impl::kABKPerLane*2>>,  // <8/2, 4, 4*2>
                                   tuple<sequence<2, 1>>,
                                   tuple<sequence<1, 1>>,
                                   sequence<1, 1, 2, 2>,
                                   sequence<0, 2, 0, 2>>;

    using BWarpDstrEncoding =
        tile_distribution_encoding<sequence<>,
                                   tuple<sequence<NRepeat_, Impl::kBNLane, NInterleave_>,   // <2, 16, 1>
                                         sequence<kKIter/2, Impl::kABKLane, Impl::kABKPerLane * 2>>, // <8/2, 4, 4*2>
                                   tuple<sequence<2, 1>>,
                                   tuple<sequence<1, 1>>,
                                   sequence<1, 1, 2, 2>,
                                   sequence<0, 2, 0, 2>>;

    // clang-format off
    // lane cluster: {Impl::kCMLane, Impl::kCNLane}
    // lane slice: {MRepeat,NRepeat, MInterleave, NInterleave, Impl::kCNPerLane, Impl::kCM0PerLane, Impl::kCM1PerLane}
    // clang-format on
    using CWarpDstrEncoding = tile_distribution_encoding<
        sequence<>,
        tuple<sequence<MRepeat, Impl::kCM0PerLane, Impl::kCMLane, Impl::kCM1PerLane, MInterleave>,  
              sequence<NRepeat, Impl::kCNLane, Impl::kCNPerLane, NInterleave>>,     // <2,16,1,1>
        tuple<sequence<1, 2>>,
        tuple<sequence<2, 1>>,
        sequence<1, 2, 1, 2, 2, 1, 1>,
        sequence<0, 0, 4, 3, 2, 1, 3>>;

    CK_TILE_DEVICE void
    operator()(CBufType& c_buf, const ABufType& a_buf, const BBufType& b_buf) const
    {
        static_for<0, MRepeat, 1>{}([&](auto iMRepeat) {
            static_for<0, NRepeat, 1>{}([&](auto iNRepeat) {
                static_for<0, MInterleave, 1>{}([&](auto iMInterleave) {
                    static_for<0, NInterleave, 1>{}([&](auto iNInterleave) {
                        static_for<0, kKIter/2, 1>{}([&](auto iKIter) {
                            // 1 mmac issue
                            Impl{}(c_buf(iMRepeat * NRepeat * MInterleave * NInterleave +
                                         iNRepeat * MInterleave * NInterleave +
                                         iMInterleave * NInterleave + iNInterleave),
                                   a_buf[iMRepeat * MInterleave * kKIter + iMInterleave * kKIter +
                                         iKIter * 2],
                                   b_buf[iNRepeat * NInterleave * kKIter + iNInterleave * kKIter +
                                         iKIter * 2]);
                            Impl{}(c_buf(iMRepeat * NRepeat * MInterleave * NInterleave +
                                        iNRepeat * MInterleave * NInterleave +
                                        iMInterleave * NInterleave + iNInterleave),
                                    a_buf[iMRepeat * MInterleave * kKIter + iMInterleave * kKIter +
                                        iKIter * 2 + 1],
                                    b_buf[iNRepeat * NInterleave * kKIter + iNInterleave * kKIter +
                                        iKIter * 2 + 1]);

                        });
                    });
                });
            });
        });
    }
};

template <typename WarpGemmAttributeMmacImpl_,
          index_t MRepeat,
          index_t NRepeat,
          index_t MInterleave,
          index_t NInterleave,
          index_t kKIter>
struct WarpGemmAttributeInt8MmacIterateK
{
    static_assert(kKIter > 0, "wrong!");

    using Impl = remove_cvref_t<WarpGemmAttributeMmacImpl_>;

    using ADataType = typename Impl::ADataType;
    using BDataType = typename Impl::BDataType;
    using CDataType = typename Impl::CDataType;

    using AComputeDataType = typename Impl::AComputeDataType;
    using BComputeDataType = typename Impl::BComputeDataType;
    using CComputeDataType = typename Impl::CComputeDataType;
    

    // ext_vector_t<datatype, ext_vector_t>, 2d array 2 x vec_a
    // Impl::AvecType is Vi2=2*int32=2*4*int8
    static_assert(vector_traits<typename Impl::AInt8VecType>::vector_size ==8,"-----------------------21------------------");
    static_assert(vector_traits<typename Impl::BInt8VecType>::vector_size ==8,"-----------------------22------------------");
    static_assert(vector_traits<typename Impl::CInt32VecType>::vector_size ==4,"-----------------------23------------------");
    using AVecType =
        ext_vector_t<ADataType, vector_traits<typename Impl::AInt8VecType>::vector_size * kKIter * MRepeat * MInterleave>;
    using BVecType =
        ext_vector_t<BDataType, vector_traits<typename Impl::BInt8VecType>::vector_size * kKIter * NRepeat * NInterleave>;
    using CVecType = 
        ext_vector_t<CDataType, vector_traits<typename Impl::CInt32VecType>::vector_size * MRepeat * NRepeat * MInterleave * NInterleave>;
    
    // using ABufType = thread_buffer<typename Impl::AVecType, kKIter * MRepeat * MInterleave>;
    // using BBufType = thread_buffer<typename Impl::BVecType, kKIter * NRepeat * NInterleave>;
    // using CBufType = thread_buffer<typename Impl::CVecType, MRepeat * NRepeat * MInterleave * NInterleave>;

    // static_assert(kKIter*MRepeat*MInterleave==1*2*1,"-----------------------31------------------");
    // static_assert(kKIter*NRepeat*NInterleave==1*1*4,"-----------------------32------------------");
    // static_assert(MRepeat*NRepeat*MInterleave*NInterleave==2*1*1*4,"-----------------------33------------------");
    static constexpr index_t kM = Impl::kM * MRepeat * MInterleave;
    static constexpr index_t kN = Impl::kN * NRepeat * NInterleave;
    static constexpr index_t kK = Impl::kK * kKIter;
    static constexpr index_t kKPerThread = Impl::kABKPerLane * kKIter;

    using AWarpDstrEncoding = tile_distribution_encoding<
        sequence<>,
        tuple<sequence<MRepeat, Impl::kAMLane, MInterleave>, sequence<Impl::kABKLane, Impl::kABKPerLane * kKIter>>,
        tuple<sequence<2, 1>>,
        tuple<sequence<0, 1>>,
        sequence<1, 1, 2>,
        sequence<0, 2, 1>>;

    using BWarpDstrEncoding = tile_distribution_encoding<
        sequence<>,
        tuple<sequence<NRepeat, Impl::kBNLane, NInterleave>, sequence<Impl::kABKLane, Impl::kABKPerLane * kKIter>>,
        tuple<sequence<2, 1>>,
        tuple<sequence<0, 1>>,
        sequence<1, 1, 2>,
        sequence<0, 2, 1>>;
    
    using CWarpDstrEncoding = tile_distribution_encoding<
        sequence<>,
        tuple<sequence<MRepeat, Impl::kCMLane, MInterleave, Impl::kCMPerLane>,
              sequence<NRepeat, Impl::kCNLane, NInterleave, Impl::kCN0PerLane, Impl::kCN1PerLane>>,
        tuple<sequence<2, 1>>,
        tuple<sequence<1, 1>>,
        sequence<1, 2, 1, 1, 2, 2, 2>, // <MRepeat, NRepeat, MInterleave, MPerLane, N0PerLane,
                                       // NmmacInterleave, N1PerLane>
        sequence<0, 0, 2, 3, 3, 2, 4>>;

    using CWarpOutputDstrEncoding = tile_distribution_encoding<
        sequence<>,
        tuple<sequence<MRepeat, Impl::kCMLane, MInterleave, Impl::kCMPerLane>,
              sequence<NRepeat, Impl::kCN0PerLane, Impl::kCNLane, NInterleave * Impl::kCN1PerLane>>,
        tuple<sequence<2, 1>>,
        tuple<sequence<2, 1>>,
        sequence<1, 2, 1, 1, 2, 2>, // <MRepeat, NRepeat, MInterleave, MPerLane, N0PerLane,
                                    // NmmacInterleave, N1PerLane>
        sequence<0, 0, 2, 3, 1, 3>>;

    // c_vec += a_vec * b_vec
    CK_TILE_DEVICE void
    operator()(CVecType& c_vec, const AVecType& a_vec, const BVecType& b_vec) const
    {
        using buf_a = thread_buffer<typename Impl::AInt8VecType, kKIter * MRepeat * MInterleave>;
        using buf_b = thread_buffer<typename Impl::BInt8VecType, kKIter * NRepeat * NInterleave>;
        using buf_c = thread_buffer<typename Impl::CInt32VecType, MRepeat * NRepeat * MInterleave * NInterleave>;
	    // static_assert(MRepeat*NRepeat*MInterleave*NInterleave==2*1*1*4,"-------------------------------41----------------------");

        static_for<0, MRepeat, 1>{}([&](auto iMRepeat) {
            static_for<0, NRepeat, 1>{}([&](auto iNRepeat) {
                static_for<0, MInterleave, 1>{}([&](auto iMInterleave) {
                    static_for<0, NInterleave, 1>{}([&](auto iNInterleave) {
                        static_for<0, kKIter, 1>{}([&](auto iKIter) {
                            //1 mmac issue
                            Impl{}(reinterpret_cast<buf_c&>(c_vec)
                                       .template get_as<typename Impl::CVecType>()[iMRepeat * NRepeat * MInterleave * NInterleave + 
                                                                                   iNRepeat * MInterleave * NInterleave +
                                                                                   iMInterleave * NInterleave + iNInterleave], 
                                   reinterpret_cast<const buf_a&>(a_vec)
                                       .template get_as<typename Impl::AVecType>()[iMRepeat * MInterleave * kKIter + 
                                                                                   iMInterleave * kKIter + iKIter],
                                   reinterpret_cast<const buf_b&>(b_vec)
                                       .template get_as<typename Impl::BVecType>()[iNRepeat * NInterleave * kKIter + 
                                                                                   iNInterleave * kKIter + iKIter]);
                        });
                    });
                });
            });
        });
    }
    
    // c_vec += a_vec * b_vec
    // CK_TILE_DEVICE void
    // operator()(CBufType& c_buf, const ABufType& a_buf, const BBufType& b_buf) const
    // {
    //     // using buf_a = thread_buffer<typename Impl::AInt8VecType, kKIter * MRepeat * MInterleave>;
    //     // using buf_b = thread_buffer<typename Impl::BInt8VecType, kKIter * NRepeat * NInterleave>;
    //     // using buf_c = thread_buffer<typename Impl::CInt32VecType, MRepeat * NRepeat * MInterleave * NInterleave>;
	//     // static_assert(MRepeat*NRepeat*MInterleave*NInterleave==2*1*1*4,"-------------------------------41----------------------");

    //     static_for<0, MRepeat, 1>{}([&](auto iMRepeat) {
    //         static_for<0, NRepeat, 1>{}([&](auto iNRepeat) {
    //             static_for<0, MInterleave, 1>{}([&](auto iMInterleave) {
    //                 static_for<0, NInterleave, 1>{}([&](auto iNInterleave) {
    //                     static_for<0, kKIter, 1>{}([&](auto iKIter) {
    //                         //1 mmac issue
    //                         Impl{}(c_buf(iMRepeat * NRepeat * MInterleave * NInterleave + 
    //                                     iNRepeat * MInterleave * NInterleave +
    //                                     iMInterleave * NInterleave + iNInterleave), 
    //                                a_buf(iMRepeat * MInterleave * kKIter + 
    //                                      iMInterleave * kKIter + iKIter),
    //                                b_buf(iNRepeat * NInterleave * kKIter + 
    //                                      iNInterleave * kKIter + iKIter));
    //                     });
    //                 });
    //             });
    //         });
    //     });
    // }

    // c_vec = a_vec * b_vec
    // AVecType = ext_vector_t<ADataType, AWarpTensor::get_thread_buffer_size()>;
    CK_TILE_DEVICE CVecType operator()(const AVecType& a_vec, const BVecType& b_vec) const
    {
        using buf_a = thread_buffer<typename Impl::AVecType, kKIter * MRepeat * MInterleave>;
        using buf_b = thread_buffer<typename Impl::BVecType, kKIter * NRepeat * NInterleave>;
        using buf_c = thread_buffer<typename Impl::CVecType, MRepeat * NRepeat * MInterleave * NInterleave>;

        buf_c c_vec;

        // c += a * b
        static_for<0, MRepeat, 1>{}([&](auto iMRepeat) {
            static_for<0, NRepeat, 1>{}([&](auto iNRepeat) {
                static_for<0, MInterleave, 1>{}([&](auto iMInterleave) {
                    static_for<0, NInterleave, 1>{}([&](auto iNInterleave) {
                        static_for<0, kKIter, 1>{}([&](auto iKIter) {
                            Impl{}(c_vec.template get_as<typename Impl::CVecType>()[iMRepeat * NRepeat * MInterleave * NInterleave + 
                                                                                   iNRepeat * MInterleave * NInterleave +
                                                                                   iMInterleave * NInterleave + 
                                                                                   iNInterleave], 
                                   reinterpret_cast<const buf_a&>(a_vec)
                                       .template get_as<typename Impl::AVecType>()[iMRepeat * MInterleave * kKIter + 
                                                                                   iMInterleave * kKIter + 
                                                                                   iKIter],
                                   reinterpret_cast<const buf_b&>(b_vec)
                                       .template get_as<typename Impl::BVecType>()[iNRepeat * NInterleave * kKIter + 
                                                                                   iNInterleave * kKIter + 
                                                                                   iKIter]);
                        });
                    });
                });
            });
        });

        return reinterpret_cast<CVecType>(c_vec);
    }

    template <typename CWarpTensor>
    CK_TILE_DEVICE auto MakeCOutputLayout(const CWarpTensor& c_warp_tensor) const
    {
        constexpr auto c_warp_output_distribution = make_static_tile_distribution(CWarpOutputDstrEncoding{});
        auto c_warp_output_tensor = make_static_distributed_tensor<CDataType>(c_warp_output_distribution);
        
        static_for<0, MRepeat, 1>{}([&](auto iMR){
            static_for<0, NRepeat, 1>{}([&](auto iNR){
                static_for<0, MInterleave, 1>{} ([&](auto iMI){
                    static_for<0, NInterleave, 1>{}([&](auto iNI){
                        static_for<0, Impl::kCN0PerLane, 1>{}([&](auto iCN0){
                            
                            c_warp_output_tensor.set_y_sliced_thread_data(
                                sequence<iMR, iNR, iMI, 0, iCN0, iNI>{}, //output_orgin
                                sequence<1, 1, 1, 1, 1, Impl::kCN1PerLane>{}, //output_set_length
                                c_warp_tensor.get_y_sliced_thread_data( //get calculate tensor and change to output layout
                                    sequence<iMR, iNR, iMI, 0, iNI, iCN0, 0>{},
                                    sequence<1, 1, 1, 1, 1, 1, Impl::kCN1PerLane>{}));

                        });

                    });
                });
            });
        });

        return c_warp_output_tensor;
    }
};

template <typename WarpGemmAttributeMmacImpl_,
          index_t MRepeat_,
          index_t NRepeat_,
          index_t MInterleave_,
          index_t NInterleave_,
          index_t kKIter_>
struct WarpGemmAttributeInt8ScaleChannelMmacIterateK
{
    static_assert(kKIter_ > 0, "wrong!");

    using Impl = remove_cvref_t<WarpGemmAttributeMmacImpl_>;
    using Base = WarpGemmAttributeInt8MmacIterateK<WarpGemmAttributeMmacImpl_,
                                                   MRepeat_,
                                                   NRepeat_,
                                                   MInterleave_,
                                                   NInterleave_,
                                                   kKIter_>;

    static constexpr index_t MRepeat     = MRepeat_;
    static constexpr index_t NRepeat     = NRepeat_;
    static constexpr index_t MInterleave = MInterleave_;
    static constexpr index_t NInterleave = NInterleave_;
    static constexpr index_t kKIter      = kKIter_;

    using ADataType = typename Impl::ADataType;
    using BDataType = typename Impl::BDataType;
    using CDataType = typename Impl::CDataType;

    using AScaleType = typename Impl::AScaleType;
    using BScaleType = typename Impl::BScaleType;

    // ext_vector_t<datatype, ext_vector_t>, 2d array 2 x vec_a
    // Impl::AvecType is Vi2=2*int32=2*4*int8
    static_assert(vector_traits<typename Impl::AInt8VecType>::vector_size ==8,"Int8VecType vector size is not 8");
    static_assert(vector_traits<typename Impl::BInt8VecType>::vector_size ==8,"Int8VecType vector size is not 8");
    static_assert(vector_traits<typename Impl::CFloat32VecType>::vector_size ==4,"Float32VecType vector size is not 4");
    using AVecType =
        ext_vector_t<ADataType, vector_traits<typename Impl::AInt8VecType>::vector_size * kKIter * MRepeat * MInterleave>;
    using BVecType =
        ext_vector_t<BDataType, vector_traits<typename Impl::BInt8VecType>::vector_size * kKIter * NRepeat * NInterleave>;
    using CVecType = 
        ext_vector_t<CDataType, vector_traits<typename Impl::CInt32VecType>::vector_size * MRepeat * NRepeat * MInterleave * NInterleave>;

    using AScaleVec = ext_vector_t<AScaleType, MRepeat * MInterleave>;
    using BScaleVec = ext_vector_t<BScaleType, NRepeat * NInterleave>;
    
    using ABufType = thread_buffer<typename Impl::AVecType, kKIter * MRepeat * MInterleave>;
    using BBufType = thread_buffer<typename Impl::BVecType, kKIter * NRepeat * NInterleave>;
    using CBufType = thread_buffer<typename Impl::CVecType, MRepeat * NRepeat * MInterleave * NInterleave>;
    using AScaleBufType = thread_buffer<AScaleType, MRepeat * MInterleave>;
    using BScaleBufType = thread_buffer<BScaleType, NRepeat * NInterleave>;


    static constexpr index_t kM = Impl::kM * MRepeat * MInterleave;
    static constexpr index_t kN = Impl::kN * NRepeat * NInterleave;
    static constexpr index_t kK = Impl::kK * kKIter;
    static constexpr index_t kKPerThread = Impl::kABKPerLane * kKIter;

    static constexpr index_t kKIteration = kKIter;
    static constexpr index_t kABKPerLane = Impl::kABKPerLane;
    static constexpr index_t kCN0PerLane = Impl::kCN0PerLane;

    using AWarpDstrEncoding = typename Base::AWarpDstrEncoding;
    using BWarpDstrEncoding = typename Base::BWarpDstrEncoding;
    using CWarpDstrEncoding = typename Base::CWarpDstrEncoding;


    static_assert(MInterleave == 1, "MInterleave must be 1 for scale channel");
    static_assert(MRepeat == 1, "MRepeat must be 1 for scale channel");

    using AScaleWarpDstrEncoding = tile_distribution_encoding<
        sequence<>,
        tuple<sequence<MRepeat, Impl::kAMLane, MInterleave>,
              sequence<Impl::kABKLane, 1>>,
        tuple<sequence<2, 1>>,
        tuple<sequence<0, 1>>,
        sequence<1, 1, 2>,
        sequence<0, 2, 1>>;

    using BScaleWarpDstrEncoding = tile_distribution_encoding<
        sequence<>,
        tuple<sequence<NRepeat, Impl::kBNLane, NInterleave>,
              sequence<Impl::kABKLane, 1>>,
        tuple<sequence<2, 1>>,
        tuple<sequence<0, 1>>,
        sequence<1, 1, 2>,
        sequence<0, 2, 1>>;

  
    // c_vec += a_vec * b_vec
    CK_TILE_DEVICE void
    operator()(CBufType& c_buf, const ABufType& a_buf, const BBufType& b_buf, const AScaleBufType& a_scale_buf, const BScaleBufType& b_scale_buf) const
    {
        static_for<0, MRepeat, 1>{}([&](auto iMRepeat) {
            static_for<0, NRepeat, 1>{}([&](auto iNRepeat) {
                static_for<0, MInterleave, 1>{}([&](auto iMInterleave) {
                    static_for<0, NInterleave, 1>{}([&](auto iNInterleave) {
                        static_for<0, kKIter, 1>{}([&](auto iKIter) {
                            //1 mmac issue
                            Impl{}(c_buf(iMRepeat * NRepeat * MInterleave * NInterleave + 
                                         iNRepeat * MInterleave * NInterleave +
                                         iMInterleave * NInterleave + iNInterleave),
                                   a_buf[iMRepeat * MInterleave * kKIter + iMInterleave * kKIter + 
                                         iKIter],
                                   b_buf[iNRepeat * NInterleave * kKIter + iNInterleave * kKIter + 
                                         iKIter],
                                   a_scale_buf[iMRepeat * MInterleave + iMInterleave],
                                   b_scale_buf[iNRepeat * NInterleave + iNInterleave]);
                        });
                    });
                });
            });
        });
    }

    // CK_TILE_DEVICE void
    // operator()(CVecType& c_vec, const AVecType& a_vec, const BVecType& b_vec, const AScaleVec& a_scale_vec, const BScaleVec& b_scale_vec) const
    // {
    //     using buf_a = thread_buffer<typename Impl::AInt8VecType, kKIter * MRepeat * MInterleave>;
    //     using buf_b = thread_buffer<typename Impl::BInt8VecType, kKIter * NRepeat * NInterleave>;
    //     using buf_a_scale = thread_buffer<AScaleType, MRepeat * MInterleave>;
    //     using buf_b_scale = thread_buffer<BScaleType, NRepeat * NInterleave>;
    //     using buf_c = thread_buffer<typename Impl::CFloat32VecType, MRepeat * NRepeat * MInterleave * NInterleave>;
    //     //using buf_c = thread_buffer<typename Impl::CVecType, MRepeat * NRepeat * MInterleave * NInterleave>;
        
    //     static_for<0, MRepeat, 1>{}([&](auto iMRepeat) {
    //         static_for<0, NRepeat, 1>{}([&](auto iNRepeat) {
    //             static_for<0, MInterleave, 1>{}([&](auto iMInterleave) {
    //                 static_for<0, NInterleave, 1>{}([&](auto iNInterleave) {
    //                     AScaleType a_scale = reinterpret_cast<const buf_a_scale&>(a_scale_vec)
    //                                            .template get_as<AScaleType>()[iMRepeat * MInterleave + iMInterleave];
    //                     BScaleType b_scale = reinterpret_cast<const buf_b_scale&>(b_scale_vec)
    //                                            .template get_as<BScaleType>()[iNRepeat * NInterleave + iNInterleave];

    //                     static_for<0, kKIter, 1>{}([&](auto iKIter) {
    //                         //1 mmac issue
    //                         Impl{}(reinterpret_cast<buf_c&>(c_vec)
    //                                    .template get_as<typename Impl::CVecType>()[iMRepeat * NRepeat * MInterleave * NInterleave + 
    //                                                                                iNRepeat * MInterleave * NInterleave +
    //                                                                                iMInterleave * NInterleave + 
    //                                                                                iNInterleave],
    //                                reinterpret_cast<const buf_a&>(a_vec)
    //                                    .template get_as<typename Impl::AVecType>()[iMRepeat * MInterleave * kKIter + 
    //                                                                                iMInterleave * kKIter + 
    //                                                                                iKIter],
    //                                reinterpret_cast<const buf_b&>(b_vec)
    //                                    .template get_as<typename Impl::BVecType>()[iNRepeat * NInterleave * kKIter + 
    //                                                                                iNInterleave * kKIter + 
    //                                                                                iKIter],
    //                                a_scale,
    //                                b_scale);
    //                     });
    //                 });
    //             });
    //         });
    //     });
    // }

    // c_vec = a_vec * b_vec
    // AVecType = ext_vector_t<ADataType, AWarpTensor::get_thread_buffer_size()>;
    CK_TILE_DEVICE CVecType operator()(const AVecType& a_vec, const BVecType& b_vec) const
    {
        using buf_a = thread_buffer<typename Impl::AVecType, kKIter * MRepeat * MInterleave>;
        using buf_b = thread_buffer<typename Impl::BVecType, kKIter * NRepeat * NInterleave>;
        using buf_c = thread_buffer<typename Impl::CVecType, MRepeat * NRepeat * MInterleave * NInterleave>;

        buf_c c_vec;

        // c += a * b
        static_for<0, MRepeat, 1>{}([&](auto iMRepeat) {
            static_for<0, NRepeat, 1>{}([&](auto iNRepeat) {
                static_for<0, MInterleave, 1>{}([&](auto iMInterleave) {
                    static_for<0, NInterleave, 1>{}([&](auto iNInterleave) {
                        static_for<0, kKIter, 1>{}([&](auto iKIter) {
                            Impl{}(c_vec.template get_as<typename Impl::CVecType>()[iMRepeat * NRepeat * MInterleave * NInterleave + 
                                                                                   iNRepeat * MInterleave * NInterleave +
                                                                                   iMInterleave * NInterleave + 
                                                                                   iNInterleave], 
                                   reinterpret_cast<const buf_a&>(a_vec)
                                       .template get_as<typename Impl::AVecType>()[iMRepeat * MInterleave * kKIter + 
                                                                                   iMInterleave * kKIter + 
                                                                                   iKIter],
                                   reinterpret_cast<const buf_b&>(b_vec)
                                       .template get_as<typename Impl::BVecType>()[iNRepeat * NInterleave * kKIter + 
                                                                                   iNInterleave * kKIter + 
                                                                                   iKIter]);
                        });
                    });
                });
            });
        });

        return reinterpret_cast<CVecType>(c_vec);
    }

};

template <typename WarpGemmAttributeMmacImpl_,
          index_t MRepeat,
          index_t NRepeat,
          index_t MInterleave,
          index_t NInterleave,
          index_t kKIter>
struct WarpGemmAttributeInt8MmacIterateKShuffle
        : public WarpGemmAttributeInt8MmacIterateK<WarpGemmAttributeMmacImpl_,
                                                    MRepeat,
                                                    NRepeat,
                                                    MInterleave,
                                                    NInterleave,
                                                    kKIter>
{
    static_assert(kKIter > 0 && kKIter % 2 == 0, "kKIter must be even and greater than 0!");

    using Impl = remove_cvref_t<WarpGemmAttributeMmacImpl_>;
    using Base = WarpGemmAttributeInt8MmacIterateK<WarpGemmAttributeMmacImpl_,
                                                    MRepeat,
                                                    NRepeat,
                                                    MInterleave,
                                                    NInterleave,
                                                    kKIter>;

    using ADataType = typename Impl::ADataType;
    using BDataType = typename Impl::BDataType;
    using CDataType = typename Impl::CDataType;

    using AComputeDataType = typename Impl::AComputeDataType;
    using BComputeDataType = typename Impl::BComputeDataType;
    using CComputeDataType = typename Impl::CComputeDataType;
    
    using AVecType =
        ext_vector_t<ADataType, vector_traits<typename Impl::AInt8VecType>::vector_size * kKIter * MRepeat * MInterleave>;
    using BVecType =
        ext_vector_t<BDataType, vector_traits<typename Impl::BInt8VecType>::vector_size * kKIter * NRepeat * NInterleave>;
    using CVecType = 
        ext_vector_t<CDataType, vector_traits<typename Impl::CInt32VecType>::vector_size * MRepeat * NRepeat * MInterleave * NInterleave>;


    using AWarpDstrEncoding = tile_distribution_encoding<
        sequence<>,
        tuple<sequence<MRepeat, Impl::kAMLane, MInterleave>, sequence<kKIter / 2, Impl::kABKLane, Impl::kABKPerLane * 2>>,
        tuple<sequence<2, 1>>,
        tuple<sequence<1, 1>>,
        sequence<1, 1, 2, 2>,
        sequence<0, 2, 0, 2>>;

    using BWarpDstrEncoding = tile_distribution_encoding<
        sequence<>,
        tuple<sequence<NRepeat, Impl::kBNLane, NInterleave>, sequence<kKIter / 2, Impl::kABKLane, Impl::kABKPerLane * 2>>,
        tuple<sequence<2, 1>>,
        tuple<sequence<1, 1>>,
        sequence<1, 1, 2, 2>,
        sequence<0, 2, 0, 2>>;
    
    using CWarpDstrEncoding = typename Base::CWarpDstrEncoding;

    // c_vec += a_vec * b_vec
    CK_TILE_DEVICE void
    operator()(CVecType& c_vec, const AVecType& a_vec, const BVecType& b_vec) const
    {
        using buf_a = thread_buffer<typename Impl::AInt8VecType, kKIter * MRepeat * MInterleave>;
        using buf_b = thread_buffer<typename Impl::BInt8VecType, kKIter * NRepeat * NInterleave>;
        using buf_c = thread_buffer<typename Impl::CInt32VecType, MRepeat * NRepeat * MInterleave * NInterleave>;
	    // static_assert(MRepeat*NRepeat*MInterleave*NInterleave==2*1*1*4,"-------------------------------41----------------------");

        static_for<0, MRepeat, 1>{}([&](auto iMRepeat) {
            static_for<0, NRepeat, 1>{}([&](auto iNRepeat) {
                static_for<0, MInterleave, 1>{}([&](auto iMInterleave) {
                    static_for<0, NInterleave, 1>{}([&](auto iNInterleave) {
                        static_for<0, kKIter/2, 1>{}([&](auto iKIter) {
                            //1 mmac issue
                            Impl{}(reinterpret_cast<buf_c&>(c_vec)
                                       .template get_as<typename Impl::CVecType>()[iMRepeat * NRepeat * MInterleave * NInterleave + 
                                                                                   iNRepeat * MInterleave * NInterleave +
                                                                                   iMInterleave * NInterleave + 
                                                                                   iNInterleave], 
                                   reinterpret_cast<const buf_a&>(a_vec)
                                       .template get_as<typename Impl::AVecType>()[iMRepeat * MInterleave * kKIter + 
                                                                                   iMInterleave * kKIter + 
                                                                                   iKIter * 2],
                                   reinterpret_cast<const buf_b&>(b_vec)
                                       .template get_as<typename Impl::BVecType>()[iNRepeat * NInterleave * kKIter + 
                                                                                   iNInterleave * kKIter + 
                                                                                   iKIter * 2]);
                            //2 mmac issue
                            Impl{}(reinterpret_cast<buf_c&>(c_vec)
                                       .template get_as<typename Impl::CVecType>()[iMRepeat * NRepeat * MInterleave * NInterleave + 
                                                                                   iNRepeat * MInterleave * NInterleave +
                                                                                   iMInterleave * NInterleave + 
                                                                                   iNInterleave], 
                                   reinterpret_cast<const buf_a&>(a_vec)
                                       .template get_as<typename Impl::AVecType>()[iMRepeat * MInterleave * kKIter + 
                                                                                   iMInterleave * kKIter + 
                                                                                   iKIter * 2 + 1],
                                   reinterpret_cast<const buf_b&>(b_vec)
                                       .template get_as<typename Impl::BVecType>()[iNRepeat * NInterleave * kKIter + 
                                                                                   iNInterleave * kKIter + 
                                                                                   iKIter * 2 + 1]);
                        });
                    });
                });
            });
        });
    }

    // c_vec = a_vec * b_vec
    // AVecType = ext_vector_t<ADataType, AWarpTensor::get_thread_buffer_size()>;
    CK_TILE_DEVICE CVecType operator()(const AVecType& a_vec, const BVecType& b_vec) const
    {
        using buf_a = thread_buffer<typename Impl::AVecType, kKIter * MRepeat * MInterleave>;
        using buf_b = thread_buffer<typename Impl::BVecType, kKIter * NRepeat * NInterleave>;
        using buf_c = thread_buffer<typename Impl::CVecType, MRepeat * NRepeat * MInterleave * NInterleave>;

        buf_c c_vec;

        // c += a * b
        static_for<0, MRepeat, 1>{}([&](auto iMRepeat) {
            static_for<0, NRepeat, 1>{}([&](auto iNRepeat) {
                static_for<0, MInterleave, 1>{}([&](auto iMInterleave) {
                    static_for<0, NInterleave, 1>{}([&](auto iNInterleave) {
                        static_for<0, kKIter/2, 1>{}([&](auto iKIter) {
                            Impl{}(c_vec.template get_as<typename Impl::CVecType>()[iMRepeat * NRepeat * MInterleave * NInterleave + 
                                                                                   iNRepeat * MInterleave * NInterleave +
                                                                                   iMInterleave * NInterleave + 
                                                                                   iNInterleave], 
                                   reinterpret_cast<const buf_a&>(a_vec)
                                       .template get_as<typename Impl::AVecType>()[iMRepeat * MInterleave * kKIter + 
                                                                                   iMInterleave * kKIter + 
                                                                                   iKIter * 2],
                                   reinterpret_cast<const buf_b&>(b_vec)
                                       .template get_as<typename Impl::BVecType>()[iNRepeat * NInterleave * kKIter + 
                                                                                   iNInterleave * kKIter + 
                                                                                   iKIter * 2]);

                            Impl{}(c_vec.template get_as<typename Impl::CVecType>()[iMRepeat * NRepeat * MInterleave * NInterleave + 
                                                                                    iNRepeat * MInterleave * NInterleave +
                                                                                    iMInterleave * NInterleave + 
                                                                                    iNInterleave], 
                                    reinterpret_cast<const buf_a&>(a_vec)
                                        .template get_as<typename Impl::AVecType>()[iMRepeat * MInterleave * kKIter + 
                                                                                    iMInterleave * kKIter + 
                                                                                    iKIter * 2 + 1],
                                    reinterpret_cast<const buf_b&>(b_vec)
                                        .template get_as<typename Impl::BVecType>()[iNRepeat * NInterleave * kKIter + 
                                                                                    iNInterleave * kKIter + 
                                                                                    iKIter * 2 + 1]);
                        });
                    });
                });
            });
        });

        return reinterpret_cast<CVecType>(c_vec);
    }


};

template <typename WarpGemmAttributeMmacImpl_,
          index_t MRepeat_,
          index_t NRepeat_,
          index_t MInterleave_,
          index_t NInterleave_,
          index_t kKIter>
struct WarpGemmAttributeInt8MmacIterateKTransC
    : public WarpGemmAttributeInt8MmacIterateK<WarpGemmAttributeMmacImpl_,
                                                MRepeat_,
                                                NRepeat_,
                                                MInterleave_,
                                                NInterleave_,
                                                kKIter>
{
    using Base = WarpGemmAttributeInt8MmacIterateK<WarpGemmAttributeMmacImpl_,
                                                    MRepeat_,
                                                    NRepeat_,
                                                    MInterleave_,
                                                    NInterleave_,
                                                    kKIter>;

    using Impl = typename Base::Impl;

    using ADataType = typename Base::ADataType;
    using BDataType = typename Base::BDataType;
    using CDataType = typename Base::CDataType;

    using AComputeDataType = typename Base::AComputeDataType;
    using BComputeDataType = typename Base::BComputeDataType;
    using CComputeDataType = typename Base::CComputeDataType;
    
    using AVecType = typename Base::AVecType;
    using BVecType = typename Base::BVecType;
    using CVecType = typename Base::CVecType;
    
    static constexpr index_t kM = Base::kM;
    static constexpr index_t kN = Base::kN;
    static constexpr index_t kK = Base::kK;
    static constexpr index_t MRepeat     = MRepeat_;
    static constexpr index_t NRepeat     = NRepeat_;
    static constexpr index_t MInterleave = MInterleave_;
    static constexpr index_t NInterleave = NInterleave_;

    using AWarpDstrEncoding = typename Base::AWarpDstrEncoding;

    using BWarpDstrEncoding = typename Base::BWarpDstrEncoding;

    using CWarpDstrEncoding = tile_distribution_encoding<
        sequence<>,
        tuple<sequence<MRepeat, Impl::kCM0PerLane, Impl::kCMLane, Impl::kCM1PerLane, MInterleave>, // <1, 4, 4, 1, 1>
              sequence<NRepeat, Impl::kCNLane, Impl::kCNPerLane, NInterleave>>,         // <2, 16, 1, 1> 
        tuple<sequence<1, 2>>,
        tuple<sequence<2, 1>>,
        sequence<1, 2, 1, 2, 2, 1, 1>,
        sequence<0, 0, 4, 3, 2, 1, 3>>;


    CK_TILE_DEVICE void
    operator()(CVecType& c_vec, const AVecType& a_vec, const BVecType& b_vec) const
    {
        using buf_a = thread_buffer<typename Impl::AInt8VecType, kKIter * MRepeat * MInterleave>;
        using buf_b = thread_buffer<typename Impl::BInt8VecType, kKIter * NRepeat * NInterleave>;
        using buf_c = thread_buffer<typename Impl::CInt32VecType, MRepeat * NRepeat * MInterleave * NInterleave>;

        static_for<0, MRepeat, 1>{}([&](auto iMRepeat) {
            static_for<0, NRepeat, 1>{}([&](auto iNRepeat) {
                static_for<0, MInterleave, 1>{}([&](auto iMInterleave) {
                    static_for<0, NInterleave, 1>{}([&](auto iNInterleave) {
                        static_for<0, kKIter, 1>{}([&](auto iKIter) {
                            //1 mmac issue
                            Impl{}(reinterpret_cast<buf_c&>(c_vec)
                                       .template get_as<typename Impl::CVecType>()[iMRepeat * NRepeat * MInterleave * NInterleave + 
                                                                                   iNRepeat * MInterleave * NInterleave +
                                                                                   iMInterleave * NInterleave + iNInterleave], 
                                   reinterpret_cast<const buf_a&>(a_vec)
                                       .template get_as<typename Impl::AVecType>()[iMRepeat * MInterleave * kKIter + 
                                                                                   iMInterleave * kKIter + iKIter],
                                   reinterpret_cast<const buf_b&>(b_vec)
                                       .template get_as<typename Impl::BVecType>()[iNRepeat * NInterleave * kKIter + 
                                                                                   iNInterleave * kKIter + iKIter]);
                        });
                    });
                });
            });
        });
    }

};


template <typename WarpGemmAttributeMmacImpl_,
          index_t MRepeat_,
          index_t NRepeat_,
          index_t MInterleave_,
          index_t NInterleave_,
          index_t kKIter>
struct WarpGemmAttributeInt8MmacIterateKTransC_Shuffle
    : public WarpGemmAttributeInt8MmacIterateK<WarpGemmAttributeMmacImpl_,
                                               MRepeat_,
                                               NRepeat_,
                                               MInterleave_,
                                               NInterleave_,
                                               kKIter>
{
    using Base = WarpGemmAttributeInt8MmacIterateKTransC<WarpGemmAttributeMmacImpl_,
                                                        MRepeat_,
                                                        NRepeat_,
                                                        MInterleave_,
                                                        NInterleave_,
                                                        kKIter>;

    using Impl = typename Base::Impl;

    using ADataType = typename Base::ADataType;
    using BDataType = typename Base::BDataType;
    using CDataType = typename Base::CDataType;

    using AComputeDataType = typename Base::AComputeDataType;
    using BComputeDataType = typename Base::BComputeDataType;
    using CComputeDataType = typename Base::CComputeDataType;
    
    using AVecType = typename Base::AVecType;
    using BVecType = typename Base::BVecType;
    using CVecType = typename Base::CVecType;
    
    static constexpr index_t kM = Base::kM;
    static constexpr index_t kN = Base::kN;
    static constexpr index_t kK = Base::kK;
    static constexpr index_t MRepeat     = MRepeat_;
    static constexpr index_t NRepeat     = NRepeat_;
    static constexpr index_t MInterleave = MInterleave_;
    static constexpr index_t NInterleave = NInterleave_;

    using AWarpDstrEncoding = tile_distribution_encoding<
        sequence<>,
        tuple<sequence<MRepeat, Impl::kAMLane, MInterleave>,
            sequence<kKIter/2, Impl::kABKLane, Impl::kABKPerLane * 2>>,
        tuple<sequence<2, 1>>,
        tuple<sequence<1, 1>>,
        sequence<1, 1, 2, 2>,
        sequence<0, 2, 0, 2>>;

    using BWarpDstrEncoding = tile_distribution_encoding<
        sequence<>,
        tuple<sequence<NRepeat, Impl::kBNLane, NInterleave>,
            sequence<kKIter/2, Impl::kABKLane, Impl::kABKPerLane * 2>>,
        tuple<sequence<2, 1>>,
        tuple<sequence<1, 1>>,
        sequence<1, 1, 2, 2>,
        sequence<0, 2, 0, 2>>;

    using CWarpDstrEncoding = tile_distribution_encoding<
        sequence<>,
        tuple<sequence<MRepeat, Impl::kCM0PerLane, Impl::kCMLane, Impl::kCM1PerLane, MInterleave>, // <1, 4, 4, 1, 1>
              sequence<NRepeat, Impl::kCNLane, Impl::kCNPerLane, NInterleave>>,         // <2, 16, 1, 1> 
        tuple<sequence<1, 2>>,
        tuple<sequence<2, 1>>,
        sequence<1, 2, 1, 2, 2, 1, 1>,
        sequence<0, 0, 4, 3, 2, 1, 3>>;


    CK_TILE_DEVICE void
    operator()(CVecType& c_vec, const AVecType& a_vec, const BVecType& b_vec) const
    {
        using buf_a = thread_buffer<typename Impl::AInt8VecType, kKIter * MRepeat * MInterleave>;
        using buf_b = thread_buffer<typename Impl::BInt8VecType, kKIter * NRepeat * NInterleave>;
        using buf_c = thread_buffer<typename Impl::CInt32VecType, MRepeat * NRepeat * MInterleave * NInterleave>;

        static_for<0, MRepeat, 1>{}([&](auto iMRepeat) {
            static_for<0, NRepeat, 1>{}([&](auto iNRepeat) {
                static_for<0, MInterleave, 1>{}([&](auto iMInterleave) {
                    static_for<0, NInterleave, 1>{}([&](auto iNInterleave) {
                        static_for<0, kKIter/2, 1>{}([&](auto iKIter) {
                            //1 mmac issue
                            Impl{}(reinterpret_cast<buf_c&>(c_vec)
                                       .template get_as<typename Impl::CVecType>()[iMRepeat * NRepeat * MInterleave * NInterleave + 
                                                                                   iNRepeat * MInterleave * NInterleave +
                                                                                   iMInterleave * NInterleave + iNInterleave], 
                                   reinterpret_cast<const buf_a&>(a_vec)
                                       .template get_as<typename Impl::AVecType>()[iMRepeat * MInterleave * kKIter + 
                                                                                   iMInterleave * kKIter + iKIter * 2],
                                   reinterpret_cast<const buf_b&>(b_vec)
                                       .template get_as<typename Impl::BVecType>()[iNRepeat * NInterleave * kKIter + 
                                                                                   iNInterleave * kKIter + iKIter * 2]);
                            //2 mmac issue
                            Impl{}(reinterpret_cast<buf_c&>(c_vec)
                                       .template get_as<typename Impl::CVecType>()[iMRepeat * NRepeat * MInterleave * NInterleave + 
                                                                                   iNRepeat * MInterleave * NInterleave +
                                                                                   iMInterleave * NInterleave + iNInterleave], 
                                   reinterpret_cast<const buf_a&>(a_vec)
                                       .template get_as<typename Impl::AVecType>()[iMRepeat * MInterleave * kKIter + 
                                                                                   iMInterleave * kKIter + iKIter * 2 + 1],
                                   reinterpret_cast<const buf_b&>(b_vec)
                                       .template get_as<typename Impl::BVecType>()[iNRepeat * NInterleave * kKIter + 
                                                                                   iNInterleave * kKIter + iKIter * 2 + 1]);
                        });
                    });
                });
            });
        });
    }

};

template <typename WarpGemmAttributeMmacImpl_,
          index_t MRepeat,
          index_t NRepeat,
          index_t MInterleave,
          index_t NInterleave,
          index_t kKIter>
struct WarpGemmAttributeInt4MmacIterateK
{
    static_assert(kKIter > 0, "wrong!");

    using Impl = remove_cvref_t<WarpGemmAttributeMmacImpl_>;

    using ADataType = typename Impl::ADataType;
    using BDataType = typename Impl::BDataType;
    using CDataType = typename Impl::CDataType;

    using AComputeDataType = typename Impl::AComputeDataType;
    using BComputeDataType = typename Impl::BComputeDataType;
    using CComputeDataType = typename Impl::CComputeDataType;

    using AVecType =
        ext_vector_t<int8_t, vector_traits<typename Impl::AInt4VecType>::vector_size * kKIter *
                                 MRepeat * MInterleave>;
    using BVecType =
        ext_vector_t<int8_t, vector_traits<typename Impl::BInt4VecType>::vector_size * kKIter *
                                 NRepeat * NInterleave>;
    using CVecType =
        ext_vector_t<CDataType,
                     vector_traits<typename Impl::CVecType>::vector_size * MRepeat * NRepeat *
                         MInterleave * NInterleave>;

    static constexpr index_t kM = Impl::kM * MRepeat * MInterleave;
    static constexpr index_t kN = Impl::kN * NRepeat * NInterleave;
    static constexpr index_t kK = Impl::kK * kKIter;
    static constexpr index_t kKPerThread = Impl::kABKPerLane * kKIter;

    using AWarpDstrEncoding =
        tile_distribution_encoding<sequence<>,
                                   tuple<sequence<MRepeat, Impl::kAMLane, MInterleave>,
                                         sequence<Impl::kABKLane, Impl::kABKPerLane * kKIter>>,
                                   tuple<sequence<2, 1>>,
                                   tuple<sequence<0, 1>>,
                                   sequence<1, 1, 2>,
                                   sequence<0, 2, 1>>;

    using BWarpDstrEncoding =
        tile_distribution_encoding<sequence<>,
                                   tuple<sequence<NRepeat, Impl::kBNLane, NInterleave>,
                                         sequence<Impl::kABKLane, Impl::kABKPerLane * kKIter>>,
                                   tuple<sequence<2, 1>>,
                                   tuple<sequence<0, 1>>,
                                   sequence<1, 1, 2>,
                                   sequence<0, 2, 1>>;

    using CWarpDstrEncoding = tile_distribution_encoding<
        sequence<>,
        tuple<sequence<MRepeat, Impl::kCMLane, MInterleave, Impl::kCMPerLane>,
              sequence<NRepeat, Impl::kCNLane, NInterleave, Impl::kCN0PerLane, Impl::kCN1PerLane>>,
        tuple<sequence<2, 1>>,
        tuple<sequence<1, 1>>,
        sequence<1, 2, 1, 1, 2, 2, 2>,
        sequence<0, 0, 2, 3, 3, 2, 4>>;

    using CWarpOutputDstrEncoding = tile_distribution_encoding<
        sequence<>,
        tuple<sequence<MRepeat, Impl::kCMLane, MInterleave, Impl::kCMPerLane>,
              sequence<NRepeat, Impl::kCN0PerLane, Impl::kCNLane, NInterleave * Impl::kCN1PerLane>>,
        tuple<sequence<2, 1>>,
        tuple<sequence<2, 1>>,
        sequence<1, 2, 1, 1, 2, 2>,
        sequence<0, 0, 2, 3, 1, 3>>;

    CK_TILE_DEVICE void
    operator()(CVecType& c_vec, const AVecType& a_vec, const BVecType& b_vec) const
    {
        using buf_a =
            thread_buffer<typename Impl::AInt4VecType, kKIter * MRepeat * MInterleave>;
        using buf_b =
            thread_buffer<typename Impl::BInt4VecType, kKIter * NRepeat * NInterleave>;
        using buf_c = thread_buffer<typename Impl::CVecType,
                                    MRepeat * NRepeat * MInterleave * NInterleave>;

        static_for<0, MRepeat, 1>{}([&](auto iMRepeat) {
            static_for<0, NRepeat, 1>{}([&](auto iNRepeat) {
                static_for<0, MInterleave, 1>{}([&](auto iMInterleave) {
                    static_for<0, NInterleave, 1>{}([&](auto iNInterleave) {
                        static_for<0, kKIter, 1>{}([&](auto iKIter) {
                            Impl{}(reinterpret_cast<buf_c&>(c_vec)
                                       .template get_as<typename Impl::CVecType>()
                                           [iMRepeat * NRepeat * MInterleave * NInterleave +
                                            iNRepeat * MInterleave * NInterleave +
                                            iMInterleave * NInterleave + iNInterleave],
                                   reinterpret_cast<const buf_a&>(a_vec)
                                       .template get_as<typename Impl::AVecType>()
                                           [iMRepeat * MInterleave * kKIter +
                                            iMInterleave * kKIter + iKIter],
                                   reinterpret_cast<const buf_b&>(b_vec)
                                       .template get_as<typename Impl::BVecType>()
                                           [iNRepeat * NInterleave * kKIter +
                                            iNInterleave * kKIter + iKIter]);
                        });
                    });
                });
            });
        });
    }

    template <typename CWarpTensor>
    CK_TILE_DEVICE auto MakeCOutputLayout(const CWarpTensor& c_warp_tensor) const
    {
        constexpr auto c_warp_output_distribution =
            make_static_tile_distribution(CWarpOutputDstrEncoding{});
        auto c_warp_output_tensor =
            make_static_distributed_tensor<CDataType>(c_warp_output_distribution);

        static_for<0, MRepeat, 1>{}([&](auto iMR) {
            static_for<0, NRepeat, 1>{}([&](auto iNR) {
                static_for<0, MInterleave, 1>{}([&](auto iMI) {
                    static_for<0, NInterleave, 1>{}([&](auto iNI) {
                        static_for<0, Impl::kCN0PerLane, 1>{}([&](auto iCN0) {
                            c_warp_output_tensor.set_y_sliced_thread_data(
                                sequence<iMR, iNR, iMI, 0, iCN0, iNI>{},
                                sequence<1, 1, 1, 1, 1, Impl::kCN1PerLane>{},
                                c_warp_tensor.get_y_sliced_thread_data(
                                    sequence<iMR, iNR, iMI, 0, iNI, iCN0, 0>{},
                                    sequence<1, 1, 1, 1, 1, 1, Impl::kCN1PerLane>{}));
                        });
                    });
                });
            });
        });

        return c_warp_output_tensor;
    }
};

template <typename WarpGemmAttributeMmacImpl_,
          index_t MRepeat,
          index_t NRepeat,
          index_t MInterleave,
          index_t NInterleave,
          index_t kKIter>
struct WarpGemmAttributeFp8Bf8MmacIterateK
{
    static_assert(kKIter > 0, "wrong!");

    using Impl = remove_cvref_t<WarpGemmAttributeMmacImpl_>;

    using ADataType = typename Impl::ADataType;
    using BDataType = typename Impl::BDataType;
    using CDataType = typename Impl::CDataType;

    using AScaleType = typename Impl::AScaleType;
    using BScaleType = typename Impl::BScaleType;

    using AComputeDataType = typename Impl::AComputeDataType;
    using BComputeDataType = typename Impl::BComputeDataType;
    using CComputeDataType = typename Impl::CComputeDataType;
    

    using AVecType =
        ext_vector_t<ADataType, vector_traits<typename Impl::AFp8Bf8VecType>::vector_size * kKIter * MRepeat * MInterleave>;
    using BVecType =
        ext_vector_t<BDataType, vector_traits<typename Impl::BFp8Bf8VecType>::vector_size * kKIter * NRepeat * NInterleave>;
    using CVecType =
        ext_vector_t<CDataType, vector_traits<typename Impl::CVecType>::vector_size * MRepeat * NRepeat * MInterleave * NInterleave>;

    using ABufType = thread_buffer<typename Impl::AVecType, kKIter * MRepeat * MInterleave>;
    using BBufType = thread_buffer<typename Impl::BVecType, kKIter * NRepeat * NInterleave>;
    using CBufType = thread_buffer<typename Impl::CVecType, MRepeat * NRepeat * MInterleave * NInterleave>;
    using AScaleBufType = thread_buffer<AScaleType, MRepeat * MInterleave>;
    using BScaleBufType = thread_buffer<BScaleType, NRepeat * NInterleave>;

    static constexpr index_t kM = Impl::kM * MRepeat * MInterleave;
    static constexpr index_t kN = Impl::kN * NRepeat * NInterleave;
    static constexpr index_t kK = Impl::kK * kKIter;
    static constexpr index_t kKPerThread = Impl::kABKPerLane * kKIter;

    using AWarpDstrEncoding = tile_distribution_encoding<
        sequence<>,
        tuple<sequence<MRepeat, Impl::kAMLane, MInterleave>, sequence<Impl::kABKLane, Impl::kABKPerLane * kKIter>>,
        tuple<sequence<2, 1>>,
        tuple<sequence<0, 1>>,
        sequence<1, 1, 2>,
        sequence<0, 2, 1>>;

    using BWarpDstrEncoding = tile_distribution_encoding<
        sequence<>,
        tuple<sequence<NRepeat, Impl::kBNLane, NInterleave>, sequence<Impl::kABKLane, Impl::kABKPerLane * kKIter>>,
        tuple<sequence<2, 1>>,
        tuple<sequence<0, 1>>,
        sequence<1, 1, 2>,
        sequence<0, 2, 1>>;
    
    using CWarpDstrEncoding = tile_distribution_encoding<
        sequence<>,
        tuple<sequence<MRepeat, Impl::kCMLane, MInterleave, Impl::kCMPerLane>,
              sequence<NRepeat, Impl::kCNLane, NInterleave, Impl::kCN0PerLane, Impl::kCN1PerLane>>,
        tuple<sequence<2, 1>>,
        tuple<sequence<1, 1>>,
        sequence<1, 2, 1, 1, 2, 2, 2>, // <MRepeat, NRepeat, MInterleave, MPerLane, N0PerLane,
                                       // NmmacInterleave, N1PerLane>
        sequence<0, 0, 2, 3, 3, 2, 4>>;

    using CWarpOutputDstrEncoding = tile_distribution_encoding<
        sequence<>,
        tuple<sequence<MRepeat, Impl::kCMLane, MInterleave, Impl::kCMPerLane>,
              sequence<NRepeat, Impl::kCN0PerLane, Impl::kCNLane, NInterleave * Impl::kCN1PerLane>>,
        tuple<sequence<2, 1>>,
        tuple<sequence<2, 1>>,
        sequence<1, 2, 1, 1, 2, 2>, // <MRepeat, NRepeat, MInterleave, MPerLane, N0PerLane,
                                    // NmmacInterleave, N1PerLane>
        sequence<0, 0, 2, 3, 1, 3>>;

    using AScaleWarpDstrEncoding = tile_distribution_encoding<
        sequence<>,
        tuple<sequence<MRepeat, Impl::kAMLane, MInterleave>,
              sequence<Impl::kABKLane, 1>>,
        tuple<sequence<2, 1>>,
        tuple<sequence<0, 1>>,
        sequence<1, 1, 2>,
        sequence<0, 2, 1>>;

    using BScaleWarpDstrEncoding = tile_distribution_encoding<
        sequence<>,
        tuple<sequence<NRepeat, Impl::kBNLane, NInterleave>,
              sequence<Impl::kABKLane, 1>>,
        tuple<sequence<2, 1>>,
        tuple<sequence<0, 1>>,
        sequence<1, 1, 2>,
        sequence<0, 2, 1>>;

    
    // c_vec += a_vec * b_vec
    CK_TILE_DEVICE void
    operator()(CBufType& c_buf, const ABufType& a_buf, const BBufType& b_buf, const AScaleBufType& a_scale_buf, const BScaleBufType& b_scale_buf) const
    {
        static_for<0, MRepeat, 1>{}([&](auto iMRepeat) {
            static_for<0, NRepeat, 1>{}([&](auto iNRepeat) {
                static_for<0, MInterleave, 1>{}([&](auto iMInterleave) {
                    static_for<0, NInterleave, 1>{}([&](auto iNInterleave) {
                        static_for<0, kKIter, 1>{}([&](auto iKIter) {
                            //1 mmac issue
                            Impl{}(c_buf(iMRepeat * NRepeat * MInterleave * NInterleave + 
                                         iNRepeat * MInterleave * NInterleave +
                                         iMInterleave * NInterleave + iNInterleave),
                                   a_buf[iMRepeat * MInterleave * kKIter + iMInterleave * kKIter + 
                                         iKIter],
                                   b_buf[iNRepeat * NInterleave * kKIter + iNInterleave * kKIter + 
                                         iKIter],
                                   a_scale_buf[iMRepeat * MInterleave + iMInterleave],
                                   b_scale_buf[iNRepeat * NInterleave + iNInterleave]);
                        });
                    });
                });
            });
        });
    }

    // c_vec += a_vec * b_vec
    CK_TILE_DEVICE void
    operator()(CVecType& c_vec, const AVecType& a_vec, const BVecType& b_vec) const
    {
        using buf_a = thread_buffer<typename Impl::AFp8Bf8VecType, kKIter * MRepeat * MInterleave>;
        using buf_b = thread_buffer<typename Impl::BFp8Bf8VecType, kKIter * NRepeat * NInterleave>;
        using buf_c = thread_buffer<typename Impl::CVecType, MRepeat * NRepeat * MInterleave * NInterleave>;

        static_for<0, MRepeat, 1>{}([&](auto iMRepeat) {
            static_for<0, NRepeat, 1>{}([&](auto iNRepeat) {
                static_for<0, MInterleave, 1>{}([&](auto iMInterleave) {
                    static_for<0, NInterleave, 1>{}([&](auto iNInterleave) {
                        static_for<0, kKIter, 1>{}([&](auto iKIter) {
                            //1 mmac issue
                            Impl{}(reinterpret_cast<buf_c&>(c_vec)
                                       .template get_as<typename Impl::CVecType>()[iMRepeat * NRepeat * MInterleave * NInterleave +
                                                                                   iNRepeat * MInterleave * NInterleave +
                                                                                   iMInterleave * NInterleave + iNInterleave], 
                                   reinterpret_cast<const buf_a&>(a_vec)
                                       .template get_as<typename Impl::AVecType>()[iMRepeat * MInterleave * kKIter +        // AVecType -> Impl::AFp8Bf8VecType perf model上性能相仿
                                                                                   iMInterleave * kKIter + iKIter],
                                   reinterpret_cast<const buf_b&>(b_vec)
                                       .template get_as<typename Impl::BVecType>()[iNRepeat * NInterleave * kKIter +
                                                                                   iNInterleave * kKIter + iKIter]);
                        });
                    });
                });
            });
        });
    }

    // c_vec = a_vec * b_vec
    // AVecType = ext_vector_t<ADataType, AWarpTensor::get_thread_buffer_size()>;
    CK_TILE_DEVICE CVecType operator()(const AVecType& a_vec, const BVecType& b_vec) const
    {
        using buf_a = thread_buffer<typename Impl::AVecType, kKIter * MRepeat * MInterleave>;
        using buf_b = thread_buffer<typename Impl::BVecType, kKIter * NRepeat * NInterleave>;
        using buf_c = thread_buffer<typename Impl::CVecType, MRepeat * NRepeat * MInterleave * NInterleave>;

        buf_c c_vec;

        // c += a * b
        static_for<0, MRepeat, 1>{}([&](auto iMRepeat) {
            static_for<0, NRepeat, 1>{}([&](auto iNRepeat) {
                static_for<0, MInterleave, 1>{}([&](auto iMInterleave) {
                    static_for<0, NInterleave, 1>{}([&](auto iNInterleave) {
                        static_for<0, kKIter, 1>{}([&](auto iKIter) {
                            Impl{}(c_vec.template get_as<typename Impl::CVecType>()[iMRepeat * NRepeat * MInterleave * NInterleave +
                                                                                   iNRepeat * MInterleave * NInterleave +
                                                                                   iMInterleave * NInterleave +
                                                                                   iNInterleave], 
                                   reinterpret_cast<const buf_a&>(a_vec)
                                       .template get_as<typename Impl::AVecType>()[iMRepeat * MInterleave * kKIter +    // AVecType -> Impl::AFp8Bf8VecType perf model上性能相仿
                                                                                   iMInterleave * kKIter +
                                                                                   iKIter],
                                   reinterpret_cast<const buf_b&>(b_vec)
                                       .template get_as<typename Impl::BVecType>()[iNRepeat * NInterleave * kKIter +
                                                                                   iNInterleave * kKIter +
                                                                                   iKIter]);
                        });
                    });
                });
            });
        });

        return reinterpret_cast<CVecType>(c_vec);
    }

};


} // namespace ck_tile
