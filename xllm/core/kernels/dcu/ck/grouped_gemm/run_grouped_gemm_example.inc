// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025, , Inc. All rights reserved.

#pragma once
template <typename Layout>
static constexpr inline auto is_row_major(Layout layout_)
{
    return ck_tile::bool_constant<std::is_same_v<ck_tile::remove_cvref_t<decltype(layout_)>,
                                                 ck_tile::tensor_layout::gemm::RowMajor>>{};
}

template <typename ADataType, typename BDataType, typename AccDataType, typename CDataType>
auto calculate_rtol_atol(const ck_tile::index_t K,
                         const ck_tile::index_t kbatch,
                         const float max_accumulated_value)
{
    using ComputeType =
        std::conditional_t<sizeof(ADataType) < sizeof(BDataType), ADataType, BDataType>;
    // Calculate thresholds
    const auto rtol = ck_tile::get_relative_threshold<ComputeType, CDataType, AccDataType>(
        ck_tile::integer_divide_ceil(K, kbatch));
    const auto atol = ck_tile::get_absolute_threshold<ComputeType, CDataType, AccDataType>(
        max_accumulated_value / kbatch, ck_tile::integer_divide_ceil(K, kbatch));
    // Calculate error due to split_k accumulation
    const auto rtol_split_k =
        ck_tile::get_relative_threshold<CDataType, CDataType, CDataType>(kbatch);
    const auto atol_split_k = ck_tile::get_absolute_threshold<CDataType, CDataType, CDataType>(
        max_accumulated_value, kbatch);
    // Use higher threshold
    return ck_tile::make_tuple(std::max(rtol, rtol_split_k), std::max(atol, atol_split_k));
}

inline float grouped_gemm_abs_float(float x)
{
    return x < 0.0f ? -x : x;
}

inline float grouped_gemm_half_bits_to_float(uint16_t bits)
{
    const uint32_t sign = static_cast<uint32_t>(bits & 0x8000u) << 16;
    const uint32_t exp  = (bits >> 10) & 0x1fu;
    const uint32_t mant = bits & 0x03ffu;

    uint32_t out_bits = 0;
    if(exp == 0)
    {
        if(mant == 0)
        {
            out_bits = sign;
        }
        else
        {
            float value = std::ldexp(static_cast<float>(mant), -24);
            if(sign != 0)
            {
                value = -value;
            }
            return value;
        }
    }
    else if(exp == 0x1fu)
    {
        out_bits = sign | 0x7f800000u | (mant << 13);
    }
    else
    {
        out_bits = sign | ((exp + 112u) << 23) | (mant << 13);
    }

    float value = 0.0f;
    std::memcpy(&value, &out_bits, sizeof(value));
    return value;
}

inline float grouped_gemm_half_to_float_raw(const ck_tile::half_t& value)
{
    uint16_t bits = 0;
    static_assert(sizeof(value) == sizeof(bits));
    std::memcpy(&bits, &value, sizeof(bits));
    return grouped_gemm_half_bits_to_float(bits);
}

inline ck_tile::half_t grouped_gemm_half_from_bits(uint16_t bits)
{
    ck_tile::half_t value;
    static_assert(sizeof(value) == sizeof(bits));
    std::memcpy(&value, &bits, sizeof(bits));
    return value;
}

inline uint16_t grouped_gemm_float_to_half_bits(float value)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));

    const uint32_t sign = (bits >> 16) & 0x8000u;
    uint32_t exp        = (bits >> 23) & 0xffu;
    uint32_t mant       = bits & 0x7fffffu;

    if(exp == 0xffu)
    {
        return static_cast<uint16_t>(sign | (mant == 0 ? 0x7c00u : 0x7e00u));
    }

    const int32_t half_exp = static_cast<int32_t>(exp) - 127 + 15;
    if(half_exp >= 0x1f)
    {
        return static_cast<uint16_t>(sign | 0x7c00u);
    }
    if(half_exp <= 0)
    {
        if(half_exp < -10)
        {
            return static_cast<uint16_t>(sign);
        }
        mant |= 0x800000u;
        const uint32_t shift = static_cast<uint32_t>(14 - half_exp);
        uint32_t half_mant   = mant >> shift;
        const uint32_t round = mant & ((1u << shift) - 1u);
        const uint32_t halfway = 1u << (shift - 1);
        if(round > halfway || (round == halfway && (half_mant & 1u) != 0))
        {
            ++half_mant;
        }
        return static_cast<uint16_t>(sign | half_mant);
    }

    uint32_t half = sign | (static_cast<uint32_t>(half_exp) << 10) | (mant >> 13);
    const uint32_t round = mant & 0x1fffu;
    if(round > 0x1000u || (round == 0x1000u && (half & 1u) != 0))
    {
        ++half;
    }
    return static_cast<uint16_t>(half);
}

inline float grouped_gemm_round_float_to_half_float(float value)
{
    return grouped_gemm_half_bits_to_float(grouped_gemm_float_to_half_bits(value));
}

inline uint32_t grouped_gemm_hash_index(uint32_t x)
{
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

inline ck_tile::pk_int4_t grouped_gemm_pack_i4_pair(int lo, int hi)
{
    const auto lo_u4 = static_cast<uint8_t>(lo) & 0x0f;
    const auto hi_u4 = static_cast<uint8_t>(hi) & 0x0f;
    return ck_tile::pk_int4_t(static_cast<int8_t>((hi_u4 << 4) | lo_u4));
}

inline float grouped_gemm_pk_int4_to_float_at(const ck_tile::pk_int4_t& value,
                                              ck_tile::index_t scalar_idx)
{
    const uint8_t packed = ck_tile::bit_cast<uint8_t>(value);
    const int nibble = (scalar_idx & 1) ? ((packed >> 4) & 0x0f) : (packed & 0x0f);
    return static_cast<float>(nibble > 7 ? nibble - 16 : nibble);
}

template <typename T>
void grouped_gemm_fill_uniform(ck_tile::HostTensor<T>& tensor, uint32_t salt)
{
    if constexpr(std::is_same_v<T, ck_tile::half_t>)
    {
        for(std::size_t i = 0; i < tensor.mData.size(); ++i)
        {
            const uint32_t h = grouped_gemm_hash_index(static_cast<uint32_t>(i) ^ salt);
            if((h & 0x3fu) == 0)
            {
                tensor.mData[i] = grouped_gemm_half_from_bits(static_cast<uint16_t>(0));
            }
            else
            {
                const uint16_t sign = (h & 0x80000000u) != 0 ? 0x8000u : 0u;
                const uint16_t exp  = static_cast<uint16_t>(14u << 10);
                const uint16_t mant = static_cast<uint16_t>((h >> 10) & 0x03ffu);
                tensor.mData[i]     = grouped_gemm_half_from_bits(sign | exp | mant);
            }
        }
    }
    else if constexpr(std::is_same_v<T, ck_tile::pk_int4_t>)
    {
        for(std::size_t i = 0; i < tensor.mData.size(); ++i)
        {
            const uint32_t h  = grouped_gemm_hash_index(static_cast<uint32_t>(i) ^ salt);
            const int lo      = static_cast<int>((h >> 0) & 0x0fu) - 8;
            const int hi      = static_cast<int>((h >> 8) & 0x0fu) - 8;
            tensor.mData[i] = grouped_gemm_pack_i4_pair(lo, hi);
        }
    }
    else
    {
        ck_tile::FillUniformDistribution<T>{-1.f, 1.f}(tensor);
    }
}

inline void grouped_gemm_fill_int4_const(ck_tile::HostTensor<ck_tile::pk_int4_t>& tensor,
                                         int value)
{
    const auto packed = grouped_gemm_pack_i4_pair(value, value);
    for(auto& item : tensor.mData)
    {
        item = packed;
    }
}

inline void grouped_gemm_set_int4_logical(ck_tile::HostTensor<ck_tile::pk_int4_t>& tensor,
                                          ck_tile::index_t row,
                                          ck_tile::index_t col,
                                          int value)
{
    const auto raw_offset = tensor.mDesc.GetOffsetFromMultiIndex(row, col);
    const auto offset     = raw_offset / 2;
    const auto nibble     = static_cast<uint8_t>(value) & 0x0f;
    auto packed           = ck_tile::bit_cast<uint8_t>(tensor.mData[offset]);
    if((raw_offset & 1) == 0)
    {
        packed = static_cast<uint8_t>((packed & 0xf0) | nibble);
    }
    else
    {
        packed = static_cast<uint8_t>((packed & 0x0f) | (nibble << 4));
    }
    tensor.mData[offset] = ck_tile::pk_int4_t(static_cast<int8_t>(packed));
}

inline void grouped_gemm_fill_int4_b_one_k(ck_tile::HostTensor<ck_tile::pk_int4_t>& tensor,
                                           ck_tile::index_t logical_k)
{
    for(auto& item : tensor.mData)
    {
        item = ck_tile::pk_int4_t{0};
    }
    const auto N = static_cast<ck_tile::index_t>(tensor.get_lengths()[1]);
    for(ck_tile::index_t n = 0; n < N; ++n)
    {
        grouped_gemm_set_int4_logical(tensor, logical_k, n, 1);
    }
}

template <typename T>
inline float grouped_gemm_to_float_for_validation(const T& value)
{
    if constexpr(std::is_same_v<T, ck_tile::half_t>)
    {
        return grouped_gemm_half_to_float_raw(value);
    }
    else
    {
        if constexpr(std::is_same_v<T, ck_tile::pk_int4_t>)
        {
            return grouped_gemm_pk_int4_to_float_at(value, 0);
        }
        else
        {
            return ck_tile::type_convert<float>(value);
        }
    }
}

template <typename T>
inline float grouped_gemm_tensor_value_for_k(const ck_tile::HostTensor<T>& tensor,
                                             ck_tile::index_t row,
                                             ck_tile::index_t col,
                                             ck_tile::index_t k,
                                             bool k_on_first_dim)
{
    if constexpr(std::is_same_v<T, ck_tile::pk_int4_t>)
    {
        const auto offset =
            k_on_first_dim ? tensor.GetOffsetFromMultiIndex(k, col)
                           : tensor.GetOffsetFromMultiIndex(row, k);
        return grouped_gemm_pk_int4_to_float_at(tensor.mData[offset], k);
    }
    else
    {
        return grouped_gemm_to_float_for_validation(
            tensor.mData[tensor.GetOffsetFromMultiIndex(row, col)]);
    }
}

template <typename ADataType,
          typename BDataType,
          typename CDataType,
          typename ALayout,
          typename BLayout>
void grouped_gemm_make_host_reference(const ck_tile::HostTensor<ADataType>& a_m_k,
                                      const ck_tile::HostTensor<BDataType>& b_k_n,
                                      ck_tile::HostTensor<CDataType>& c_m_n,
                                      ck_tile::index_t M,
                                      ck_tile::index_t N,
                                      ck_tile::index_t K,
                                      [[maybe_unused]] const ALayout a_layout,
                                      [[maybe_unused]] const BLayout b_layout)
{
    for(ck_tile::index_t m = 0; m < M; ++m)
    {
        for(ck_tile::index_t n = 0; n < N; ++n)
        {
            float ref = 0.0f;
            for(ck_tile::index_t k = 0; k < K; ++k)
            {
                const float a = grouped_gemm_tensor_value_for_k(a_m_k, m, k, k, false);
                const float b = grouped_gemm_tensor_value_for_k(b_k_n, k, n, k, true);
                ref += a * b;
            }
            c_m_n.mData[c_m_n.GetOffsetFromMultiIndex(m, n)] =
                ck_tile::type_convert<CDataType>(ref);
        }
    }
}

template <typename ADataType,
          typename BDataType,
          typename ALayout,
          typename BLayout>
float grouped_gemm_half_reference_max(const ck_tile::HostTensor<ADataType>& a_m_k,
                                      const ck_tile::HostTensor<BDataType>& b_k_n,
                                      ck_tile::index_t M,
                                      ck_tile::index_t N,
                                      ck_tile::index_t K,
                                      [[maybe_unused]] const ALayout a_layout,
                                      [[maybe_unused]] const BLayout b_layout)
{
    ck_tile::HostTensor<float> c_m_n({M, N});
    grouped_gemm_make_host_reference(
        a_m_k, b_k_n, c_m_n, M, N, K, a_layout, b_layout);

    float max_accumulated_value = 0.0f;
    for(const auto& value : c_m_n.mData)
    {
        max_accumulated_value =
            std::max(max_accumulated_value, grouped_gemm_abs_float(value));
    }
    return max_accumulated_value;
}

template <typename ADataType,
          typename BDataType,
          typename CDataType,
          typename ALayout,
          typename BLayout,
          typename CLayout>
bool grouped_gemm_check_half_reference(
    const ck_tile::HostTensor<ADataType>& a_m_k,
    const ck_tile::HostTensor<BDataType>& b_k_n,
    const ck_tile::HostTensor<CDataType>& c_m_n,
    ck_tile::index_t M,
    ck_tile::index_t N,
    ck_tile::index_t K,
    [[maybe_unused]] const ALayout a_layout,
    [[maybe_unused]] const BLayout b_layout,
    [[maybe_unused]] const CLayout c_layout,
    const std::string& msg,
    double rtol,
    double atol,
    bool debug)
{
    bool pass       = true;
    int err_count   = 0;
    double max_err  = 0.0;
    constexpr int error_detail_limit = 5;
    float first_ref = 0.0f;
    for(ck_tile::index_t m = 0; m < M; ++m)
    {
        for(ck_tile::index_t n = 0; n < N; ++n)
        {
            float ref = 0.0f;
            for(ck_tile::index_t k = 0; k < K; ++k)
            {
                const float a = grouped_gemm_tensor_value_for_k(a_m_k, m, k, k, false);
                const float b = grouped_gemm_tensor_value_for_k(b_k_n, k, n, k, true);
                ref += a * b;
            }

            if(m == 0 && n == 0)
            {
                first_ref = ref;
            }

            const auto c_offset = c_m_n.GetOffsetFromMultiIndex(m, n);
            const double out    = grouped_gemm_to_float_for_validation(c_m_n.mData[c_offset]);
            const double err    = std::abs(out - static_cast<double>(ref));
            if(err > atol + rtol * std::abs(ref) || !std::isfinite(out) || !std::isfinite(ref))
            {
                max_err = std::max(max_err, err);
                ++err_count;
                if(err_count < error_detail_limit)
                {
                    std::cerr << msg << std::setw(12) << std::setprecision(7) << " out["
                              << c_offset << "] != ref[" << c_offset << "]: " << out
                              << " != " << ref << std::endl;
                }
                pass = false;
            }
        }
    }

    if(debug)
    {
        std::cout << "debug raw-half first A/B/C/ref: "
                  << grouped_gemm_to_float_for_validation(a_m_k.mData[0]) << ", "
                  << grouped_gemm_to_float_for_validation(b_k_n.mData[0]) << ", "
                  << grouped_gemm_to_float_for_validation(c_m_n.mData[0]) << ", "
                  << first_ref << std::endl;
    }

    if(!pass)
    {
        std::cerr << "max err: " << max_err << ", number of errors: " << err_count << ", "
                  << (static_cast<double>(err_count) / static_cast<double>(M * N) * 100.0)
                  << "% wrong values" << std::endl;
    }

    return pass;
}

template <typename GemmConfig,
          typename ADataType,
          typename BDataType,
          typename DsDataType,
          typename AccDataType,
          typename CDataType,
          typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename CLayout,
          typename CDEElementWise = ck_tile::element_wise::PassThrough>
float invoke_gemm(int n_warmup,
                  int n_repeat,
                  int group_count,
                  const std::vector<grouped_gemm_kargs>& args)
{
    (void)group_count;

    // Workspace memory allocated to hold the gemm descriptions.
    ck_tile::DeviceMem gemm_workspace;
    gemm_workspace.Realloc(get_workspace_size(args));

    float ave_time = 0;
    if constexpr(!GemmConfig::Persistent)
    {
        ave_time = grouped_gemm<GemmConfig,
                                ADataType,
                                BDataType,
                                DsDataType,
                                AccDataType,
                                CDataType,
                                ALayout,
                                BLayout,
                                DsLayout,
                                CLayout,
                                CDEElementWise>(
            args,
            ck_tile::stream_config{nullptr, true, 1, n_warmup, n_repeat},
            gemm_workspace.GetDeviceBuffer());
    }
    else
    {
        if(GemmConfig::Preshuffle)
        {
            // not supported yet
            throw std::runtime_error(
                "Persistent grouped gemm with preshuffle is not supported yet");
        }

        // NOTE: With the persistent TileLoop kernel, we do not necessarily need to haveCollapse
        // commentComment on line L74tenpercent commented on Sep 5, 2025 tenpercenton Sep 5,
        // 2025ContributorMore actionsdid you intend to remove the comment?Write a replyResolve
        // commentCode has comments. Press enter to view. the gemm problems known on the host.
        // Instead, we can just pass the pointer to the kernel and let the workgroups figure out
        // which tiles to work on. This is useful when the gemm problems are generated dynamically.
        // In this example however, we generate the `kargs` using the known gemm_descs,
        // and copy the gemm descriptions to the device memory.
        // The contents of the memory pointed to by `kargs_ptr` pointer could be
        // written by e.g. another kernel from earlier stage.

        std::vector<ck_tile::GemmTransKernelArg> kargs;
        void* kargs_ptr   = gemm_workspace.GetDeviceBuffer();
        const bool splitk = args[0].k_batch > 1;
        for(const auto& arg : args)
        {
            kargs.emplace_back(ck_tile::UniversalGemmKernelArgs<>{{arg.a_ptr},
                                                                  {arg.b_ptr},
                                                                  {/*arg.ds_ptr*/},
                                                                  arg.e_ptr,
                                                                  arg.M,
                                                                  arg.N,
                                                                  arg.K,
                                                                  {arg.stride_A},
                                                                  {arg.stride_B},
                                                                  {/*arg.stride_Ds*/},
                                                                  arg.stride_E,
                                                                  arg.k_batch});
        }
        const auto stream = ck_tile::stream_config{nullptr, true, 1, n_warmup, n_repeat};
        HIP_CHECK_ERROR(hipMemcpyWithStream(kargs_ptr,
                                            kargs.data(),
                                            kargs.size() * sizeof(ck_tile::GemmTransKernelArg),
                                            hipMemcpyHostToDevice,
                                            stream.stream_id_));

        ave_time = grouped_gemm_tileloop<GemmConfig,
                                         ALayout,
                                         BLayout,
                                         CLayout,
                                         ADataType,
                                         BDataType,
                                         AccDataType,
                                         CDataType>(stream, group_count, kargs_ptr, splitk);
    }

    return ave_time;
}

template <typename GemmConfig,
          typename ADataType,
          typename BDataType,
          typename CDataType,
          typename AccDataType,
          typename ALayout,
          typename BLayout,
          typename CLayout>
int run_grouped_gemm_example_with_layouts(int argc,
                                          char* argv[],
                                          const ALayout a_layout                  = ALayout{},
                                          const BLayout b_layout                  = BLayout{},
                                          [[maybe_unused]] const CLayout c_layout = CLayout{})
{

    auto [result, arg_parser] = create_args(argc, argv);

    auto valid_input_data = [&](int group_count, const auto&... args) {
        return !(args.empty() || ...) && group_count == (args.size() == ...);
    };

    const int group_count = arg_parser.get_int("group_count");
    const int repeat      = arg_parser.get_int("repeat");
    const int warmup      = arg_parser.get_int("warmup");
    const int kbatch      = arg_parser.get_int("kbatch");
    const bool debug      = arg_parser.get_bool("debug");
    bool validate         = arg_parser.get_bool("validate");
    const std::string int4_const = arg_parser.get_str("int4_const");
    const std::string int4_const_a = arg_parser.get_str("int4_const_a");
    const std::string int4_const_b = arg_parser.get_str("int4_const_b");
    const ck_tile::index_t int4_b_one_k = arg_parser.get_int("int4_b_one_k");

    if(kbatch > 1 && validate && warmup + repeat > 1)
    {
        std::cout << "WARNING: Data validation enabled with SplitK and more than"
                  << "1 warmup/repeat. Disabling validation." << std::endl;
        validate = false;
    }

    std::vector<ck_tile::index_t> Ms        = arg_parser.get_int_vec("Ms");
    std::vector<ck_tile::index_t> Ns        = arg_parser.get_int_vec("Ns");
    std::vector<ck_tile::index_t> Ks        = arg_parser.get_int_vec("Ks");
    std::vector<ck_tile::index_t> stride_As = arg_parser.get_int_vec("stride_As");
    std::vector<ck_tile::index_t> stride_Bs = arg_parser.get_int_vec("stride_Bs");
    std::vector<ck_tile::index_t> stride_Cs = arg_parser.get_int_vec("stride_Cs");

    if(!valid_input_data(group_count, Ms, Ns, Ks, stride_As, stride_Bs, stride_Cs))
    {
        std::cout << "Please check the input data. Default values will be used." << std::endl;
        std::cout << "Default values: Ms (256, 512, 768, 1024..), Ns (256, 768, 1280..), Ks (512, "
                     "896, 1280..)"
                  << std::endl;
        for(int i = 0; i < group_count; i++)
        {
            //Non Persitent kernel模式：必须保证所有组的 K % K_Tile % 2 的 tailNumber 一致，因为目前所有group
            //                          使用tail_num是启动kernel前通过group[0]计算得到的, 见grouped_gemm()
            //Persistent Kernel模式：不受此限制，因为每个cu启动的block都会动态计算自己的tail_num
            //因目前采用double buffer配置，且pipeline存在预取2个tile的设计，K维度需要满足 K / K_Tile >= 2，且 K / K_Tile != 3.
            Ms.push_back(256 + 256 * i);
            Ns.push_back(256 + 128 * i);
            Ks.push_back(640 + 256 * i);

            stride_As.push_back(Ks[i]);
            stride_Bs.push_back(Ks[i]);
            stride_Cs.push_back(Ns[i]);
        }
    }

    std::vector<ck_tile::HostTensor<ADataType>> a_m_k_tensors;
    std::vector<ck_tile::HostTensor<BDataType>> b_k_n_tensors;
    std::vector<ck_tile::HostTensor<CDataType>> c_m_n_tensors;

    a_m_k_tensors.reserve(group_count);
    b_k_n_tensors.reserve(group_count);
    c_m_n_tensors.reserve(group_count);

    std::vector<std::unique_ptr<ck_tile::DeviceMem>> a_m_k_dev_buf;
    std::vector<std::unique_ptr<ck_tile::DeviceMem>> b_k_n_dev_buf;
    std::vector<std::unique_ptr<ck_tile::DeviceMem>> c_m_n_dev_buf;

    a_m_k_dev_buf.reserve(group_count);
    b_k_n_dev_buf.reserve(group_count);
    c_m_n_dev_buf.reserve(group_count);

    std::vector<grouped_gemm_kargs> gemm_descs;
    gemm_descs.reserve(group_count);

    for(int i = 0; i < group_count; ++i)
    {

        const ck_tile::index_t M = Ms[i];
        const ck_tile::index_t N = Ns[i];
        const ck_tile::index_t K = Ks[i];

        stride_As[i] = ck_tile::get_default_stride(M, K, stride_As[i], is_row_major(a_layout));
        stride_Bs[i] = ck_tile::get_default_stride(K, N, stride_Bs[i], is_row_major(b_layout));
        stride_Cs[i] = ck_tile::get_default_stride(M, N, stride_Cs[i], is_row_major(CLayout{}));

        ck_tile::index_t host_stride_A = stride_As[i];
        ck_tile::index_t host_stride_B = stride_Bs[i];
        if constexpr(std::is_same_v<ADataType, ck_tile::pk_int4_t>)
        {
            host_stride_A *= ck_tile::numeric_traits<ADataType>::PackedSize;
        }
        if constexpr(std::is_same_v<BDataType, ck_tile::pk_int4_t>)
        {
            host_stride_B *= ck_tile::numeric_traits<BDataType>::PackedSize;
        }

        a_m_k_tensors.push_back(ck_tile::HostTensor<ADataType>(
            ck_tile::host_tensor_descriptor(M, K, host_stride_A, is_row_major(a_layout))));
        b_k_n_tensors.push_back(ck_tile::HostTensor<BDataType>(
            ck_tile::host_tensor_descriptor(K, N, host_stride_B, is_row_major(b_layout))));
        c_m_n_tensors.push_back(ck_tile::HostTensor<CDataType>(
            ck_tile::host_tensor_descriptor(M, N, stride_Cs[i], is_row_major(CLayout{}))));

        std::cout << "gemm[" << i << "]" << " a_m_k: " << a_m_k_tensors[i].mDesc
                  << " b_k_n: " << b_k_n_tensors[i].mDesc << " c_m_n: " << c_m_n_tensors[i].mDesc
                  << std::endl;

        if constexpr(std::is_same_v<ADataType, ck_tile::pk_int4_t>)
        {
            if(!int4_const_a.empty())
            {
                grouped_gemm_fill_int4_const(a_m_k_tensors[i], std::atoi(int4_const_a.c_str()));
            }
            else if(!int4_const.empty())
            {
                grouped_gemm_fill_int4_const(a_m_k_tensors[i], std::atoi(int4_const.c_str()));
            }
            else
            {
                grouped_gemm_fill_uniform<ADataType>(a_m_k_tensors[i],
                                                     static_cast<uint32_t>(0x11939u + i));
            }
        }
        else if constexpr(std::is_integral_v<ADataType>)
        {
            ck_tile::FillUniformDistributionIntegerValue<ADataType>{-20, 20}(a_m_k_tensors[i]);
        } else {
            grouped_gemm_fill_uniform<ADataType>(a_m_k_tensors[i],
                                                 static_cast<uint32_t>(0x11939u + i));
        }

        if constexpr(std::is_same_v<BDataType, ck_tile::pk_int4_t>)
        {
            if(int4_b_one_k >= 0)
            {
                grouped_gemm_fill_int4_b_one_k(b_k_n_tensors[i], int4_b_one_k);
            }
            else if(!int4_const_b.empty())
            {
                grouped_gemm_fill_int4_const(b_k_n_tensors[i], std::atoi(int4_const_b.c_str()));
            }
            else if(!int4_const.empty())
            {
                grouped_gemm_fill_int4_const(b_k_n_tensors[i], std::atoi(int4_const.c_str()));
            }
            else
            {
                grouped_gemm_fill_uniform<BDataType>(b_k_n_tensors[i],
                                                     static_cast<uint32_t>(0x23872u + i));
            }
        }
        else if constexpr(std::is_integral_v<BDataType>)
        {
            ck_tile::FillUniformDistributionIntegerValue<BDataType>{-20, 20}(b_k_n_tensors[i]);
        } else {
            grouped_gemm_fill_uniform<BDataType>(b_k_n_tensors[i],
                                                 static_cast<uint32_t>(0x23872u + i));
        }

        a_m_k_dev_buf.push_back(std::make_unique<ck_tile::DeviceMem>(a_m_k_tensors[i]));

        // Perform preshuffle for B tensor
        // if constexpr(GemmConfig::Preshuffle)
        // {
        //     ck_tile::HostTensor<BDataType> b_shuffle_host = shuffle_b<GemmConfig>(b_k_n_tensors[i]);
        //     b_k_n_dev_buf.push_back(std::make_unique<ck_tile::DeviceMem>(b_shuffle_host));
        // }
        // else
        // {
            b_k_n_dev_buf.push_back(std::make_unique<ck_tile::DeviceMem>(b_k_n_tensors[i]));
        // }

        c_m_n_dev_buf.push_back(std::make_unique<ck_tile::DeviceMem>(c_m_n_tensors[i]));

        c_m_n_dev_buf[i]->SetZero();
        c_m_n_tensors[i].SetZero();

        const void* p_a = a_m_k_dev_buf[i]->GetDeviceBuffer();
        const void* p_b = b_k_n_dev_buf[i]->GetDeviceBuffer();
        void* p_c       = c_m_n_dev_buf[i]->GetDeviceBuffer();

        gemm_descs.push_back(
            {p_a, p_b, p_c, kbatch, M, N, K, stride_As[i], stride_Bs[i], stride_Cs[i]});
    }

    float ave_time = invoke_gemm<GemmConfig,
                                 ADataType,
                                 BDataType,
                                 ck_tile::tuple<>,
                                 AccDataType,
                                 CDataType,
                                 ALayout,
                                 BLayout,
                                 ck_tile::tuple<>,
                                 CLayout>(warmup, repeat, group_count, gemm_descs);

    std::string op_name{"Grouped Gemm"};

    std::size_t flop = 0, num_btype = 0;
    for(int j = 0; j < group_count; ++j)
    {
        flop += std::size_t(2) * gemm_descs[j].M * gemm_descs[j].N * gemm_descs[j].K;

        num_btype += sizeof(ADataType) * gemm_descs[j].M * gemm_descs[j].K +
                     sizeof(BDataType) * gemm_descs[j].K * gemm_descs[j].N +
                     sizeof(CDataType) * gemm_descs[j].M * gemm_descs[j].N;
    }

    float tflops     = static_cast<float>(flop) / 1.E9 / ave_time;
    float gb_per_sec = num_btype / 1.E6 / ave_time;

    std::cout << "Perf: " << std::setw(10) << ave_time << " ms, " << tflops << " TFlops, "
              << gb_per_sec << " GB/s, " << op_name << std::endl;

    for(int i = 0; i < group_count; i++)
    {
        c_m_n_dev_buf[i]->FromDevice(c_m_n_tensors[i].data());
    }

    bool pass{true};
    if(validate)
    {
        for(int i = 0; i < group_count; ++i)
        {
            float max_accumulated_value = 0.0f;
            if constexpr(std::is_same_v<ADataType, ck_tile::half_t> &&
                         std::is_same_v<BDataType, ck_tile::half_t> &&
                         std::is_same_v<CDataType, ck_tile::half_t>)
            {
                max_accumulated_value =
                    grouped_gemm_half_reference_max(a_m_k_tensors[i],
                                                    b_k_n_tensors[i],
                                                    Ms[i],
                                                    Ns[i],
                                                    Ks[i],
                                                    ALayout{},
                                                    BLayout{});
            }
            else
            {
                ck_tile::HostTensor<CDataType> c_m_n_host_ref(ck_tile::host_tensor_descriptor(
                    Ms[i], Ns[i], stride_Cs[i], is_row_major(CLayout{})));
                c_m_n_host_ref.SetZero();
                if constexpr(std::is_same_v<ADataType, ck_tile::pk_int4_t> ||
                             std::is_same_v<BDataType, ck_tile::pk_int4_t>)
                {
                    grouped_gemm_make_host_reference(a_m_k_tensors[i],
                                                     b_k_n_tensors[i],
                                                     c_m_n_host_ref,
                                                     Ms[i],
                                                     Ns[i],
                                                     Ks[i],
                                                     ALayout{},
                                                     BLayout{});
                    if(debug)
                    {
                        int q00 = 0, q01 = 0, q10 = 0, q11 = 0;
                        std::array<int, 16> n_bands{};
                        std::array<int, 8 * 16> mn_bands{};
                        std::array<int, 256> n_columns{};
                        std::array<long long, 256> n_column_first_out{};
                        std::array<long long, 256> n_column_first_ref{};
                        std::array<ck_tile::index_t, 8> sample_m{};
                        std::array<ck_tile::index_t, 8> sample_n{};
                        std::array<long long, 8> sample_out{};
                        std::array<long long, 8> sample_ref{};
                        std::size_t sample_count = 0;
                        const auto m_mid = Ms[i] / 2;
                        const auto n_mid = Ns[i] / 2;
                        for(ck_tile::index_t m = 0; m < Ms[i]; ++m)
                        {
                            for(ck_tile::index_t n = 0; n < Ns[i]; ++n)
                            {
                                const auto c_offset =
                                    c_m_n_tensors[i].GetOffsetFromMultiIndex(m, n);
                                const auto ref_offset =
                                    c_m_n_host_ref.GetOffsetFromMultiIndex(m, n);
                                const auto out = static_cast<long long>(
                                    grouped_gemm_to_float_for_validation(
                                        c_m_n_tensors[i].mData[c_offset]));
                                const auto ref = static_cast<long long>(
                                    grouped_gemm_to_float_for_validation(
                                        c_m_n_host_ref.mData[ref_offset]));
                                if(out != ref)
                                {
                                    if(m < m_mid && n < n_mid)
                                        ++q00;
                                    else if(m < m_mid)
                                        ++q01;
                                    else if(n < n_mid)
                                        ++q10;
                                    else
                                        ++q11;
                                    const auto band =
                                        static_cast<std::size_t>(n / 16 < 16 ? n / 16 : 15);
                                    ++n_bands[band];
                                    const auto column = static_cast<std::size_t>(
                                        n < static_cast<ck_tile::index_t>(n_columns.size())
                                            ? n
                                            : static_cast<ck_tile::index_t>(n_columns.size() - 1));
                                    if(n_columns[column] == 0)
                                    {
                                        n_column_first_out[column] = out;
                                        n_column_first_ref[column] = ref;
                                    }
                                    ++n_columns[column];
                                    const auto m_band_raw = static_cast<std::size_t>(m / 16);
                                    const auto m_band_count = mn_bands.size() / n_bands.size();
                                    const auto m_band =
                                        m_band_raw < m_band_count ? m_band_raw : m_band_count - 1;
                                    ++mn_bands[m_band * n_bands.size() + band];
                                    if(sample_count < sample_m.size())
                                    {
                                        sample_m[sample_count]   = m;
                                        sample_n[sample_count]   = n;
                                        sample_out[sample_count] = out;
                                        sample_ref[sample_count] = ref;
                                        ++sample_count;
                                    }
                                }
                            }
                        }
                        std::cout << "debug gemm[" << i
                                  << "] int4 mismatch quadrants: " << q00 << ", " << q01
                                  << ", " << q10 << ", " << q11 << std::endl;
                        std::cout << "debug gemm[" << i << "] int4 mismatch n16:";
                        for(std::size_t band = 0; band < n_bands.size(); ++band)
                        {
                            if(band * 16 < static_cast<std::size_t>(Ns[i]))
                                std::cout << " " << n_bands[band];
                        }
                        std::cout << std::endl;
                        std::cout << "debug gemm[" << i << "] int4 mismatch columns:";
                        for(std::size_t column = 0; column < n_columns.size(); ++column)
                        {
                            if(column < static_cast<std::size_t>(Ns[i]) && n_columns[column] != 0)
                            {
                                std::cout << " " << column << ":" << n_columns[column] << "("
                                          << n_column_first_out[column] << " vs "
                                          << n_column_first_ref[column] << ")";
                            }
                        }
                        std::cout << std::endl;
                        std::cout << "debug gemm[" << i << "] int4 mismatch m16n16:";
                        for(std::size_t m_band = 0; m_band < mn_bands.size() / n_bands.size();
                            ++m_band)
                        {
                            if(m_band * 16 < static_cast<std::size_t>(Ms[i]))
                            {
                                std::cout << " |";
                                for(std::size_t n_band = 0; n_band < n_bands.size(); ++n_band)
                                {
                                    if(n_band * 16 < static_cast<std::size_t>(Ns[i]))
                                        std::cout << " "
                                                  << mn_bands[m_band * n_bands.size() + n_band];
                                }
                            }
                        }
                        std::cout << std::endl;
                        std::cout << "debug gemm[" << i << "] int4 mismatch samples:";
                        for(std::size_t sample = 0; sample < sample_count; ++sample)
                        {
                            std::cout << " (" << sample_m[sample] << "," << sample_n[sample]
                                      << ": " << sample_out[sample] << " vs "
                                      << sample_ref[sample] << ")";
                        }
                        std::cout << std::endl;
                    }
                }
                else
                {
                    ck_tile::reference_gemm<ADataType, BDataType, AccDataType, CDataType>(
                        a_m_k_tensors[i], b_k_n_tensors[i], c_m_n_host_ref);
                }
                for(const auto& value : c_m_n_host_ref.mData)
                {
                    const float value_f = grouped_gemm_to_float_for_validation(value);
                    max_accumulated_value =
                        std::max(max_accumulated_value, grouped_gemm_abs_float(value_f));
                }
            }

            const auto rtol_atol =
                calculate_rtol_atol<ADataType, BDataType, AccDataType, CDataType>(
                    Ks[i], kbatch, max_accumulated_value);
            if(debug)
            {
                float max_a = 0.0f;
                float max_b = 0.0f;
                float max_c = 0.0f;
                for(const auto& value : a_m_k_tensors[i].mData)
                {
                    max_a = std::max(
                        max_a,
                        grouped_gemm_abs_float(grouped_gemm_to_float_for_validation(value)));
                }
                for(const auto& value : b_k_n_tensors[i].mData)
                {
                    max_b = std::max(
                        max_b,
                        grouped_gemm_abs_float(grouped_gemm_to_float_for_validation(value)));
                }
                for(const auto& value : c_m_n_tensors[i].mData)
                {
                    max_c = std::max(
                        max_c,
                        grouped_gemm_abs_float(grouped_gemm_to_float_for_validation(value)));
                }
                std::cout << "debug gemm[" << i << "] max_abs A/B/C/ref: " << max_a << ", "
                          << max_b << ", " << max_c << ", " << max_accumulated_value
                          << std::endl;
            }

            if constexpr(std::is_same_v<ADataType, ck_tile::half_t> &&
                         std::is_same_v<BDataType, ck_tile::half_t> &&
                         std::is_same_v<CDataType, ck_tile::half_t>)
            {
                pass &= grouped_gemm_check_half_reference(
                    a_m_k_tensors[i],
                    b_k_n_tensors[i],
                    c_m_n_tensors[i],
                    Ms[i],
                    Ns[i],
                    Ks[i],
                    ALayout{},
                    BLayout{},
                    CLayout{},
                    "Error: Incorrect results! in group [" + std::to_string(i) + "]",
                    rtol_atol.at(ck_tile::number<0>{}),
                    rtol_atol.at(ck_tile::number<1>{}),
                    debug);
            }
            else
            {
                ck_tile::HostTensor<CDataType> c_m_n_host_ref(ck_tile::host_tensor_descriptor(
                    Ms[i], Ns[i], stride_Cs[i], is_row_major(CLayout{})));
                c_m_n_host_ref.SetZero();
                if constexpr(std::is_same_v<ADataType, ck_tile::pk_int4_t> ||
                             std::is_same_v<BDataType, ck_tile::pk_int4_t>)
                {
                    grouped_gemm_make_host_reference(a_m_k_tensors[i],
                                                     b_k_n_tensors[i],
                                                     c_m_n_host_ref,
                                                     Ms[i],
                                                     Ns[i],
                                                     Ks[i],
                                                     ALayout{},
                                                     BLayout{});
                }
                else
                {
                    ck_tile::reference_gemm<ADataType, BDataType, AccDataType, CDataType>(
                        a_m_k_tensors[i], b_k_n_tensors[i], c_m_n_host_ref);
                }
                pass &= ck_tile::check_err(
                    c_m_n_tensors[i],
                    c_m_n_host_ref,
                    "Error: Incorrect results! in group [" + std::to_string(i) + "]",
                    rtol_atol.at(ck_tile::number<0>{}),
                    rtol_atol.at(ck_tile::number<1>{}));
            }
            std::cout << "gemm[" << i
                      << "] Relative error threshold: " << rtol_atol.at(ck_tile::number<0>{})
                      << " Absolute error threshold: " << rtol_atol.at(ck_tile::number<1>{})
                      << std::endl;
        }
        std::cout << "The CPU verification result is:" << (pass ? "correct" : "fail") << std::endl;
    }

    if(arg_parser.get_int("json") == 1)
    {
        dump_grouped_gemm_json_results<ALayout, BLayout, CLayout>(arg_parser.get_str("jsonfile"),
                                                                  op_name,
                                                                  group_count,
                                                                  pass,
                                                                  ave_time,
                                                                  tflops,
                                                                  gb_per_sec);
    }

    return pass;
}

template <typename GemmConfig,
          typename ADataType,
          typename BDataType,
          typename CDataType,
          typename AccDataType,
          typename CDEElementWise = ck_tile::element_wise::AddAdd,
          typename ALayout,
          typename BLayout,
          typename CLayout>
int run_grouped_gemm_multiple_d_example_with_layouts(
    int argc,
    char* argv[],
    const ALayout a_layout                  = ALayout{},
    const BLayout b_layout                  = BLayout{},
    [[maybe_unused]] const CLayout c_layout = CLayout{})
{
    auto [result, arg_parser] = create_args(argc, argv);

    auto valid_input_data = [&](int group_count, const auto&... args) {
        return !(args.empty() || ...) && group_count == (args.size() == ...);
    };

    const int group_count = arg_parser.get_int("group_count");
    const int repeat      = arg_parser.get_int("repeat");
    const int warmup      = arg_parser.get_int("warmup");
    const int kbatch      = arg_parser.get_int("kbatch");
    const bool debug      = arg_parser.get_bool("debug");
    const bool validate   = arg_parser.get_bool("validate");

    if(kbatch != 1)
    {
        throw std::runtime_error("multiple-D grouped_gemm currently supports only kbatch=1");
    }

    std::vector<ck_tile::index_t> Ms        = arg_parser.get_int_vec("Ms");
    std::vector<ck_tile::index_t> Ns        = arg_parser.get_int_vec("Ns");
    std::vector<ck_tile::index_t> Ks        = arg_parser.get_int_vec("Ks");
    std::vector<ck_tile::index_t> stride_As = arg_parser.get_int_vec("stride_As");
    std::vector<ck_tile::index_t> stride_Bs = arg_parser.get_int_vec("stride_Bs");
    std::vector<ck_tile::index_t> stride_Cs = arg_parser.get_int_vec("stride_Cs");

    if(!valid_input_data(group_count, Ms, Ns, Ks, stride_As, stride_Bs, stride_Cs))
    {
        std::cout << "Please check the input data. Default values will be used." << std::endl;
        for(int i = 0; i < group_count; i++)
        {
            Ms.push_back(256 + 256 * i);
            Ns.push_back(256 + 128 * i);
            Ks.push_back(640 + 256 * i);
            stride_As.push_back(Ks[i]);
            stride_Bs.push_back(Ks[i]);
            stride_Cs.push_back(Ns[i]);
        }
    }

    std::vector<ck_tile::HostTensor<ADataType>> a_m_k_tensors;
    std::vector<ck_tile::HostTensor<BDataType>> b_k_n_tensors;
    std::vector<ck_tile::HostTensor<CDataType>> d0_m_n_tensors;
    std::vector<ck_tile::HostTensor<CDataType>> d1_m_n_tensors;
    std::vector<ck_tile::HostTensor<CDataType>> c_m_n_tensors;

    a_m_k_tensors.reserve(group_count);
    b_k_n_tensors.reserve(group_count);
    d0_m_n_tensors.reserve(group_count);
    d1_m_n_tensors.reserve(group_count);
    c_m_n_tensors.reserve(group_count);

    std::vector<std::unique_ptr<ck_tile::DeviceMem>> a_m_k_dev_buf;
    std::vector<std::unique_ptr<ck_tile::DeviceMem>> b_k_n_dev_buf;
    std::vector<std::unique_ptr<ck_tile::DeviceMem>> d0_m_n_dev_buf;
    std::vector<std::unique_ptr<ck_tile::DeviceMem>> d1_m_n_dev_buf;
    std::vector<std::unique_ptr<ck_tile::DeviceMem>> c_m_n_dev_buf;

    a_m_k_dev_buf.reserve(group_count);
    b_k_n_dev_buf.reserve(group_count);
    d0_m_n_dev_buf.reserve(group_count);
    d1_m_n_dev_buf.reserve(group_count);
    c_m_n_dev_buf.reserve(group_count);

    std::vector<ck_tile::GroupedGemmHostArgsImpl<2>> gemm_descs;
    gemm_descs.reserve(group_count);

    for(int i = 0; i < group_count; ++i)
    {
        const ck_tile::index_t M = Ms[i];
        const ck_tile::index_t N = Ns[i];
        const ck_tile::index_t K = Ks[i];

        stride_As[i] = ck_tile::get_default_stride(M, K, stride_As[i], is_row_major(a_layout));
        stride_Bs[i] = ck_tile::get_default_stride(K, N, stride_Bs[i], is_row_major(b_layout));
        stride_Cs[i] = ck_tile::get_default_stride(M, N, stride_Cs[i], is_row_major(CLayout{}));

        a_m_k_tensors.push_back(ck_tile::HostTensor<ADataType>(
            ck_tile::host_tensor_descriptor(M, K, stride_As[i], is_row_major(a_layout))));
        b_k_n_tensors.push_back(ck_tile::HostTensor<BDataType>(
            ck_tile::host_tensor_descriptor(K, N, stride_Bs[i], is_row_major(b_layout))));
        d0_m_n_tensors.push_back(ck_tile::HostTensor<CDataType>(
            ck_tile::host_tensor_descriptor(M, N, stride_Cs[i], is_row_major(CLayout{}))));
        d1_m_n_tensors.push_back(ck_tile::HostTensor<CDataType>(
            ck_tile::host_tensor_descriptor(M, N, stride_Cs[i], is_row_major(CLayout{}))));
        c_m_n_tensors.push_back(ck_tile::HostTensor<CDataType>(
            ck_tile::host_tensor_descriptor(M, N, stride_Cs[i], is_row_major(CLayout{}))));

        std::cout << "gemm[" << i << "] a_m_k: " << a_m_k_tensors[i].mDesc
                  << " b_k_n: " << b_k_n_tensors[i].mDesc << " d0_m_n: "
                  << d0_m_n_tensors[i].mDesc << " d1_m_n: " << d1_m_n_tensors[i].mDesc
                  << " c_m_n: " << c_m_n_tensors[i].mDesc << std::endl;

        grouped_gemm_fill_uniform<ADataType>(a_m_k_tensors[i], static_cast<uint32_t>(0x11939u + i));
        grouped_gemm_fill_uniform<BDataType>(b_k_n_tensors[i], static_cast<uint32_t>(0x23872u + i));
        grouped_gemm_fill_uniform<CDataType>(d0_m_n_tensors[i],
                                             static_cast<uint32_t>(0x34567u + i));
        grouped_gemm_fill_uniform<CDataType>(d1_m_n_tensors[i],
                                             static_cast<uint32_t>(0x45678u + i));

        a_m_k_dev_buf.push_back(std::make_unique<ck_tile::DeviceMem>(a_m_k_tensors[i]));
        b_k_n_dev_buf.push_back(std::make_unique<ck_tile::DeviceMem>(b_k_n_tensors[i]));
        d0_m_n_dev_buf.push_back(std::make_unique<ck_tile::DeviceMem>(d0_m_n_tensors[i]));
        d1_m_n_dev_buf.push_back(std::make_unique<ck_tile::DeviceMem>(d1_m_n_tensors[i]));
        c_m_n_dev_buf.push_back(std::make_unique<ck_tile::DeviceMem>(c_m_n_tensors[i]));
        c_m_n_dev_buf[i]->SetZero();
        c_m_n_tensors[i].SetZero();

        gemm_descs.emplace_back(a_m_k_dev_buf[i]->GetDeviceBuffer(),
                                b_k_n_dev_buf[i]->GetDeviceBuffer(),
                                std::array<const void*, 2>{d0_m_n_dev_buf[i]->GetDeviceBuffer(),
                                                           d1_m_n_dev_buf[i]->GetDeviceBuffer()},
                                c_m_n_dev_buf[i]->GetDeviceBuffer(),
                                kbatch,
                                M,
                                N,
                                K,
                                stride_As[i],
                                stride_Bs[i],
                                std::array<ck_tile::index_t, 2>{stride_Cs[i], stride_Cs[i]},
                                stride_Cs[i]);
    }

    ck_tile::DeviceMem gemm_workspace;
    gemm_workspace.Realloc(get_workspace_size(gemm_descs));
    const float ave_time = grouped_gemm<GemmConfig,
                                        ADataType,
                                        BDataType,
                                        ck_tile::tuple<CDataType, CDataType>,
                                        AccDataType,
                                        CDataType,
                                        ALayout,
                                        BLayout,
                                        ck_tile::tuple<CLayout, CLayout>,
                                        CLayout,
                                        CDEElementWise>(
        gemm_descs,
        ck_tile::stream_config{nullptr, true, 1, warmup, repeat},
        gemm_workspace.GetDeviceBuffer());

    std::size_t flop = 0, num_btype = 0;
    for(int j = 0; j < group_count; ++j)
    {
        flop += std::size_t(2) * gemm_descs[j].M * gemm_descs[j].N * gemm_descs[j].K;
        num_btype += sizeof(ADataType) * gemm_descs[j].M * gemm_descs[j].K +
                     sizeof(BDataType) * gemm_descs[j].K * gemm_descs[j].N +
                     2 * sizeof(CDataType) * gemm_descs[j].M * gemm_descs[j].N +
                     sizeof(CDataType) * gemm_descs[j].M * gemm_descs[j].N;
    }

    const float tflops     = static_cast<float>(flop) / 1.E9 / ave_time;
    const float gb_per_sec = num_btype / 1.E6 / ave_time;
    const char* op_name = std::is_same_v<CDEElementWise, ck_tile::element_wise::MultiplyMultiply>
                              ? "Grouped Gemm Multiple-D Multiply"
                              : "Grouped Gemm Multiple-D Add";
    std::cout << "Perf: " << std::setw(10) << ave_time << " ms, " << tflops << " TFlops, "
              << gb_per_sec << " GB/s, " << op_name << std::endl;

    for(int i = 0; i < group_count; i++)
    {
        c_m_n_dev_buf[i]->FromDevice(c_m_n_tensors[i].data());
    }

    bool pass{true};
    if(validate)
    {
        constexpr bool is_multiply =
            std::is_same_v<CDEElementWise, ck_tile::element_wise::MultiplyMultiply>;
        constexpr bool is_half_output = std::is_same_v<CDataType, ck_tile::half_t> ||
                                        std::is_same_v<CDataType, ck_tile::bf16_t>;
        for(int i = 0; i < group_count; ++i)
        {
            if constexpr(is_half_output)
            {
                float max_accumulated_value =
                    grouped_gemm_half_reference_max(a_m_k_tensors[i],
                                                    b_k_n_tensors[i],
                                                    Ms[i],
                                                    Ns[i],
                                                    Ks[i],
                                                    ALayout{},
                                                    BLayout{});
                if constexpr(is_multiply)
                {
                    max_accumulated_value *= 2.0f;
                }
                else
                {
                    max_accumulated_value += 2.0f;
                }

                const auto rtol_atol =
                    calculate_rtol_atol<ADataType, BDataType, AccDataType, CDataType>(
                        Ks[i], kbatch, max_accumulated_value);

                int err_count   = 0;
                double max_err  = 0.0;
                constexpr int error_detail_limit = 5;
                for(ck_tile::index_t m = 0; m < Ms[i]; ++m)
                {
                    for(ck_tile::index_t n = 0; n < Ns[i]; ++n)
                    {
                        float ref = 0.0f;
                        for(ck_tile::index_t k = 0; k < Ks[i]; ++k)
                        {
                            const float a =
                                grouped_gemm_tensor_value_for_k(a_m_k_tensors[i], m, k, k, false);
                            const float b =
                                grouped_gemm_tensor_value_for_k(b_k_n_tensors[i], k, n, k, true);
                            ref += a * b;
                        }
                        const auto c_offset = c_m_n_tensors[i].GetOffsetFromMultiIndex(m, n);
                        ref = grouped_gemm_round_float_to_half_float(ref);
                        if constexpr(is_multiply)
                        {
                            ref = grouped_gemm_round_float_to_half_float(
                                ref * grouped_gemm_to_float_for_validation(d0_m_n_tensors[i].mData[c_offset]));
                            ref = grouped_gemm_round_float_to_half_float(
                                ref * grouped_gemm_to_float_for_validation(d1_m_n_tensors[i].mData[c_offset]));
                        }
                        else
                        {
                            ref = grouped_gemm_round_float_to_half_float(
                                ref + grouped_gemm_to_float_for_validation(d0_m_n_tensors[i].mData[c_offset]));
                            ref = grouped_gemm_round_float_to_half_float(
                                ref + grouped_gemm_to_float_for_validation(d1_m_n_tensors[i].mData[c_offset]));
                        }
                        const double out =
                            grouped_gemm_to_float_for_validation(c_m_n_tensors[i].mData[c_offset]);
                        const double err = std::abs(out - static_cast<double>(ref));
                        const double rtol = rtol_atol.at(ck_tile::number<0>{});
                        const double atol = 4.0 * rtol_atol.at(ck_tile::number<1>{});
                        if(err > atol + rtol * std::abs(ref) || !std::isfinite(out) ||
                           !std::isfinite(ref))
                        {
                            max_err = std::max(max_err, err);
                            ++err_count;
                            if(err_count < error_detail_limit)
                            {
                                std::cerr << "Error: Incorrect multiple-D results! in group [" << i
                                          << "] out[" << c_offset << "] != ref[" << c_offset
                                          << "]: " << out << " != " << ref << std::endl;
                            }
                            pass = false;
                        }
                    }
                }
                if(!pass)
                {
                    std::cerr << "max err: " << max_err << ", number of errors: " << err_count
                              << std::endl;
                }
                if(debug)
                {
                    std::cout << "debug multiple-D gemm[" << i << "] max_ref_bound: "
                              << max_accumulated_value << std::endl;
                }
                std::cout << "gemm[" << i
                          << "] Relative error threshold: " << rtol_atol.at(ck_tile::number<0>{})
                          << " Absolute error threshold: "
                          << 4.0 * rtol_atol.at(ck_tile::number<1>{})
                          << std::endl;
            }
            else
            {
                // Non-half output (float for fp8/bf8, int32 for int8)
                ck_tile::HostTensor<CDataType> c_m_n_host_ref(ck_tile::host_tensor_descriptor(
                    Ms[i], Ns[i], stride_Cs[i], is_row_major(CLayout{})));
                c_m_n_host_ref.SetZero();
                if constexpr(std::is_same_v<ADataType, ck_tile::pk_int4_t> ||
                             std::is_same_v<BDataType, ck_tile::pk_int4_t>)
                {
                    grouped_gemm_make_host_reference(a_m_k_tensors[i],
                                                     b_k_n_tensors[i],
                                                     c_m_n_host_ref,
                                                     Ms[i],
                                                     Ns[i],
                                                     Ks[i],
                                                     ALayout{},
                                                     BLayout{});
                }
                else
                {
                    ck_tile::reference_gemm<ADataType, BDataType, AccDataType, CDataType>(
                        a_m_k_tensors[i], b_k_n_tensors[i], c_m_n_host_ref);
                }

                float max_accumulated_value = 0.0f;
                const std::size_t total_elems = c_m_n_host_ref.mData.size();
                for(std::size_t idx = 0; idx < total_elems; ++idx)
                {
                    if constexpr(is_multiply)
                    {
                        c_m_n_host_ref.mData[idx] = c_m_n_host_ref.mData[idx] * d0_m_n_tensors[i].mData[idx] * d1_m_n_tensors[i].mData[idx];
                    }
                    else
                    {
                        c_m_n_host_ref.mData[idx] = c_m_n_host_ref.mData[idx] + d0_m_n_tensors[i].mData[idx] + d1_m_n_tensors[i].mData[idx];
                    }
                    max_accumulated_value = std::max(max_accumulated_value,
                        grouped_gemm_abs_float(grouped_gemm_to_float_for_validation(c_m_n_host_ref.mData[idx])));
                }

                const auto rtol_atol =
                    calculate_rtol_atol<ADataType, BDataType, AccDataType, CDataType>(
                        Ks[i], kbatch, max_accumulated_value);
                pass &= ck_tile::check_err(
                    c_m_n_tensors[i],
                    c_m_n_host_ref,
                    "Error: Incorrect multiple-D results! in group [" + std::to_string(i) + "]",
                    rtol_atol.at(ck_tile::number<0>{}),
                    rtol_atol.at(ck_tile::number<1>{}));
                std::cout << "gemm[" << i
                          << "] Relative error threshold: " << rtol_atol.at(ck_tile::number<0>{})
                          << " Absolute error threshold: " << rtol_atol.at(ck_tile::number<1>{})
                          << std::endl;
            }
        }
        std::cout << "The CPU verification result is:" << (pass ? "correct" : "fail") << std::endl;
    }

    return pass;
}

template <typename GemmConfig,
          typename ADataType,
          typename BDataType,
          typename CDataType,
          typename AccDataType,
          typename ALayout,
          typename BLayout,
          typename CLayout>
int run_grouped_gemm_bias_example_with_layouts(
    int argc,
    char* argv[],
    const ALayout a_layout                  = ALayout{},
    const BLayout b_layout                  = BLayout{},
    [[maybe_unused]] const CLayout c_layout = CLayout{})
{
    auto [result, arg_parser] = create_args(argc, argv);

    auto valid_input_data = [&](int group_count, const auto&... args) {
        return !(args.empty() || ...) && group_count == (args.size() == ...);
    };

    const int group_count = arg_parser.get_int("group_count");
    const int repeat      = arg_parser.get_int("repeat");
    const int warmup      = arg_parser.get_int("warmup");
    const int kbatch      = arg_parser.get_int("kbatch");
    const bool debug      = arg_parser.get_bool("debug");
    const bool validate   = arg_parser.get_bool("validate");

    if(kbatch != 1)
    {
        throw std::runtime_error("bias grouped_gemm currently supports only kbatch=1");
    }

    std::vector<ck_tile::index_t> Ms        = arg_parser.get_int_vec("Ms");
    std::vector<ck_tile::index_t> Ns        = arg_parser.get_int_vec("Ns");
    std::vector<ck_tile::index_t> Ks        = arg_parser.get_int_vec("Ks");
    std::vector<ck_tile::index_t> stride_As = arg_parser.get_int_vec("stride_As");
    std::vector<ck_tile::index_t> stride_Bs = arg_parser.get_int_vec("stride_Bs");
    std::vector<ck_tile::index_t> stride_Cs = arg_parser.get_int_vec("stride_Cs");

    if(!valid_input_data(group_count, Ms, Ns, Ks, stride_As, stride_Bs, stride_Cs))
    {
        std::cout << "Please check the input data. Default values will be used." << std::endl;
        for(int i = 0; i < group_count; i++)
        {
            Ms.push_back(256 + 256 * i);
            Ns.push_back(256 + 128 * i);
            Ks.push_back(640 + 256 * i);
            stride_As.push_back(Ks[i]);
            stride_Bs.push_back(Ks[i]);
            stride_Cs.push_back(Ns[i]);
        }
    }

    std::vector<ck_tile::HostTensor<ADataType>> a_m_k_tensors;
    std::vector<ck_tile::HostTensor<BDataType>> b_k_n_tensors;
    std::vector<ck_tile::HostTensor<CDataType>> d0_m_n_tensors;
    std::vector<ck_tile::HostTensor<CDataType>> c_m_n_tensors;

    std::vector<std::unique_ptr<ck_tile::DeviceMem>> a_m_k_dev_buf;
    std::vector<std::unique_ptr<ck_tile::DeviceMem>> b_k_n_dev_buf;
    std::vector<std::unique_ptr<ck_tile::DeviceMem>> d0_m_n_dev_buf;
    std::vector<std::unique_ptr<ck_tile::DeviceMem>> c_m_n_dev_buf;

    std::vector<ck_tile::GroupedGemmHostArgsImpl<1>> gemm_descs;
    gemm_descs.reserve(group_count);

    for(int i = 0; i < group_count; ++i)
    {
        const ck_tile::index_t M = Ms[i];
        const ck_tile::index_t N = Ns[i];
        const ck_tile::index_t K = Ks[i];

        stride_As[i] = ck_tile::get_default_stride(M, K, stride_As[i], is_row_major(a_layout));
        stride_Bs[i] = ck_tile::get_default_stride(K, N, stride_Bs[i], is_row_major(b_layout));
        stride_Cs[i] = ck_tile::get_default_stride(M, N, stride_Cs[i], is_row_major(CLayout{}));

        a_m_k_tensors.push_back(ck_tile::HostTensor<ADataType>(
            ck_tile::host_tensor_descriptor(M, K, stride_As[i], is_row_major(a_layout))));
        b_k_n_tensors.push_back(ck_tile::HostTensor<BDataType>(
            ck_tile::host_tensor_descriptor(K, N, stride_Bs[i], is_row_major(b_layout))));
        d0_m_n_tensors.push_back(ck_tile::HostTensor<CDataType>(
            ck_tile::host_tensor_descriptor(M, N, stride_Cs[i], is_row_major(CLayout{}))));
        c_m_n_tensors.push_back(ck_tile::HostTensor<CDataType>(
            ck_tile::host_tensor_descriptor(M, N, stride_Cs[i], is_row_major(CLayout{}))));

        std::cout << "gemm[" << i << "] a_m_k: " << a_m_k_tensors[i].mDesc
                  << " b_k_n: " << b_k_n_tensors[i].mDesc << " d0_m_n: "
                  << d0_m_n_tensors[i].mDesc << " c_m_n: " << c_m_n_tensors[i].mDesc << std::endl;

        grouped_gemm_fill_uniform<ADataType>(a_m_k_tensors[i], static_cast<uint32_t>(0x11939u + i));
        grouped_gemm_fill_uniform<BDataType>(b_k_n_tensors[i], static_cast<uint32_t>(0x23872u + i));
        grouped_gemm_fill_uniform<CDataType>(d0_m_n_tensors[i],
                                             static_cast<uint32_t>(0x34567u + i));

        a_m_k_dev_buf.push_back(std::make_unique<ck_tile::DeviceMem>(a_m_k_tensors[i]));
        b_k_n_dev_buf.push_back(std::make_unique<ck_tile::DeviceMem>(b_k_n_tensors[i]));
        d0_m_n_dev_buf.push_back(std::make_unique<ck_tile::DeviceMem>(d0_m_n_tensors[i]));
        c_m_n_dev_buf.push_back(std::make_unique<ck_tile::DeviceMem>(c_m_n_tensors[i]));
        c_m_n_dev_buf[i]->SetZero();
        c_m_n_tensors[i].SetZero();

        gemm_descs.emplace_back(a_m_k_dev_buf[i]->GetDeviceBuffer(),
                                b_k_n_dev_buf[i]->GetDeviceBuffer(),
                                std::array<const void*, 1>{d0_m_n_dev_buf[i]->GetDeviceBuffer()},
                                c_m_n_dev_buf[i]->GetDeviceBuffer(),
                                kbatch,
                                M,
                                N,
                                K,
                                stride_As[i],
                                stride_Bs[i],
                                std::array<ck_tile::index_t, 1>{stride_Cs[i]},
                                stride_Cs[i]);
    }

    ck_tile::DeviceMem gemm_workspace;
    gemm_workspace.Realloc(get_workspace_size(gemm_descs));
    const float ave_time = grouped_gemm<GemmConfig,
                                        ADataType,
                                        BDataType,
                                        ck_tile::tuple<CDataType>,
                                        AccDataType,
                                        CDataType,
                                        ALayout,
                                        BLayout,
                                        ck_tile::tuple<CLayout>,
                                        CLayout,
                                        ck_tile::element_wise::Add>(
        gemm_descs,
        ck_tile::stream_config{nullptr, true, 1, warmup, repeat},
        gemm_workspace.GetDeviceBuffer());

    std::size_t flop = 0, num_btype = 0;
    for(int j = 0; j < group_count; ++j)
    {
        flop += std::size_t(2) * gemm_descs[j].M * gemm_descs[j].N * gemm_descs[j].K;
        num_btype += sizeof(ADataType) * gemm_descs[j].M * gemm_descs[j].K +
                     sizeof(BDataType) * gemm_descs[j].K * gemm_descs[j].N +
                     sizeof(CDataType) * gemm_descs[j].M * gemm_descs[j].N +
                     sizeof(CDataType) * gemm_descs[j].M * gemm_descs[j].N;
    }

    const float tflops     = static_cast<float>(flop) / 1.E9 / ave_time;
    const float gb_per_sec = num_btype / 1.E6 / ave_time;
    std::cout << "Perf: " << std::setw(10) << ave_time << " ms, " << tflops << " TFlops, "
              << gb_per_sec << " GB/s, Grouped Gemm Bias Add" << std::endl;

    for(int i = 0; i < group_count; i++)
    {
        c_m_n_dev_buf[i]->FromDevice(c_m_n_tensors[i].data());
    }

    bool pass{true};
    if(validate)
    {
        constexpr bool is_half_output = std::is_same_v<CDataType, ck_tile::half_t> ||
                                        std::is_same_v<CDataType, ck_tile::bf16_t>;
        for(int i = 0; i < group_count; ++i)
        {
            if constexpr(is_half_output)
            {
                float max_accumulated_value =
                    grouped_gemm_half_reference_max(a_m_k_tensors[i],
                                                    b_k_n_tensors[i],
                                                    Ms[i],
                                                    Ns[i],
                                                    Ks[i],
                                                    ALayout{},
                                                    BLayout{});
                max_accumulated_value += 1.0f;

                const auto rtol_atol =
                    calculate_rtol_atol<ADataType, BDataType, AccDataType, CDataType>(
                        Ks[i], kbatch, max_accumulated_value);

                int err_count   = 0;
                double max_err  = 0.0;
                constexpr int error_detail_limit = 5;
                for(ck_tile::index_t m = 0; m < Ms[i]; ++m)
                {
                    for(ck_tile::index_t n = 0; n < Ns[i]; ++n)
                    {
                        float ref = 0.0f;
                        for(ck_tile::index_t k = 0; k < Ks[i]; ++k)
                        {
                            const float a =
                                grouped_gemm_tensor_value_for_k(a_m_k_tensors[i], m, k, k, false);
                            const float b =
                                grouped_gemm_tensor_value_for_k(b_k_n_tensors[i], k, n, k, true);
                            ref += a * b;
                        }
                        const auto c_offset = c_m_n_tensors[i].GetOffsetFromMultiIndex(m, n);
                        ref = grouped_gemm_round_float_to_half_float(ref);
                        ref = grouped_gemm_round_float_to_half_float(
                            ref + grouped_gemm_to_float_for_validation(d0_m_n_tensors[i].mData[c_offset]));
                        const double out =
                            grouped_gemm_to_float_for_validation(c_m_n_tensors[i].mData[c_offset]);
                        const double err = std::abs(out - static_cast<double>(ref));
                        const double rtol = rtol_atol.at(ck_tile::number<0>{});
                        const double atol = 4.0 * rtol_atol.at(ck_tile::number<1>{});
                        if(err > atol + rtol * std::abs(ref) || !std::isfinite(out) ||
                           !std::isfinite(ref))
                        {
                            max_err = std::max(max_err, err);
                            ++err_count;
                            if(err_count < error_detail_limit)
                            {
                                std::cerr << "Error: Incorrect bias results! in group [" << i
                                          << "] out[" << c_offset << "] != ref[" << c_offset
                                          << "]: " << out << " != " << ref << std::endl;
                            }
                            pass = false;
                        }
                    }
                }
                if(!pass)
                {
                    std::cerr << "max err: " << max_err << ", number of errors: " << err_count
                              << std::endl;
                }
                if(debug)
                {
                    std::cout << "debug bias gemm[" << i << "] max_ref_bound: "
                              << max_accumulated_value << std::endl;
                }
                std::cout << "gemm[" << i
                          << "] Relative error threshold: " << rtol_atol.at(ck_tile::number<0>{})
                          << " Absolute error threshold: "
                          << 4.0 * rtol_atol.at(ck_tile::number<1>{})
                          << std::endl;
            }
            else
            {
                ck_tile::HostTensor<CDataType> c_m_n_host_ref(ck_tile::host_tensor_descriptor(
                    Ms[i], Ns[i], stride_Cs[i], is_row_major(CLayout{})));
                c_m_n_host_ref.SetZero();
                if constexpr(std::is_same_v<ADataType, ck_tile::pk_int4_t> ||
                             std::is_same_v<BDataType, ck_tile::pk_int4_t>)
                {
                    grouped_gemm_make_host_reference(a_m_k_tensors[i],
                                                     b_k_n_tensors[i],
                                                     c_m_n_host_ref,
                                                     Ms[i],
                                                     Ns[i],
                                                     Ks[i],
                                                     ALayout{},
                                                     BLayout{});
                }
                else
                {
                    ck_tile::reference_gemm<ADataType, BDataType, AccDataType, CDataType>(
                        a_m_k_tensors[i], b_k_n_tensors[i], c_m_n_host_ref);
                }

                float max_accumulated_value = 0.0f;
                const std::size_t total_elems = c_m_n_host_ref.mData.size();
                for(std::size_t idx = 0; idx < total_elems; ++idx)
                {
                    c_m_n_host_ref.mData[idx] = c_m_n_host_ref.mData[idx] + d0_m_n_tensors[i].mData[idx];
                    max_accumulated_value = std::max(max_accumulated_value,
                        grouped_gemm_abs_float(grouped_gemm_to_float_for_validation(c_m_n_host_ref.mData[idx])));
                }

                const auto rtol_atol =
                    calculate_rtol_atol<ADataType, BDataType, AccDataType, CDataType>(
                        Ks[i], kbatch, max_accumulated_value);
                pass &= ck_tile::check_err(
                    c_m_n_tensors[i],
                    c_m_n_host_ref,
                    "Error: Incorrect bias results! in group [" + std::to_string(i) + "]",
                    rtol_atol.at(ck_tile::number<0>{}),
                    rtol_atol.at(ck_tile::number<1>{}));
                std::cout << "gemm[" << i
                          << "] Relative error threshold: " << rtol_atol.at(ck_tile::number<0>{})
                          << " Absolute error threshold: " << rtol_atol.at(ck_tile::number<1>{})
                          << std::endl;
            }
        }
        std::cout << "The CPU verification result is:" << (pass ? "correct" : "fail") << std::endl;
    }

    return pass;
}
