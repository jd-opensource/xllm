#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wzero-as-null-pointer-constant"
#include "rapidjson/writer.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/document.h"
#include "rapidjson/rapidjson.h"
// #include <fstream>
#pragma GCC diagnostic pop

#define START_JSON_DUMP_FILE(file_name)                                             \
    std::string file_str(file_name);                                                \
    std::ofstream file(file_str);                                                   \
    if(!file.is_open())                                                             \
    {                                                                               \
        throw std::runtime_error("Could not open file: " + std::string(file_name)); \
    }                                                                               \
    rapidjson::StringBuffer s;                                                      \
    rapidjson::Writer<rapidjson::StringBuffer> writer(s);                           \
    writer.StartObject();

#define END_JSON_DUMP_FILE() \
    writer.EndObject();      \
    file << s.GetString();   \
    file.close();            \
    std::cout << "Results written to " << file_str << " successfully" << std::endl;

#define ADD_KEY_VALUE(key, value) add_key_value_pair(writer, key, value);
#define ADD_PERF_TO_JSON(_time, tflops, gbytes) add_perf_to_json(writer, _time, tflops, gbytes);

template <typename T>
void add_key_value_pair(rapidjson::Writer<rapidjson::StringBuffer>& writer,
                        const char* key,
                        T value)
{
    writer.Key(key);
    if constexpr(std::is_same<T, const char*>::value)
    {
        writer.String(value, static_cast<rapidjson::SizeType>(std::strlen(value)));
    }
    else if constexpr(std::is_same<T, std::string>::value)
    {
        writer.String(value.c_str(), static_cast<rapidjson::SizeType>(value.length()));
    }
    else if constexpr(std::is_floating_point<T>::value)
    {
        writer.Double(static_cast<double>(value));
    }
    else if constexpr(std::is_integral<T>::value)
    {
        writer.Int64(static_cast<int64_t>(value));
    }
    else
    {
        static_assert(std::is_same<T, const char*>::value || std::is_floating_point<T>::value ||
                          std::is_integral<T>::value,
                      "Unsupported type for JSON serialization");
    }
}

static void add_perf_to_json(rapidjson::Writer<rapidjson::StringBuffer>& writer,
                             float time,
                             float tflops,
                             float gbytes)
{
    std::string roster("perf");
    writer.String(roster.c_str(), static_cast<rapidjson::SizeType>(roster.length()));

    writer.StartArray();
    writer.StartObject();

    add_key_value_pair(writer, "time", time);
    add_key_value_pair(writer, "tflops", tflops);
    add_key_value_pair(writer, "gbytes", gbytes);

    writer.EndObject();
    writer.EndArray();
}

// Helper traits to check for static member existence
template <typename T, typename = void>
struct has_warp_tile_members : std::false_type
{
};

template <typename T>
struct has_warp_tile_members<
    T,
    std::void_t<decltype(T::M_Warp_Tile), decltype(T::N_Warp_Tile), decltype(T::K_Warp_Tile)>>
    : std::true_type
{
};

template <typename ALayout,
          typename BLayout,
          typename CLayout,
          typename ADataType,
          typename BDataType,
          typename CDataType,
          typename GemmConfig,
          template <typename>
          typename DTypeTraits>
inline void dump_gemm_json_results(const std::string& json_filename,
                            int M,
                            int N,
                            int K,
                            int stride_A,
                            int stride_B,
                            int stride_C,
                            bool persistent,
                            bool pass,
                            float ave_time,
                            float tflops,
                            float gb_per_sec,
                            const std::string& kernel_name = "gemm_basic")
{
    START_JSON_DUMP_FILE(json_filename);
    ADD_KEY_VALUE("name", kernel_name);
    ADD_KEY_VALUE("M", M);
    ADD_KEY_VALUE("N", N);
    ADD_KEY_VALUE("K", K);
    ADD_KEY_VALUE("stride_A", stride_A);
    ADD_KEY_VALUE("stride_B", stride_B);
    ADD_KEY_VALUE("stride_C", stride_C);
    ADD_KEY_VALUE("A_layout", ALayout::name);
    ADD_KEY_VALUE("B_layout", BLayout::name);
    ADD_KEY_VALUE("C_layout", CLayout::name);
    using TraitsADataType = DTypeTraits<ADataType>;
    using TraitsBDataType = DTypeTraits<BDataType>;
    using TraitsCDataType = DTypeTraits<CDataType>;
    ADD_KEY_VALUE("A_type", TraitsADataType::name);
    ADD_KEY_VALUE("B_type", TraitsBDataType::name);
    ADD_KEY_VALUE("C_type", TraitsCDataType::name);
    ADD_KEY_VALUE("structured_sparsity", GemmConfig::UseStructuredSparsity ? "on" : "off");

    if constexpr(has_warp_tile_members<GemmConfig>::value)
    {
        ADD_KEY_VALUE("warp_tile",
                      std::to_string(GemmConfig::M_Warp_Tile) + "x" +
                          std::to_string(GemmConfig::N_Warp_Tile) + "x" +
                          std::to_string(GemmConfig::K_Warp_Tile));
    }
    ADD_KEY_VALUE("persistent", persistent ? "on" : "off");
    ADD_KEY_VALUE("verification", pass ? "pass" : "fail");
    ADD_PERF_TO_JSON(ave_time, tflops, gb_per_sec);
    END_JSON_DUMP_FILE();
}

inline void dump_batched_gemm_json_results(const std::string& json_filename,
                                    const std::string& op_name,
                                    int M,
                                    int N,
                                    int K,
                                    int stride_A,
                                    int stride_B,
                                    int stride_C,
                                    int batch_stride_A,
                                    int batch_stride_B,
                                    int batch_stride_C,
                                    int batch_count,
                                    bool pass,
                                    float ave_time,
                                    float tflops,
                                    float gb_per_sec,
                                    const std::string& kernel_name = "batched_gemm_basic")
{
    START_JSON_DUMP_FILE(json_filename);
    ADD_KEY_VALUE("name", kernel_name);
    ADD_KEY_VALUE("op_name", op_name);
    ADD_KEY_VALUE("M", M);
    ADD_KEY_VALUE("N", N);
    ADD_KEY_VALUE("K", K);
    ADD_KEY_VALUE("stride_A", stride_A);
    ADD_KEY_VALUE("stride_B", stride_B);
    ADD_KEY_VALUE("stride_C", stride_C);
    ADD_KEY_VALUE("batch_stride_A", batch_stride_A);
    ADD_KEY_VALUE("batch_stride_B", batch_stride_B);
    ADD_KEY_VALUE("batch_stride_C", batch_stride_C);
    ADD_KEY_VALUE("batch_count", batch_count);
    ADD_KEY_VALUE("verification", pass ? "pass" : "fail");
    ADD_PERF_TO_JSON(ave_time, tflops, gb_per_sec)
    END_JSON_DUMP_FILE();
}

template <typename ALayout, typename BLayout, typename CLayout>
inline void dump_grouped_gemm_json_results(const std::string& json_filename,
                                    const std::string& op_name,
                                    int group_count,
                                    bool pass,
                                    float ave_time,
                                    float tflops,
                                    float gb_per_sec,
                                    const std::string& kernel_name = "grouped_gemm")
{
    START_JSON_DUMP_FILE(json_filename);
    ADD_KEY_VALUE("name", kernel_name);
    ADD_KEY_VALUE("op_name", op_name);
    ADD_KEY_VALUE("group_count", group_count);
    ADD_KEY_VALUE("A_layout", ALayout::name);
    ADD_KEY_VALUE("B_layout", BLayout::name);
    ADD_KEY_VALUE("C_layout", CLayout::name);
    ADD_KEY_VALUE("verification", pass ? "pass" : "fail");
    ADD_PERF_TO_JSON(ave_time, tflops, gb_per_sec)
    END_JSON_DUMP_FILE();
}

inline void dump_flatmm_json_results(const std::string& json_filename,
                              const std::string& datatype,
                              int M,
                              int N,
                              int K,
                              int stride_A,
                              int stride_B,
                              int stride_C,
                              int kbatch,
                              bool pass,
                              float ave_time,
                              float tflops,
                              float gb_per_sec,
                              const std::string& kernel_name = "flatmm_basic")
{
    START_JSON_DUMP_FILE(json_filename);
    ADD_KEY_VALUE("name", kernel_name);
    ADD_KEY_VALUE("DataType", datatype);
    ADD_KEY_VALUE("M", M);
    ADD_KEY_VALUE("N", N);
    ADD_KEY_VALUE("K", K);
    ADD_KEY_VALUE("StrideA", stride_A);
    ADD_KEY_VALUE("StrideB", stride_B);
    ADD_KEY_VALUE("StrideC", stride_C);
    ADD_KEY_VALUE("kbatch", kbatch);
    ADD_KEY_VALUE("verification", pass ? "pass" : "fail");
    ADD_PERF_TO_JSON(ave_time, tflops, gb_per_sec)
    END_JSON_DUMP_FILE();
}

inline void dump_gemm_multi_d_fp16_json_results(const std::string& json_filename,
                                         const std::string& op_name,
                                         int M,
                                         int N,
                                         int K,
                                         int StrideA,
                                         int StrideB,
                                         int StrideD0,
                                         int StrideD1,
                                         int StrideE,
                                         bool pass,
                                         float ave_time,
                                         float tflops,
                                         float gb_per_sec,
                                         const std::string& kernel_name = "gemm_multi_d_fp16")
{
    START_JSON_DUMP_FILE(json_filename);
    ADD_KEY_VALUE("name", kernel_name);
    ADD_KEY_VALUE("op_name", op_name);
    ADD_KEY_VALUE("M", M);
    ADD_KEY_VALUE("N", N);
    ADD_KEY_VALUE("K", K);
    ADD_KEY_VALUE("StrideA", StrideA);
    ADD_KEY_VALUE("StrideB", StrideB);
    ADD_KEY_VALUE("StrideD0", StrideD0);
    ADD_KEY_VALUE("StrideD1", StrideD1);
    ADD_KEY_VALUE("StrideE", StrideE);
    ADD_KEY_VALUE("verification", pass ? "pass" : "fail");
    ADD_PERF_TO_JSON(ave_time, tflops, gb_per_sec)
    END_JSON_DUMP_FILE();
}

inline void dump_elementwise_json_results(const std::string& json_filename,
                                   const std::string& prec,
                                   int grid_size,
                                   int block_size,
                                   float ave_time,
                                   float tflops,
                                   float gb_per_sec,
                                   const std::string& kernel_name = "elementwise")
{
    START_JSON_DUMP_FILE(json_filename);
    ADD_KEY_VALUE("name", kernel_name);
    ADD_KEY_VALUE("prec", prec);
    ADD_KEY_VALUE("grid_size", grid_size);
    ADD_KEY_VALUE("block_size", block_size);
    ADD_PERF_TO_JSON(ave_time, tflops, gb_per_sec)
    END_JSON_DUMP_FILE();
}

inline void dump_layernorm2d_fwd_json_results(const std::string& json_filename,
                                       const std::string& prec_i,
                                       const std::string& prec_o,
                                       const std::string& prec_sm,
                                       const std::string& prec_sy,
                                       int m,
                                       int n,
                                       int x_stride,
                                       int xr_stride,
                                       int y_stride,
                                       int yr_stride,
                                       bool pass,
                                       float ave_time,
                                       float tflops,
                                       float gb_per_sec,
                                       const std::string& kernel_name = "layernorm2d_fwd")
{
    START_JSON_DUMP_FILE(json_filename);
    ADD_KEY_VALUE("name", kernel_name);
    ADD_KEY_VALUE("prec_i", prec_i);
    ADD_KEY_VALUE("prec_o", prec_o);
    ADD_KEY_VALUE("prec_sm", prec_sm);
    ADD_KEY_VALUE("prec_sy", prec_sy);
    ADD_KEY_VALUE("m", m);
    ADD_KEY_VALUE("n", n);
    ADD_KEY_VALUE("x_stride", x_stride);
    ADD_KEY_VALUE("xr_stride", xr_stride);
    ADD_KEY_VALUE("y_stride", y_stride);
    ADD_KEY_VALUE("yr_stride", yr_stride);
    ADD_KEY_VALUE("verification", pass ? "pass" : "fail");
    ADD_PERF_TO_JSON(ave_time, tflops, gb_per_sec)
    END_JSON_DUMP_FILE();
}

template <typename DataType, template <typename> typename DTypeTraits>
inline void dump_reduce_json_results(const std::string& json_filename,
                              int N,
                              int C,
                              int H,
                              int W,
                              bool pass,
                              float ave_time,
                              float tflops,
                              float gb_per_sec,
                              const std::string& kernel_name = "reduce")
{
    START_JSON_DUMP_FILE(json_filename);
    ADD_KEY_VALUE("name", kernel_name);
    using Traits = DTypeTraits<DataType>;
    ADD_KEY_VALUE("data_type", Traits::name);
    ADD_KEY_VALUE("N", N);
    ADD_KEY_VALUE("C", C);
    ADD_KEY_VALUE("H", H);
    ADD_KEY_VALUE("W", W);
    ADD_KEY_VALUE("verification", pass ? "pass" : "fail");
    ADD_PERF_TO_JSON(ave_time, tflops, gb_per_sec)
    END_JSON_DUMP_FILE();
}

inline void dump_permute_json_results(const std::string& json_filename,
                               const std::string& data_type,
                               bool pass,
                               float ave_time,
                               float tflop,
                               float gb_per_sec,
                               const std::string& kernel_name = "permute")
{
    START_JSON_DUMP_FILE(json_filename);
    ADD_KEY_VALUE("name", kernel_name);
    ADD_KEY_VALUE("data_type", data_type);
    ADD_KEY_VALUE("verification", pass ? "pass" : "fail");
    ADD_PERF_TO_JSON(ave_time, tflop, gb_per_sec)
    END_JSON_DUMP_FILE();
}

inline void dump_topk_softmax_json(const std::string& json_filename,
                            const std::string& input_prec,
                            const std::string& weight_prec,
                            int tokens,
                            int experts,
                            int topk,
                            int stride_input,
                            int stride_output,
                            float ave_time,
                            float tflop,
                            float gb_per_sec,
                            bool pass,
                            const std::string& kernel_name = "topk_softmax")
{
    START_JSON_DUMP_FILE(json_filename);
    ADD_KEY_VALUE("name", kernel_name);
    ADD_KEY_VALUE("input_prec", input_prec);
    ADD_KEY_VALUE("weight_prec", weight_prec);
    ADD_KEY_VALUE("tokens", tokens);
    ADD_KEY_VALUE("experts", experts);
    ADD_KEY_VALUE("topk", topk);
    ADD_KEY_VALUE("stride_input", stride_input);
    ADD_KEY_VALUE("stride_output", stride_output);
    ADD_KEY_VALUE("verification", pass ? "pass" : "fail");
    ADD_PERF_TO_JSON(ave_time, tflop, gb_per_sec);
    END_JSON_DUMP_FILE();
}

inline void dump_rmsnorm2d_fwd_json(const std::string& json_filename,
                             const std::string& prec_str,
                             int m,
                             int n,
                             int x_stride,
                             int xr_stride,
                             int y_stride,
                             int yr_stride,
                             int use_model_sensitive_rmsnorm,
                             float ave_time,
                             float tflops,
                             float gb_per_sec,
                             bool pass,
                             const std::string& kernel_name = "rmsnorm2d_fwd")
{
    START_JSON_DUMP_FILE(json_filename);
    ADD_KEY_VALUE("name", kernel_name);
    ADD_KEY_VALUE("prec", prec_str);
    ADD_KEY_VALUE("m", m);
    ADD_KEY_VALUE("n", n);
    ADD_KEY_VALUE("x_stride", x_stride);
    ADD_KEY_VALUE("xr_stride", xr_stride);
    ADD_KEY_VALUE("y_stride", y_stride);
    ADD_KEY_VALUE("yr_stride", yr_stride);
    ADD_KEY_VALUE("use_model_sensitive_rmsnorm", use_model_sensitive_rmsnorm);
    ADD_KEY_VALUE("verification", pass ? "pass" : "fail");
    ADD_PERF_TO_JSON(ave_time, tflops, gb_per_sec);
    END_JSON_DUMP_FILE();
}

inline void dump_add_rmsnorm2d_rdquant_fwd_json(
    const std::string& json_filename,
    const std::string& input_data_type,
    const std::string& quantized_data_type,
    int m,
    int n,
    int stride,
    float epsilon,
    float ave_time,
    float tflops,
    float gb_per_sec,
    bool pass,
    const std::string& kernel_name = "add_rmsnorm2d_rdquant_fwd")
{
    START_JSON_DUMP_FILE(json_filename);
    ADD_KEY_VALUE("name", kernel_name);
    ADD_KEY_VALUE("input_data_type", input_data_type);
    ADD_KEY_VALUE("quantized_data_type", quantized_data_type);
    ADD_KEY_VALUE("m", m);
    ADD_KEY_VALUE("n", n);
    ADD_KEY_VALUE("stride", stride);
    ADD_KEY_VALUE("epsilon", epsilon);
    ADD_KEY_VALUE("verification", pass ? "pass" : "fail");
    ADD_PERF_TO_JSON(ave_time, tflops, gb_per_sec);
    END_JSON_DUMP_FILE();
}

inline void dump_smoothquant_json(const std::string& json_filename,
                           const std::string& prec_str,
                           int m,
                           int n,
                           int x_stride,
                           int y_stride,
                           float ave_time,
                           float tflops,
                           float gb_per_sec,
                           bool pass,
                           const std::string& kernel_name = "smoothquant")
{
    START_JSON_DUMP_FILE(json_filename);
    ADD_KEY_VALUE("name", kernel_name);
    ADD_KEY_VALUE("prec", prec_str);
    ADD_KEY_VALUE("m", m);
    ADD_KEY_VALUE("n", n);
    ADD_KEY_VALUE("x_stride", x_stride);
    ADD_KEY_VALUE("y_stride", y_stride);
    ADD_KEY_VALUE("verification", pass ? "pass" : "fail");
    ADD_PERF_TO_JSON(ave_time, tflops, gb_per_sec);
    END_JSON_DUMP_FILE();
}

inline void dump_moe_sorting_json(const std::string& json_filename,
                           const std::string& index_prec,
                           const std::string& weight_prec,
                           const std::string& workspace_size,
                           int dispatch_policy,
                           int tokens,
                           int num_experts,
                           int topk,
                           float ave_time,
                           float tflops,
                           float gb_per_sec,
                           bool pass,
                           const std::string& kernel_name = "moe_sorting")
{
    START_JSON_DUMP_FILE(json_filename);
    ADD_KEY_VALUE("name", kernel_name);
    ADD_KEY_VALUE("index_prec", index_prec);
    ADD_KEY_VALUE("weight_prec", weight_prec);
    ADD_KEY_VALUE("workspace_size", workspace_size);
    ADD_KEY_VALUE("dispatch_policy", dispatch_policy);
    ADD_KEY_VALUE("tokens", tokens);
    ADD_KEY_VALUE("num_experts", num_experts);
    ADD_KEY_VALUE("topk", topk);
    ADD_KEY_VALUE("verification", pass ? "pass" : "fail");
    ADD_PERF_TO_JSON(ave_time, tflops, gb_per_sec)
    END_JSON_DUMP_FILE();
}

inline void dump_batched_transpose_json(const std::string& json_filename,
                                 int N,
                                 int C,
                                 int H,
                                 int W,
                                 const std::string& layout_in,
                                 const std::string& layout_out,
                                 const std::string& prec,
                                 float ave_time,
                                 float tflops,
                                 float gb_per_sec,
                                 bool pass,
                                 const std::string& kernel_name = "batched_transpose")
{
    START_JSON_DUMP_FILE(json_filename);
    ADD_KEY_VALUE("name", kernel_name);
    ADD_KEY_VALUE("N", N);
    ADD_KEY_VALUE("C", C);
    ADD_KEY_VALUE("H", H);
    ADD_KEY_VALUE("W", W);
    ADD_KEY_VALUE("LayoutIn", layout_in);
    ADD_KEY_VALUE("LayoutOut", layout_out);
    ADD_KEY_VALUE("Precision", prec);
    ADD_KEY_VALUE("verification", pass ? "pass" : "fail");
    ADD_PERF_TO_JSON(ave_time, tflops, gb_per_sec)
    END_JSON_DUMP_FILE();
}

inline void dump_moe_smoothquant_json(const std::string& json_filename,
                               const std::string& prec_i,
                               const std::string& prec_o,
                               int tokens,
                               int hidden_size,
                               int stride,
                               int experts,
                               int topk,
                               bool pass,
                               float ave_time,
                               float tflops,
                               float gb_per_sec,
                               const std::string& kernel_name = "moe_smoothquant")
{
    START_JSON_DUMP_FILE(json_filename);
    ADD_KEY_VALUE("name", kernel_name);
    ADD_KEY_VALUE("prec_i", prec_i);
    ADD_KEY_VALUE("prec_o", prec_o);
    ADD_KEY_VALUE("tokens", tokens);
    ADD_KEY_VALUE("hidden_size", hidden_size);
    ADD_KEY_VALUE("stride", stride);
    ADD_KEY_VALUE("experts", experts);
    ADD_KEY_VALUE("topk", topk);
    ADD_KEY_VALUE("verification", pass ? "pass" : "fail");
    ADD_PERF_TO_JSON(ave_time, tflops, gb_per_sec)
    END_JSON_DUMP_FILE();
}

inline void dump_fused_moe_json(const std::string& json_filename,
                         const std::string& api_str,
                         const std::string& prec_str,
                         int tokens,
                         bool is_local_token,
                         int local_tokens,
                         int experts,
                         int topk,
                         int hidden_size,
                         int intermediate_size,
                         int stride,
                         int block_m,
                         int activation,
                         bool gate_only,
                         bool fused_quant,
                         bool pass,
                         float ave_time,
                         float tflops,
                         float tb_per_sec,
                         const std::string& kernel_name = "fused_moe")
{
    START_JSON_DUMP_FILE(json_filename);
    ADD_KEY_VALUE("name", kernel_name);
    ADD_KEY_VALUE("api", api_str);
    ADD_KEY_VALUE("prec", prec_str);
    ADD_KEY_VALUE("tokens", tokens);
    if(is_local_token)
    {
        ADD_KEY_VALUE("local_tokens", local_tokens);
    }
    ADD_KEY_VALUE("experts", experts);
    ADD_KEY_VALUE("topk", topk);
    ADD_KEY_VALUE("hidden_size", hidden_size);
    ADD_KEY_VALUE("intermediate_size", intermediate_size);
    ADD_KEY_VALUE("stride", stride);
    ADD_KEY_VALUE("block_m", block_m);
    ADD_KEY_VALUE("activation", activation);
    ADD_KEY_VALUE("gate_only", gate_only);
    ADD_KEY_VALUE("fused_quant", fused_quant);
    ADD_KEY_VALUE("verification", pass ? "pass" : "fail");
    ADD_PERF_TO_JSON(ave_time, tflops, (tb_per_sec * 1024.0f))
    END_JSON_DUMP_FILE();
}

inline void dump_fmha_fwd_json_results(const std::string& json_filename,
                                const std::string& prec,
                                const std::string& mode,
                                const std::string& io_layout,
                                int batch,
                                int nhead,
                                int nhead_k,
                                int seqlen_qs,
                                int seqlen_ks,
                                int seqlen_kpads,
                                int hdim_q,
                                int hdim_v,
                                float scale_s,
                                float p_drop,
                                bool lse,
                                bool squant,
                                const std::string& bias,
                                const std::string& vlayout,
                                bool pass,
                                float ave_time,
                                float tflops,
                                float gb_per_sec,
                                const std::string& kernel_name = "fmha_fwd")
{
    START_JSON_DUMP_FILE(json_filename);
    ADD_KEY_VALUE("name", kernel_name);
    ADD_KEY_VALUE("prec", prec);
    ADD_KEY_VALUE("mode", mode);
    ADD_KEY_VALUE("io_layout", io_layout);
    ADD_KEY_VALUE("batch", batch);
    ADD_KEY_VALUE("nhead", nhead);
    ADD_KEY_VALUE("nhead_k", nhead_k);
    ADD_KEY_VALUE("seqlen_q", seqlen_qs);
    ADD_KEY_VALUE("seqlen_k", seqlen_ks);
    ADD_KEY_VALUE("seqlen_kpads", seqlen_kpads);
    ADD_KEY_VALUE("hdim_q", hdim_q);
    ADD_KEY_VALUE("hdim_v", hdim_v);
    ADD_KEY_VALUE("scale_s", scale_s);
    ADD_KEY_VALUE("p_drop", p_drop);
    ADD_KEY_VALUE("lse", lse);
    ADD_KEY_VALUE("squant", squant);
    ADD_KEY_VALUE("bias", bias);
    ADD_KEY_VALUE("vlayout", vlayout);
    ADD_KEY_VALUE("verification", pass ? "pass" : "fail");
    ADD_PERF_TO_JSON(ave_time, tflops, gb_per_sec)
    END_JSON_DUMP_FILE();
}

inline void dump_fmha_bwd_json_results(const std::string& json_filename,
                                const std::string& data_type,
                                const std::string& mode,
                                const std::string& i_perm,
                                const std::string& o_perm,
                                int batch,
                                int nhead,
                                int nhead_k,
                                int seqlen_q,
                                int seqlen_k,
                                int hdim_q,
                                int hdim_v,
                                float scale,
                                const std::string& bias,
                                bool use_dbias,
                                float p_drop,
                                bool s_randval,
                                bool deterministic,
                                const std::string& mask,
                                int mask_left,
                                int mask_right,
                                int workspace_size,
                                bool pass,
                                float ave_time,
                                float tflops,
                                float gb_per_sec,
                                const std::string& kernel_name = "fmha_bwd")
{
    START_JSON_DUMP_FILE(json_filename);
    ADD_KEY_VALUE("name", kernel_name);
    ADD_KEY_VALUE("prec", data_type);
    ADD_KEY_VALUE("mode", mode);
    ADD_KEY_VALUE("i_perm", i_perm);
    ADD_KEY_VALUE("o_perm", o_perm);
    ADD_KEY_VALUE("batch", batch);
    ADD_KEY_VALUE("nhead", nhead);
    ADD_KEY_VALUE("nhead_k", nhead_k);
    ADD_KEY_VALUE("seqlen_q", seqlen_q);
    ADD_KEY_VALUE("seqlen_k", seqlen_k);
    ADD_KEY_VALUE("hdim_q", hdim_q);
    ADD_KEY_VALUE("hdim_v", hdim_v);
    ADD_KEY_VALUE("scale", scale);
    ADD_KEY_VALUE("bias", bias);
    ADD_KEY_VALUE("use_dbias", use_dbias);
    ADD_KEY_VALUE("p_drop", p_drop);
    ADD_KEY_VALUE("s_randval", s_randval);
    ADD_KEY_VALUE("deterministic", deterministic ? "true" : "false");
    ADD_KEY_VALUE("mask", mask);
    ADD_KEY_VALUE("mask_left", mask_left);
    ADD_KEY_VALUE("mask_right", mask_right);
    ADD_KEY_VALUE("workspace_size", workspace_size);
    ADD_KEY_VALUE("verification", pass ? "pass" : "fail");
    ADD_PERF_TO_JSON(ave_time, tflops, gb_per_sec)
    END_JSON_DUMP_FILE();
}
