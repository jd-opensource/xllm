// 这个文件用于注册所有 kernel 的 launcher
// 用户只需要使用 REG_KERNEL_LAUNCHER 宏即可

#ifndef KERNEL_LAUNCHERS_H
#define KERNEL_LAUNCHERS_H

#include "macros.h"

namespace TritonLauncher {
/* add_kernel */
#define ADD_KERNEL_ARG_LIST(OP) \
  OP(void*, arg0)               \
  OP(void*, arg1)               \
  OP(void*, arg2)               \
  OP(int32_t, arg3)

// param 1 must be same as "kernel_name" in ${kernel_name}.json
REG_KERNEL_ARGS(add_kernel, ADD_KERNEL_ARG_LIST)
REG_KERNEL_LAUNCHER(add_kernel, ADD_KERNEL_ARG_LIST)
// 调用REG_KERNEL_LAUNCHER后 会产生相应的kernel接口 与triton python接口一致
 // add_kernel(rtStream_t stream, \
        int32_t gridX, int32_t gridY, int32_t gridZ, \
        void* workspace_addr, \
        void* syncBlockLock, \
        void* arg0, void* arg1, void* arg2, int32_t arg3)
/* end of add_kernel */

/* fused_gdn_gating_kernel */
#define FUSED_GDN_GATING_ARG_LIST(OP) \
  OP(void*, g)                        \
  OP(void*, beta_output)              \
  OP(void*, A_log)                    \
  OP(void*, a)                        \
  OP(void*, b) \        
    OP(void*, dt_bias) \ 
    OP(int32_t, seq_len)

REG_KERNEL_ARGS(fused_gdn_gating_head8_kernel, FUSED_GDN_GATING_ARG_LIST)
REG_KERNEL_LAUNCHER(fused_gdn_gating_head8_kernel, FUSED_GDN_GATING_ARG_LIST)
}  // namespace TritonLauncher
/* end of fused_gdn_gating_kernel */
#endif  // KERNEL_LAUNCHERS_H
