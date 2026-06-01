#pragma once

#include "ck_tile/core/arch/amd_buffer_addressing.hpp"
#include "ck_tile/core/arch/hcu_matrix_addressing.hpp"
#include "ck_tile/core/numeric/integral_constant.hpp"
#include "ck_tile/core/container/multi_index.hpp"

namespace ck_tile {

template <typename MlsAtom, index_t Alt>
struct mls_traits;

} // namespace ck_tile
