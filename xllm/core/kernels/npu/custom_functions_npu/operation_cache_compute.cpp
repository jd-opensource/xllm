/* Copyright 2025 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "operation_cache_compute.h"

namespace atb {

thread_local char g_hash_buf[g_hash_buf_size];
thread_local int g_hash_offset = 0;
constexpr int g_rShift33Bits = 33;
constexpr uint64_t MIX_STEP1 = 18397679294719823053LLU;
constexpr uint64_t MIX_STEP2 = 14181476777654086739LLU;

void add_param_to_buf(const string& s) {
  MEMCPY_TO_BUF(s.c_str(), static_cast<int64_t>(s.size()));
}

void add_param_to_buf(const c10::optional<at::Tensor>& t) {}
void add_param_to_buf(const at::Tensor& t) {}

void add_param_to_buf() {}

inline uint64_t rotating_left(uint64_t x, uint8_t n) {
  return (x << n) | (x >> (64 - n));
}

inline uint64_t mixture(uint64_t x) {
  x ^= x >> g_rShift33Bits;
  x *= MIX_STEP1;
  x ^= x >> g_rShift33Bits;
  x *= MIX_STEP2;
  x ^= x >> g_rShift33Bits;

  return x;
}

uint64_t gen_hash(const void* key,
                  const int len,
                  const uint32_t seed = 0xdeadb0d7) {
  const uint8_t* data = static_cast<const uint8_t*>(key);
  const int block_num = len / 16;
  uint64_t has = seed;
  uint64_t hax = seed;

  const uint64_t c1 = 9782798678568883157LLU;
  const uint64_t c2 = 5545529020109919103LLU;

  const uint64_t* blocks =
      static_cast<const uint64_t*>(static_cast<const void*>(data));

  for (int i = 0; i < block_num; i++) {
    int even_num = 2;
    uint64_t tmp1 = blocks[i * even_num];
    uint64_t tmp2 = blocks[i * even_num + 1];

    int8_t bits_31 = 31;
    tmp1 *= c1;
    tmp1 = rotating_left(tmp1, bits_31);
    tmp1 *= c2;
    has ^= tmp1;

    int8_t bits_27 = 27;
    has = rotating_left(has, bits_27);
    has += hax;
    has = has * 5 + 1390208809;

    int8_t bits_33 = 33;
    tmp2 *= c2;
    tmp2 = rotating_left(tmp2, bits_33);
    tmp2 *= c1;
    hax ^= tmp2;

    hax = rotating_left(hax, bits_31);
    hax += has;
    hax = hax * 5 + 944331445;
  }

  const uint8_t* tail = data + block_num * 16;
  uint64_t t1 = 0;
  uint64_t t2 = 0;
  switch (static_cast<uint64_t>(len) & 15) {
    case 15:
      t2 ^= (static_cast<uint64_t>(tail[14])) << 48;
      [[fallthrough]];
      ;
    case 14:
      t2 ^= (static_cast<uint64_t>(tail[13])) << 40;
      [[fallthrough]];
      ;
    case 13:
      t2 ^= (static_cast<uint64_t>(tail[12])) << 32;
      [[fallthrough]];
      ;
    case 12:
      t2 ^= (static_cast<uint64_t>(tail[11])) << 24;
      [[fallthrough]];
      ;
    case 11:
      t2 ^= (static_cast<uint64_t>(tail[10])) << 16;
      [[fallthrough]];
      ;
    case 10:
      t2 ^= (static_cast<uint64_t>(tail[9])) << 8;
      [[fallthrough]];
      ;
    case 9:
      t2 ^= (static_cast<uint64_t>(tail[8])) << 0;
      t2 *= c2;
      t2 = rotating_left(t2, 33);
      t2 *= c1;
      hax ^= t2;
      [[fallthrough]];
      ;
    case 8:
      t1 ^= (static_cast<uint64_t>(tail[7])) << 56;
      [[fallthrough]];
      ;
    case 7:
      t1 ^= (static_cast<uint64_t>(tail[6])) << 48;
      [[fallthrough]];
      ;
    case 6:
      t1 ^= (static_cast<uint64_t>(tail[5])) << 40;
      [[fallthrough]];
      ;
    case 5:
      t1 ^= (static_cast<uint64_t>(tail[4])) << 32;
      [[fallthrough]];
      ;
    case 4:
      t1 ^= (static_cast<uint64_t>(tail[3])) << 24;
      [[fallthrough]];
      ;
    case 3:
      t1 ^= (static_cast<uint64_t>(tail[2])) << 16;
      [[fallthrough]];
      ;
    case 2:
      t1 ^= (static_cast<uint64_t>(tail[1])) << 8;
      [[fallthrough]];
      ;
    case 1:
      t1 ^= (static_cast<uint64_t>(tail[0])) << 0;
      t1 *= c1;
      t1 = rotating_left(t1, 31);
      t1 *= c2;
      has ^= t1;
      [[fallthrough]];
      ;
    default:
      break;
  };

  has ^= static_cast<uint64_t>(len);
  hax ^= static_cast<uint64_t>(len);

  has += hax;
  hax += has;

  has = mixture(has);
  hax = mixture(hax);

  has += hax;
  hax += has;
  return hax;
}

uint64_t calc_hash_id() {
  if (g_hash_offset == g_hash_buf_max_size) {
    return 0;
  }
  uint64_t hash_id = gen_hash(g_hash_buf, g_hash_offset);
  return hash_id;
}

}  // namespace atb
