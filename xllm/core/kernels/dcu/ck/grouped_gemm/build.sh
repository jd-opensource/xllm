

echo "build ./instances/grouped_gemm_fp16.cpp"
/opt/dtk/bin/aicc \
-DCK_EXPERIMENTAL_BIT_INT_EXTENSION \
-D__HIP_PLATFORM_AMD__=1 -D__HIP_PLATFORM_HCC__=1 -D__HIP_ROCclr__=1 \
-I../include \
-isystem /opt/dtk/llvm/lib/clang/17.0.0/include/.. \
-O3 -DNDEBUG -std=c++17   \
-Wall -Wextra -Wcomment -Wendif-labels -Wformat -Winit-self -Wreturn-type -Wsequence-point -Wswitch \
-Wtrigraphs -Wundef -Wuninitialized -Wunreachable-code -Wunused -Wno-reserved-identifier -Werror \
-Wsign-compare -Wno-extra-semi-stmt -Wno-missing-field-initializers -Wno-deprecated-declarations \
-Wno-c++98-compat -Wno-c++98-compat-pedantic -Wno-conversion -Wno-double-promotion -Wno-exit-time-destructors \
-Wno-extra-semi -Wno-float-conversion -Wno-gnu-anonymous-struct -Wno-gnu-zero-variadic-macro-arguments \
-Wno-missing-prototypes -Wno-nested-anon-types -Wno-padded -Wno-return-std-move-in-c++11 -Wno-shorten-64-to-32 \
-Wno-sign-conversion -Wno-unknown-warning-option -Wno-unused-command-line-argument -Wno-weak-vtables \
-Wno-covered-switch-default -Wno-unsafe-buffer-usage -Wno-bit-int-extension -Weverything \
-mllvm -support-768-vgprs=true \
-xhip --offload-arch=gfx936 -MD -MT -MF \
-o ./instances/grouped_gemm_fp16.cpp.o \
-c ./instances/grouped_gemm_fp16.cpp

echo "build ./instances/grouped_gemm_fp8.cpp"
/opt/dtk/bin/aicc \
-DCK_EXPERIMENTAL_BIT_INT_EXTENSION \
-D__HIP_PLATFORM_AMD__=1 -D__HIP_PLATFORM_HCC__=1 -D__HIP_ROCclr__=1 \
-I../include \
-isystem /opt/dtk/llvm/lib/clang/17.0.0/include/.. \
-O3 -DNDEBUG -std=c++17   \
-Wall -Wextra -Wcomment -Wendif-labels -Wformat -Winit-self -Wreturn-type -Wsequence-point -Wswitch \
-Wtrigraphs -Wundef -Wuninitialized -Wunreachable-code -Wunused -Wno-reserved-identifier -Werror \
-Wsign-compare -Wno-extra-semi-stmt -Wno-missing-field-initializers -Wno-deprecated-declarations \
-Wno-c++98-compat -Wno-c++98-compat-pedantic -Wno-conversion -Wno-double-promotion -Wno-exit-time-destructors \
-Wno-extra-semi -Wno-float-conversion -Wno-gnu-anonymous-struct -Wno-gnu-zero-variadic-macro-arguments \
-Wno-missing-prototypes -Wno-nested-anon-types -Wno-padded -Wno-return-std-move-in-c++11 -Wno-shorten-64-to-32 \
-Wno-sign-conversion -Wno-unknown-warning-option -Wno-unused-command-line-argument -Wno-weak-vtables \
-Wno-covered-switch-default -Wno-unsafe-buffer-usage -Wno-bit-int-extension -Weverything \
-mllvm -support-768-vgprs=true \
-xhip --offload-arch=gfx936 -MD -MT -MF \
-o ./instances/grouped_gemm_fp8.cpp.o \
-c ./instances/grouped_gemm_fp8.cpp

echo "build ./instances/grouped_gemm_bf16.cpp"
/opt/dtk/bin/aicc \
-DCK_EXPERIMENTAL_BIT_INT_EXTENSION \
-D__HIP_PLATFORM_AMD__=1 -D__HIP_PLATFORM_HCC__=1 -D__HIP_ROCclr__=1 \
-I../include \
-isystem /opt/dtk/llvm/lib/clang/17.0.0/include/.. \
-O3 -DNDEBUG -std=c++17   \
-Wall -Wextra -Wcomment -Wendif-labels -Wformat -Winit-self -Wreturn-type -Wsequence-point -Wswitch \
-Wtrigraphs -Wundef -Wuninitialized -Wunreachable-code -Wunused -Wno-reserved-identifier -Werror \
-Wsign-compare -Wno-extra-semi-stmt -Wno-missing-field-initializers -Wno-deprecated-declarations \
-Wno-c++98-compat -Wno-c++98-compat-pedantic -Wno-conversion -Wno-double-promotion -Wno-exit-time-destructors \
-Wno-extra-semi -Wno-float-conversion -Wno-gnu-anonymous-struct -Wno-gnu-zero-variadic-macro-arguments \
-Wno-missing-prototypes -Wno-nested-anon-types -Wno-padded -Wno-return-std-move-in-c++11 -Wno-shorten-64-to-32 \
-Wno-sign-conversion -Wno-unknown-warning-option -Wno-unused-command-line-argument -Wno-weak-vtables \
-Wno-covered-switch-default -Wno-unsafe-buffer-usage -Wno-bit-int-extension -Weverything \
-mllvm -support-768-vgprs=true \
-xhip --offload-arch=gfx936 -MD -MT -MF \
-o ./instances/grouped_gemm_bf16.cpp.o \
-c ./instances/grouped_gemm_bf16.cpp

echo "build ./instances/grouped_gemm_bf8.cpp"
/opt/dtk/bin/aicc \
-DCK_EXPERIMENTAL_BIT_INT_EXTENSION \
-D__HIP_PLATFORM_AMD__=1 -D__HIP_PLATFORM_HCC__=1 -D__HIP_ROCclr__=1 \
-I../include \
-isystem /opt/dtk/llvm/lib/clang/17.0.0/include/.. \
-O3 -DNDEBUG -std=c++17   \
-Wall -Wextra -Wcomment -Wendif-labels -Wformat -Winit-self -Wreturn-type -Wsequence-point -Wswitch \
-Wtrigraphs -Wundef -Wuninitialized -Wunreachable-code -Wunused -Wno-reserved-identifier -Werror \
-Wsign-compare -Wno-extra-semi-stmt -Wno-missing-field-initializers -Wno-deprecated-declarations \
-Wno-c++98-compat -Wno-c++98-compat-pedantic -Wno-conversion -Wno-double-promotion -Wno-exit-time-destructors \
-Wno-extra-semi -Wno-float-conversion -Wno-gnu-anonymous-struct -Wno-gnu-zero-variadic-macro-arguments \
-Wno-missing-prototypes -Wno-nested-anon-types -Wno-padded -Wno-return-std-move-in-c++11 -Wno-shorten-64-to-32 \
-Wno-sign-conversion -Wno-unknown-warning-option -Wno-unused-command-line-argument -Wno-weak-vtables \
-Wno-covered-switch-default -Wno-unsafe-buffer-usage -Wno-bit-int-extension -Weverything \
-mllvm -support-768-vgprs=true \
-xhip --offload-arch=gfx936 -MD -MT -MF \
-o ./instances/grouped_gemm_bf8.cpp.o \
-c ./instances/grouped_gemm_bf8.cpp

echo "build ./instances/grouped_gemm_int8.cpp"
/opt/dtk/bin/aicc \
-DCK_EXPERIMENTAL_BIT_INT_EXTENSION \
-D__HIP_PLATFORM_AMD__=1 -D__HIP_PLATFORM_HCC__=1 -D__HIP_ROCclr__=1 \
-I../include \
-isystem /opt/dtk/llvm/lib/clang/17.0.0/include/.. \
-O3 -DNDEBUG -std=c++17   \
-Wall -Wextra -Wcomment -Wendif-labels -Wformat -Winit-self -Wreturn-type -Wsequence-point -Wswitch \
-Wtrigraphs -Wundef -Wuninitialized -Wunreachable-code -Wunused -Wno-reserved-identifier -Werror \
-Wsign-compare -Wno-extra-semi-stmt -Wno-missing-field-initializers -Wno-deprecated-declarations \
-Wno-c++98-compat -Wno-c++98-compat-pedantic -Wno-conversion -Wno-double-promotion -Wno-exit-time-destructors \
-Wno-extra-semi -Wno-float-conversion -Wno-gnu-anonymous-struct -Wno-gnu-zero-variadic-macro-arguments \
-Wno-missing-prototypes -Wno-nested-anon-types -Wno-padded -Wno-return-std-move-in-c++11 -Wno-shorten-64-to-32 \
-Wno-sign-conversion -Wno-unknown-warning-option -Wno-unused-command-line-argument -Wno-weak-vtables \
-Wno-covered-switch-default -Wno-unsafe-buffer-usage -Wno-bit-int-extension -Weverything \
-mllvm -support-768-vgprs=true \
-xhip --offload-arch=gfx936 -MD -MT -MF \
-o ./instances/grouped_gemm_int8.cpp.o \
-c ./instances/grouped_gemm_int8.cpp

echo "build ./instances/grouped_gemm_int4.cpp"
/opt/dtk/bin/aicc \
-DCK_EXPERIMENTAL_BIT_INT_EXTENSION \
-D__HIP_PLATFORM_AMD__=1 -D__HIP_PLATFORM_HCC__=1 -D__HIP_ROCclr__=1 \
-I../include \
-isystem /opt/dtk/llvm/lib/clang/17.0.0/include/.. \
-O3 -DNDEBUG -std=c++17   \
-Wall -Wextra -Wcomment -Wendif-labels -Wformat -Winit-self -Wreturn-type -Wsequence-point -Wswitch \
-Wtrigraphs -Wundef -Wuninitialized -Wunreachable-code -Wunused -Wno-reserved-identifier -Werror \
-Wsign-compare -Wno-extra-semi-stmt -Wno-missing-field-initializers -Wno-deprecated-declarations \
-Wno-c++98-compat -Wno-c++98-compat-pedantic -Wno-conversion -Wno-double-promotion -Wno-exit-time-destructors \
-Wno-extra-semi -Wno-float-conversion -Wno-gnu-anonymous-struct -Wno-gnu-zero-variadic-macro-arguments \
-Wno-missing-prototypes -Wno-nested-anon-types -Wno-padded -Wno-return-std-move-in-c++11 -Wno-shorten-64-to-32 \
-Wno-sign-conversion -Wno-unknown-warning-option -Wno-unused-command-line-argument -Wno-weak-vtables \
-Wno-covered-switch-default -Wno-unsafe-buffer-usage -Wno-bit-int-extension -Weverything \
-mllvm -support-768-vgprs=true \
-xhip --offload-arch=gfx936 -MD -MT -MF \
-o ./instances/grouped_gemm_int4.cpp.o \
-c ./instances/grouped_gemm_int4.cpp

echo "build ./grouped_gemm.cpp"
/opt/dtk/bin/aicc \
-DCK_EXPERIMENTAL_BIT_INT_EXTENSION \
-D__HIP_PLATFORM_AMD__=1 -D__HIP_PLATFORM_HCC__=1 -D__HIP_ROCclr__=1 \
-I../include \
-isystem /opt/dtk/llvm/lib/clang/17.0.0/include/.. \
-O3 -DNDEBUG -std=c++17   \
-Wall -Wextra -Wcomment -Wendif-labels -Wformat -Winit-self -Wreturn-type -Wsequence-point -Wswitch \
-Wtrigraphs -Wundef -Wuninitialized -Wunreachable-code -Wunused -Wno-reserved-identifier -Werror \
-Wsign-compare -Wno-extra-semi-stmt -Wno-missing-field-initializers -Wno-deprecated-declarations \
-Wno-c++98-compat -Wno-c++98-compat-pedantic -Wno-conversion -Wno-double-promotion -Wno-exit-time-destructors \
-Wno-extra-semi -Wno-float-conversion -Wno-gnu-anonymous-struct -Wno-gnu-zero-variadic-macro-arguments \
-Wno-missing-prototypes -Wno-nested-anon-types -Wno-padded -Wno-return-std-move-in-c++11 -Wno-shorten-64-to-32 \
-Wno-sign-conversion -Wno-unknown-warning-option -Wno-unused-command-line-argument -Wno-weak-vtables \
-Wno-covered-switch-default -Wno-unsafe-buffer-usage -Wno-bit-int-extension -Weverything \
-mllvm -support-768-vgprs=true \
-xhip --offload-arch=gfx936 -MD -MT -MF \
-o ./grouped_gemm.cpp.o \
-c ./grouped_gemm.cpp

echo "build ./tile_example_grouped_gemm"
/opt/dtk/bin/aicc \
-O3 -DNDEBUG \
./grouped_gemm.cpp.o \
./instances/grouped_gemm_bf16.cpp.o \
./instances/grouped_gemm_bf8.cpp.o \
./instances/grouped_gemm_fp16.cpp.o \
./instances/grouped_gemm_fp8.cpp.o \
./instances/grouped_gemm_int4.cpp.o \
./instances/grouped_gemm_int8.cpp.o \
-o ./tile_example_grouped_gemm  \
/opt/dtk/hip/lib/libgalaxyhip.so.5.2.26093.2057-4d75eef5 \
--hip-link --offload-arch=gfx936 -L"/opt/dtk/llvm/lib/clang/17.0.0/include/../lib/linux" -lclang_rt.builtins-x86_64

