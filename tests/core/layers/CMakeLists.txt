if(USE_CUDA)
  add_subdirectory(cuda)
endif()

if(USE_MLU)
  add_subdirectory(mlu)
endif()
