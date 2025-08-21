include(proto_library)

proto_library(
  NAME
    xllm_proto
  SRCS
    common.proto
    completion.proto
    chat.proto
    multimodal.proto
    embedding.proto
    models.proto
    worker.proto
    disagg_pd.proto
    xllm_service.proto
    xservice.proto
    page_manager.proto
)
