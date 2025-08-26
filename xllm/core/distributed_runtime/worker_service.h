#pragma once

#include <string>

#include "runtime/worker.h"
#include "worker.pb.h"

namespace xllm {

class WorkerService : public proto::DistributeWorker {
 public:
  WorkerService(runtime::Options options, const torch::Device& device);
  WorkerService(runtime::Options options,
                const torch::Device& device,
                std::unique_ptr<Worker> worker);

  virtual ~WorkerService();

  void set_worker(std::unique_ptr<Worker> worker);

  // service functions
  void Hello(::google::protobuf::RpcController* controller,
             const proto::Status* request,
             proto::Status* response,
             ::google::protobuf::Closure* done) override;

  void InitModel(::google::protobuf::RpcController* controller,
                 const proto::ModelPath* request,
                 proto::Status* response,
                 ::google::protobuf::Closure* done) override;

  void ProcessGroupTest(::google::protobuf::RpcController* controller,
                        const proto::Empty* request,
                        proto::Status* response,
                        ::google::protobuf::Closure* done) override;

  void ProfileDeviceMemory(::google::protobuf::RpcController* controller,
                           const proto::Empty* request,
                           proto::DeviceMemory* response,
                           ::google::protobuf::Closure* done) override;

  void AllocateKVCache(::google::protobuf::RpcController* controller,
                       const proto::KVCacheShape* request,
                       proto::Status* response,
                       ::google::protobuf::Closure* done) override;

  void AllocateKVCacheWithTransfer(
      ::google::protobuf::RpcController* controller,
      const proto::AllocateKVCacheWithTransferRequest* req,
      proto::Status* resp,
      ::google::protobuf::Closure* done) override;

  void PullKVCache(::google::protobuf::RpcController* controller,
                   const proto::PullKVCacheRequest* req,
                   proto::Status* resp,
                   ::google::protobuf::Closure* done) override;

  void GetDeviceInfo(::google::protobuf::RpcController* controller,
                     const proto::Empty* req,
                     proto::DeviceInfo* resp,
                     ::google::protobuf::Closure* done) override;

  void GetCacheInfo(::google::protobuf::RpcController* controller,
                    const proto::Empty* req,
                    proto::CacheInfo* resp,
                    ::google::protobuf::Closure* done) override;

  void LinkCluster(::google::protobuf::RpcController* controller,
                   const proto::ClusterInfo* req,
                   proto::Status* resp,
                   ::google::protobuf::Closure* done) override;

  void UnlinkCluster(::google::protobuf::RpcController* controller,
                     const proto::ClusterInfo* req,
                     proto::Status* resp,
                     ::google::protobuf::Closure* done) override;

  void ExecuteModel(::google::protobuf::RpcController* controller,
                    const proto::ForwardInput* pb_forward_input,
                    proto::ForwardOutput* pb_forward_output,
                    ::google::protobuf::Closure* done) override;

  void GetLastStepResult(::google::protobuf::RpcController* controller,
                         const proto::Empty* req,
                         proto::ForwardOutput* pb_forward_output,
                         ::google::protobuf::Closure* done) override;

  void GetActiveActivationMemory(::google::protobuf::RpcController* controller,
                                 const proto::Empty* req,
                                 proto::ActivationMemory* resp,
                                 ::google::protobuf::Closure* done) override;

 private:
  DISALLOW_COPY_AND_ASSIGN(WorkerService);

 private:
  // runtime options
  runtime::Options options_;

  bool initialized_;
  torch::Device device_;

  std::unique_ptr<Worker> worker_;

  ThreadPool threadpool_{5};

  // a walkaround to avoid compilation conflict involved by
  // c10_npu::NPUStream related files.
#if defined(USE_NPU)
  struct NPUStreamHelper;
  std::unique_ptr<NPUStreamHelper> npu_stream_helper_;
#elif defined(USE_MLU)
  // TODO(mlu): implement mlu stream helper
  struct MLUStreamHelper;
  std::unique_ptr<MLUStreamHelper> mlu_stream_helper_;
#endif
};

}  // namespace xllm
