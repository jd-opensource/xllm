/* Copyright 2025 The xLLM Authors. All Rights Reserved.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/channel_layout.h>
#include <libavutil/samplefmt.h>
#include <libswresample/swresample.h>
#include <libswscale/swscale.h>
}
#include "mm_codec.h"

namespace xllm {

struct MemCtx {
  const uint8_t* mem_ptr;
  int64_t size;
  int64_t offset;
};

struct Reader {
  static int read(void* opaque, uint8_t* buf, int buf_size) {
    auto* mc = static_cast<MemCtx*>(opaque);
    if (mc->offset < 0) return AVERROR(EINVAL);
    int64_t remain = mc->size - mc->offset;
    int n = (int)std::min(remain, (int64_t)buf_size);
    if (n <= 0) return AVERROR_EOF;
    memcpy(buf, mc->mem_ptr + mc->offset, n);
    mc->offset += (int64_t)n;
    return n;
  }

  static int64_t seek(void* opaque, int64_t offset, int whence) {
    auto* mc = static_cast<MemCtx*>(opaque);

    if (whence == AVSEEK_SIZE) {
      return (int64_t)mc->size;
    }

    int64_t pos = 0;
    switch (whence) {
      case SEEK_SET:
        pos = offset;
        break;
      case SEEK_CUR:
        pos = (int64_t)mc->offset + offset;
        break;
      case SEEK_END:
        pos = (int64_t)mc->size + offset;
        break;
      default:
        return AVERROR(EINVAL);
    }

    if (pos < 0 || pos > mc->size) return AVERROR(EINVAL);

    mc->offset = pos;
    return pos;
  }
};

class MemoryMediaReaderBase {
 public:
  MemoryMediaReaderBase(const uint8_t* data, size_t size) {
    mc_.mem_ptr = data;
    if (size > static_cast<size_t>(INT64_MAX)) {
      LOG(ERROR) << "MemCtx size too large";
    }
    mc_.size = static_cast<int64_t>(size);
    mc_.offset = 0;
  }

  ~MemoryMediaReaderBase() {
    if (frm_) av_frame_free(&frm_);
    if (pkt_) av_packet_free(&pkt_);
    if (codec_ctx_) avcodec_free_context(&codec_ctx_);
    if (fmt_ctx_) {
      if (opened_)
        avformat_close_input(&fmt_ctx_);
      else
        avformat_free_context(fmt_ctx_);
    }
    if (avio_ctx_) {
      av_freep(&avio_ctx_->buffer);
      avio_context_free(&avio_ctx_);
    }
  }

  bool init_base(AVMediaType type) {
    fmt_ctx_ = avformat_alloc_context();
    constexpr int avio_buf_sz = 1 << 16;
    uint8_t* avio_buf = (uint8_t*)av_malloc(avio_buf_sz);
    if (!fmt_ctx_ || !avio_buf) {
      if (fmt_ctx_) {
        avformat_free_context(fmt_ctx_);
        fmt_ctx_ = nullptr;
      }
      if (avio_buf) av_free(avio_buf);
      return false;
    }

    avio_ctx_ = avio_alloc_context(
        avio_buf, avio_buf_sz, 0, &mc_, &Reader::read, nullptr, &Reader::seek);
    if (!avio_ctx_) {
      av_free(avio_buf);
      avformat_free_context(fmt_ctx_);
      fmt_ctx_ = nullptr;
      return false;
    }

    avio_ctx_->seekable = AVIO_SEEKABLE_NORMAL;
    fmt_ctx_->pb = avio_ctx_;
    fmt_ctx_->flags |= AVFMT_FLAG_CUSTOM_IO;

    if (avformat_open_input(&fmt_ctx_, nullptr, nullptr, nullptr) < 0)
      return false;
    opened_ = true;

    if (avformat_find_stream_info(fmt_ctx_, nullptr) < 0) return false;

    stream_index_ = av_find_best_stream(fmt_ctx_, type, -1, -1, nullptr, 0);
    if (stream_index_ < 0) return false;

    AVStream* st = fmt_ctx_->streams[stream_index_];
    const AVCodec* codec = avcodec_find_decoder(st->codecpar->codec_id);
    if (!codec) return false;

    codec_ctx_ = avcodec_alloc_context3(codec);
    if (!codec_ctx_) return false;

    if (avcodec_parameters_to_context(codec_ctx_, st->codecpar) < 0 ||
        avcodec_open2(codec_ctx_, codec, nullptr) < 0)
      return false;

    return true;
  }

  bool decode_all() {
    if (!fmt_ctx_ || !codec_ctx_ || stream_index_ < 0) return false;

    if (!pkt_) pkt_ = av_packet_alloc();
    if (!frm_) frm_ = av_frame_alloc();
    if (!pkt_ || !frm_) return false;

    while (av_read_frame(fmt_ctx_, pkt_) >= 0) {
      if (pkt_->stream_index == stream_index_) {
        if (avcodec_send_packet(codec_ctx_, pkt_) == 0) {
          while (avcodec_receive_frame(codec_ctx_, frm_) == 0) {
            if (!handle_frame(frm_)) {
              av_packet_unref(pkt_);
              return false;
            }
          }
        }
      }
      av_packet_unref(pkt_);
    }

    // flush
    avcodec_send_packet(codec_ctx_, nullptr);
    while (avcodec_receive_frame(codec_ctx_, frm_) == 0) {
      if (!handle_frame(frm_)) return false;
    }

    return true;
  }

  virtual bool handle_frame(AVFrame* f) = 0;

 protected:
  AVFormatContext* fmt_ctx_ = nullptr;
  AVIOContext* avio_ctx_ = nullptr;
  AVCodecContext* codec_ctx_ = nullptr;
  AVPacket* pkt_ = nullptr;
  AVFrame* frm_ = nullptr;
  MemCtx mc_{nullptr, 0, 0};
  int stream_index_ = -1;
  bool opened_ = false;
};

class MemoryVideoReader : public MemoryMediaReaderBase {
 public:
  using MemoryMediaReaderBase::MemoryMediaReaderBase;

  ~MemoryVideoReader() {
    if (sws_ctx_) sws_freeContext(sws_ctx_);
  }

  bool init(VideoMetadata& metadata) {
    if (!init_base(AVMEDIA_TYPE_VIDEO)) return false;

    AVStream* st = fmt_ctx_->streams[stream_index_];
    AVRational r =
        st->avg_frame_rate.num ? st->avg_frame_rate : st->r_frame_rate;
    metadata.fps = (r.num && r.den) ? av_q2d(r) : 0.0;
    metadata.total_num_frames = 0;
    metadata.duration = 0.0;
    return true;
  }

  bool read_all(torch::Tensor& video_tensor, VideoMetadata& metadata) {
    frames_.clear();

    if (!decode_all()) return false;
    if (frames_.empty()) return false;

    video_tensor = torch::stack(frames_);  // [T,C,H,W]
    metadata.total_num_frames = (int64_t)frames_.size();
    metadata.duration = (metadata.fps > 0.0)
                            ? (double)metadata.total_num_frames / metadata.fps
                            : 0.0;
    return true;
  }

  bool handle_frame(AVFrame* f) override {
    if (!sws_ctx_) {
      sws_ctx_ = sws_getContext(f->width,
                                f->height,
                                (AVPixelFormat)f->format,
                                f->width,
                                f->height,
                                AV_PIX_FMT_RGB24,
                                SWS_BILINEAR,
                                nullptr,
                                nullptr,
                                nullptr);
      if (!sws_ctx_) return false;
    }

    torch::Tensor rgb = torch::empty({f->height, f->width, 3}, torch::kUInt8);

    uint8_t* dst_data[4] = {rgb.data_ptr<uint8_t>(), nullptr, nullptr, nullptr};
    int dst_linesize[4] = {(int)rgb.stride(0), 0, 0, 0};

    sws_scale(
        sws_ctx_, f->data, f->linesize, 0, f->height, dst_data, dst_linesize);

    frames_.emplace_back(rgb.permute({2, 0, 1}).clone());  // [C,H,W]
    return true;
  }

 private:
  SwsContext* sws_ctx_ = nullptr;
  std::vector<torch::Tensor> frames_;
};

class MemoryAudioReader : public MemoryMediaReaderBase {
 public:
  using MemoryMediaReaderBase::MemoryMediaReaderBase;

  ~MemoryAudioReader() {
    if (swr_ctx_) swr_free(&swr_ctx_);
  }

  bool init(AudioMetadata& metadata) {
    target_sr_ = 16000;
    target_ch_ = 1;

    if (!init_base(AVMEDIA_TYPE_AUDIO)) return false;

    AVStream* st = fmt_ctx_->streams[stream_index_];
    codec_ctx_->pkt_timebase = st->time_base;

    swr_ctx_ = swr_alloc();
    if (!swr_ctx_) return false;

    AVChannelLayout in_layout;
    av_channel_layout_copy(&in_layout, &codec_ctx_->ch_layout);

    AVChannelLayout out_layout;
    av_channel_layout_default(&out_layout, target_ch_);

    if (swr_alloc_set_opts2(&swr_ctx_,
                            &out_layout,
                            AV_SAMPLE_FMT_FLT,
                            target_sr_,
                            &in_layout,
                            codec_ctx_->sample_fmt,
                            codec_ctx_->sample_rate,
                            0,
                            nullptr) < 0) {
      av_channel_layout_uninit(&out_layout);
      av_channel_layout_uninit(&in_layout);
      return false;
    }

    av_channel_layout_uninit(&out_layout);
    av_channel_layout_uninit(&in_layout);

    if (swr_init(swr_ctx_) < 0) {
      swr_free(&swr_ctx_);
      return false;
    }

    metadata.sample_rate = target_sr_;
    metadata.num_channels = target_ch_;
    metadata.duration = 0.0;
    return true;
  }

  bool read_all(torch::Tensor& audio_tensor, AudioMetadata& metadata) {
    pcm_.clear();

    if (!swr_ctx_) return false;
    if (!decode_all()) return false;
    // flush
    while (true) {
      int out_nb = swr_get_out_samples(swr_ctx_, 0);
      if (out_nb <= 0) break;

      std::vector<float> out_buf((size_t)out_nb * (size_t)target_ch_);
      uint8_t* out_data[1] = {reinterpret_cast<uint8_t*>(out_buf.data())};

      int converted = swr_convert(swr_ctx_, out_data, out_nb, nullptr, 0);
      if (converted < 0) return false;
      if (converted == 0) break;

      pcm_.insert(
          pcm_.end(),
          out_buf.begin(),
          out_buf.begin() + (ptrdiff_t)converted * (ptrdiff_t)target_ch_);
    }

    if (pcm_.empty()) return false;

    if (target_ch_ == 1) {
      audio_tensor =
          torch::from_blob(pcm_.data(),
                           {(int64_t)pcm_.size()},
                           torch::TensorOptions().dtype(torch::kFloat32))
              .clone();
      metadata.duration = (double)pcm_.size() / (double)target_sr_;
    } else {
      int64_t T = (int64_t)(pcm_.size() / (size_t)target_ch_);
      torch::Tensor t = torch::empty({target_ch_, T}, torch::kFloat32);

      float* dst = t.data_ptr<float>();
      for (int c = 0; c < target_ch_; ++c) {
        for (int64_t i = 0; i < T; ++i) {
          dst[c * T + i] = pcm_[(size_t)i * (size_t)target_ch_ + (size_t)c];
        }
      }
      audio_tensor = t;
      metadata.duration = (double)T / (double)target_sr_;
    }
    metadata.sample_rate = target_sr_;
    metadata.num_channels = target_ch_;
    return true;
  }

  bool handle_frame(AVFrame* f) override {
    int out_nb = swr_get_out_samples(swr_ctx_, f->nb_samples);
    if (out_nb <= 0) return false;

    std::vector<float> out_buf((size_t)out_nb * (size_t)target_ch_);
    uint8_t* out_data[1] = {reinterpret_cast<uint8_t*>(out_buf.data())};
    const uint8_t** in_data = (const uint8_t**)f->extended_data;

    int converted =
        swr_convert(swr_ctx_, out_data, out_nb, in_data, f->nb_samples);
    if (converted < 0) return false;

    pcm_.insert(pcm_.end(),
                out_buf.begin(),
                out_buf.begin() + (ptrdiff_t)converted * (ptrdiff_t)target_ch_);
    return true;
    return true;
  }

 private:
  SwrContext* swr_ctx_ = nullptr;
  int target_sr_ = 16000;
  int target_ch_ = 1;
  std::vector<float> pcm_;
};

bool OpenCVImageDecoder::decode(const std::string& raw_data, torch::Tensor& t) {
  cv::Mat buffer(1, raw_data.size(), CV_8UC1, (void*)raw_data.data());
  cv::Mat image = cv::imdecode(buffer, cv::IMREAD_COLOR);
  if (image.empty()) {
    LOG(INFO) << " opencv image decode failed";
    return false;
  }

  cv::cvtColor(image, image, cv::COLOR_BGR2RGB);  // RGB

  torch::Tensor tensor =
      torch::from_blob(image.data, {image.rows, image.cols, 3}, torch::kUInt8);

  t = tensor.permute({2, 0, 1}).clone();  // [C, H, W]
  return true;
}

bool OpenCVImageEncoder::encode(const torch::Tensor& t, std::string& raw_data) {
  if (!valid(t)) {
    return false;
  }

  auto img = t.permute({1, 2, 0}).contiguous();
  cv::Mat mat(img.size(0), img.size(1), CV_32FC3, img.data_ptr<float>());

  cv::Mat mat_8u;
  mat.convertTo(mat_8u, CV_8UC3, 255.0);

  // rgb -> bgr
  cv::cvtColor(mat_8u, mat_8u, cv::COLOR_RGB2BGR);

  std::vector<uchar> data;
  if (!cv::imencode(".png", mat_8u, data)) {
    LOG(ERROR) << "image encode faild";
    return false;
  }

  raw_data.assign(data.begin(), data.end());
  return true;
}

bool OpenCVImageEncoder::valid(const torch::Tensor& t) {
  if (t.dim() != 3 || t.size(0) != 3) {
    LOG(ERROR) << "input tensor must be 3HW  tensor";
    return false;
  }

  if (t.scalar_type() != torch::kFloat32 || !t.device().is_cpu()) {
    LOG(ERROR) << "tensor must be cpu float32";
    return false;
  }

  return true;
}

bool OpenCVVideoDecoder::decode(const std::string& raw_data,
                                torch::Tensor& t,
                                VideoMetadata& metadata) {
  MemoryVideoReader reader(reinterpret_cast<const uint8_t*>(raw_data.data()),
                           raw_data.size());

  if (!reader.init(metadata)) return false;
  if (!reader.read_all(t, metadata)) return false;
  return true;
}

bool FFmpegAudioDecoder::decode(const std::string& raw_data,
                                torch::Tensor& t,
                                AudioMetadata& metadata) {
  MemoryAudioReader reader(reinterpret_cast<const uint8_t*>(raw_data.data()),
                           raw_data.size());
  if (!reader.init(metadata)) return false;
  if (!reader.read_all(t, metadata)) return false;
  auto cpu_embeds = t.to(torch::kCPU);
  torch::save(cpu_embeds,
              "/export/home/wangziyue.28/test/mainxllm/xllm/xllm.pt");
  return true;
}

}  // namespace xllm
