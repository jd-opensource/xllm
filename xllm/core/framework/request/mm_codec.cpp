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
#include "mm_codec.h"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/channel_layout.h>
#include <libavutil/samplefmt.h>
#include <libswresample/swresample.h>
#include <libswscale/swscale.h>
}

namespace xllm {

namespace {

struct MemCtx {
  const uint8_t* mem_ptr;
  int64_t size;
  int64_t offset;
};

struct Reader {
  // AVIO read callback
  static int32_t read(void* opaque, uint8_t* buf, int32_t buf_size) {
    auto* mc = static_cast<MemCtx*>(opaque);
    if (mc->offset < 0) return AVERROR(EINVAL);
    int64_t remain = mc->size - mc->offset;
    int64_t n = std::min(remain, static_cast<int64_t>(buf_size));
    if (n <= 0) return AVERROR_EOF;
    std::memcpy(buf, mc->mem_ptr + mc->offset, static_cast<size_t>(n));
    mc->offset += n;
    return static_cast<int32_t>(n);
  }

  // AVIO seek callback
  static int64_t seek(void* opaque, int64_t offset, int32_t whence) {
    auto* mc = static_cast<MemCtx*>(opaque);

    if (whence == AVSEEK_SIZE) {
      return mc->size;
    }

    int64_t pos = 0;
    switch (whence) {
      case SEEK_SET:
        pos = offset;
        break;
      case SEEK_CUR:
        pos = mc->offset + offset;
        break;
      case SEEK_END:
        pos = mc->size + offset;
        break;
      default:
        return AVERROR(EINVAL);
    }

    if (pos < 0 || pos > mc->size) return AVERROR(EINVAL);

    mc->offset = pos;
    return pos;
  }
};

}  // namespace

class MemoryMediaReader {
 public:
  MemoryMediaReader(const uint8_t* data, size_t size) {
    mc_.mem_ptr = data;
    if (size > static_cast<size_t>(INT64_MAX)) {
      LOG(FATAL) << "MemCtx size too large";
    }
    mc_.size = static_cast<int64_t>(size);
    mc_.offset = 0;
  }

  ~MemoryMediaReader() {
    if (frm_) {
      av_frame_free(&frm_);
    }
    if (pkt_) {
      av_packet_free(&pkt_);
    }
    if (codec_ctx_) {
      avcodec_free_context(&codec_ctx_);
    }
    // if opened via avformat_open_input, close with avformat_close_input
    // otherwise free with avformat_free_context
    if (fmt_ctx_) {
      if (opened_) {
        avformat_close_input(&fmt_ctx_);
      } else {
        avformat_free_context(fmt_ctx_);
      }
    }
    if (avio_ctx_) {
      av_freep(&avio_ctx_->buffer);
      avio_context_free(&avio_ctx_);
    }
  }

  bool init(AVMediaType type) {
    fmt_ctx_ = avformat_alloc_context();
    constexpr int32_t avio_buf_sz = 1 << 16;
    uint8_t* avio_buf =
        static_cast<uint8_t*>(av_malloc(static_cast<size_t>(avio_buf_sz)));
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

    if (avformat_find_stream_info(fmt_ctx_, nullptr) < 0) {
      return false;
    }

    stream_index_ = av_find_best_stream(fmt_ctx_, type, -1, -1, nullptr, 0);
    if (stream_index_ < 0) return false;

    AVStream* st = fmt_ctx_->streams[stream_index_];
    const AVCodec* codec = avcodec_find_decoder(st->codecpar->codec_id);
    if (!codec) {
      return false;
    }

    codec_ctx_ = avcodec_alloc_context3(codec);
    if (!codec_ctx_) {
      return false;
    }

    if (avcodec_parameters_to_context(codec_ctx_, st->codecpar) < 0 ||
        avcodec_open2(codec_ctx_, codec, nullptr) < 0) {
      return false;
    }

    return true;
  }

  bool decode() {
    if (!fmt_ctx_ || !codec_ctx_ || stream_index_ < 0) {
      return false;
    }

    if (!pkt_) {
      pkt_ = av_packet_alloc();
    }
    if (!frm_) {
      frm_ = av_frame_alloc();
    }
    if (!pkt_ || !frm_) {
      return false;
    }

    // Read packets, send to decoder, pull frames
    while (av_read_frame(fmt_ctx_, pkt_) >= 0) {
      if (pkt_->stream_index == stream_index_) {
        if (avcodec_send_packet(codec_ctx_, pkt_) == 0) {
          while (avcodec_receive_frame(codec_ctx_, frm_) == 0) {
            // handle_frame: video->push RGB frame, audio->append PCM
            if (!handle_frame(frm_)) {
              av_packet_unref(pkt_);
              return false;
            }
          }
        }
      }
      av_packet_unref(pkt_);
    }

    // flush decoder at end of stream
    avcodec_send_packet(codec_ctx_, nullptr);
    while (avcodec_receive_frame(codec_ctx_, frm_) == 0) {
      if (!handle_frame(frm_)) {
        return false;
      }
    }

    return true;
  }

  // video->RGB tensor, audio->PCM samples
  virtual bool handle_frame(AVFrame* f) = 0;

 protected:
  AVFormatContext* fmt_ctx_ = nullptr;
  AVIOContext* avio_ctx_ = nullptr;
  AVCodecContext* codec_ctx_ = nullptr;
  AVPacket* pkt_ = nullptr;
  AVFrame* frm_ = nullptr;
  MemCtx mc_{nullptr, 0, 0};
  int32_t stream_index_ = -1;
  bool opened_ = false;
};

class MemoryVideoReader : public MemoryMediaReader {
 public:
  MemoryVideoReader(const uint8_t* data, size_t size)
      : MemoryMediaReader(data, size) {}

  ~MemoryVideoReader() {
    if (sws_ctx_) {
      sws_freeContext(sws_ctx_);
    }
    if (rgb_frame_) {
      av_frame_free(&rgb_frame_);
    }
  }

  bool init(VideoMetadata& metadata) {
    if (!MemoryMediaReader::init(AVMEDIA_TYPE_VIDEO)) {
      return false;
    }

    // init VideoMetadata
    AVStream* st = fmt_ctx_->streams[stream_index_];
    AVRational r =
        st->avg_frame_rate.num ? st->avg_frame_rate : st->r_frame_rate;
    metadata.fps = (r.num && r.den) ? av_q2d(r) : 0.0;
    metadata.total_num_frames = 0;
    metadata.duration = 0.0;
    return true;
  }

  bool read(torch::Tensor& tensor, VideoMetadata& metadata) {
    frames_.clear();

    if (!decode()) {
      return false;
    }
    if (frames_.empty()) {
      return false;
    }

    tensor = torch::stack(frames_);  // [T,C,H,W]
    metadata.total_num_frames = static_cast<int64_t>(frames_.size());
    metadata.duration = (metadata.fps > 0.0)
                            ? (double)metadata.total_num_frames / metadata.fps
                            : 0.0;
    return true;
  }

  bool handle_frame(AVFrame* f) override {
    // init colorspace converter once based on first frame
    if (!sws_ctx_) {
      sws_ctx_ = sws_getContext(f->width,
                                f->height,
                                static_cast<AVPixelFormat>(f->format),
                                f->width,
                                f->height,
                                AV_PIX_FMT_RGB24,
                                SWS_BILINEAR,
                                nullptr,
                                nullptr,
                                nullptr);
      if (!sws_ctx_) {
        return false;
      }
    }

    // use an FFmpeg-allocated frame so sws_scale writes into a buffer with the
    // correct padded linesize
    if (!rgb_frame_) {
      rgb_frame_ = av_frame_alloc();
      if (!rgb_frame_) {
        return false;
      }
    }

    // (re)allocate the RGB buffer when input changes
    if (rgb_frame_->width != f->width || rgb_frame_->height != f->height ||
        rgb_frame_->format != AV_PIX_FMT_RGB24 || !rgb_frame_->data[0]) {
      av_frame_unref(rgb_frame_);
      rgb_frame_->format = AV_PIX_FMT_RGB24;
      rgb_frame_->width = f->width;
      rgb_frame_->height = f->height;
      if (av_frame_get_buffer(rgb_frame_, 0) < 0) {
        return false;
      }
    }
    if (av_frame_make_writable(rgb_frame_) < 0) return false;

    // convert the current decoded frame into RGB24
    if (sws_scale(sws_ctx_,
                  f->data,
                  f->linesize,
                  0,
                  f->height,
                  rgb_frame_->data,
                  rgb_frame_->linesize) != f->height) {
      return false;
    }

    // build CHW uint8 tensor
    const int64_t H = f->height;
    const int64_t W = f->width;
    const int64_t src_ls = rgb_frame_->linesize[0];

    auto rgb = torch::from_blob(rgb_frame_->data[0],
                                {3, H, W},  // [C,H,W]
                                {1, src_ls, 3},
                                torch::TensorOptions().dtype(torch::kUInt8))
                   .contiguous();

    frames_.emplace_back(rgb.clone());
    return true;
  }

 private:
  SwsContext* sws_ctx_ = nullptr;
  AVFrame* rgb_frame_ = nullptr;
  std::vector<torch::Tensor> frames_;
};

class MemoryAudioReader : public MemoryMediaReader {
 public:
  MemoryAudioReader(const uint8_t* data, size_t size)
      : MemoryMediaReader(data, size) {}

  ~MemoryAudioReader() {
    if (swr_ctx_) {
      swr_free(&swr_ctx_);
    }
  }

  bool init(AudioMetadata& metadata) {
    target_sr_ = 16000;
    target_ch_ = 1;

    if (!MemoryMediaReader::init(AVMEDIA_TYPE_AUDIO)) {
      return false;
    }

    AVStream* st = fmt_ctx_->streams[stream_index_];
    codec_ctx_->pkt_timebase = st->time_base;

    // setup resampler
    swr_ctx_ = swr_alloc();
    if (!swr_ctx_) {
      return false;
    }

    AVChannelLayout in_layout;
    av_channel_layout_copy(&in_layout, &codec_ctx_->ch_layout);

    AVChannelLayout out_layout;
    av_channel_layout_default(&out_layout, static_cast<int32_t>(target_ch_));

    if (swr_alloc_set_opts2(&swr_ctx_,
                            &out_layout,
                            AV_SAMPLE_FMT_FLT,
                            static_cast<int32_t>(target_sr_),
                            &in_layout,
                            codec_ctx_->sample_fmt,
                            codec_ctx_->sample_rate,
                            0,
                            nullptr) < 0) {
      av_channel_layout_uninit(&out_layout);
      av_channel_layout_uninit(&in_layout);
      swr_free(&swr_ctx_);
      return false;
    }

    av_channel_layout_uninit(&out_layout);
    av_channel_layout_uninit(&in_layout);

    // if downmixing stereo -> mono, use customized remix matrix (L+R)/2
    int32_t in_ch = codec_ctx_->ch_layout.nb_channels;
    if (target_ch_ == 1 && in_ch == 2) {
      constexpr double matrix[2] = {0.5, 0.5};
      if (swr_set_matrix(swr_ctx_, matrix, in_ch) < 0) {
        swr_free(&swr_ctx_);
        return false;
      }
    }

    if (swr_init(swr_ctx_) < 0) {
      swr_free(&swr_ctx_);
      return false;
    }

    // init AudioMetadata
    metadata.sample_rate = target_sr_;
    metadata.num_channels = target_ch_;
    metadata.duration = 0.0;
    return true;
  }

  bool read(torch::Tensor& tensor, AudioMetadata& metadata) {
    pcm_.clear();

    if (!swr_ctx_) {
      return false;
    }
    if (!decode()) {
      return false;
    }

    // flush resampler buffered samples after decoder flush
    while (true) {
      int32_t out_nb = swr_get_out_samples(swr_ctx_, 0);
      if (out_nb <= 0) {
        break;
      }

      std::vector<float> out_buf(static_cast<size_t>(out_nb) *
                                 static_cast<size_t>(target_ch_));
      uint8_t* out_data[1] = {reinterpret_cast<uint8_t*>(out_buf.data())};

      int32_t converted = swr_convert(swr_ctx_, out_data, out_nb, nullptr, 0);
      if (converted < 0) {
        return false;
      }
      if (converted == 0) {
        break;
      }

      const int64_t n = static_cast<int64_t>(converted) * target_ch_;
      pcm_.reserve(pcm_.size() + static_cast<size_t>(n));
      pcm_.insert(pcm_.end(), out_buf.data(), out_buf.data() + n);
    }

    if (pcm_.empty()) {
      return false;
    }

    // build output tensor and compute duration from samples
    if (target_ch_ == 1) {
      tensor = torch::from_blob(pcm_.data(),
                                {static_cast<int64_t>(pcm_.size())},
                                torch::TensorOptions().dtype(torch::kFloat32))
                   .clone();
      metadata.duration = (double)pcm_.size() / (double)target_sr_;
    } else {
      int64_t T =
          static_cast<int64_t>(pcm_.size() / static_cast<size_t>(target_ch_));
      tensor = torch::from_blob(pcm_.data(),
                                {T, target_ch_},
                                torch::TensorOptions().dtype(torch::kFloat32))
                   .permute({1, 0})
                   .clone();
      metadata.duration = (double)T / (double)target_sr_;
    }
    metadata.sample_rate = target_sr_;
    metadata.num_channels = target_ch_;
    return true;
  }

  bool handle_frame(AVFrame* f) override {
    int32_t out_nb = swr_get_out_samples(swr_ctx_, f->nb_samples);
    if (out_nb < 0) {
      return false;
    }
    if (out_nb == 0) {
      return true;
    }

    std::vector<float> out_buf(static_cast<size_t>(out_nb) *
                               static_cast<size_t>(target_ch_));
    uint8_t* out_data[1] = {reinterpret_cast<uint8_t*>(out_buf.data())};
    const uint8_t** in_data = (const uint8_t**)f->extended_data;

    // convert input frame samples to target format
    int32_t converted =
        swr_convert(swr_ctx_, out_data, out_nb, in_data, f->nb_samples);

    if (converted < 0) {
      return false;
    }

    // append converted samples to pcm buffer
    const int64_t n = static_cast<int64_t>(converted) * target_ch_;
    pcm_.reserve(pcm_.size() + static_cast<size_t>(n));
    pcm_.insert(pcm_.end(), out_buf.data(), out_buf.data() + n);
    return true;
  }

 private:
  SwrContext* swr_ctx_ = nullptr;
  int64_t target_sr_ = 16000;
  int64_t target_ch_ = 1;
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

bool FFmpegVideoDecoder::decode(const std::string& raw_data,
                                torch::Tensor& t,
                                VideoMetadata& metadata) {
  MemoryVideoReader reader(reinterpret_cast<const uint8_t*>(raw_data.data()),
                           raw_data.size());

  if (!reader.init(metadata) || !reader.read(t, metadata)) {
    LOG(INFO) << "video decode faild";
    return false;
  }
  return true;
}

bool FFmpegAudioDecoder::decode(const std::string& raw_data,
                                torch::Tensor& t,
                                AudioMetadata& metadata) {
  MemoryAudioReader reader(reinterpret_cast<const uint8_t*>(raw_data.data()),
                           raw_data.size());

  if (!reader.init(metadata) || !reader.read(t, metadata)) {
    LOG(INFO) << "audio decode faild";
    return false;
  }
  return true;
}

}  // namespace xllm
