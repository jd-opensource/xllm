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
#include <libswscale/swscale.h>
}
#include "mm_codec.h"

namespace xllm {

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
  struct MemCtx {
    const uint8_t* p;
    size_t sz;
    size_t off;
  };

  struct Reader {
    static int read(void* opaque, uint8_t* buf, int buf_size) {
      auto* mc = static_cast<MemCtx*>(opaque);
      size_t remain = mc->sz - mc->off;
      int n = (int)std::min(remain, (size_t)buf_size);
      if (n <= 0) return AVERROR_EOF;
      memcpy(buf, mc->p + mc->off, n);
      mc->off += (size_t)n;
      return n;
    }

    static int64_t seek(void* opaque, int64_t offset, int whence) {
      auto* mc = static_cast<MemCtx*>(opaque);

      if (whence == AVSEEK_SIZE) {
        return (int64_t)mc->sz;
      }

      int64_t pos = 0;
      switch (whence) {
        case SEEK_SET:
          pos = offset;
          break;
        case SEEK_CUR:
          pos = (int64_t)mc->off + offset;
          break;
        case SEEK_END:
          pos = (int64_t)mc->sz + offset;
          break;
        default:
          return AVERROR(EINVAL);
      }

      if (pos < 0 || pos > (int64_t)mc->sz) return AVERROR_EOF;

      mc->off = (size_t)pos;
      return pos;
    }
  };

  AVFormatContext* fmt = avformat_alloc_context();
  const int avio_buf_sz = 1 << 16;
  uint8_t* avio_buf = (uint8_t*)av_malloc(avio_buf_sz);
  if (!fmt || !avio_buf) {
    if (fmt) avformat_free_context(fmt);
    if (avio_buf) av_free(avio_buf);
    return false;
  }

  MemCtx mc{(const uint8_t*)raw_data.data(), raw_data.size(), 0};

  AVIOContext* avio = avio_alloc_context(
      avio_buf, avio_buf_sz, 0, &mc, &Reader::read, nullptr, &Reader::seek);
  if (!avio) {
    av_free(avio_buf);
    avformat_free_context(fmt);
    return false;
  }

  avio->seekable = AVIO_SEEKABLE_NORMAL;

  fmt->pb = avio;
  fmt->flags |= AVFMT_FLAG_CUSTOM_IO;

  fmt->probesize = std::min<size_t>(raw_data.size(), 20 * 1024 * 1024);
  fmt->max_analyze_duration = 5LL * AV_TIME_BASE;

  bool ok = false;

  AVDictionary* opts = nullptr;
  av_dict_set(&opts, "probesize", "20000000", 0);
  av_dict_set(&opts, "analyzeduration", "5000000", 0);

  const AVInputFormat* in_fmt = av_find_input_format("mp4");
  int ret = avformat_open_input(&fmt, nullptr, in_fmt, &opts);
  av_dict_free(&opts);

  if (ret < 0) {
    av_freep(&avio->buffer);
    avio_context_free(&avio);
    avformat_free_context(fmt);
    return false;
  }

  ret = avformat_find_stream_info(fmt, nullptr);
  if (ret < 0) {
    av_freep(&avio->buffer);
    avio_context_free(&avio);
    avformat_close_input(&fmt);
    return false;
  }

  int vs = av_find_best_stream(fmt, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
  if (vs < 0) {
    av_freep(&avio->buffer);
    avio_context_free(&avio);
    avformat_close_input(&fmt);
    return false;
  }

  AVStream* st = fmt->streams[vs];
  AVCodecParameters* par = st->codecpar;
  const AVCodec* dec = avcodec_find_decoder(par->codec_id);
  if (!dec) {
    av_freep(&avio->buffer);
    avio_context_free(&avio);
    avformat_close_input(&fmt);
    return false;
  }

  AVCodecContext* cc = avcodec_alloc_context3(dec);
  if (!cc) {
    av_freep(&avio->buffer);
    avio_context_free(&avio);
    avformat_close_input(&fmt);
    return false;
  }

  if (avcodec_parameters_to_context(cc, par) < 0 ||
      avcodec_open2(cc, dec, nullptr) < 0) {
    avcodec_free_context(&cc);
    av_freep(&avio->buffer);
    avio_context_free(&avio);
    avformat_close_input(&fmt);
    return false;
  }

  AVRational r = st->avg_frame_rate.num ? st->avg_frame_rate : st->r_frame_rate;
  double fps = (r.num && r.den) ? av_q2d(r) : 0.0;
  metadata.fps = fps;

  SwsContext* sws = nullptr;
  AVPacket* pkt = av_packet_alloc();
  AVFrame* frm = av_frame_alloc();
  std::vector<torch::Tensor> frames;

  auto push_frame = [&](AVFrame* f) -> bool {
    if (!sws) {
      sws = sws_getContext(f->width,
                           f->height,
                           (AVPixelFormat)f->format,
                           f->width,
                           f->height,
                           AV_PIX_FMT_RGB24,
                           SWS_BILINEAR,
                           nullptr,
                           nullptr,
                           nullptr);
      if (!sws) return false;
    }

    torch::Tensor rgb = torch::empty({f->height, f->width, 3}, torch::kUInt8);
    uint8_t* dst_data[4] = {rgb.data_ptr<uint8_t>(), nullptr, nullptr, nullptr};
    int dst_linesize[4] = {(int)rgb.stride(0), 0, 0, 0};

    sws_scale(sws, f->data, f->linesize, 0, f->height, dst_data, dst_linesize);

    frames.emplace_back(rgb.permute({2, 0, 1}).clone());  // [C,H,W]
    return true;
  };

  while (av_read_frame(fmt, pkt) >= 0) {
    if (pkt->stream_index == vs) {
      if (avcodec_send_packet(cc, pkt) == 0) {
        while (avcodec_receive_frame(cc, frm) == 0) {
          if (!push_frame(frm)) break;
        }
      }
    }
    av_packet_unref(pkt);
  }

  // flush
  avcodec_send_packet(cc, nullptr);
  while (avcodec_receive_frame(cc, frm) == 0) {
    if (!push_frame(frm)) break;
  }

  if (!frames.empty()) {
    t = torch::stack(frames);  // [T,C,H,W]
    metadata.total_num_frames = static_cast<int64_t>(frames.size());
    if (metadata.fps > 0.0) {
      metadata.duration = metadata.total_num_frames / metadata.fps;
    } else {
      metadata.duration = 0.0;
    }
    ok = true;
  }

  if (sws) sws_freeContext(sws);
  av_frame_free(&frm);
  av_packet_free(&pkt);
  avcodec_free_context(&cc);

  av_freep(&avio->buffer);
  avio_context_free(&avio);
  avformat_close_input(&fmt);

  return ok;
}

}  // namespace xllm
