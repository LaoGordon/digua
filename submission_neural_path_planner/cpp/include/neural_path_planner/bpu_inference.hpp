// bpu_inference.hpp — BPU C++ 推理 wrapper。
//
// 封装 hb_dnn C++ API, 提供简单的 load + forward 接口。
// 只能在板子上编译(依赖 libdnn.so + BPU驱动)。

#ifndef NEURAL_PATH_PLANNER__BPU_INFERENCE_HPP_
#define NEURAL_PATH_PLANNER__BPU_INFERENCE_HPP_

#include <hobot/dnn/hb_dnn.h>
#include <hobot/hb_ucp.h>
#include <hobot/hb_ucp_sys.h>

#include <string>
#include <vector>
#include <cstdint>
#include <cstring>

namespace neural_path_planner
{

class BpuInference
{
public:
  BpuInference() = default;

  ~BpuInference()
  {
    if (packed_handle_) {
      hbDNNRelease(packed_handle_);
    }
  }

  bool load(const std::string & model_path)
  {
    const char * path = model_path.c_str();
    const char ** paths = &path;
    int32_t ret = hbDNNInitializeFromFiles(&packed_handle_, paths, 1);
    if (ret != 0) return false;

    const char ** name_list = nullptr;
    int32_t name_count = 0;
    hbDNNGetModelNameList(&name_list, &name_count, packed_handle_);
    if (name_count == 0) return false;

    ret = hbDNNGetModelHandle(&dnn_handle_, packed_handle_, name_list[0]);
    if (ret != 0) return false;

    hbDNNGetInputTensorProperties(&input_prop_, dnn_handle_, 0);
    hbDNNGetOutputTensorProperties(&output_prop_, dnn_handle_, 0);

    loaded_ = true;
    return true;
  }

  std::vector<float> forward(const std::vector<float> & input, int H, int W)
  {
    if (!loaded_) return {};

    // 输入tensor
    hbDNNTensor in_tensor;
    std::memset(&in_tensor, 0, sizeof(in_tensor));
    in_tensor.properties = input_prop_;
    int64_t in_size = input_prop_.alignedByteSize;
    hbUCPMalloc(&in_tensor.sysMem, in_size, 0);
    std::memcpy(in_tensor.sysMem.virAddr, input.data(),
                std::min(static_cast<size_t>(in_size), input.size() * sizeof(float)));

    // 输出tensor
    hbDNNTensor out_tensor;
    std::memset(&out_tensor, 0, sizeof(out_tensor));
    out_tensor.properties = output_prop_;
    int64_t out_size = output_prop_.alignedByteSize;
    hbUCPMalloc(&out_tensor.sysMem, out_size, 0);

    // 推理 (taskHandle=nullptr → API自动创建新task)
    hbUCPTaskHandle_t task_handle = nullptr;
    int32_t ret = hbDNNInferV2(&task_handle, &out_tensor, &in_tensor, dnn_handle_);
    if (ret != 0) {
      hbUCPFree(&in_tensor.sysMem);
      hbUCPFree(&out_tensor.sysMem);
      return {};
    }
    // 提交task到BPU执行, 然后等待完成
    hbUCPSchedParam sched_param;
    std::memset(&sched_param, 0, sizeof(sched_param));
    sched_param.priority = 0;
    hbUCPSubmitTask(task_handle, &sched_param);
    hbUCPWaitTaskDone(task_handle, 0);  // 等待推理完成

    // 提取输出
    std::vector<float> output(H * W);
    float * out_data = static_cast<float *>(out_tensor.sysMem.virAddr);
    int copy_n = std::min(static_cast<int>(H * W),
                          static_cast<int>(out_size / sizeof(float)));
    std::memcpy(output.data(), out_data, copy_n * sizeof(float));

    hbUCPReleaseTask(task_handle);
    hbUCPFree(&in_tensor.sysMem);
    hbUCPFree(&out_tensor.sysMem);
    return output;
  }

  bool isLoaded() const { return loaded_; }

private:
  hbDNNPackedHandle_t packed_handle_{nullptr};
  hbDNNHandle_t dnn_handle_{nullptr};
  hbDNNTensorProperties input_prop_{};
  hbDNNTensorProperties output_prop_{};
  bool loaded_{false};
};

}  // namespace neural_path_planner

#endif  // NEURAL_PATH_PLANNER__BPU_INFERENCE_HPP_
