#define STRIP_FLAG_HELP 1
#include <gflags/gflags.h>
#include <thread>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "cyto3.h"

DEFINE_string(encoder, "../cyto3.onnx", "Path to the encoder model");
DEFINE_string(image, "../demo_images/img02.png", "Path to the image");
DEFINE_string(device, "cpu", "cpu or cuda:0(1,2,3...)");
DEFINE_bool(h, false, "Show help");

int main(int argc, char** argv) {
  gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
  if(FLAGS_h){
    std::cout<<"Example: ./build/cyto3_cpp_test -encoder=\"../cyto3.onnx\" "
               "-image=\"../demo_images/img02.png\" -device=\"cpu\""<< std::endl;
    return 0;
  }
  Cyto3 cyto3;
  std::cout<<"loadModel started"<<std::endl;
  bool successLoadModel = cyto3.loadModel(FLAGS_encoder, std::thread::hardware_concurrency(), FLAGS_device);
  if(!successLoadModel){
    std::cout<<"loadModel error"<<std::endl;
    return 1;
  }
  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
  std::cout<<"preprocessImage started"<<std::endl;
  cv::Mat image = cv::imread(FLAGS_image, cv::IMREAD_COLOR);
  cv::Size imageSize = cv::Size(image.cols, image.rows);
  cv::Size inputSize = cyto3.getInputSize();
  cv::resize(image, image, inputSize);
  std::vector<int64_t> channels = {1, 2};
  int diameter = 30;
  int niter = 200;
  float flow_threshold = 0.4;
  int min_size = 15;
  torch::Tensor mask = cyto3.preprocessImage(image, inputSize, channels, diameter, niter, flow_threshold, min_size);
  if(mask.numel() == 0){
    std::cout<<"preprocessImage error"<<std::endl;
    return 1;
  }
  saveOutputMask(mask, imageSize, flow_threshold, min_size);
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  std::cout << "sec = " << (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) / 1000000.0 <<std::endl;

  begin = std::chrono::steady_clock::now();
  flow_threshold = 0.8;
  min_size = 100;
  mask = cyto3.changeFlowThreshold(flow_threshold, min_size);
  saveOutputMask(mask, imageSize, flow_threshold, min_size);
  end = std::chrono::steady_clock::now();
  std::cout << "sec = " << (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) / 1000000.0 <<std::endl;

  return 0;
}
