#ifndef CYTO3CPP_H_
#define CYTO3CPP_H_

#include <list>
#include <fstream>
#include <iostream>
#include <opencv2/core.hpp>
#include <onnxruntime_cxx_api.h>
#include "fill_voids.hpp"

class Cyto3 {
  std::unique_ptr<Ort::Session> sessionEncoder, sessionAfterRunNet;
  Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "test"};
  Ort::SessionOptions sessionOptions[1];
  Ort::RunOptions runOptionsEncoder;
  Ort::MemoryInfo memoryInfo{Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)};
  std::vector<int64_t> inputShapeEncoder;
  std::vector<unsigned short> maskVector;
  std::vector<float> flowErrorsVector;
  std::vector<float> cellProbVector;
  std::vector<float> dPVector;

  bool loadingModel = false;
  bool preprocessing = false;
  bool terminating = false;
  
 public:
  Cyto3();
  ~Cyto3();
  bool clearLoadModel();
  void terminatePreprocessing();
  bool loadModel(const std::string& encoderPath, const std::string& afterRunNetPath, int threadsNumber, std::string device);
  void loadingStart();
  void loadingEnd();
  cv::Size getInputSize();
  cv::Mat changeFlowThreshold(float flow_threshold, int min_size);
  std::tuple<cv::Mat, cv::Mat> preprocessImage(const cv::Mat& image, const cv::Size &imageSize, const std::vector<int64_t> &channels, int diameter, float cellprob_threshold, int niter, float flow_threshold, int min_size);
  cv::Mat afterRunNet(const cv::Size &imageSize, float cellprob_threshold, int niter, float flow_threshold, int min_size);
  void preprocessingStart();
  void preprocessingEnd();
};

void saveOutputMask(cv::Mat mask, cv::Size imageSize, float cellprob_threshold, float flow_threshold, int min_size);

#endif
