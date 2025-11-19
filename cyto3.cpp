#include "cyto3.h"
#include <opencv2/opencv.hpp>

Cyto3::Cyto3(){}
Cyto3::~Cyto3(){
  if(loadingModel){
    return;
  }
  if(preprocessing){
    return;
  }
  clearLoadModelCP();
}

bool Cyto3::clearLoadModelCP(){
  try{
    Ort::Session* pre = sessionEncoder.release();
    delete pre;
    inputShapeEncoder.resize(0);
    maskVector.resize(0);
    flowErrorsVector.resize(0);
  }catch(Ort::Exception& e){
    return false;
  }
  return true;
}

void Cyto3::terminatePreprocessingCP(){
  runOptionsEncoder.SetTerminate();
  terminating = true;
}

bool modelExistsCP(const std::string& modelPath){
  std::ifstream f(modelPath);
  if (!f.good()) {
    return false;
  }
  return true;
}

bool Cyto3::loadModel(const std::string& encoderPath, int threadsNumber, std::string device){
  try{
    loadingStart();
    if(!clearLoadModelCP()){
      loadingEnd();
      return false;
    }
    if(!modelExistsCP(encoderPath)){
      loadingEnd();
      return false;
    }
    for(int i = 0; i < 1; i++){
      auto& option = sessionOptions[i];
      option.SetIntraOpNumThreads(threadsNumber);
      option.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
      if(device == "cpu"){
        continue;
      }
      if(device.substr(0, 5) == "cuda:"){
        int gpuDeviceId = std::stoi(device.substr(5));
        OrtCUDAProviderOptions options;
        options.device_id = gpuDeviceId;
        options.arena_extend_strategy = 1;
        option.AppendExecutionProvider_CUDA(options);
      }
    }
    sessionEncoder = std::make_unique<Ort::Session>(env, encoderPath.c_str(), sessionOptions[0]);
    inputShapeEncoder = sessionEncoder->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
  }catch(Ort::Exception& e){
    std::cout << e.what() << std::endl;
    loadingEnd();
    return false;
  }
  if(terminating){
    loadingEnd();
    return false;
  }
  loadingEnd();
  return true;
}

void Cyto3::loadingStart(){
  loadingModel = true;
}

void Cyto3::loadingEnd(){
  loadingModel = false;
  terminating = false;
}

cv::Size Cyto3::getInputSize(){
  return cv::Size((int)inputShapeEncoder[3], (int)inputShapeEncoder[2]);
}

std::vector<const char*> getInputNamesCP(std::unique_ptr<Ort::Session> &session){
  std::vector<const char*> inputNames;
  Ort::AllocatorWithDefaultOptions allocator;
  for (size_t i = 0; i < session->GetInputCount(); ++i) {
    Ort::AllocatedStringPtr name_Ptr = session->GetInputNameAllocated(i, allocator);
    char* name = name_Ptr.get();
    size_t name_length = strlen(name) + 1;
    char* name_new = new char[name_length];
    strncpy(name_new, name, name_length);
    inputNames.push_back(name_new);
  }
  return inputNames;
}

std::vector<const char*> getOutputNamesCP(std::unique_ptr<Ort::Session> &session){
  std::vector<const char*> outputNames;
  Ort::AllocatorWithDefaultOptions allocator;
  for (size_t i = 0; i < session->GetOutputCount(); ++i) {
    Ort::AllocatedStringPtr name_Ptr = session->GetOutputNameAllocated(i, allocator);
    char* name = name_Ptr.get();
    size_t name_length = strlen(name) + 1;
    char* name_new = new char[name_length];
    strncpy(name_new, name, name_length);
    outputNames.push_back(name_new);
  }
  return outputNames;
}

std::tuple<cv::Mat, cv::Mat> Cyto3::preprocessImage(const cv::Mat& image, const cv::Size &imageSize, const std::vector<int64_t> &channels, int diameter, float cellprob_threshold, int niter, float flow_threshold, int min_size){
  try{
    preprocessingStart();
    if(image.size() != cv::Size((int)inputShapeEncoder[3], (int)inputShapeEncoder[2])){
      preprocessingEnd();
      cv::Mat m1, m2;
      return std::make_tuple(m1, m2);
    }
    if(image.channels() != 3){
      preprocessingEnd();
      cv::Mat m1, m2;
      return std::make_tuple(m1, m2);
    }
    std::vector<Ort::Value> inputTensors;
    std::vector<float> inputTensorValuesFloat;
    inputTensorValuesFloat.resize(inputShapeEncoder[0] * inputShapeEncoder[1] * inputShapeEncoder[2] * inputShapeEncoder[3]);
    for(int i = 0; i < inputShapeEncoder[2]; i++){
      for(int j = 0; j < inputShapeEncoder[3]; j++){
        int64_t pos = i * inputShapeEncoder[3] + j;
        int64_t size = inputShapeEncoder[2] * inputShapeEncoder[3];
        inputTensorValuesFloat[pos + size * 0] = image.at<cv::Vec3b>(i, j)[0];
        inputTensorValuesFloat[pos + size * 1] = image.at<cv::Vec3b>(i, j)[1];
        inputTensorValuesFloat[pos + size * 2] = image.at<cv::Vec3b>(i, j)[2];
      }
    }
    inputTensors.push_back(Ort::Value::CreateTensor<float>(memoryInfo, inputTensorValuesFloat.data(), inputTensorValuesFloat.size(), inputShapeEncoder.data(), inputShapeEncoder.size()));
    
    std::vector<int64_t> orig_im_size_values = {imageSize.height, imageSize.width};
    std::vector<int64_t> origImSizeShape = {2};
    inputTensors.push_back(Ort::Value::CreateTensor<int64_t>(memoryInfo, orig_im_size_values.data(), 2, origImSizeShape.data(), origImSizeShape.size()));
    
    std::vector<int64_t> channels_values = {channels[0], channels[1]};
    std::vector<int64_t> channelsShape = {2};
    inputTensors.push_back(Ort::Value::CreateTensor<int64_t>(memoryInfo, channels_values.data(), 2, channelsShape.data(), channelsShape.size()));
    
    std::vector<int64_t> diameter_values = {diameter};
    std::vector<int64_t> diameterShape = {1};
    inputTensors.push_back(Ort::Value::CreateTensor<int64_t>(memoryInfo, diameter_values.data(), 1, diameterShape.data(), diameterShape.size()));
   
    std::vector<float> cellprob_threshold_values = {cellprob_threshold};
    std::vector<int64_t> cellprob_thresholdShape = {1};
    inputTensors.push_back(Ort::Value::CreateTensor<float>(memoryInfo, cellprob_threshold_values.data(), 1, cellprob_thresholdShape.data(), cellprob_thresholdShape.size()));
   
    std::vector<int64_t> niter_values_int64 = {niter};
    std::vector<int64_t> niterShape = {1};
    inputTensors.push_back(Ort::Value::CreateTensor<int64_t>(memoryInfo, niter_values_int64.data(), 1, niterShape.data(), niterShape.size()));

    if(terminating){
      preprocessingEnd();
      cv::Mat m1, m2;
      return std::make_tuple(m1, m2);
    }
    runOptionsEncoder.UnsetTerminate();
    std::vector<const char*> inputNames = getInputNamesCP(sessionEncoder);
    std::vector<const char*> outputNames = getOutputNamesCP(sessionEncoder);
    auto outputTensors =  sessionEncoder->Run(runOptionsEncoder, inputNames.data(), inputTensors.data(), inputTensors.size(), outputNames.data(), outputNames.size());
    for (size_t i = 0; i < inputNames.size(); ++i) {
      delete [] inputNames[i];
    }
    for (size_t i = 0; i < outputNames.size(); ++i) {
      delete [] outputNames[i];
    }

    auto maskValues = outputTensors[0].GetTensorMutableData<int64_t>();
    auto maskShape = outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();
    maskVector.assign(maskValues, maskValues + maskShape[0] * maskShape[1]);

    auto flowErrorsValues = outputTensors[1].GetTensorMutableData<float>();
    auto flowErrorsShape = outputTensors[1].GetTensorTypeAndShapeInfo().GetShape();
    flowErrorsVector.assign(flowErrorsValues, flowErrorsValues + flowErrorsShape[0]);

    auto rgbOfFlowsValues = outputTensors[2].GetTensorMutableData<float>();
    auto rgbOfFlowsShape = outputTensors[2].GetTensorTypeAndShapeInfo().GetShape();
    cv::Mat rgbOfFlows = cv::Mat((int)rgbOfFlowsShape[0], (int)rgbOfFlowsShape[1], CV_8UC3, rgbOfFlowsValues);

    cv::Mat mask = changeFlowThreshold(flow_threshold, min_size);

    preprocessingEnd();
    return std::make_tuple(mask, rgbOfFlows);
  }catch(Ort::Exception& e){
    std::cout << e.what() << std::endl;
    preprocessingEnd();
    cv::Mat m1, m2;
    return std::make_tuple(m1, m2);
  }
}

cv::Mat Cyto3::changeFlowThreshold(float flow_threshold, int min_size){
  std::vector<unsigned short> maskVectorCopy(maskVector);
  cv::Mat mask = cv::Mat((int)inputShapeEncoder[2], (int)inputShapeEncoder[3], CV_16U, maskVectorCopy.data());
  int label = 0;
  for (size_t k = 0; k < flowErrorsVector.size(); k++) {
    if(flowErrorsVector[k] > flow_threshold){
      mask.setTo(0, mask == (k + 1));
      continue;
    }
    cv::Mat maskk = cv::Mat((int)inputShapeEncoder[2], (int)inputShapeEncoder[3], CV_16U, cv::Scalar(0));
    maskk.setTo(1, mask == (k + 1));
    cv::Mat Points;
    cv::findNonZero(maskk, Points);
    if(!Points.data){
      continue;
    }
    cv::Rect Min_Rect = cv::boundingRect(Points);
    cv::Mat roi_tmp = maskk(Min_Rect);
    cv::Mat roi = roi_tmp.clone();
    int npix = (int)Points.total();
    if(npix < min_size){
      mask.setTo(0, mask == (k + 1));
      continue;
    }
    fill_voids::binary_fill_holes<unsigned short>((unsigned short *)roi.data, roi.cols, roi.rows);
    for(int i = 0; i < roi.rows; i++){
      for(int j = 0; j < roi.cols; j++){
        unsigned short value = roi.at<unsigned short>(i, j);
        if(value > 0){
          mask.at<unsigned short>(i + Min_Rect.y, j + Min_Rect.x) = label + 1;
        }
      }
    }
    label++;
  }
  return mask.clone();
}

void Cyto3::preprocessingStart(){
  preprocessing = true;
}

void Cyto3::preprocessingEnd(){
  preprocessing = false;
  terminating = false;
}

void HSVtoRGB(float& fR, float& fG, float& fB, float& fH, float& fS, float& fV) {
  float fC = fV * fS; // Chroma
  float fHPrime = fmod(fH / 60.0, 6);
  float fX = fC * (1 - fabs(fmod(fHPrime, 2) - 1));
  float fM = fV - fC;
  if(0 <= fHPrime && fHPrime < 1) {
    fR = fC;
    fG = fX;
    fB = 0;
  } else if(1 <= fHPrime && fHPrime < 2) {
    fR = fX;
    fG = fC;
    fB = 0;
  } else if(2 <= fHPrime && fHPrime < 3) {
    fR = 0;
    fG = fC;
    fB = fX;
  } else if(3 <= fHPrime && fHPrime < 4) {
    fR = 0;
    fG = fX;
    fB = fC;
  } else if(4 <= fHPrime && fHPrime < 5) {
    fR = fX;
    fG = 0;
    fB = fC;
  } else if(5 <= fHPrime && fHPrime < 6) {
    fR = fC;
    fG = 0;
    fB = fX;
  } else {
    fR = 0;
    fG = 0;
    fB = 0;
  }
  fR += fM;
  fG += fM;
  fB += fM;
}

void saveOutputMask(cv::Mat mask, cv::Size imageSize, std::string path){
  double minVal;
  double maxVal;
  cv::Point minLoc;
  cv::Point maxLoc;
  minMaxLoc(mask, &minVal, &maxVal, &minLoc, &maxLoc);
  int labels_num = maxVal;
  std::cout<<"max = " <<labels_num<<std::endl;
  cv::Mat outputMask = cv::Mat(mask.rows, mask.cols, CV_8UC3, cv::Scalar(0, 0, 0));
  for (int i = 0; i < labels_num; i++) {
    float fR = 0, fG = 0, fB = 0, fH = 0, fS = 0, fV = 0;
    fH = 360.0 * i / labels_num;
    fS = 0.5;
    fV = 0.5;
    HSVtoRGB(fR, fG, fB, fH, fS, fV);
    outputMask.setTo(cv::Scalar(fR * 255, fG * 255, fB * 255), mask == (i + 1));
  }
  cv::resize(outputMask, outputMask, imageSize);
  cv::imwrite(path, outputMask);
}










































