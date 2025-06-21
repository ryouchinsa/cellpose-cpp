#include "cyto3.h"
#include <opencv2/opencv.hpp>
using namespace torch::indexing;

Cyto3::Cyto3(){}
Cyto3::~Cyto3(){
  if(loadingModel){
    return;
  }
  if(preprocessing){
    return;
  }
  clearLoadModel();
}

bool Cyto3::clearLoadModel(){
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

void Cyto3::terminatePreprocessing(){
  runOptionsEncoder.SetTerminate();
  terminating = true;
}

bool modelExists(const std::string& modelPath){
  std::ifstream f(modelPath);
  if (!f.good()) {
    return false;
  }
  return true;
}

bool Cyto3::loadModel(const std::string& encoderPath, int threadsNumber, std::string device){
  try{
    loadingStart();
    if(!clearLoadModel()){
      loadingEnd();
      return false;
    }
    if(!modelExists(encoderPath)){
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
        option.AppendExecutionProvider_CUDA(options);
      }
    }
    sessionEncoder = std::make_unique<Ort::Session>(env, encoderPath.c_str(), sessionOptions[0]);
    inputShapeEncoder = sessionEncoder->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
  }catch(Ort::Exception& e){
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

std::vector<const char*> getInputNames(std::unique_ptr<Ort::Session> &session){
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

std::vector<const char*> getOutputNames(std::unique_ptr<Ort::Session> &session){
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

int getLabelsNum(torch::Tensor mask){
  auto m = torch::max(mask);
  int labels_num = m.item<int>();
  return labels_num;
}

torch::Tensor findObjects(torch::Tensor mask){
  int labels_num = getLabelsNum(mask);
  auto slices = torch::zeros({labels_num, 4}, torch::kInt);
  for (int i = 1; i < labels_num + 1; i++) {
    auto mask_i = torch::where(mask == i);
    auto ymin = torch::min(mask_i[0]);
    auto ymax = torch::max(mask_i[0]);
    auto xmin = torch::min(mask_i[1]);
    auto xmax = torch::max(mask_i[1]);
    slices[i - 1][0] = ymin;
    slices[i - 1][1] = ymax;
    slices[i - 1][2] = xmin;
    slices[i - 1][3] = xmax;
  }
  return slices;
}

torch::Tensor removeBadFlowMasks(torch::Tensor mask, torch::Tensor flowErrors, float flow_threshold){
  auto flowErrors_index = torch::where(flowErrors > flow_threshold);
  for (int i = 0; i < flowErrors_index[0].size(0); i++) {
    int label = flowErrors_index[0][i].item<int>() + 1;
    auto mask_index = torch::where(mask == label);
    mask.index_put_({mask_index[0], mask_index[1]}, 0);
  }
  auto tensors = at::_unique(mask, true, true);
  auto inverse_indices = std::get<1>(tensors);
  return inverse_indices;
}

void fillHolesAndRemoveSmallMasks(torch::Tensor mask, int min_size){
  auto slices = findObjects(mask);
  int j = 0;
  int labels_num = getLabelsNum(mask);
  for (int i = 0; i < labels_num; i++) {
    int ymin = slices[i][0].item<int>();
    int ymax = slices[i][1].item<int>() + 1;
    int xmin = slices[i][2].item<int>();
    int xmax = slices[i][3].item<int>() + 1;
    auto msk = mask.index({Slice(ymin, ymax), Slice(xmin, xmax)});
    // msk.index_put_({Slice(msk.size(0) * 0.425, msk.size(0) * 0.575), Slice(msk.size(1) * 0.425, msk.size(1) * 0.575)}, 0);
    auto msk_index = torch::where(msk == (i + 1));
    int npix = msk_index[0].size(0);
    if(npix < min_size){
      msk.index_put_({msk_index[0], msk_index[1]}, 0);
    }else{
      auto msk_i_plus_1 = torch::zeros({msk.size(0), msk.size(1)}, torch::kInt);
      msk_i_plus_1.index_put_({msk_index[0], msk_index[1]}, 1);
      msk_i_plus_1 = fill_voids(msk_i_plus_1);
      msk_index = torch::where(msk_i_plus_1 > 0);
      msk.index_put_({msk_index[0], msk_index[1]}, j + 1);
      j++;
    }
    mask.index_put_({Slice(ymin, ymax), Slice(xmin, xmax)}, msk);
  }
}

torch::Tensor postProcess(torch::Tensor mask, torch::Tensor flowErrors, float flow_threshold, int min_size){
  torch::set_num_threads(1);
  mask = removeBadFlowMasks(mask, flowErrors, flow_threshold);
  fillHolesAndRemoveSmallMasks(mask, min_size);
  return mask;
}

torch::Tensor Cyto3::changeFlowThreshold(float flow_threshold, int min_size){
  torch::Tensor mask = torch::from_blob(maskVector.data(), {inputShapeEncoder[2], inputShapeEncoder[3]}, torch::kInt64);
  torch::Tensor flowErrors = torch::from_blob(flowErrorsVector.data(), {(int64_t)flowErrorsVector.size()}, torch::kFloat64);
  return postProcess(mask, flowErrors, flow_threshold, min_size);
}

std::tuple<torch::Tensor , torch::Tensor > Cyto3::preprocessImage(const cv::Mat& image, const cv::Size &imageSize, const std::vector<int64_t> &channels, int diameter, int niter, float flow_threshold, int min_size){
  try{
    preprocessingStart();
    if(image.size() != cv::Size((int)inputShapeEncoder[3], (int)inputShapeEncoder[2])){
      preprocessingEnd();
      return std::make_tuple(torch::zeros(0), torch::zeros(0));
    }
    if(image.channels() != 3){
      preprocessingEnd();
      return std::make_tuple(torch::zeros(0), torch::zeros(0));
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
    
    std::vector<int64_t> orig_im_size_values_int64 = {imageSize.height, imageSize.width};
    std::vector<int64_t> origImSizeShape = {2};
    inputTensors.push_back(Ort::Value::CreateTensor<int64_t>(memoryInfo, orig_im_size_values_int64.data(), 2, origImSizeShape.data(), origImSizeShape.size()));
    
    std::vector<int64_t> channels_values_int64 = {channels[0], channels[1]};
    std::vector<int64_t> channelsShape = {2};
    inputTensors.push_back(Ort::Value::CreateTensor<int64_t>(memoryInfo, channels_values_int64.data(), 2, channelsShape.data(), channelsShape.size()));
    
    std::vector<int64_t> diameter_values_int64 = {diameter};
    std::vector<int64_t> diameterShape = {1};
    inputTensors.push_back(Ort::Value::CreateTensor<int64_t>(memoryInfo, diameter_values_int64.data(), 1, diameterShape.data(), diameterShape.size()));
    
    std::vector<int64_t> niter_values_int64 = {niter};
    std::vector<int64_t> niterShape = {1};
    inputTensors.push_back(Ort::Value::CreateTensor<int64_t>(memoryInfo, niter_values_int64.data(), 1, niterShape.data(), niterShape.size()));
   
    if(terminating){
      preprocessingEnd();
      return std::make_tuple(torch::zeros(0), torch::zeros(0));
    }
    runOptionsEncoder.UnsetTerminate();
    std::vector<const char*> inputNames = getInputNames(sessionEncoder);
    std::vector<const char*> outputNames = getOutputNames(sessionEncoder);
    auto outputTensors =  sessionEncoder->Run(runOptionsEncoder, inputNames.data(), inputTensors.data(), inputTensors.size(), outputNames.data(), outputNames.size());
    for (size_t i = 0; i < inputNames.size(); ++i) {
      delete [] inputNames[i];
    }
    for (size_t i = 0; i < outputNames.size(); ++i) {
      delete [] outputNames[i];
    }

    auto maskValues = outputTensors[0].GetTensorMutableData<int64_t>();
    auto maskShape = outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();
    torch::Tensor mask = torch::from_blob(maskValues, {maskShape[0], maskShape[1]}, torch::kInt64);
    maskVector.assign(mask.data_ptr<int64_t>(), mask.data_ptr<int64_t>() + mask.numel());

    auto flowErrorsValues = outputTensors[1].GetTensorMutableData<double>();
    auto flowErrorsShape = outputTensors[1].GetTensorTypeAndShapeInfo().GetShape();
    torch::Tensor flowErrors = torch::from_blob(flowErrorsValues, {flowErrorsShape[0]}, torch::kFloat64);
    flowErrorsVector.assign(flowErrors.data_ptr<double>(), flowErrors.data_ptr<double>() + flowErrors.numel());

    auto rgbOfFlowsValues = outputTensors[2].GetTensorMutableData<double>();
    auto rgbOfFlowsShape = outputTensors[2].GetTensorTypeAndShapeInfo().GetShape();
    torch::Tensor rgbOfFlows = torch::from_blob(rgbOfFlowsValues, {rgbOfFlowsShape[0], rgbOfFlowsShape[1], rgbOfFlowsShape[2]}, torch::kU8);

    mask = postProcess(mask, flowErrors, flow_threshold, min_size);
    preprocessingEnd();
    return std::make_tuple(mask, rgbOfFlows);
  }catch(Ort::Exception& e){
    std::cout << e.what() << std::endl;
    preprocessingEnd();
    return std::make_tuple(torch::zeros(0), torch::zeros(0));
  }
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

void saveOutputMask(torch::Tensor mask, cv::Size imageSize, float flow_threshold, int min_size){
  int labels_num = getLabelsNum(mask);
  std::cout<<labels_num<<std::endl;
  torch::Tensor maskRGB = torch::zeros({mask.size(0), mask.size(1), 3}, torch::kU8);
  for (int i = 0; i < labels_num; i++) {
    auto msk_index = torch::where(mask == (i + 1));
    float fR = 0, fG = 0, fB = 0, fH = 0, fS = 0, fV = 0;
    fH = 360.0 * i / labels_num;
    fS = 0.5;
    fV = 0.5;
    HSVtoRGB(fR, fG, fB, fH, fS, fV);
    maskRGB.index_put_({msk_index[0], msk_index[1], 0}, fR * 255);
    maskRGB.index_put_({msk_index[0], msk_index[1], 1}, fG * 255);
    maskRGB.index_put_({msk_index[0], msk_index[1], 2}, fB * 255);
  }
  cv::Mat outputMask = cv::Mat(mask.size(0), mask.size(1), CV_8UC3, maskRGB.data_ptr());
  std::stringstream ss;
  ss << std::fixed << std::setprecision(2) << flow_threshold;  
  std::string fileName = "outputMask" + ss.str() + "_" + std::to_string(min_size);
  cv::imwrite(fileName + ".png", outputMask);
  cv::resize(outputMask, outputMask, imageSize);
  cv::imwrite(fileName + "_original.png", outputMask);
}

void saveRGBOfFlows(torch::Tensor rgbOfFlows, cv::Size imageSize){
  cv::Mat outputMask = cv::Mat(rgbOfFlows.size(0), rgbOfFlows.size(1), CV_8UC3, rgbOfFlows.data_ptr());
  std::string fileName = "rgbOfFlows";
  cv::imwrite(fileName + ".jpg", outputMask);
  cv::resize(outputMask, outputMask, imageSize);
  cv::imwrite(fileName + "_original.jpg", outputMask);
}
























