## Cellpose CPP Wrapper for macOS and Ubuntu GPU
This code is to run a [Cellpose](https://github.com/MouseLand/cellpose) ONNX model in c++ code and will be implemented on the macOS app [RectLabel](https://rectlabel.com).

![img00](https://github.com/user-attachments/assets/7c78cb97-12fe-41b5-80d7-bd2e12179313)

Install [Cellpose](https://github.com/MouseLand/cellpose/tree/v3.1.1).
```bash
git clone https://github.com/MouseLand/cellpose.git --branch v3.1.1
cd cellpose
python -m pip install 'cellpose[gui]'
```

Put [cyto3_onnx.py](https://github.com/ryouchinsa/cellpose-cpp/blob/master/cyto3_onnx.py) and [demo_images](https://huggingface.co/rectlabel/cellpose/resolve/main/demo_images.zip) to the root folder.

![cellpose](https://github.com/user-attachments/assets/6a6cbb75-a190-48cc-9d4e-546c79c0aae9)

To export an ONNX model.

```bash
python cyto3_onnx.py --mode export
```

To check how the ONNX model works.

```bash
python cyto3_onnx.py --mode import
```

Download an exported Cellpose cyte3 ONNX model.
- [Cellpose cyte3](https://huggingface.co/rectlabel/cellpose/resolve/main/cyto3.onnx.zip)

Download ONNX Runtime.
- [onnxruntime-osx-universal2-1.20.0.tgz](https://github.com/microsoft/onnxruntime/releases/download/v1.20.0/onnxruntime-osx-universal2-1.20.0.tgz) for macOS
- [onnxruntime-linux-x64-gpu-1.20.0.tgz](https://github.com/microsoft/onnxruntime/releases/download/v1.20.0/onnxruntime-linux-x64-gpu-1.20.0.tgz) for Ubuntu GPU

![cellpose_cpp](https://github.com/user-attachments/assets/3a93316d-d205-45c5-8ebe-eb2e0cbf09ef)

For Ubuntu GPU, install packages including gflags and opencv.
```bash
sudo apt-get update
sudo apt-get install build-essential tar curl zip unzip autopoint libtool bison libx11-dev libxft-dev libxext-dev libxrandr-dev libxi-dev libxcursor-dev libxdamage-dev libxinerama-dev libxtst-dev cmake libgflags-dev libopencv-dev python3-dev
```

For Ubuntu GPU, install CUDA and cuDNN.
```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt install cuda-drivers
nvidia-smi

sudo apt install cuda-toolkit-12-8
vi ~/.bashrc
export PATH="/usr/local/cuda/bin${PATH:+:${PATH}}"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
source ~/.bashrc
nvcc --version

apt-cache search libcudnn
sudo apt install libcudnn9-cuda-12
sudo apt install libcudnn9-dev-cuda-12
```

For Ubuntu GPU, download LibTorch for C++.
```
https://docs.pytorch.org/cppdocs/installing.html
https://pytorch.org/get-started/locally/

# Ubuntu GPU
wget https://download.pytorch.org/libtorch/cu128/libtorch-cxx11-abi-shared-with-deps-2.7.0%2Bcu128.zip
unzip libtorch-cxx11-abi-shared-with-deps-2.7.0+cu128.zip
```

Build and run.

```bash
# macOS
cmake -S . -B build -DONNXRUNTIME_ROOT_DIR=/Users/ryo/Downloads/onnxruntime-osx-universal2-1.20.0
# Ubuntu GPU
cmake -S . -B build -DONNXRUNTIME_ROOT_DIR=/root/onnxruntime-linux-x64-gpu-1.20.0 -DCMAKE_PREFIX_PATH=/root/libtorch

cmake --build build

# macOS
./build/cyto3_cpp_test -encoder="cyto3.onnx" -image="demo_images/img02.png" -device="cpu"
# Ubuntu GPU
./build/cyto3_cpp_test -encoder="cyto3.onnx" -image="demo_images/img02.png" -device="cuda:0"
```
