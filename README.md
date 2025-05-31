# cellpose-cpp
[Cellpose](https://github.com/MouseLand/cellpose) CPP Wrapper for macOS and Ubuntu CPU/GPU.

![img00](https://github.com/user-attachments/assets/7c78cb97-12fe-41b5-80d7-bd2e12179313)

You can download an exported Cellpose cyte3 ONNX model.
- [cyto3](https://huggingface.co/rectlabel/cellpose/resolve/main/cyto3.onnx.zip)

Download an ONNX Runtime folder.
- [onnxruntime-osx-universal2-1.20.0.tgz](https://github.com/microsoft/onnxruntime/releases/download/v1.20.0/onnxruntime-osx-universal2-1.20.0.tgz) for macOS
- [onnxruntime-linux-x64-1.20.0.tgz](https://github.com/microsoft/onnxruntime/releases/download/v1.20.0/onnxruntime-linux-x64-1.20.0.tgz) for Ubuntu CPU
- [onnxruntime-linux-x64-gpu-1.20.0.tgz](https://github.com/microsoft/onnxruntime/releases/download/v1.20.0/onnxruntime-linux-x64-gpu-1.20.0.tgz) for Ubuntu GPU

For Ubuntu, install gflags and opencv through [vcpkg](https://github.com/microsoft/vcpkg).
```bash
git clone https://github.com/microsoft/vcpkg.git
./vcpkg/bootstrap-vcpkg.sh
./vcpkg/vcpkg install gflags
./vcpkg/vcpkg install opencv
```

For Ubuntu GPU, install cuda and cudnn.
```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install cuda-drivers
reboot
nvidia-smi

sudo apt install cuda-toolkit-11-8
vi ~/.bashrc
export PATH="/usr/local/cuda/bin${PATH:+:${PATH}}"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
source ~/.bashrc
which nvcc
nvcc --version

apt list libcudnn8 -a
cudnn_version=8.9.7.29
cuda_version=cuda11.8
sudo apt install libcudnn8=${cudnn_version}-1+${cuda_version}
sudo apt install libcudnn8-dev=${cudnn_version}-1+${cuda_version}
sudo apt install libcudnn8-samples=${cudnn_version}-1+${cuda_version}
```

Build and run.

```bash
# macOS
cmake -S . -B build -DONNXRUNTIME_ROOT_DIR=/Users/ryo/Downloads/onnxruntime-osx-universal2-1.20.0
# Ubuntu CPU
cmake -S . -B build -DONNXRUNTIME_ROOT_DIR=/root/onnxruntime-linux-x64-1.20.0 -DCMAKE_TOOLCHAIN_FILE=/root/vcpkg/scripts/buildsystems/vcpkg.cmake
# Ubuntu GPU
cmake -S . -B build -DONNXRUNTIME_ROOT_DIR=/root/onnxruntime-linux-x64-gpu-1.20.0 -DCMAKE_TOOLCHAIN_FILE=/root/vcpkg/scripts/buildsystems/vcpkg.cmake

cmake --build build

# macOS and Ubuntu CPU
./build/cyto3_cpp_test -encoder="cyto3.onnx" -image="demo_images/img02.png" -device="cpu"

# Ubuntu GPU
./build/cyto3_cpp_test -encoder="cyto3.onnx" -image="demo_images/img02.png" -device="cuda:0"
```
