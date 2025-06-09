## Cellpose CPP Wrapper for macOS and Ubuntu GPU
This code is to run a [Cellpose](https://github.com/MouseLand/cellpose) ONNX model in c++ code and will be implemented on the macOS app [RectLabel](https://rectlabel.com).

![img00](https://github.com/user-attachments/assets/7c78cb97-12fe-41b5-80d7-bd2e12179313)

Install [CUDA, cuDNN, PyTorch, LibTorch, and ONNX Runtime](https://rectlabel.com/pytorch/).

Install [Cellpose](https://github.com/MouseLand/cellpose/tree/v3.1.1).
```bash
git clone https://github.com/MouseLand/cellpose.git --branch v3.1.1
cd cellpose
python -m pip install 'cellpose[gui]'
```

Install Cellpose CPP.
```bash
git clone https://github.com/ryouchinsa/cellpose-cpp.git
cp cellpose-cpp/cyto3_onnx.py .
cp cellpose-cpp/resnet_torch.py cellpose

wget https://huggingface.co/rectlabel/cellpose/resolve/main/demo_images.zip
unzip demo_images.zip
cp -r demo_images cellpose-cpp/
```

Export an ONNX model and check how the ONNX model works.

```bash
python cyto3_onnx.py --mode export
python cyto3_onnx.py --mode import --device cuda:0
cp cyto3.onnx cellpose-cpp/
cd cellpose-cpp
```

Download an exported Cellpose cyto3 ONNX model.
- [Cellpose cyto3](https://huggingface.co/rectlabel/cellpose/resolve/main/cyto3.onnx.zip)

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
