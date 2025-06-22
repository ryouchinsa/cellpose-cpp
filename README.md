## Cellpose CPP Wrapper for macOS and Ubuntu GPU
This code is to run a [Cellpose](https://github.com/MouseLand/cellpose) ONNX model in c++ code and will be implemented on the macOS app [RectLabel](https://rectlabel.com).

![cellpose](https://github.com/user-attachments/assets/d19d935c-b26b-416a-9eb6-5e43348d1ac3)

Install [CUDA, cuDNN, PyTorch, LibTorch, and ONNX Runtime](https://rectlabel.com/pytorch/).

Install Cellpose3 and Cellpose4.
```bash
git clone https://github.com/MouseLand/cellpose.git cellpose3 --branch v3.1.1
git clone https://github.com/MouseLand/cellpose.git cellpose4
git clone https://github.com/ryouchinsa/cellpose-cpp.git
cp cellpose-cpp/cyto3_onnx.py cellpose3
cp cellpose-cpp/resnet_torch.py cellpose3/cellpose
cp cellpose-cpp/cpsam_onnx.py cellpose4
cd cellpose4
python -m pip install 'cellpose[gui]'
cd ..
wget https://huggingface.co/rectlabel/cellpose/resolve/main/demo_images.zip
unzip demo_images.zip
```

Export each ONNX model and check how the ONNX model works.

```bash
cd cellpose3
python cyto3_onnx.py --mode export --image ../demo_images/img02.png --device cuda:0
python cyto3_onnx.py --mode import --image ../demo_images/img00.png --device cuda:0
cd ..
cd cellpose4
python cpsam_onnx.py --mode export --image ../demo_images/img02.png --device cuda:0
python cpsam_onnx.py --mode import --image ../demo_images/img00.png --device cuda:0
cd ..
cd cellpose-cpp
```

Build and run.

```bash
# macOS
cmake -S . -B build -DONNXRUNTIME_ROOT_DIR=/Users/ryo/Downloads/onnxruntime-osx-universal2-1.20.0
# Ubuntu GPU
cmake -S . -B build -DONNXRUNTIME_ROOT_DIR=/root/onnxruntime-linux-x64-gpu-1.20.0 -DCMAKE_PREFIX_PATH=/root/libtorch

cmake --build build

# macOS
./build/cyto3_cpp_test -encoder="../cellpose3/cyto3.onnx" -image="../demo_images/img00.png" -device="cpu"
./build/cyto3_cpp_test -encoder="../cellpose4/cpsam.onnx" -image="../demo_images/img00.png" -device="cpu"
# Ubuntu GPU
./build/cyto3_cpp_test -encoder="../cellpose3/cyto3.onnx" -image="../demo_images/img00.png" -device="cuda:0"
./build/cyto3_cpp_test -encoder="../cellpose4/cpsam.onnx" -image="../demo_images/img00.png" -device="cuda:0"
```

| Model | Size | Time on Apple M1 | Time on g4dn.xlarge |
| :---: | :---: | :---: | :---: |
| cyto3.onnx <br>([download](https://huggingface.co/rectlabel/cellpose/resolve/main/cyto3.onnx.zip)) | 27.5MB | 5s | 2s |
| cpsam.onnx <br>([download](https://huggingface.co/rectlabel/cellpose/resolve/main/cpsam.onnx.zip)) | 1.22GB | 40s | 5s |

