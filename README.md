# SOLOv2 ONNX Converter (Based on MMDETECTION)

Converter of SOLOv2 ([paper](https://arxiv.org/abs/2003.10152)) instance segmentation model based on [`mmdetection`](https://github.com/open-mmlab/mmdetection) codebase's model.

## Environment Setup

Before begin, setup your virtual environment, either using `conda` or `virtualenv`. I used `virtualenv` with Python 3.8. First, install basic dependencies

```
pip install -r requirements.txt
```

Then install PyTorch. I tested this project with torch v1.12.1. Older and newer version may also be compatible. Check also [PyTorch installation guide](https://pytorch.org/get-started/locally/).

```
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
```

Install `mmcv` and `mmdet`. You can check the [official documentation](https://github.com/open-mmlab/mmdetection/blob/master/docs/en/get_started.md/#Installation).

```
pip install -U openmim
mim install mmcv-full==1.6.1
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
git checkout v2.25.1
pip install -v -e .
cd ..
```

## Download Models

Create a folder named `checkpoints` inside this directory. Put the models checkpoint that you want to use inside it. Please check [this page](https://github.com/open-mmlab/mmdetection/tree/master/configs/solov2) to get the pre-trained checkpoints, or you can also use your own model that was trained using `mmdetection`.

## Usage

To convert a SOLOv2 model to ONNX run `convert.py`.

```
python convert.py --cfg path/to/model/config.py --ckpt path/to/model/checkpoint.pth --img path/to/test/image.jpg --out path/to/output.onnx
```

If you are using pre-trained model, normally the config path is like `mmdetection/configs/solov2/[$your_model_config]` and you can use the test image in `mmdetection\demo\demo.jpg`. See [this page](https://github.com/open-mmlab/mmdetection/tree/master/configs/solov2) again to see the list of the available config files.

## Known issue(s)

- The ONNX model cannot be used with CPU Execution Provider of ONNXRuntime. The following error will be returned

```txt
onnxruntime.capi.onnxruntime_pybind11_state.NotImplemented: [ONNXRuntimeError] : 9 : NOT_IMPLEMENTED : Could not find an implementation for Trilu(14) node with name 'Trilu_1625'
```
