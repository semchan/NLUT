# NLUT: Neural-based 3D Lookup Tables for Video Photorealistic Style Transfer

## Overview 
NLUT(see [our paper](https://arxiv.org/pdf/2303.09170) and [project page](https://semchan.github.io/NLUT_Project/) )is a super fast photorealistic style transfer method for video. We build a neural network to generate a stylized 3D LUT. The goal is to realize fast photorealistic style transfer for video. Specifically, we train the neural network that produces 3D LUT on a large dataset and then fine-tune it in test-time training to generate a stylized 3D LUT of a specific style image and video content. Although our method needs fine-tuning when used, it is more effective than other methods and is super fast in video processing. For example, it can process 8K video in less than 2 milliseconds. In the future, we will explore ways to generate 3D LUTs in arbitrary styles even more quickly.
<div align=center><img height="400" src="./teaser.png"/></div>


## Preparation 

### Enviroment
Please ensure that you have correctly configured the following environment and you can quickly install the required environment through the following command. 

	pip install -r requirements.txt

- matplotlib==3.5.1
- numpy==1.22.4
- opencv_python==4.5.5.62
- Pillow==9.4.0
- plotly==5.13.0
- scipy==1.7.3
- setuptools==58.0.4
- torch==1.10.1
- torchvision==0.11.2
- tqdm==4.62.3

The fast deployment of 3D LUT relies on the CUDA implementation of trilinear interpolation in [Image-Adaptive-3DLUT](https://github.com/HuiZeng/Image-Adaptive-3DLUT).
To install their **trilinear** library: 

    cd trilinear_cpp
    sh setup.sh

	

### data
Training dataset.

You can download the training dataset through the link below

- [Microsoft COCO](https://cocodataset.org/#download) 

pre-trained checkpoint: link：https://pan.baidu.com/s/1VddHbq2cBy5RcKOp8S5eSg
extraction code：1234 

## training
All the appropriate hyper-parameters have been set as default,Only the content_path and style_path needs to be modified before training.

You can train with the following commands	

	python train.py --content_dir <path> --style_dir <path>
## test

We have set the appropriate hyper-parameters as the default，Only the content_path and style_path needs to be modified before testing.

generate stylized image



	python inference_finetuning_image.py --content_path <path> --style_path <path> --output_path <path>

generate stylized video



	python inference_finetuning_video.py --content_path <path> --style_path <path> --src_video <path> --dst_video <path>

## License
This algorithm is licensed under the MIT License.See the [LICENSE](LICENSE) file for details.
