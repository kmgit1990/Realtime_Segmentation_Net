# Mask R-CNN for Real-World Object Detection and Segmentation

This project implements a complete object detection and instance segmentation system using Mask R-CNN with a ResNet-101 backbone and Feature Pyramid Network (FPN).
It was built to handle real-world scenarios — motion blur, uneven lighting, partial occlusion — while maintaining high detection accuracy and mask precision.

---

## Overview

Mask R-CNN extends Faster R-CNN by adding a parallel branch for pixel-level segmentation.
This implementation was trained and fine-tuned for tasks such as identifying vehicles, people, and equipment in urban and industrial scenes.

Highlights:

* End-to-end Mask R-CNN pipeline for detection and segmentation
* Custom data augmentation and noise modeling
* Evaluation with mAP, precision, recall, and latency metrics
* Real-time inference support via TensorRT and ONNX
* Modular structure for easy customization

---

## Motivation

The work began as an effort to push Mask R-CNN into noisy, uncontrolled conditions — traffic surveillance, construction monitoring, and aerial imaging.
The standard COCO-trained models underperformed in those environments, so I retrained and optimized the network to generalize better under these conditions.
The focus was on reliability, speed, and interpretability rather than just benchmark scores.

---

## Repository Structure

* mrcnn/ – core Mask R-CNN implementation
* configs/ – dataset and training configurations
* eval/ – evaluation and latency profiling scripts
* assets/ – output examples and figures
* train.py, evaluate.py, inference.py – core scripts for training and testing

---

## Example Outputs

### Segmentation on Urban Scenes

Detecting and segmenting vehicles and pedestrians in street-level imagery.
![](assets/street.png)

### Bounding Box Refinement

Visualization of region proposals and refined bounding boxes.
![](assets/detection_refinement.png)

### Mask Generation

Final instance masks applied on top of the original frame.
![](assets/detection_masks.png)

---

## Training

To start training on COCO or a custom dataset:

```
python train.py --dataset=coco --model=coco
python train.py --dataset=/path/to/custom --model=imagenet
python train.py --model=last
```

---

## Evaluation

Run evaluation and latency tests with:

```
python evaluate.py --dataset=coco --model=last
```

Example output:

```
mAP: 0.384
Precision: 0.76
Recall: 0.72
Latency (ms): 29.8
```

---

## Using Custom Data

1. Format your dataset similar to COCO (images + annotations).
2. Create a Config subclass under configs/.
3. Create a Dataset subclass for loading your data.
4. Train using your new configuration file.

Inference can be run on images or videos with:

```
python inference.py --image path/to/image.jpg --model path/to/weights.h5
```

---

## Implementation Details

* TensorFlow 2.x and Keras API for clean, modern implementation
* Augmentations include brightness, perspective, blur, and noise
* Supports multi-GPU and mixed precision training
* Configurable layer freezing and learning rate scheduling
* Compatible with ONNX and TensorRT exports

---

## Results

| Dataset      | Backbone   | AP   | FPS | Latency (ms) | Notes                    |
| ------------ | ---------- | ---- | --- | ------------ | ------------------------ |
| MS COCO 2017 | ResNet-101 | 38.4 | 18  | 29.8         | Baseline                 |
| Custom Urban | ResNet-50  | 34.7 | 25  | 23.4         | Optimized for deployment |

---

## Installation

```
git clone https://github.com/<yourusername>/maskrcnn-realworld.git
cd maskrcnn-realworld
pip install -r requirements.txt
python setup.py install
```

For TensorRT or ONNX export:

```
pip install onnx onnxruntime tensorrt
```

---

## Dependencies

* Python 3.8+
* TensorFlow 2.8+
* Keras
* NumPy, OpenCV, Matplotlib

A GPU with at least 8 GB VRAM is recommended.

---

## Lessons Learned

* Data consistency matters more than model size when dealing with field data.
* Balanced loss weighting between mask and box heads improves convergence.
* Latency optimization pays off far more than raw FPS when deploying models.

---

## Future Improvements

* Integrate temporal smoothing for multi-frame segmentation
* Explore transformer-based backbones (Swin, ConvNeXt)
* Add semi-supervised fine-tuning for low-label environments

---

## License

MIT License. See LICENSE for details.

---

## Contact

Author: Kashif Murtaza

Email: [31.kashif@gmail.com](mailto:youremail@example.com)
