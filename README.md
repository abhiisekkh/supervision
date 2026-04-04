# Crowd Flow Analysis Tool

Real-time person detection, tracking, and movement analysis for crowd monitoring and flow prediction.

## 🎯 Features

**Person Detection & Tracking:**
- YOLOv8 small model for robust person detection
- ByteTrack for persistent ID tracking across frames
- Detects small people, children, and partial bodies (confidence threshold: 0.35)

**Movement Visualization:**
- Arrow vectors showing direction of movement
- Speed-based color coding:
  - 🔵 Blue: Stationary/Slow (0-3 px/frame)
  - 🟢 Green: Normal walking (3-8 px/frame)
  - 🟡 Yellow: Fast walking (8-15 px/frame)
  - 🔴 Red: Running (15+ px/frame)

**Zone Analysis:**
- Configurable grid zones (2x2, 3x3, 4x4, etc)
- Real-time people count per zone
- Congestion status detection (LOW/MEDIUM/HIGH/CRITICAL)

**Data Export:**
- `flow_vectors.csv` with complete tracking data
- MP4 video with all annotations
- Organized output folders per input video

## 🚀 Quick Start

### Install
```bash
cd /Users/abhisekh/Desktop/supervision
uv sync
```

### Run
```bash
uv run python detect_people.py
```

Select your video file. Output saved in `./video_name/` folder.

### Output
```
video_name/
├── video_name_result.mp4  (annotated video)
└── flow_vectors.csv       (ML-ready tracking data)
```

## ⌨️ Controls
- **P** - Pause/Resume
- **Q** - Quit

## 📊 CSV Export (LSTM-Ready)

```csv
frame,tracker_id,cx,cy,arrow_dx,arrow_dy,speed,direction_degrees,zone_number,congestion_status
0,1,960,540,-50,20,5.38,158.2,5,LOW
```

Perfect for Phase 2 machine learning models!

---

## Supervision Library Documentation

This project uses [Roboflow Supervision](https://supervision.roboflow.com).

## 💻 Install Supervision

Pip install the supervision package in a
[**Python>=3.9**](https://www.python.org/) environment.

```bash
pip install supervision
```

## 🔥 Quickstart

### models

Supervision was designed to be model agnostic. Just plug in any classification, detection, or segmentation model. For your convenience, we have created [connectors](https://supervision.roboflow.com/latest/detection/core/#detections) for the most popular libraries like Ultralytics, Transformers, MMDetection, or Inference. Other integrations, like `rfdetr`, already return `sv.Detections` directly.

Install the optional dependencies for this example with `pip install pillow rfdetr`.

```python
import supervision as sv
from PIL import Image
from rfdetr import RFDETRSmall

image = Image.open(...)
model = RFDETRSmall()
detections = model.predict(image, threshold=0.5)

len(detections)
# 5
```

<details>
<summary>👉 more model connectors</summary>

- inference

    Running with [Inference](https://github.com/roboflow/inference) requires a [Roboflow API KEY](https://docs.roboflow.com/api-reference/authentication#retrieve-an-api-key).

    ```python
    import supervision as sv
    from PIL import Image
    from inference import get_model

    image = Image.open(...)
    model = get_model(model_id="rfdetr-small", api_key="ROBOFLOW_API_KEY")
    result = model.infer(image)[0]
    detections = sv.Detections.from_inference(result)

    len(detections)
    # 5
    ```

</details>

### annotators

Supervision offers a wide range of highly customizable [annotators](https://supervision.roboflow.com/latest/detection/annotators/), allowing you to compose the perfect visualization for your use case.

```python
import cv2
import supervision as sv

image = cv2.imread(...)
detections = sv.Detections(...)

box_annotator = sv.BoxAnnotator()
annotated_frame = box_annotator.annotate(scene=image.copy(), detections=detections)
```

https://github.com/roboflow/supervision/assets/26109316/691e219c-0565-4403-9218-ab5644f39bce

### datasets

Supervision provides a set of [utils](https://supervision.roboflow.com/latest/datasets/core/) that allow you to load, split, merge, and save datasets in one of the supported formats.

```python
import supervision as sv
from roboflow import Roboflow

project = Roboflow().workspace("WORKSPACE_ID").project("PROJECT_ID")
dataset = project.version("PROJECT_VERSION").download("coco")

ds = sv.DetectionDataset.from_coco(
    images_directory_path=f"{dataset.location}/train",
    annotations_path=f"{dataset.location}/train/_annotations.coco.json",
)

path, image, annotation = ds[0]
# loads image on demand

for path, image, annotation in ds:
    # loads image on demand
    pass
```

<details close>
<summary>👉 more dataset utils</summary>

- load

    ```python
    dataset = sv.DetectionDataset.from_yolo(
        images_directory_path=...,
        annotations_directory_path=...,
        data_yaml_path=...,
    )

    dataset = sv.DetectionDataset.from_pascal_voc(
        images_directory_path=...,
        annotations_directory_path=...,
    )

    dataset = sv.DetectionDataset.from_coco(
        images_directory_path=...,
        annotations_path=...,
    )
    ```

- split

    ```python
    train_dataset, test_dataset = dataset.split(split_ratio=0.7)
    test_dataset, valid_dataset = test_dataset.split(split_ratio=0.5)

    len(train_dataset), len(test_dataset), len(valid_dataset)
    # (700, 150, 150)
    ```

- merge

    ```python
    ds_1 = sv.DetectionDataset(...)
    len(ds_1)
    # 100
    ds_1.classes
    # ['dog', 'person']

    ds_2 = sv.DetectionDataset(...)
    len(ds_2)
    # 200
    ds_2.classes
    # ['cat']

    ds_merged = sv.DetectionDataset.merge([ds_1, ds_2])
    len(ds_merged)
    # 300
    ds_merged.classes
    # ['cat', 'dog', 'person']
    ```

- save

    ```python
    dataset.as_yolo(
        images_directory_path=...,
        annotations_directory_path=...,
        data_yaml_path=...,
    )

    dataset.as_pascal_voc(
        images_directory_path=...,
        annotations_directory_path=...,
    )

    dataset.as_coco(
        images_directory_path=...,
        annotations_path=...,
    )
    ```

- convert

    ```python
    sv.DetectionDataset.from_yolo(
        images_directory_path=...,
        annotations_directory_path=...,
        data_yaml_path=...,
    ).as_pascal_voc(
        images_directory_path=...,
        annotations_directory_path=...,
    )
    ```

</details>

## 🎬 tutorials

Want to learn how to use Supervision? Explore our [how-to guides](https://supervision.roboflow.com/develop/how_to/detect_and_annotate/), [end-to-end examples](./examples), [cheatsheet](https://roboflow.github.io/cheatsheet-supervision/), and [cookbooks](https://supervision.roboflow.com/develop/cookbooks/)!

<br/>

<p align="left">
<a href="https://youtu.be/hAWpsIuem10" title="Dwell Time Analysis with Computer Vision | Real-Time Stream Processing"><img src="https://github.com/SkalskiP/SkalskiP/assets/26109316/a742823d-c158-407d-b30f-063a5d11b4e1" alt="Dwell Time Analysis with Computer Vision | Real-Time Stream Processing" width="300px" align="left" /></a>
<a href="https://youtu.be/hAWpsIuem10" title="Dwell Time Analysis with Computer Vision | Real-Time Stream Processing"><strong>Dwell Time Analysis with Computer Vision | Real-Time Stream Processing</strong></a>
<div><strong>Created: 5 Apr 2024</strong></div>
<br/>Learn how to use computer vision to analyze wait times and optimize processes. This tutorial covers object detection, tracking, and calculating time spent in designated zones. Use these techniques to improve customer experience in retail, traffic management, or other scenarios.</p>

<br/>

<p align="left">
<a href="https://youtu.be/uWP6UjDeZvY" title="Speed Estimation & Vehicle Tracking | Computer Vision | Open Source"><img src="https://github.com/SkalskiP/SkalskiP/assets/26109316/61a444c8-b135-48ce-b979-2a5ab47c5a91" alt="Speed Estimation & Vehicle Tracking | Computer Vision | Open Source" width="300px" align="left" /></a>
<a href="https://youtu.be/uWP6UjDeZvY" title="Speed Estimation & Vehicle Tracking | Computer Vision | Open Source"><strong>Speed Estimation & Vehicle Tracking | Computer Vision | Open Source</strong></a>
<div><strong>Created: 11 Jan 2024</strong></div>
<br/>Learn how to track and estimate the speed of vehicles using YOLO, ByteTrack, and Roboflow Inference. This comprehensive tutorial covers object detection, multi-object tracking, filtering detections, perspective transformation, speed estimation, visualization improvements, and more.</p>

## 💜 built with supervision

Did you build something cool using supervision? [Let us know!](https://github.com/roboflow/supervision/discussions/categories/built-with-supervision)

https://user-images.githubusercontent.com/26109316/207858600-ee862b22-0353-440b-ad85-caa0c4777904.mp4

https://github.com/roboflow/supervision/assets/26109316/c9436828-9fbf-4c25-ae8c-60e9c81b3900

https://github.com/roboflow/supervision/assets/26109316/3ac6982f-4943-4108-9b7f-51787ef1a69f

## 📚 documentation

Visit our [documentation](https://roboflow.github.io/supervision) page to learn how supervision can help you build computer vision applications faster and more reliably.

## 🏆 contribution

We love your input! Please see our [contributing guide](.github/CONTRIBUTING.md) to get started. Thank you 🙏 to all our contributors!

<p align="center">
    <a href="https://github.com/roboflow/supervision/graphs/contributors">
      <img src="https://contrib.rocks/image?repo=roboflow/supervision" />
    </a>
</p>

<br>

<div align="center">

<div align="center">
      <a href="https://youtube.com/roboflow">
          <img
            src="https://media.roboflow.com/notebooks/template/icons/purple/youtube.png?ik-sdk-version=javascript-1.4.3&updatedAt=1672949634652"
            width="3%"
          />
      </a>
      <img src="https://raw.githubusercontent.com/ultralytics/assets/main/social/logo-transparent.png" width="3%"/>
      <a href="https://roboflow.com">
          <img
            src="https://media.roboflow.com/notebooks/template/icons/purple/roboflow-app.png?ik-sdk-version=javascript-1.4.3&updatedAt=1672949746649"
            width="3%"
          />
      </a>
      <img src="https://raw.githubusercontent.com/ultralytics/assets/main/social/logo-transparent.png" width="3%"/>
      <a href="https://www.linkedin.com/company/roboflow-ai/">
          <img
            src="https://media.roboflow.com/notebooks/template/icons/purple/linkedin.png?ik-sdk-version=javascript-1.4.3&updatedAt=1672949633691"
            width="3%"
          />
      </a>
      <img src="https://raw.githubusercontent.com/ultralytics/assets/main/social/logo-transparent.png" width="3%"/>
      <a href="https://docs.roboflow.com">
          <img
            src="https://media.roboflow.com/notebooks/template/icons/purple/knowledge.png?ik-sdk-version=javascript-1.4.3&updatedAt=1672949634511"
            width="3%"
          />
      </a>
      <img src="https://raw.githubusercontent.com/ultralytics/assets/main/social/logo-transparent.png" width="3%"/>
      <a href="https://discuss.roboflow.com">
          <img
            src="https://media.roboflow.com/notebooks/template/icons/purple/forum.png?ik-sdk-version=javascript-1.4.3&updatedAt=1672949633584"
            width="3%"
          />
      <img src="https://raw.githubusercontent.com/ultralytics/assets/main/social/logo-transparent.png" width="3%"/>
      <a href="https://blog.roboflow.com">
          <img
            src="https://media.roboflow.com/notebooks/template/icons/purple/blog.png?ik-sdk-version=javascript-1.4.3&updatedAt=1672949633605"
            width="3%"
          />
      </a>
      </a>
  </div>
</div>
