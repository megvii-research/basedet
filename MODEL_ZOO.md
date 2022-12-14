# Model Zoo

##  ImageNet Pretrained Models

<details open>
<summary>weights used to finetune</summary>

|Model name  | weights |
| ------     | :----:  |
| ResNet-50  | [github](https://github.com/megvii-research/basedet/releases/download/0.3.0rc/resnet50_fbaug_633cb650.pkl)  |
| ResNet-101 | [github](https://github.com/megvii-research/basedet/releases/download/0.3.0rc/resnet101_fbaug_77944b79.pkl) |
| DarkNet-53 | [github](https://github.com/megvii-research/basedet/releases/download/0.3.0rc/darknet53.pkl) |

</details>

For more pretrained backbone weights, please check @[basecls](https://basecls.readthedocs.io/zh_CN/latest/zoo/index.html).

## COCO Baselines

<details open>
<summary>model zoo</summary>

|     Model name       | input size | lr sched | box mAP | weights |
| ----------------     |   :----:   |  :----:  | :----:  | :----:  |
| Faster R-CNN R50-FPN |    800     |  1x(12e) |  37.7   | [github](https://github.com/megvii-research/basedet/releases/download/0.3.0rc/faster_rcnn_r50_fpn_1x.pkl)  |
|  RetinaNet R50-FPN   |    800     |  1x(12e) |  36.2   | [github](https://github.com/megvii-research/basedet/releases/download/0.3.0rc/retinanet_r50_fpn_1x.pkl)    |
|  FreeAnchor R50-FPN  |    800     |  1x(12e) |  38.4   | [github](https://github.com/megvii-research/basedet/releases/download/0.3.0rc/free_anchor_r50_fpn_1x.pkl)  |
|     FCOS R50-FPN     |    800     |  1x(12e) |  39.0   | [github](https://github.com/megvii-research/basedet/releases/download/0.3.0rc/fcos_r50_fpn_1x.pkl)         |
|     ATSS R50-FPN     |    800     |  1x(12e) |  39.5   | [github](https://github.com/megvii-research/basedet/releases/download/0.3.0rc/atss_r50_fpn_1x.pkl)  |
|      OTA R50-FPN     |    800     |  1x(12e) |  41.0   | [github](https://github.com/megvii-research/basedet/releases/download/0.3.0rc/ota_r50_fpn_1x.pkl)  |
|       DETR R50       |    800     |  150e    |  39.9   | [github](https://github.com/megvii-research/basedet/releases/download/0.3.0rc/detr_r50_150e.pkl)  |

</details>
