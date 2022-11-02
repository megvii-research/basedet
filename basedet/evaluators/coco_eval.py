#!/usr/bin/python3

import json
import os
import random
import cv2
import megfile
import numpy as np
from loguru import logger
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import megengine.functional as F

from basecore.engine import BaseEvaluator

from basedet.data.build import build_transform, get_images_dir_and_anno_path
from basedet.layers import is_empty_tensor
from basedet.utils import ensure_dir, redirect_to_loguru, registers

__all__ = ["COCOEvaluator"]


def visualize_detection(
    img, dets, is_show_label=True, classes=None, thresh=0.3, name="detection", return_img=True,
):
    img = np.array(img)
    colors = dict()
    font = cv2.FONT_HERSHEY_SIMPLEX

    for det in dets:
        bb = det[:4].astype(int)
        if is_show_label:
            cls_id = int(det[5])
            score = det[4]

            if cls_id == 0:
                continue
            if score > thresh:
                if cls_id not in colors:
                    colors[cls_id] = (
                        random.random() * 255,
                        random.random() * 255,
                        random.random() * 255,
                    )

                cv2.rectangle(img, (bb[0], bb[1]), (bb[2], bb[3]), colors[cls_id], 3)

                if classes and len(classes) > cls_id:
                    cls_name = classes[cls_id]
                else:
                    cls_name = str(cls_id)
                cv2.putText(
                    img, "{:s} {:.3f}".format(cls_name, score), (bb[0], bb[1] - 2),
                    font, 0.5, (255, 255, 255), 1,
                )
        else:
            cv2.rectangle(img, (bb[0], bb[1]), (bb[2], bb[3]), (0, 0, 255), 2)

    if return_img:
        return img
    cv2.imshow(name, img)
    while True:
        c = cv2.waitKey(100000)
        if c == ord("d"):
            return None
        elif c == ord("n"):
            break


@registers.evalutors.register()
class COCOEvaluator(BaseEvaluator):
    """
    Evalutor of COCO dataset.
    """
    def __init__(self, config):
        super().__init__()
        self.cfg = config
        self.tta = build_transform(self.cfg, mode="test")

    def preprocess(self, input_data):
        image, im_info = self.tta(input_data[0][0])
        return {"data": image, "im_info": im_info}

    def postprocess(self, model_outputs, input_data=None):
        """
        Args:
            model_outputs: outputs of evaluated model.
            input_data: inputs of evaluated model.

        Returns:
            results boxes: detection model output
        """
        # TODO thinking of instance strures here.
        image_id = int(input_data[1][2][0].split(".")[0].split("_")[-1])

        if is_empty_tensor(model_outputs.boxes):
            return {
                "det_res": np.array([], dtype=np.float),
                "image_id": image_id,
            }
        else:
            scores = model_outputs.box_scores.reshape(-1, 1)
            labels = model_outputs.box_labels.reshape(-1, 1)
            eval_boxes = F.concat([model_outputs.boxes, scores, labels], axis=1)
            return {
                # since tensor.numpy() is read only ndarray, using numpy to make it writeable.
                "det_res": np.array(eval_boxes.numpy(), dtype=np.float),
                "image_id": image_id,
            }

    @staticmethod
    def format(results, cfg):
        dataset_full_name = cfg.DATA.TEST.name
        dataset_name = dataset_full_name.split("_")[0].upper()
        dataset_class_name = registers.datasets_info.get(dataset_name)["dataset_type"]
        dataset_class = registers.datasets.get(dataset_class_name)

        all_results = []
        for record in results:
            image_filename = record["image_id"]
            boxes = record["det_res"]
            if len(boxes) <= 0:
                continue
            boxes[:, 2:4] = boxes[:, 2:4] - boxes[:, 0:2]
            for box in boxes:
                elem = dict()
                elem["image_id"] = image_filename
                elem["bbox"] = box[:4].tolist()
                elem["score"] = box[4]
                if hasattr(dataset_class, "classes_originID"):
                    elem["category_id"] = dataset_class.classes_originID[
                        dataset_class.class_names[int(box[5])]
                    ]
                else:
                    elem["category_id"] = int(box[5]) + 1
                all_results.append(elem)
        return all_results

    def save_results(self, results_list):
        filename = os.path.join(self.cfg.GLOBAL.OUTPUT_DIR, "predict_coco.json")

        all_results = self.format(results_list, self.cfg)
        all_results = json.dumps(all_results, indent=4)

        ensure_dir(os.path.dirname(filename))
        with open(filename, "w") as fo:
            fo.write(all_results)

        logger.info("Save results to {}.".format(filename))
        return filename

    def evaluate(self, results):
        """
        evaluate provided results and log results.
        """
        assert isinstance(results, str)
        filename = results

        logger.info("Start evaluation...")
        _, anno_path = get_images_dir_and_anno_path(self.cfg.DATA.TEST.name)
        if anno_path.startswith("s3"):
            cache_file = megfile.smart_cache(anno_path)
            anno_path = cache_file.cache_path
        eval_gt = COCO(anno_path)

        eval_dt = eval_gt.loadRes(filename)
        with redirect_to_loguru():
            cocoEval = COCOeval(eval_gt, eval_dt, iouType="bbox")
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()
