import os
import argparse
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from collections import defaultdict

def transform_bbox(bbox, bbox_format):
    if bbox_format == 'xyxy':
        return [bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]]
    return bbox

def parse_file(filepath, bbox_format, is_pred=False):
    with open(filepath, 'r') as f:
        data = f.readlines()
        annotations = []
        for line in data:
            elems = line.split(' ')
            class_name = elems[0]
            bbox = transform_bbox(list(map(float, elems[2:] if is_pred else elems[1:])), bbox_format)
            if is_pred:  # if it's a prediction file
                score = float(elems[1])
                annotations.append({'class_name': class_name, 'score': score, 'bbox': bbox})
            else:  # for ground truth file
                annotations.append({'class_name': class_name, 'bbox': bbox})
    return annotations

def calculate_metrics(pred_dir, gt_dir, bbox_format):
    image_id = 0
    ann_id = 0
    class_name_to_id = {}

    gt_anns = []
    pred_anns = []
    images = []
    categories = []

    gt_files = [f for f in os.listdir(gt_dir) if f.endswith('.txt')]
    pred_files = [f for f in os.listdir(pred_dir) if f.endswith('.txt')]

    assert len(gt_files) == len(pred_files), "The number of ground truth and prediction files should be same."
    assert set(gt_files) == set(pred_files), "The names of ground truth and prediction files should match."

    for file_name in gt_files:
        gt_annotations = parse_file(os.path.join(gt_dir, file_name), bbox_format, is_pred=False)
        pred_annotations = parse_file(os.path.join(pred_dir, file_name), bbox_format, is_pred=True)

        for gt in gt_annotations:
            if gt['class_name'] not in class_name_to_id:
                class_name_to_id[gt['class_name']] = len(class_name_to_id) + 1
                categories.append({"id": class_name_to_id[gt['class_name']], "name": gt['class_name']})

            gt_anns.append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": class_name_to_id[gt['class_name']],
                "bbox": gt['bbox'],
                "iscrowd": 0,
                "area": gt['bbox'][2] * gt['bbox'][3]
            })
            ann_id += 1

        for dt in pred_annotations:
            if dt['class_name'] not in class_name_to_id:
                class_name_to_id[dt['class_name']] = len(class_name_to_id) + 1
                categories.append({"id": class_name_to_id[dt['class_name']], "name": dt['class_name']})

            pred_anns.append({
                "image_id": image_id,
                "category_id": class_name_to_id[dt['class_name']],
                "bbox": dt['bbox'],
                "score": dt['score']
            })

        images.append({
            "id": image_id,
            "width": 0,
            "height": 0
        })

        image_id += 1

    coco_gt = COCO()
    coco_gt.dataset = {"annotations": gt_anns, "images": images, "categories": categories}
    coco_gt.createIndex()

    coco_dt = coco_gt.loadRes(pred_anns)

    cocoEval = COCOeval(cocoGt=coco_gt, cocoDt=coco_dt, iouType='bbox')
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()


    # Calculate mAP for each class:
    for key, value in class_name_to_id.items():
        print("---------------")
        print("mAP FOR CLASS {}".format(key))

        gt_class_anns = [anno for anno in gt_anns if anno["category_id"] == value]
        pred_class_anns = [anno for anno in pred_anns if anno["category_id"] == value]
        coco_gt = COCO()
        coco_gt.dataset = {"annotations": gt_class_anns, "images": images, "categories": categories}
        coco_gt.createIndex()

        coco_dt = coco_gt.loadRes(pred_class_anns)

        cocoEval = COCOeval(cocoGt=coco_gt, cocoDt=coco_dt, iouType='bbox')
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate COCO Metrics')
    parser.add_argument('pred_dir', type=str, help='Directory path for prediction files')
    parser.add_argument('gt_dir', type=str, help='Directory path for ground truth files')
    parser.add_argument('--bbox_format', type=str, default='xywh', choices=['xywh', 'xyxy'],
                        help='Input bounding box format. xywh is x_center, y_center, width, height. '
                        'xyxy is x_min, y_min, x_max, y_max. xyxy is x_left, y_top, x_right, y_bottom. '
                        'All in relative coordinates')

    args = parser.parse_args()
    calculate_metrics(args.pred_dir, args.gt_dir, args.bbox_format)
