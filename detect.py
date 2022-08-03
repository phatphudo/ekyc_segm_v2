"""
Usage:
    python detect.py --img path/to/img --mode {inference_mode}
    {inference_mode}:
        save_segm: save segmented/ROI image to output directory
        save_pred: save segmented/ROI image including detections on image to output directory
        get_segm: return the segmented/ROI image as numpy array and predicted class if model is multi
        eval: return predictions and segmented/ROI image including detections for evaluation
"""

import sys, os
import argparse

from entities import Detector
from tool import Config


def main(opt):
    img_path = opt.img
    inf_mode = opt.mode

    config = Config.load_yaml('config.yml')
    
    detector = Detector(config)
    results = detector.inf_single_image(img_path, inf_mode)
    

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', type=str, help='path to image')
    parser.add_argument('--mode', type=str, default='save_segm', help='inference mode')
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)