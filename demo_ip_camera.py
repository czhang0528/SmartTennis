#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

import os
from timeit import time
import warnings
import sys
import cv2
import numpy as np
from PIL import Image
from src.yolo import YOLO
import colorsys
import argparse
# from src.demo_image import POSE
from src.tf_pose.estimator import TfPoseEstimator
from src.tf_pose.networks import get_graph_path, model_wh
import logging

warnings.filterwarnings('ignore')


def arg_parse():
    parser = argparse.ArgumentParser(description='openpose Video Detection Module')

    parser.add_argument("--video", dest='video', help="Video to run detection upon",
                        default="/home/ftp/DataSet/Hand-Dataset/video/tennis.mp4", type=str)
    parser.add_argument("--url", dest='url', help="Video to run in url",
                        default="rtsp://admin:password@192.168.1.104:554/11", type=str)
    parser.add_argument('--yolo-model', dest='model1', type=str, default='model/keras/model.h5',
                        help='path to the weights file')
    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin')
    parser.add_argument('--resize', type=str, default='656x368',
                        help='network input resolution. default=432x368,656x368,1312x736')
    parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')
    return parser.parse_args()


def main():
    # Generate colors for drawing bounding boxes.
    hsv_tuples = [(x / 80, 1., 1.)
                  for x in range(80)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(
        map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
            colors))
    # tracker, encoder, nms_max_overlap = deep_sort()
    classes = load_classes('model/coco_classes.txt')
    writeVideo_flag = False  # True
    args = arg_parse()
    #videofile = args.video
    url_flie = args.url
    print('start load')
    w, h = model_wh(args.resize)
    if w > 0 and h > 0:
        print('11111111111')
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
    else:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368))
    print('loaded openpose!')

    yolo = YOLO()
    video_capture = cv2.VideoCapture(url_flie)
    assert video_capture.isOpened(), 'Cannot capture source'

    if writeVideo_flag:
        # Define the codec and create VideoWriter object
        wh = int(video_capture.get(3))
        hi = int(video_capture.get(4))
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        outname = os.path.join('result_video', 'result.mp4')
        out = cv2.VideoWriter(outname, fourcc, 25, (wh, hi))
        list_file = open('detection.txt', 'w')
        frame_index = -1

    fps = 0.0
    fps_time = 0
    # athlete_flag = [0,0]
    result_frame = 0
    result_save = ''
    Distance = ''
    Spead = ''
    while video_capture.isOpened():
        ret, frame = video_capture.read()  # frame shape 640*480*3
        frame = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_LINEAR)  # (720,1280,3)
        b = 2
        if ret != True:
            break
        # t1 = time.time()

        image = Image.fromarray(frame)
        boxs = yolo.detect_image_track(image)
        boxs = np.array(boxs)

        flag_boxs = 0
        for bbox in boxs:
            label = "{0}".format(classes[bbox[-1]])
            c1 = (int(bbox[0]), int(bbox[1]))
            c2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            if label == 'person':
                if c1[0] > int(256 / b) and c1[1] > int(290 / b) and c2[0] < int(1238 / b) and c2[1] < int(
                        704 / b):  # c1[1]>int(95/b)
                    label = 'athlete'
                    if c1[0] > int(141 / b) and c1[1] > int(367 / b) and c2[0] < int(1238 / b) and c2[1] < int(704 / b):
                        # print(str(c1)+str(c2))
                        humans = e.inference(frame[c1[1] - 15:c2[1] + 15, c1[0] - 40:c2[0] + 30, :],
                                             resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
                        frame[c1[1] - 15:c2[1] + 15, c1[0] - 40:c2[0] + 30,
                        :], result_txt, mDistance, mSpead = TfPoseEstimator.draw_humans(
                            frame[c1[1] - 15:c2[1] + 15, c1[0] - 40:c2[0] + 30, :], humans, imgcopy=False)
                        if not mDistance == '':
                            Distance = mDistance
                        if not mSpead == '':
                            Spead = mSpead
                        if not result_txt == '':
                            result_save = result_txt
                            print('==============================result:' + result_save)
                            result_frame = 0
                        cv2.rectangle(frame, (c1[0] - 40, c1[1] - 15), (c2[0] + 30, c2[1] + 15), (0, 255, 0), 1)
                elif c1[0] > int(261 / b) and c1[1] > int(20 / b) and c2[0] < int(1012 / b) and c2[1] < int(162 / b):
                    label = 'baseline_referee'
                elif c1[0] > 1 and c1[1] > int(95 / b) and c2[0] < int(175 / b) and c2[1] < int(413 / b):
                    label = 'sideline_referee'
                elif c1[0] > int(1002 / b) and c1[1] > int(128 / b) and c2[0] < int(1181 / b) and c2[1] < int(387 / b):
                    label = 'referee'
                elif c1[0] > int(1002 / b) and c1[1] > int(245 / b) and c2[0] < int(1170 / b) and c2[1] < int(387 / b):
                    label = 'Caddy'
                elif c1[0] > int(166 / b) and c1[1] > int(232 / b) and c2[0] < int(295 / b) and c2[1] < int(347 / b):
                    label = 'Caddy'

            color = colors[int(bbox[-1])]
            cv2.rectangle(frame, c1, c2, color, 2)
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 2)[0]
            name = label
            c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
            cv2.rectangle(frame, c1, c2, color, -1)
            cv2.putText(frame, name, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1);
            flag_boxs += 1
        color1 = [0, 255, 0]
        if result_frame < 10:
            cv2.putText(frame, 'Action:' + result_save, (int(80 / b), int(100 / b)), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        color1, 2)
        else:
            cv2.putText(frame, 'Action:' + result_save, (int(80 / b), int(100 / b)), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        color1, 2)
        cv2.putText(frame, 'Distance:' + Distance + 'm', (int(80 / b), int(100 / b) + 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, color1, 2)
        cv2.putText(frame, 'Spead:' + Spead + 'm/s', (int(80 / b), int(100 / b) + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    color1, 2)
        result_frame += 1
        cv2.putText(frame, "FPS: %.2f" % (1.0 / (time.time() - fps_time)), (int(80 / b), int(100 / b) + 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color1, 2)
        cv2.imshow('', frame)
        fps_time = time.time()
        if writeVideo_flag:
            # save a frame
            out.write(frame)
            frame_index = frame_index + 1
            list_file.write(str(frame_index) + ' ')
            if len(boxs) != 0:
                for i in range(0, len(boxs)):
                    list_file.write(
                        str(boxs[i][0]) + ' ' + str(boxs[i][1]) + ' ' + str(boxs[i][2]) + ' ' + str(boxs[i][3]) + ' ')
            list_file.write('\n')

        # Press Q to stop!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    if writeVideo_flag:
        out.release()
        list_file.close()
    cv2.destroyAllWindows()


def load_classes(namesfile):
    fp = open(namesfile, "r")
    names = fp.read().split("\n")[:-1]
    return names


if __name__ == '__main__':
    main()
