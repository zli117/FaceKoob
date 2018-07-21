"""Capture images for registration
"""
import argparse
import logging
import os
import sys

import numpy as np

import cv2
from preprocess import INNER_EYES_AND_BOTTOM_LIP, CropAndAlign


def capture_images(out_path, identity, face_predictor_path, output_dim, logger):
    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1240)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    aligner = CropAndAlign(face_predictor_path, INNER_EYES_AND_BOTTOM_LIP,
                           logger)
    counter = 0
    dir = os.path.join(out_path, identity)
    if not os.path.exists(dir):
        os.makedirs(dir)
    else:
        logger.warning('Identity %s already exists' % identity)
    while camera.isOpened():
        _, image = camera.read()
        logger.debug(image.shape == (480, 640, 3))
        faces = aligner.find_all_bounding_boxes(image)
        k = cv2.waitKey(30) & 0xff
        if k == ord('q'):  # press 'q' to quit
            break
        elif k == ord('c'):
            aligned = aligner.align_biggest_face(image, output_dim)
            if aligned is not None:
                path = os.path.join(dir, '%02d.jpg' % counter)
                cv2.imwrite(path, aligned)
                logger.info('Saving to %s' % path)
                counter += 1

        if len(faces) > 0:
            landmarks = aligner.find_landmarks(image, faces[0])
            for i in range(landmarks.shape[0]):
                cv2.circle(image, (landmarks[i, 0], landmarks[i, 1]), 1,
                           (0, 255, 0))

        for face in faces:
            l, t, r, b = face.left(), face.top(), face.right(), face.bottom()
            cv2.rectangle(image, (l, t), (r, b), (255, 0, 0), 2)
        cv2.imshow('video', image)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    parser = argparse.ArgumentParser('Registering face')
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--identity', type=str, required=True)
    parser.add_argument('--face_predictor', type=str, required=True)
    parser.add_argument('--output_dim', type=int, default=72)

    if len(sys.argv) == 1:
        parser.print_help()

    args = parser.parse_args()

    capture_images(args.output_dir, args.identity, args.face_predictor,
                   args.output_dim, logger)
