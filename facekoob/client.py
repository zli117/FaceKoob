"""The client for face recognition
"""
import argparse
import logging
import socket
import ssl
import os
from threading import Thread

import numpy as np

import cv2
import dlib
from numpy_socket import NumpySocket
from preprocess import *


class Client:
    def __init__(self, registered_dir, server_cert, server_hostname,
                 server_port, face_prediction, logger):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.ssl_sock = ssl.wrap_socket(
            s, ca_certs=server_cert, cert_reqs=ssl.CERT_REQUIRED)
        self.ssl_sock.connect((server_hostname, server_port))
        self.numpy_sock = NumpySocket(self.ssl_sock)
        identities = os.listdir(registered_dir)
        self.identity_map = {}
        for identity in identities:
            cur_dir = os.path.join(registered_dir, identity)
            images = os.listdir(cur_dir)
            embeddings = []
            for image in images:
                img_numpy = dlib.load_rgb_image(os.path.join(cur_dir, image))
                self.numpy_sock.send_numpy(img_numpy)
                embedding = np.array(self.numpy_sock.receive_numpy())
                embeddings.append(embedding)
            if len(embeddings) > 0:
                total_embedding = embeddings[0]
                for i in range(len(embeddings)):
                    total_embedding += embeddings[i]
                ave_embedding = total_embedding / len(embeddings)
                self.identity_map[identity] = ave_embedding
        self.prediction_path = face_prediction
        self.logger = logger

    def start(self):
        camera = cv2.VideoCapture(0)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1240)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
        aligner = CropAndAlign(self.prediction_path, INNER_EYES_AND_BOTTOM_LIP,
                               logger)
        while camera.isOpened():
            _, image = camera.read()
            faces = aligner.align_all_faces(image, 72)
            logger.info('Found %d faces' % len(faces))
            k = cv2.waitKey(1) & 0xff
            if k == ord('q'):
                break
            for face in faces:
                self.numpy_sock.send_numpy(face)
                embedding = np.array(self.numpy_sock.receive_numpy())
                found = False
                for identity in self.identity_map:
                    distance = np.linalg.norm(embedding -
                                              self.identity_map[identity])
                    if distance < 0.8:
                        self.logger.info('found %s' % identity)
                        found = True
                        break
                if not found:
                    self.logger.info('Not found')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    parser = argparse.ArgumentParser('Dataset preprocessor')
    parser.add_argument('--register_dir', type=str, required=True)
    parser.add_argument('--server_cert', type=str, required=True)
    parser.add_argument('--server_hostname', type=str, required=True)
    parser.add_argument('--server_port', type=int, required=True)
    parser.add_argument('--prediction_path', type=str, required=True)

    if len(sys.argv) == 1:
        parser.print_help()

    args = parser.parse_args()

    client = Client(args.register_dir, args.server_cert, args.server_hostname,
                    args.server_port, args.prediction_path, logger)
    client.start()
