"""Train the embedder
"""
import random

import torch
import cv2
import os


def loss_fn(anchor_emb, positive_emb, negative_emb, alpha=0.2):
    batch_size = anchor_emb.shape[0]
    loss = (torch.norm(anchor_emb - positive_emb)**2 -
            torch.norm(anchor_emb - negative_emb)**2) / batch_size + alpha
    return loss


class DataGenerator:
    def __init__(self, data_dir, batch_size, logger, gpu=True):
        self.data_dir = data_dir
        self.face_images = []
        if not os.path.exists(data_dir):
            logger.error('Data dir %s doesn\'t exist' % data_dir)
            return
        people = os.listdir(data_dir)
        for person in people:
            person_dir = os.path.join(data_dir, person)
            faces = os.listdir(person_dir)
            if len(faces) > 0:
                face_images = list(
                    map(faces,
                        lambda img_name: os.path.join(person_dir, img_name)))
                self.face_images.append(face_images)
        self.batch_size = batch_size
        self.logger = logger
        self.gpu = gpu
        self.person_idx = 0
        self.image_idx = 0

    def __iter__(self):
        return self

    def __next__(self):
        anchors = []
        positives = []
        negatives = []

        batch_counter = 0
        while batch_counter < self.batch_size:
            if self.person_idx == 0:
                random.shuffle(self.face_images)
            if self.image_idx == 0:
                random.shuffle(self.face_images[self.person_idx])
                self.image_idx = 1

            if len(self.face_images[self.person_idx]) > 1:
                anchors.append()

            batch_counter += 1
            self.image_idx += 1
            if self.image_idx >= len(self.face_images[self.person_idx]):
                self.image_idx = 0
                self.person_idx += 1
                self.person_idx %= len(self.face_images)
