"""Train the embedder
"""
import logging
import os
import random
import sys
import time

import numpy as np
import torch
import torch.multiprocessing as mp
from torch import optim

import cv2
from model import Embedder


class DataGenerator:
    def __init__(self, data_dir, batch_size, logger):
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
                    map(lambda img_name: os.path.join(person_dir, img_name),
                        faces))
                self.face_images.append(face_images)
        self.batch_size = batch_size
        self.logger = logger
        self.person_idx = 0
        self.image_idx = 0

    def __iter__(self):
        return self

    def __next__(self):
        anchors = []
        positives = []
        negatives = []

        batch_counter = 0
        while (batch_counter < self.batch_size) and len(self.face_images) > 1:
            if self.person_idx == 0:
                random.shuffle(self.face_images)
                self.person_idx += 1
            if self.image_idx == 0:
                random.shuffle(self.face_images[self.person_idx])
                self.image_idx = 1

            if len(self.face_images[self.person_idx]) > 1:
                anchors.append(
                    cv2.imread(self.face_images[self.person_idx][
                        self.image_idx]).astype(np.float))
                positives.append(
                    cv2.imread(self.face_images[self.person_idx][self.image_idx
                                                                 - 1]).astype(
                                                                     np.float))
                negative_idx = random.randint(
                    0,
                    len(self.face_images[self.person_idx - 1]) - 1)
                negatives.append(
                    cv2.imread(self.face_images[self.person_idx -
                                                1][negative_idx]).astype(
                                                    np.float))
                batch_counter += 1

            self.image_idx += 1
            if self.image_idx >= len(self.face_images[self.person_idx]):
                self.image_idx = 0
                self.person_idx += 1
                self.person_idx %= len(self.face_images)

        anchors = torch.Tensor(anchors)
        positives = torch.Tensor(positives)
        negatives = torch.Tensor(negatives)
        batch = torch.cat((anchors, positives, negatives), dim=0)
        batch = torch.transpose(batch, 1, 3)
        batch = torch.transpose(batch, 2, 3)
        return batch


class LoadingWorker(mp.Process):
    def __init__(self, data_dir, batch_size, queue):
        super().__init__()
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.generator = DataGenerator(data_dir, batch_size, self.logger)
        self.queue = queue
        self.exit = mp.Event()

    def run(self):
        while not self.exit.is_set():
            batch = self.generator.__next__()
            self.queue.put(batch)
        time.sleep(2)

    def terminate(self):
        self.logger.info('Shutting down loader')
        self.exit.set()
        while not self.queue.empty():
            self.queue.get()


def loss_fn(batch_size, embeddings, alpha=0.2):
    anchor = embeddings[:batch_size]
    positive = embeddings[batch_size:2 * batch_size]
    negative = embeddings[2 * batch_size:]
    loss = (
        (torch.norm(anchor - positive)**2 - torch.norm(anchor - negative)**2) /
        batch_size + alpha)
    return loss


def train(data_dir,
          batch_size,
          total_iter,
          kernel_sizes,
          logger,
          model_save_path,
          input_shape=(72, 72),
          cuda=True):
    queue = mp.Queue(10)
    process = LoadingWorker(data_dir, batch_size, queue)
    process.start()
    model = Embedder(input_shape, kernel_sizes)
    if cuda:
        model = model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=5e-6)
    for i in range(total_iter):
        optimizer.zero_grad()
        images = queue.get()
        if cuda:
            images = images.cuda()
        embeddings = model(images)
        loss = loss_fn(batch_size, embeddings)
        loss.backward()
        optimizer.step()
        if i % 50 == 0:
            logger.info('loss at step %04d is: %f' % (i, loss))
    process.terminate()
    process.join()
    torch.save(model.cpu().state_dict(), model_save_path)


if __name__ == '__main__':
    DATA_DIR = 'data/processed'
    BATCH_SIZE = 4
    TOTAL_ITER = 200
    KERNEL_SIZES = [5, 3, 5, 3, 3, 3, 3, 3]
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    train(DATA_DIR, BATCH_SIZE, TOTAL_ITER, KERNEL_SIZES, logger,
          'model-weights')
