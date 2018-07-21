"""Utility class for transporting numpy arrays through socket
"""
import socket
import struct
import pickle

import numpy as np

class NumpySocket:
    def __init__(self, sock):
        self.sock = sock

    def send_numpy(self, array):
        byte_array = pickle.dumps(array)
        byte_array = struct.pack('>I', len(byte_array)) + byte_array
        self.sock.sendall(byte_array)

    def receive_numpy(self):
        data_len = self.receive_helper(4)
        if data_len is None:
            return None
        length = struct.unpack('>I', data_len)
        dump = self.receive_helper(length)
        if dump is None:
            return None
        return pickle.loads(dump)

    def receive_helper(self, n):
        byte_array = b''
        while len(byte_array) < n:
            packet = self.sock.recv(n - len(byte_array))
            if packet is None:
                return None
            byte_array += packet
        return byte_array
