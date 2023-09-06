from abc import ABC
from abc import abstractmethod

import cv2
import numpy as np
from PIL import Image


DATETIME_TAG = 36867  # Datetime Original tag for EXIF
DATETIME_FMT = '%Y-%m-%d %H:%M:%S.%f'


class Camera(ABC):
    @abstractmethod
    def capture(self):
        pass

    @abstractmethod
    def save(self, fname, dt):
        pass


class Lepton:
    def __init__(self, q, ignore=False):
        self.size = (120, 160)
        self.q = q
        self.ignore = ignore
        self.frame = np.zeros(self.size, dtype=np.uint8)
        self.capture()  # Check if camera initialized

    def capture(self):
        if self.ignore:
            return self.frame
        im = self.q.get(timeout=10)
        self.frame = im
        return self.frame
    
    def save(self, fname, dt):
        if self.ignore:
            return
        arr = np.array(self.frame).astype(np.uint16)
        pil = Image.fromarray(arr)
        exif = pil.getexif()
        exif.update({DATETIME_TAG: dt.strftime(DATETIME_FMT)})
        pil.save(fname, format='png', exif=exif)
        return 1


class ArduCam:
    def __init__(self, ignore=False):
        self.ignore = ignore
        self.size = (1080, 1920, 3)
        self.cam = None
        self.frame = np.zeros(self.size, dtype=np.uint8)
        for i in range(3):
            cam = cv2.VideoCapture(i)
            ret, frame = cam.read()
            if ret and frame.shape == self.size:
                self.cam = cam
                break
            cam.release()
        if self.cam is None:
            raise IOError('ArduCam not found!')

    def capture(self):
        if self.ignore:
            return self.frame
        _, im = self.cam.read()
        self.frame = im
        return self.frame

    def save(self, fname, dt):
        if self.ignore:
            return
        cv2.imwrite(fname, self.frame)
        return 1
