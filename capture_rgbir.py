# flake8: noqa

import argparse
import os
import platform
from datetime import datetime
from queue import Queue

import cv2
import imutils
import numpy as np
from ctypes import *
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

import utils
from cam import ArduCam
from cam import Lepton

try:
    if platform.system() == 'Darwin':
        print('here')
        libuvc = cdll.LoadLibrary('libuvc.dylib')
    elif platform.system() == 'Linux':
        libuvc = cdll.LoadLibrary('libuvc.so')
    else:
        libuvc = cdll.LoadLibrary('libuvc')
except OSError:
    print('Error: could not find libuvc!')
    exit(1)

from uvctypes import print_device_formats
from uvctypes import print_device_info
from uvctypes import PT_USB_PID
from uvctypes import PT_USB_VID
from uvctypes import uvc_context
from uvctypes import uvc_device
from uvctypes import uvc_device_handle
from uvctypes import uvc_frame
from uvctypes import UVC_FRAME_FORMAT_Y16
from uvctypes import uvc_get_frame_formats_by_guid
from uvctypes import uvc_stream_ctrl
from uvctypes import VS_FMT_GUID_Y16


BUF_SIZE = 2
q = Queue(BUF_SIZE)

DISPRATE = 1   # update/second
FRAMERATE = 4  # frames/second, max 8.7 for Lepton
FONT = ImageFont.truetype('IBMPlexMono-SemiBold.ttf', 10)
BLUE = (255, 150, 0)


def py_frame_callback(frame, ptr):
    array_pointer = cast(
        frame.contents.data,
        POINTER(c_uint16 * (frame.contents.width * frame.contents.height))
    )
    data = np.frombuffer(
        array_pointer.contents, dtype=np.dtype(np.uint16)
    ).reshape(
        frame.contents.height, frame.contents.width
    )

    if frame.contents.data_bytes != (2 * frame.contents.width * frame.contents.height):
        return

    if not q.full():
        q.put(data)


PTR_PY_FRAME_CALLBACK = CFUNCTYPE(None, POINTER(uvc_frame), c_void_p)(py_frame_callback)


def listdir(dir):
    return [f for f in os.listdir(dir) if not f.startswith('.')]


def mean_temp(raw):
    temps = utils.raw2temp(raw)
    return f'{int(temps.min())},{int(temps.max())}'


def display(raw, rgb, i, ignore_ir):
    total_secs = i // FRAMERATE

    # Convert to 8bit
    if ignore_ir:
        ir = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
        temp = 'NA'
    else:
        ir = utils.raw2viz(raw)
        ir = cv2.cvtColor(ir, cv2.COLOR_GRAY2BGR)
        ir = cv2.flip(ir, 1)
        temp = mean_temp(raw)

    msg = f'{total_secs // 60:01d}:{total_secs % 60:02d} \n {temp}'
    ir = Image.fromarray(ir)
    draw = ImageDraw.Draw(ir)
    draw.text((3, 0), msg, fill=BLUE, font=FONT)
    draw.rectangle((40, 10, 120, 110), outline=BLUE)

    rgb = imutils.resize(rgb, height=ir.size[1])
    rgb = cv2.flip(rgb, 1)
    return np.hstack((np.array(ir), rgb))


def capture_loop(save_dir, capture_time, cam_ir, cam_rgb, ignore_ir):
    for i in range(int(capture_time * FRAMERATE)):
        ir = cam_ir.capture()
        rgb = cam_rgb.capture()
        dt = datetime.now()

        # Save data
        cam_ir.save(os.path.join(save_dir, f'ir{i}.png'), dt)
        cam_rgb.save(os.path.join(save_dir, f'rgb{i}.jpg'), dt)
        
        # Display data
        if i % FRAMERATE % DISPRATE == 0:
            wname = 'IR / RGB'
            cv2.namedWindow(wname, cv2.WINDOW_KEEPRATIO)
            cv2.imshow(wname, display(ir, rgb, i, ignore_ir=ignore_ir))
            cv2.resizeWindow(wname, 1125, 360)
            cv2.setWindowProperty(wname, cv2.WND_PROP_TOPMOST, 1)
        cv2.waitKey(int(1000 / FRAMERATE))

    cv2.destroyAllWindows()


def main(save_dir, capture_time, ignore_ir=True, ignore_rgb=False):
    ctx = POINTER(uvc_context)()
    dev = POINTER(uvc_device)()
    devh = POINTER(uvc_device_handle)()
    ctrl = uvc_stream_ctrl()

    res = libuvc.uvc_init(byref(ctx), 0)
    if res < 0 and not ignore_ir:
        print('uvc_init error')
        exit(1)

    try:
        res = libuvc.uvc_find_device(ctx, byref(dev), PT_USB_VID, PT_USB_PID, 0)
        if res < 0:
            print('uvc_find_device error')
            exit(1)

        try:
            res = libuvc.uvc_open(dev, byref(devh))
            if res < 0 and not ignore_ir:
                print('uvc_open error: did you run with `sudo`?')
                exit(1)

            print('Device opened!')
            print_device_info(devh)
            print_device_formats(devh)
            frame_formats = uvc_get_frame_formats_by_guid(devh, VS_FMT_GUID_Y16)
            if len(frame_formats) == 0 and not ignore_ir:
                print('Device does not support Y16')
                exit(1)

            libuvc.uvc_get_stream_ctrl_format_size(
                devh, byref(ctrl), UVC_FRAME_FORMAT_Y16,
                frame_formats[0].wWidth, frame_formats[0].wHeight,
                int(1e7 / frame_formats[0].dwDefaultFrameInterval),
            )
            res = libuvc.uvc_start_streaming(devh, byref(ctrl), PTR_PY_FRAME_CALLBACK, None, 0)
            if res < 0:
                print('uvc_start_streaming failed: {0}'.format(res))
                exit(1)

            try:
                cam_ir = Lepton(q, ignore=ignore_ir)
                cam_rgb = ArduCam(ignore=ignore_rgb)
                capture_loop(save_dir, capture_time, cam_ir, cam_rgb, ignore_ir)
            finally:
                libuvc.uvc_stop_streaming(devh)
            print('Done')
        finally:
            libuvc.uvc_unref_device(dev)
    finally:
        libuvc.uvc_exit(ctx)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dir', type=str)
    parser.add_argument('--base', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--ignore_ir', action='store_true')
    parser.add_argument('--ignore_rgb', action='store_true')
    args = parser.parse_args()

    dir = os.path.join(args.dir, 'cool')
    capture_time = 5 * 60
    if args.base:
        dir = os.path.join(args.dir, 'base')
        capture_time = 60
    
    dir = os.path.join('../data', dir)
    overwrite = args.overwrite or (os.path.isdir(dir) and len(listdir(dir)) == 0)
    os.makedirs(dir, exist_ok=overwrite)

    main(dir, capture_time, args.ignore_ir, args.ignore_rgb)
