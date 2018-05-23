from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
from omnistereo import common_cv
from threading import Timer

class RepeatedTimer(object):

    def __init__(self, interval, function, *args, **kwargs):
        self._timer = None
        self.interval = interval
        self.function = function
        self.args = args
        self.kwargs = kwargs
        self.is_running = False
        self.start()

    def _run(self):
        self.is_running = False
        self.start()
        self.function(*self.args, **self.kwargs)

    def start(self):
        if not self.is_running:
            self._timer = Timer(self.interval, self._run)
            self._timer.start()
            self.is_running = True

    def stop(self):
        self._timer.cancel()
        self.is_running = False

class WebcamLive:

    def __init__(self, cam_index = 0, mirror_image = False, win_name = "Webcam Video", img_file_name = "", **kwargs):
        self.frame = None
        self.cam_index = cam_index
        self.mirror_image = mirror_image
        self.current_gain = 0
        self.exposure = 0
        if img_file_name == "":
            import time
            localtime = time.localtime()
            time_str = time.strftime("%Y-%m-%d-%H-%M-%S", localtime)
            img_file_name = "img-" + time_str  # Use current date and time
        self.img_file_name = img_file_name
        self.cam_model = kwargs.get("cam_model", "GENERIC")
        self.save_key = kwargs.get("save_key", 's')
        self.show_img = kwargs.get("show_img", True)
        self.save_img_interval = kwargs.get("save_img_interval", -1)
        self.auto_save_trigger = False  # Will trigger the save flag to True at the specified interval

        # Start Frame Capturing
        self.capture = cv2.VideoCapture(self.cam_index)
        # Setting desired frame width and height (if any particular known model)
        if self.cam_model == "CHAMELEON":
            self.capture.set(cv2.CAP_PROP_MODE, cv2.CAP_MODE_GRAY)
#             self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1296)  # Is it 1280 or 1296?
            self.needs_bayer_conversion = True
        elif self.cam_model == "BLACKFLY":
            # self.capture.set(cv2.CAP_PROP_MODE, cv2.CAP_MODE_GRAY)
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            self.capture.set(cv2.CAP_PROP_FPS, -1)
            self.needs_bayer_conversion = True
        else:
            self.needs_bayer_conversion = False

        self.print_cam_info()
        if self.capture.isOpened():
            # For some reason, USB3 Point Grey cameras require a few trials to read the capture
            success = False
            failure_count = 0
            while not success:
                success, frame = self.capture.read()
                cv2.waitKey(1000)
                if not success:
                    failure_count += 1
                    print("Failed to read (%d). Try reading again" % (failure_count))
            if success:
                h, w = frame.shape[0:2]
                self.channels = frame.ndim
                self.img_size = (w, h)
                print("Successfully initialize video source %s using %d color channels" % (str(self.cam_index), self.channels))
        else:
            print("Failed to initialize video source %s" % (str(self.cam_index)))

        self.win_name = win_name
        if self.show_img:
            cv2.namedWindow(self.win_name, cv2.WINDOW_NORMAL)
            # cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)
            # cv2.namedWindow(win_name, cv2.WINDOW_OPENGL)
            self.win_handler = common_cv.PointClicker(self.win_name)

    def trigger_auto_save(self):
        self.auto_save_trigger = True

    def print_cam_info(self):
#         capture_mode = self.capture.get(cv2.CAP_PROP_MODE)
#         print("Capture mode:", capture_mode)
#         codec = self.capture.get(cv2.CAP_PROP_FOURCC)
#         print("4-character code of codec:", codec)
        fps = self.capture.get(cv2.CAP_PROP_FPS)
        print("Framerate = %0.2f FPS" % (fps))
        self.current_gain = self.capture.get(cv2.CAP_PROP_GAIN)
        print("Gain = %0.2f" % (self.current_gain))
        print("Aperture = ", self.capture.get(cv2.CAP_PROP_APERTURE))
        print("Auto Exposure = ", self.capture.get(cv2.CAP_PROP_AUTO_EXPOSURE))
        print("Exposure = ", self.capture.get(cv2.CAP_PROP_EXPOSURE))
        print("Brightness = ", self.capture.get(cv2.CAP_PROP_BRIGHTNESS))
        print("Buffer Size = ", self.capture.get(cv2.CAP_PROP_BUFFERSIZE))
        print("Gamma = ", self.capture.get(cv2.CAP_PROP_GAMMA))
        print("Hue = ", self.capture.get(cv2.CAP_PROP_HUE))
        print("ISO Speed = ", self.capture.get(cv2.CAP_PROP_ISO_SPEED))
        print("Saturation = ", self.capture.get(cv2.CAP_PROP_SATURATION))
        print("Sharpness = ", self.capture.get(cv2.CAP_PROP_SHARPNESS))
        print("White Balance Blue U = ", self.capture.get(cv2.CAP_PROP_WHITE_BALANCE_BLUE_U))
        print("White Balance Red V = ", self.capture.get(cv2.CAP_PROP_WHITE_BALANCE_RED_V))

    def demo_live(self):
        success, _ = self.get_single_frame()
        if self.show_img:
            # Create window
            cv2.namedWindow(self.win_name, cv2.WINDOW_NORMAL)
            self.win_handler = common_cv.PointClicker(self.win_name)

        rt = None
        if self.save_img_interval > 0:
            rt = RepeatedTimer(interval = self.save_img_interval, function = self.trigger_auto_save)  # it auto-starts, no need of rt.start()

        while success and self.capture.isOpened():
            success, frame = self.get_single_frame()
            if self.show_img and success:
                cv2.imshow(self.win_name, frame)

            ch_pressed_waitkey = cv2.waitKey(1)

            if (ch_pressed_waitkey & 255) == ord(self.save_key) or self.auto_save_trigger:
                self.win_handler.save_image(self.frame, img_name = self.img_file_name)
                self.auto_save_trigger = False  # Reset save trigger

            if ch_pressed_waitkey == 27:  # Pressing the Escape key breaks the loop
                success = False
                break
        if self.show_img:
            # Destroy window
            cv2.destroyWindow(self.win_name)

        if self.save_img_interval > 0:
            rt.stop()  # better in a try/finally block to make sure the program ends!

#             self.capture.get(cv2.CAP_PVAPI_PIXELFORMAT_BAYER8)
#             self.capture.get(cv2.CAP_PVAPI_PIXELFORMAT_BAYER16)
#             self.capture.get(cv2.CAP_PVAPI_PIXELFORMAT_BGR24)
#             self.capture.get(cv2.CAP_PROP_MONOCHROME)

#===============================================================================
# # The PointGrey BlackFly that reads as three 8-bit channels in Python's Numpy array of shape (rows, cols, 3).
# # However, each cell (pixel) has the same value, so I end up with a gray-scale image-like by default.
# # I tried several conversions of color, but Bayer cannot go from 3 channels to anything else.
# # Does anyone know which property to set for this VideoCapture object or any other suggestion? I'm not sure which Video Data Output it's using, perhaps it's 24-bit digital data it's giving 24-bits per pixel as (8,8,8), but maybe not. How to find out the video format within OpenCV? If I ask for get(CAP_PVAPI_PIXELFORMAT_RGB24) it give 15.0, get(CAP_PVAPI_PIXELFORMAT_BAYER16) gives 480.0, and get(CAP_PVAPI_PIXELFORMAT_BAYER8) gives 640.0. Any other PIXELFORMAT gives -1. I don't understand it.
#===============================================================================

    def get_single_frame(self):
        self.frame = None
        success = False

        try:
            # Grabs, decodes and returns the next video frame
            success, frame_raw = self.capture.read()
            if success:
                if self.needs_bayer_conversion:
                    if self.cam_model == "CHAMELEON":
                        self.frame = cv2.cvtColor(frame_raw, cv2.COLOR_BAYER_GR2BGR)  # TODO: figure out the right scheme for the 2nd COLOR conversion chameleon
                    elif self.cam_model == "BLACKFLY":
                        self.frame = cv2.cvtColor(frame_raw[..., 0], cv2.COLOR_BAYER_BG2BGR)
                else:
                    self.frame = frame_raw

                if self.mirror_image:
                    self.frame = self.frame[:, -1::-1]

        except:
            print("Failed to read frame!")

        return success, self.frame

class WebcamLiveDuo:

    def __init__(self, cam_indices, mirror_image = False, win_name = "Webcam Video", img_file_name = "", cam_models = [], save_keys = [], **kwargs):
        self.cams = []
        self.save_key = kwargs.get("save_key", 's')
        self.show_img = kwargs.get("show_img", True)
        self.save_img_interval = kwargs.get("save_img_interval", -1)
        self.auto_save_triggers = [False, False]  # Will trigger the save flag to True at the specified interval
        self.win_names = [None, None]
        self.file_names = [None, None]
        self.win_handlers = [None, None]
        self.frames = [None, None]

        for i, c_model, s_key in zip(cam_indices, cam_models, save_keys):
            self.win_names[i] = "%s-%d" % (win_name, i)
            self.cams.append(WebcamLive(cam_index = i, mirror_image = mirror_image, win_name = self.win_names[i], img_file_name = "%s-cam_%d" % (img_file_name, i), cam_model = c_model, save_key = s_key))

        self.success_list = np.zeros(len(cam_indices), dtype = "bool")
        self.open_list = np.zeros(len(cam_indices), dtype = "bool")

    def trigger_auto_save_duo(self):
        self.auto_save_triggers = [True, True]

    def demo_live(self):
        for i, c in enumerate(self.cams):
            self.success_list[i], frame = c.get_single_frame()
            self.open_list[i] = c.capture.isOpened()

            if self.show_img:
                # Create window
                cv2.namedWindow(self.win_names[i], cv2.WINDOW_NORMAL)
                self.win_handlers[i] = common_cv.PointClicker(self.win_names[i])

        rt = None
        if self.save_img_interval > 0:
            rt = RepeatedTimer(interval = self.save_img_interval, function = self.trigger_auto_save_duo)  # it auto-starts, no need of rt.start()

        while np.all(self.success_list) and np.all(self.open_list):
            for i, c in enumerate(self.cams):
                # TODO: It should be using threads instead
                self.success_list[i], self.frames[i] = c.get_single_frame()
                self.open_list[i] = c.capture.isOpened()

            for i, c in enumerate(self.cams):
                if self.show_img and self.success_list[i]:
                    cv2.imshow(self.win_names[i], self.frames[i])

            ch_pressed_waitkey = cv2.waitKey(1)
            for i, c in enumerate(self.cams):
                if (ch_pressed_waitkey & 255) == ord(c.save_key) or self.auto_save_triggers[i]:
                    self.win_handlers[i].save_image(c.frame, img_name = c.img_file_name)
                    self.auto_save_triggers[i] = False  # Reset save trigger

            if ch_pressed_waitkey == 27:  # Pressing the Escape key breaks the loop
                break

        for i, c in enumerate(self.cams):
            if self.show_img:
                # Destroy window
                cv2.destroyWindow(self.win_names[i])

        if self.save_img_interval > 0:
            rt.stop()  # better in a try/finally block to make sure the program ends!

from threading import Thread

class CamAsWorkingThread(Thread):

    def __init__(self, cam):
        Thread.__init__(self)
        self.cam = cam

        if self.cam.show_img:
            # Create window
            cv2.namedWindow(self.cam.win_name, cv2.WINDOW_NORMAL)
            self.cam.win_handler = common_cv.PointClicker(self.cam.win_name)

            # General Trackbars (sliders)
            # NOTE: Only those properties that I was able to SET successfully have trackbars:
            cv2.createTrackbar("Brigthness", self.cam.win_name, int(self.cam.capture.get(cv2.CAP_PROP_BRIGHTNESS)), 1000, self.cam_prop_brightness_callback)
            self.tb_name_cam_prop_gain = "Gain"
            cv2.createTrackbar(self.tb_name_cam_prop_gain, self.cam.win_name, int(self.cam.capture.get(cv2.CAP_PROP_GAIN)), 1000, self.cam_prop_gain_callback)
            cv2.createTrackbar("Auto Exposure", self.cam.win_name, int(self.cam.capture.get(cv2.CAP_PROP_AUTO_EXPOSURE)), 5000, self.cam_prop_auto_exposure_callback)
            # cv2.createTrackbar("Exposure", self.cam.win_name, int(self.cam.capture.get(cv2.CAP_PROP_EXPOSURE)), 5000, self.cam_prop_exposure_callback)

        self.rt = None
        if self.cam.save_img_interval > 0:
            self.rt = RepeatedTimer(interval = self.cam.save_img_interval, function = self.cam.trigger_auto_save)  # it auto-starts, no need of rt.start()

        self.current_frame = None
        self.quit_flag = False

    def run(self):  # the entry point for a thread called when start() is invoked
        # Just continuously poll images from the camera and keep the most recent frame from the buffer
        success, _ = self.cam.get_single_frame()
        while success and self.cam.capture.isOpened():
            if self.quit_flag:
                print("Quit capturing from camera")
                break
            success, self.current_frame = self.cam.get_single_frame()
            if not success:
                self.current_frame = None

    # Camera Properties' Trackbar Callbacks:
    def cam_prop_brightness_callback(self, pos):
        if not self.cam.capture.set(cv2.CAP_PROP_BRIGHTNESS, pos):
            print("Error setting Brightness to value", pos)

    def cam_prop_gain_callback(self, pos):
        if not self.cam.capture.set(cv2.CAP_PROP_GAIN, pos):
            print("Error setting Gain to value", pos)

    def cam_prop_auto_exposure_callback(self, pos):
        if not self.cam.capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, pos):
            print("Error setting Auto Exposure to value", pos)

    def cam_prop_exposure_callback(self, pos):
        if not self.cam.capture.set(cv2.CAP_PROP_EXPOSURE, pos):
            print("Error setting Exposure to value", pos)

def demo_single_cam_as_thread(cam):
    cam_working_thread = CamAsWorkingThread(cam = cam)
    cam_working_thread.start()
    keep_working = True
    while keep_working:
        if cam_working_thread.current_frame is not None:
            cv2.imshow(cam_working_thread.cam.win_name, cam_working_thread.current_frame)

        if cam_working_thread.cam.show_img:
            ch_pressed_waitkey = cv2.waitKey(1)

            if ch_pressed_waitkey == 27:  # Pressing the Escape key breaks the loop
                keep_working = False
                break
            if (ch_pressed_waitkey & 255) == ord(cam_working_thread.cam.save_key) or cam_working_thread.cam.auto_save_trigger:
                cam_working_thread.cam.win_handler.save_image(cam_working_thread.current_frame, img_name = cam_working_thread.cam.img_file_name)
                cam_working_thread.cam.auto_save_trigger = False  # Reset save trigger

    if cam_working_thread.cam.show_img:
        # Destroy window
        cv2.destroyWindow(cam_working_thread.cam.win_name)

    if cam_working_thread.rt is not None:
        cam_working_thread.rt.stop()  # better in a try/finally block to make sure the program ends!

    cam_working_thread.quit_flag = True  # Indicate camera to stop capturing and quit
    cam_working_thread.join()

    print("Finished camera as working thread DEMO")

if __name__ == '__main__':
    from omnistereo.common_tools import get_current_os_name, make_sure_path_exists
    from os import path
    if get_current_os_name() == "linux":
        cam_name = "/dev/blackfly"
    else:
        cam_name = 0

#     img_file_name = "../data/real/rr/test/scene/office/image"

#     img_file_name = "../data/real/new/VICON/scene/office/image"
#     img_file_name = "../data/real/new/simple/calibration/chessboard"
#     img_file_name = "../data/real/new/simple/scene/office/image"
#     img_file_name = "../data/real/new/simple/scene/home/image"
#     img_file_name = "../data/real/new/simple/scene/outdoors/image"
#     img_file_name = "../data/real/rr/simple/calibration_test/charuco/chessboard"

    files_path_root = path.dirname(path.abspath(path.realpath(file_name)))
    file_basename = path.basename(file_name)

    make_sure_path_exists(files_path_root)
    file_name_resolved = path.join(files_path_root, file_basename)

    save_img_interval = -1  # -1: for manual saving, otherwise, specify the timing here
    # A Single Blackfly chamera
    cam = WebcamLive(cam_index = cam_name, mirror_image = False, file_name = file_name_resolved, cam_model = "BLACKFLY", save_img_interval = save_img_interval)

    # A Single Chameleon chamera
    # cam = WebcamLive(cam_index=cam_name, mirror_image=False, img_file_name=file_name_resolved, cam_model="CHAMELEON")
    # A Duo Chameleon chameras
    # cam = WebcamLiveDuo(cam_indices=[0, 1], mirror_image=False, img_file_name=file_name_resolved, cam_models=["CHAMELEON", "CHAMELEON"], save_keys=['a', 's'], save_img_interval=save_img_interval)

    # Using a working thread
    # TODO: only for single camera at the moment
    demo_single_cam_as_thread(cam = cam)
    # OR
#     cam.demo_live()

    cv2.destroyAllWindows()
