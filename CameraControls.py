import ThorlabsCam as TC
import numpy as np
import threading
import time
import os
from time import sleep
import pickle
from copy import copy, deepcopy
import cv2
from pypylon import pylon
from datetime import datetime
from queue import Queue
import skvideo.io
from MiscFunctions import subtract_bg

class VideoFormatError(Exception):
    """
    Raised when a video format is not supported.
    """
    pass

def get_camera_c_p():
    '''
    Function for retrieving the c_p relevant for controlling the camera
    '''
    # TODO make it so that the AOI of the basler camera is not hardcoded. Maybe
    # make a camera model class?
    # Make it easier to change objective?
    camera_c_p = {
        'new_video': False,
        'recording_duration': 3000,
        'exposure_time': 30_000,  # ExposureTime in ms for thorlabs,
        # mus for basler
        'fps': 15,
        'max_fps': 10_000, # maximum fps allowed.
        'recording': False,  # True if recording is on
        'AOI': [0, 480, 0, 480],  # Default for
        'zoomed_in': False,  # Keeps track of whether the image is cropped or
        'camera_model': 'basler_large',  #basler_large, basler_fast, thorlabs are the options
        'camera_orientatation': 'down',  # direction camera is mounted in.
        'default_offset_x':0, # Used to center the camera on the sample
        'default_offset_y':0,
        'bitrate': '300000000', # Default value 3e8 default
        'saving_video': False, # indicator for when a video is being saved
        'video_format': 'avi',
        'bg_removal': False,
        'video_name': 'video'
        # Needed for not
    }
    # TODO Fix so that the software recoginze the camera and use it to
    # Determine camera width etc. May still need to calibrate for pixel-size
    # Add custom parameters for different cameras.
    if camera_c_p['camera_model'] == 'basler_large':
        # Correct value for QD setup
        camera_c_p['mmToPixel'] = 45_000 # TODO: Need to measure this!
        camera_c_p['camera_width'] = 4096
        camera_c_p['camera_height'] = 3040
        camera_c_p['default_offset_x'] = 0#1000

    elif camera_c_p['camera_model'] == 'thorlabs':
        camera_c_p['mmToPixel'] = 17736/0.7
        # TODO check why we need to set th correct sensor size here
        camera_c_p['camera_width'] = 1936
        camera_c_p['camera_height'] = 1216

    elif camera_c_p['camera_model'] == 'basler_fast':
        camera_c_p['mmToPixel'] = 21_500
        camera_c_p['camera_width'] = 672
        camera_c_p['camera_height'] = 512

    camera_c_p['slm_to_pixel'] = 5_000_000 if camera_c_p['camera_model'] == 'basler_fast' else 4_550_000
    camera_c_p['AOI'] = [0, camera_c_p['camera_width'], 0, camera_c_p['camera_height']]
    camera_c_p['frame_queue'] = Queue(maxsize=2_000_000)

    return camera_c_p

def get_video_name(c_p, base_name=''):
    """
    Returns an auto-generated name of the video. The name has the time of creation
    in the title to be easy to locate.
    """
    now = datetime.now()
    # base_name +
    print(c_p['measurement_name'], base_name)
    name = 'video-'+ c_p['measurement_name'] + '-' + str(now.hour)
    name += '-' + str(now.minute) + '-' + str(now.second)+'-fps-'
    name += str(c_p['fps'])
    return name

def create_avi_video_writer(c_p, video_name, image_width, image_height):
    '''
    Funciton for creating a VideoWriter.
    Will also save the relevant parameters of the experiments.
    Returns
    -------
    video : VideoWriter
        A video writer for creating a video.
    experiment_info_name : String
        Name of experiment being run.
    exp_info_params : Dictionary
        Dictionary with controlparameters describing the experiment.
    '''
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')

    video_name = c_p['recording_path'] + '/' + video_name + '.avi'
    video = cv2.VideoWriter(video_name, fourcc, min(500, c_p['fps']),
                        (image_height, image_width), isColor=False)
    return video

def create_mp4_video_writer(c_p, video_name=None, image_width=None,
        image_height=None):
    """
    Crates a high quality video writer for lossless recording.
    """

    if video_name is None:
        video_name = get_video_name(c_p=c_p)
    tmp = min(500, int(c_p['fps']))
    frame_rate = str(max(25, tmp)) # Can in principle reach 500fps
    if tmp < 25:
        print('Warning, skvideo cannot handle framerates below 25 fps so\
        reverting to 25.')

    video_name = c_p['recording_path'] + '/' + video_name + '.mp4'
    # TODO fix so that exceptions in recording path can be handled
    video = skvideo.io.FFmpegWriter(video_name, outputdict={
                                     '-b':c_p['bitrate'],
                                     '-r':frame_rate, # Does not like this
                                     # specifying codec and bitrate, 'vcodec': 'libx264',
                                    })
    return video

class VideoWriterThread(threading.Thread):
    """
    A class which simply deques the latest frame and prints it to a video
    """
    def __init__(self , threadID, name, c_p):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.c_p = c_p
        self.setDaemon(True)

        self.sleep_time = 0.1
        self.frame = None
        self.video_width = self.c_p['AOI'][1] - self.c_p['AOI'][0]
        self.video_height = self.c_p['AOI'][3] - self.c_p['AOI'][2]
        self.video_created = False
        self.video_name = None
        self.frame_buffer = []
        self.frame_buffer_size = 100
        self.frame_count = 0
        self.VideoWriter = None
        # self.video_format = 'avi' # TODO use only the c_p version. Format of current video being recorded
        self.np_save_path = None

    def close_video(self):
        """
        Closes and the current video and deletes the python object.
        """
        self.video_created = False
        try:
            if self.c_p['video_format'] == 'mp4':
                if self.VideoWriter is not None:
                    self.VideoWriter.close()
                del self.VideoWriter
            elif self.c_p['video_format'] == 'avi':
                if self.VideoWriter is not None:
                    self.VideoWriter.release()
                del self.VideoWriter
            else:
                # is npy, save what remains of buffer then clear it
                self.np_save_frames()
                self.frame_count = 0

        except Exception as err:
            # The program tries to close a
            print(f"No video to close {err}")
    # TODO add so that background is removed from videos and photos as well when
    # saving! Would probably need a separate queue for the BGs to do this well.

    def np_save_frames(self):
        """
        Saves all the frames in the buffer to a .npy file.
        """
        nbr_frames = self.frame_count % self.frame_buffer_size
        if nbr_frames == 0:
            nbr_frames = self.frame_buffer_size
        lower_lim = str(max(self.frame_count-nbr_frames, 0))
        upper_lim = str(self.frame_count)

        filename = lower_lim + '-' + upper_lim + '.npy'
        with open(self.np_save_path+filename, 'wb') as f:
            np.save(f, self.frame_buffer[:nbr_frames])
        self.frame_buffer *= 0

    def create_NP_writer(self, video_name):
        """
        Creates a folder and saves path to it for saving the numpy images
        """
        if video_name is None:
                video_name = get_video_name(c_p=self.c_p)
        self.np_save_path = self.c_p['recording_path'] + '/' + video_name + '/'
        try:
            os.mkdir(self.np_save_path)
        except Exception as ex:
            print(f"Directory already exist, {ex}")

    def write_to_NPY(self):
        nbr_frames = self.frame_count % self.frame_buffer_size
        if nbr_frames == 0 and self.frame_count > 0:
            # Save the frames into target folder and with suitable name
            self.np_save_frames()
        try:
            self.frame_buffer[nbr_frames,:,:] = deepcopy(self.frame)
            self.frame_count += 1
        except Exception as ex:
            print(f"Trouble writing frame, {ex}")
            self.close_video()
            self.frame_count = 0


    def write_frame(self):
        """
        Writes a frame to the current video_writer.
        If the format is "npy" we instead put it in our np-array of images.
            If the np-array of images reaches a special threshold then it is
            automatically saved.
            # Reasonable threshold perhaps 100_000 frames?
        """
        # TODO add bg removal as parameter for each saved frame
        if self.c_p['video_format'] == 'mp4':
            self.VideoWriter.writeFrame(self.frame)
        elif self.c_p['video_format'] == 'avi':
            self.VideoWriter.write(self.frame)
        else:
            self.write_to_NPY()

        # Let the caller know that a frame was successfully added to the output
        return True

    def create_video_writer(self, video_name):
        """
        Creates a video writer for the current video.
        """
        # Adjust the video shape to match the images
        image_shape = np.shape(self.frame)
        self.video_width = int(image_shape[0])
        self.video_height = int(image_shape[1])
        if self.c_p['video_format'] == 'mp4':
            self.VideoWriter = create_mp4_video_writer(c_p=self.c_p,
                video_name=video_name)
            #self.video_format = 'mp4'

        elif self.c_p['video_format'] == 'avi':
            self.VideoWriter = create_avi_video_writer(self.c_p,
                video_name, self.video_width, self.video_height)
            #self.video_format = 'avi'

        elif self.c_p['video_format'] == 'npy':
            #self.video_format = 'npy'
            # calculate an appropriate buffer size based on the size in memory
            # the frames take up.
            self.frame_buffer_size = int(501760000/( self.video_width*self.video_height))
            # print(f'Buffer_size: {self.frame_buffer_size}')
            self.frame_buffer = np.uint8(np.zeros([self.frame_buffer_size, self.video_height, self.video_width]))
            self.frame_count = 0

            self.create_NP_writer(video_name)
        else:
            raise VideoFormatError(f"Video format{self.c_p['video_format']} not recognized!")

        self.video_created = True

    def run(self):

        while self.c_p['program_running']:
            sleep(self.sleep_time)
            idx = 0
            self.c_p['saving_video']  = False

            while self.c_p['recording'] or not self.c_p['frame_queue'].empty() and self.c_p['program_running']:
                self.c_p['saving_video'] = True

                # Check empty twice since we don't want to wait longer than
                # necessary for the image to be printed to the videowriter

                if not self.c_p['frame_queue'].empty():

                    [self.frame, source_video] = self.c_p['frame_queue'].get()

                    # Check that name and size are correct, if not create a new
                    image_shape = np.shape(self.frame)
                    if image_shape[0] != self.video_width or image_shape[1] != self.video_height:
                        self.video_width = image_shape[0]
                        self.video_height = image_shape[1]
                        self.close_video()
                    # Check if name is ok
                    if self.video_name != self.c_p['video_name'] and self.video_created:
                        self.close_video()
                        self.video_name = self.c_p['video_name']

                    if not self.video_created:
                        size = '_' + str(self.video_width) + 'x' + str(self.video_height)
                        video_name = source_video + size
                        self.video_name = self.c_p['video_name']
                        self.create_video_writer(video_name)
                    self.write_frame()
                else:
                    # Queue empty
                    sleep(0.001)
            if self.video_created:
                self.close_video()


class CameraThread(threading.Thread):

    def __init__(self, threadID, name, c_p):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.c_p = c_p
        self.setDaemon(True)
        # Initalize camera
        # TODO make it so that we can connect/disconnect cameras on the go
        self.connect_basler_camera()
        if self.cam is None:
            try:
                # Get a thorlabs camera
                self.cam = TC.get_camera()
                #camera_c_p['camera_width'] = int(self.cam.max_width)
                #camera_c_p['camera_height'] = int(self.cam.max_height)
                #c_p['AOI'] = [0, c_p['camera_width'], 0, c_p['camera_height']]
                self.cam.set_defaults(left=c_p['AOI'][0], right=c_p['AOI'][1],
                                      top=c_p['AOI'][2], bot=c_p['AOI'][3],
                                      n_frames=1)
                c_p['exposure_time'] = 20
                #

            except Exception as ex:
                print('Could not connect to camera, ex')
                self.__del__()
        self.video_width = self.c_p['AOI'][1] - self.c_p['AOI'][0]
        self.video_height = self.c_p['AOI'][3] - self.c_p['AOI'][2]
        # Could change here to get color
        c_p['image'] = np.ones((self.c_p['AOI'][3], self.c_p['AOI'][1], 1))

    def __del__(self):
        """
        Closes the camera in preparation of terminating the thread.
        """
        if self.c_p['camera_model'] == 'thorlabs':
            self.cam.close()
        else:
            try:
                self.cam.Close()
            except Exception as ex:
                print(ex)

    def connect_basler_camera(self):
        """
        Connects to the basler camera.
        """
        try:
            tlf = pylon.TlFactory.GetInstance()
            self.cam = pylon.InstantCamera(tlf.CreateFirstDevice())
            self.cam.Open()
            sleep(0.2)
            self.c_p['camera_width'] = int(self.cam.Width.GetMax())
            self.c_p['camera_height'] = int(self.cam.Height.GetMax())
            if self.c_p['camera_width'] < 1000:
                self.c_p['camera_model'] = 'basler_fast'
            else:
                self.c_p['camera_model'] = 'basler_large'
            return True
        except Exception as ex:
            self.cam = None
            return False

    def get_important_parameters(self):
        '''
        Gets the parameters needed to describe the experiment. Used for svaing
        experiment setup.
        Returns
        -------
        parameter_dict : Dictionary
            Dictionary containing the control parameters needed to describe
            the experiment.
        '''
        c_p = self.c_p
        parameter_dict = {
            'xm': c_p['xm'],
            'ym': c_p['ym'],
            'zm': c_p['zm'],
            'use_LGO': c_p['use_LGO'],
            'LGO_order': c_p['LGO_order'],
            'max_fps': c_p['max_fps'],
            'setpoint_temperature': c_p['setpoint_temperature'],
            'target_experiment_z': c_p['target_experiment_z'],
            'temperature_output_on': c_p['temperature_output_on'],
            'exposure_time': c_p['exposure_time'],
            'starting_temperature': c_p['current_temperature'],
            'phasemask': c_p['phasemask'],
        }
        return parameter_dict

    # TODO integrate thorlabs-capture with the new videowriter thread.
    def thorlabs_capture(self):
        number_images_saved = 0
        video_created = False
        c_p = self.c_p

        while c_p['program_running']:
            # Set defaults for camera, aknowledge that this has been done

            self.cam.set_defaults(left=c_p['AOI'][0],
                right=c_p['AOI'][1],
                top=c_p['AOI'][2],
                bot=c_p['AOI'][3],
                exposure_time=TC.number_to_millisecond(c_p['exposure_time']))
            c_p['new_settings_camera'] = False

            image_count = 0

            # Setting  maximum fps. Will cap it to make it stable
            # Start livefeed from the camera
            self.cam.start_live_video()
            start = time.time()

            # Start continously capturing images
            while c_p['program_running'] and not c_p['new_settings_camera']:
                self.cam.wait_for_frame(timeout=None)
                if c_p['recording']:

                    if not video_created:
                        video, experiment_info_name, exp_info_params = self.create_video_writer()

                        video_created = True
                    video.write(c_p['image'])

                # Capture an image and update the image count
                image_count = image_count + 1
                c_p['image'] = self.cam.latest_frame()[:, :]
            # Close the livefeed and calculate the fps of the captures
            end = time.time()
            self.cam.stop_live_video()
            try:
                fps = image_count/(end-start)
            except ZeroDivisionError as e:
                fps = 1

            if video_created:
                video.release()
                del video
                video_created = False
                # Save the experiment data in a pickled dict.
                outfile = open(experiment_info_name, 'wb')
                exp_info_params['fps'] = fps
                pickle.dump(exp_info_params, outfile)
                outfile.close()

    def set_basler_AOI(self):
        '''
        Function for setting AOI of basler camera to c_p['AOI']
        '''
        c_p = self.c_p
        try:
            '''
            The order in which you set the size and offset parameters matter.
            If you ever get the offset + width greater than max width the
            camera won't accept your valuse. Thereof the if-else-statements
            below. Conditions might need to be changed if the usecase of this
            funciton change
            '''

            width = int(c_p['AOI'][1] - c_p['AOI'][0])
            offset_x = c_p['AOI'][0]
            height = int(c_p['AOI'][3] - c_p['AOI'][2])
            offset_y = c_p['AOI'][2]
            #print(self.cam.Width.GetMax(), self.cam.Height.GetMax())
            self.video_width = width
            self.video_height = height
            self.cam.OffsetX = 0
            self.cam.OffsetY = 0
            time.sleep(0.1)
            self.cam.Width = width
            self.cam.Height = height
            self.cam.OffsetX = c_p['default_offset_x'] + offset_x
            self.cam.OffsetY = c_p['default_offset_y'] + offset_y

        except Exception as ex:
            print(f"AOI not accepted, AOI: {c_p['AOI']}, error {ex}")

    def update_basler_exposure(self):
        try:
            self.cam.ExposureTime = self.c_p['exposure_time']
            self.c_p['fps'] = round(float(self.cam.ResultingFrameRate.GetValue()), 1)
            print('FPS is ', self.c_p['fps'])
        except Exception as ex:
            print(f"Exposure time not accepted by camera, {ex}")


    def add_image_to_queue(self):
        """
        Adds a copy of the current image from the control parameters to the
        recording queue.
        """
        c_p = self.c_p
        img = copy(c_p['image'])
        # TODO: Here could be a good place to use # no wait
        if c_p['bg_removal']:
            try:
                img = subtract_bg(c_p['image'],
                    c_p['raw_background'][c_p['AOI'][2]:c_p['AOI'][3], c_p['AOI'][0]:c_p['AOI'][1]]) # This could be a quite expensive operation
            except AssertionError:
                print('Could not subtract bg from image queue')
        # TODO fix bug in saving video with npy format!
        c_p['frame_queue'].put([img, copy(c_p['video_name'])])

    def basler_capture(self):
        '''
        Function for live capture using the basler camera. Also allows for
        change of AOI, exposure and saving video on the fly.
        Returns
        -------
        None.
        '''
        video_created = False
        c_p = self.c_p
        img = pylon.PylonImage()

        while c_p['program_running']:
            # Set defaults for camera, aknowledge that this has been done

            self.set_basler_AOI()
            c_p['new_settings_camera'] = False

            #TODO replace c_p['new_settings_camera'] with two parameters and
            # one for expsore and one for AOI
            self.update_basler_exposure()
            image_count = 0

            self.cam.StartGrabbing()
            start = time.perf_counter()

            # Start continously capturing images
            while c_p['program_running']\
                 and not c_p['new_settings_camera']:
                 with self.cam.RetrieveResult(3000) as result:
                    img.AttachGrabResultBuffer(result)
                    if result.GrabSucceeded():
                        c_p['image'] = np.uint8(img.GetArray())
                        img.Release()
                        # I do these checks here to make the program a tiny bit
                        # faster

                        if c_p['recording']:
                            self.add_image_to_queue()
                        # Capture an image and update the image count
                        image_count = image_count+1
                 if c_p['new_settings_camera']:
                    w = c_p['AOI'][1] - c_p['AOI'][0]
                    h = c_p['AOI'][3] - c_p['AOI'][2]
                    self.cam.AcquisitionFrameRateEnable.SetValue(True);
                    self.cam.AcquisitionFrameRate.SetValue(c_p['max_fps'])
                    if w == self.video_width and h == self.video_height:
                        self.update_basler_exposure()
                        c_p['new_settings_camera'] = False
            self.cam.StopGrabbing()

            # Close the livefeed and calculate the fps of the captures
            end = time.perf_counter()
            try:
                fps = image_count/(end-start)
            except ZeroDivisionError:
                fps = -1
            print(" Measured fps was:", fps, " Indicated by camera", c_p['fps'])

    def run(self):
        if self.c_p['camera_model'] == 'thorlabs':
            self.thorlabs_capture()
        elif self.c_p['camera_model'] == 'basler_large' or 'basler_fast':
            self.basler_capture()
        return


def set_AOI(c_p, left=None, right=None, up=None, down=None):
    '''
    Function for changing the Area Of Interest for the camera to the box
    specified by left,right,top,bottom.
    Parameters
    ----------
    c_p : Dictionary
        Dictionary with control parameters.
    left : INT, optional
        Left position of camera AOI in pixels. The default is None.
    right : INT, optional
        Right position of camera AOI in pixels. The default is None.
    up : INT, optional
        Top position of camera AOI in pixels. The default is None.
    down : INT, optional
        Bottom position of camera AOI in pixels. The default is None.
    Returns
    -------
    None.
    '''

    # If exact values have been provided for all the corners change AOI
    h_max = c_p['camera_width']
    v_max = c_p['camera_height']
    if left is not None and right is not None and up is not None and down is not None:
        if 0<=left<=h_max and left<=right<=h_max and 0<=up<=v_max and up<=down<=v_max:
            c_p['AOI'][0] = left
            c_p['AOI'][1] = right
            c_p['AOI'][2] = up
            c_p['AOI'][3] = down
        else:
            print("Trying to set invalid area")
            return

    # Inform the camera and display thread about the updated AOI
    c_p['new_settings_camera'] = True
    c_p['new_AOI_display'] = True

    # Update trap relative position
    update_traps_relative_pos(c_p)

    # Threads time to catch up
    time.sleep(0.1)


def update_traps_relative_pos(c_p):
    '''
    Updates the relative position of the traps when zooming in/out.
    '''
    tmp_x = [x - c_p['AOI'][0] for x in c_p['traps_absolute_pos'][0]]
    tmp_y = [y - c_p['AOI'][2] for y in c_p['traps_absolute_pos'][1]]
    tmp = np.asarray([tmp_x, tmp_y])
    c_p['traps_relative_pos'] = tmp


def zoom_out(c_p):
    '''
    Zooming out the camera AOI to the maximum allowed.
    '''
    # Reset camera to fullscreen view
    set_AOI(c_p, left=0, right=int(c_p['camera_width']), up=0,
            down=int(c_p['camera_height']))
    c_p['AOI'] = [0, int(c_p['camera_width']), 0, int(c_p['camera_height'])]
    print('Zoomed out', c_p['AOI'])
    c_p['new_settings_camera'] = True
