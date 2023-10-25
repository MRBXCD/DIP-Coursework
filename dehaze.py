import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from PIL import Image
import matplotlib
import tqdm
#from pylab import *


def DarkChannel(img):
    image = img
    b, g, r = cv2.split(image)
    min_channel = cv2.min(cv2.min(r, g), b)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    dark_channel = cv2.erode(min_channel, kernel)
    return dark_channel


def GuideFilter(img, window_size, guide_img, correction_parameters):
    image_mean = cv2.boxFilter(img, cv2.CV_64F, (window_size, window_size))
    guide_mean = cv2.boxFilter(guide_img, cv2.CV_64F, (window_size, window_size))
    guide_image_mean = cv2.boxFilter(guide_img * img, cv2.CV_64F, (window_size, window_size))
    covariance_image = guide_image_mean - image_mean * guide_mean

    image_square_mean = cv2.boxFilter(img * img, cv2.CV_64F, (window_size, window_size))

    variance_image = image_square_mean - image_mean * image_mean

    filter_para_a = covariance_image / (variance_image + correction_parameters)
    filter_para_b = guide_mean - filter_para_a * image_mean

    para_a_mean_img = cv2.boxFilter(filter_para_a, cv2.CV_64F, (window_size, window_size))
    para_b_mean_img = cv2.boxFilter(filter_para_b, cv2.CV_64F, (window_size, window_size))

    output = para_a_mean_img * img + para_b_mean_img
    return output


def RecoverImage(img, trans_est_refined, atmos_illu, lower_boundary=0.1):
    enhanced_img = np.empty(img.shape, img.dtype)
    t = cv2.max(trans_est_refined, lower_boundary)

    for i in range(0, 3):
        enhanced_img[:, :, i] = (img[:, :, i] - atmos_illu[0, i]) / t + atmos_illu[0, i]

    return enhanced_img


class DehazeProcess:
    def __init__(self, parameters):
        # Load parameters
        self.p = parameters

        # Load raw images and videos
        self.raw_image = cv2.imread(os.path.join(self.p.raw_pic_dir, self.p.file_name_pic))
        self.raw_video = cv2.VideoCapture(os.path.join(self.p.raw_video_dir, self.p.file_name_video))

        # Prepare the dictionary to store the pics
        self.output_image_path = self.p.dehazed_pic_path
        self.output_video_path = self.p.dehazed_video_path
        os.makedirs(self.p.dir_to_save_histograms, exist_ok=True)
        os.makedirs(self.output_image_path, exist_ok=True)
        os.makedirs(self.output_video_path, exist_ok=True)

        # Prepare variables required to enhance the image
        [image_high, image_width] = self.raw_image.shape[:2]
        self.image_size = image_high * image_width
        self.image_processing = self.raw_image
        self.image_vector = self.raw_image.reshape(self.image_size, 3)

        # Prepare variables required to enhance the frame
        self.frame = np.empty(self.p.frame_shape)
        [self.frame_height, self.frame_width] = self.p.frame_shape
        self.frame_size = self.frame_height * self.frame_width

        # Prepare images to print histogram
        self.image_H22 = cv2.imread(os.path.join(self.p.raw_pic_dir, 'H22.png'))
        self.image_H26 = cv2.imread(os.path.join(self.p.raw_pic_dir, 'H26.jpg'))
        self.image_R1 = cv2.imread(os.path.join(self.p.raw_pic_dir, 'R1.jpg'))
        # Check



        H22_rgb = cv2.cvtColor(self.image_H22, cv2.COLOR_BGR2RGB)
        H22_gray = cv2.cvtColor(H22_rgb, cv2.COLOR_BGR2GRAY)
        H22_hist = cv2.calcHist(H22_gray, [0], None, [256], [0, 256])

        H26_rgb = cv2.cvtColor(self.image_H26, cv2.COLOR_BGR2RGB)
        H26_gray = cv2.cvtColor(H26_rgb, cv2.COLOR_BGR2GRAY)
        H26_hist = cv2.calcHist(H26_gray, [0], None, [256], [0, 256])

        R1_rgb = cv2.cvtColor(self.image_R1, cv2.COLOR_BGR2RGB)
        R1_gray = cv2.cvtColor(R1_rgb, cv2.COLOR_BGR2GRAY)
        R1_hist = cv2.calcHist(R1_gray, [0], None, [256], [0, 256])

        # Set resolution
        width_pixels = 1920
        height_pixels = 1080

        # Set figure size
        dpi = 100
        width_inches = width_pixels / dpi
        height_inches = height_pixels / dpi

        # Build sub figure
        fig, axes = plt.subplots(3, 3, figsize=(width_inches, height_inches), dpi=dpi)
        fig.suptitle('Histogram of Figures', x=0.5, y=0.97, fontsize=21, fontweight='bold')

        # H22 image plot
        plt.subplot(3, 3, 1)
        plt.imshow(H22_rgb)
        plt.title('H22.png')

        plt.subplot(3, 3, 4)
        plt.imshow(H22_gray, cmap='gray')
        plt.title('Gray image of H22')

        plt.subplot(3, 3, 7)
        plt.plot(H22_hist)
        plt.title("Histogram of H22 image")
        plt.xlabel("Pixel Value")
        plt.ylabel("Frequency")
        plt.xlim([0, 256])
        plt.ylim([0, 100])

        # H26 image plot
        plt.subplot(3, 3, 2)
        plt.imshow(H26_rgb)
        plt.title('H26.jpg')

        plt.subplot(3, 3, 5)
        plt.imshow(H26_gray, cmap='gray')
        plt.title('Gray image of H26')

        plt.subplot(3, 3, 8)
        plt.plot(H26_hist)
        plt.title("Histogram of H26 image")
        plt.xlabel("Pixel Value")
        plt.ylabel("Frequency")
        plt.xlim([0, 256])
        plt.ylim([0, 100])

        # R1 image plot
        plt.subplot(3, 3, 3)
        plt.imshow(R1_rgb)
        plt.title('R1.jpg')

        plt.subplot(3, 3, 6)
        plt.imshow(R1_gray, cmap='gray')
        plt.title('Gray image of R1')

        plt.subplot(3, 3, 9)
        plt.plot(R1_hist)
        plt.title("Histogram of R1 image")
        plt.xlabel("Pixel Value")
        plt.ylabel("Frequency")
        plt.xlim([0, 256])
        plt.ylim([0, 100])

        plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)
        plt.savefig(os.path.join(self.p.dir_to_save_histograms, 'Histograms of several figure.png'))
        plt.show()

    def BasicProcessing(self):
        if self.p.median == "image":
            image_basic_processed = self.raw_image.astype('float64') / 255
            return image_basic_processed
        elif self.p.median == "video":
            frame_basic_processed = self.frame.astype('float64') / 255
            return frame_basic_processed

    def AtmosphericIllumination(self, dark):
        # Calc the number of top 0.1% light pixels
        dark_channel = dark
        total_pixels = dark_channel.size
        top_percent = 0.1
        top_pixels_num = int(total_pixels * (top_percent / 100))
        # Flatten the pixels from dark channel
        flat_dark_channel = dark_channel.reshape(total_pixels)

        # Get top 0.1% pixels
        pixels_index = flat_dark_channel.argsort()
        brightest_pixels_index = pixels_index[total_pixels - top_pixels_num::]

        # Form an array to store Atmospheric Illumination
        atmos_illum = np.zeros([1, 3])
        if self.p.median == "image":
            image_vector = self.image_vector.astype('float64') / 255
            for i in range(1, top_pixels_num):
                atmos_illum = atmos_illum + image_vector[brightest_pixels_index[i]]
        elif self.p.median == "video":
            frame_vector = self.frame.reshape(self.frame_size, 3).astype('float64') / 255
            for i in range(1, top_pixels_num):
                atmos_illum = atmos_illum + frame_vector[brightest_pixels_index[i]]

        # Calculate "A"
        A = atmos_illum / top_pixels_num
        return Acal

    def TransmissionEstimate(self, A):
        # In this function, we calculate the transmission estimate "t"
        omega = 0.95
        image = self.BasicProcessing()
        image_processing = np.empty(image.shape, image.dtype)

        for i in range(0, 3):
            image_processing[:, :, i] = image[:, :, i] / A[0, i]

        dark_channel_for_transEst = DarkChannel(image_processing)
        trans_estimate = 1 - omega * dark_channel_for_transEst
        return trans_estimate

    def TransmissionEstimateRefine(self, trans_estimate):
        if self.p.median == "image":
            gray_image = cv2.cvtColor(self.raw_image, cv2.COLOR_BGR2GRAY)
        elif self.p.median == "video":
            gray_image = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        gray_image_normalized = np.float64(gray_image) / 255
        window_size = 60
        correction_para = 0.0001
        transmission_estimation_refined = GuideFilter(gray_image_normalized, window_size, trans_estimate,
                                                      correction_para)
        return transmission_estimation_refined

    def ResultShow(self, dark_channel, image_enhanced):
        width_pixels = 1280
        height_pixels = 720

        # Set figure size
        dpi = 100
        width_inches = width_pixels / dpi
        height_inches = height_pixels / dpi

        # Build sub figure
        fig, axes = plt.subplots(1, 3, figsize=(width_inches, height_inches), dpi=dpi)

        fig.suptitle('Result of ' + self.p.file_name_pic, x=0.5, y=0.9, fontsize=16, fontweight='bold')

        axes[0].imshow(self.raw_image)
        axes[0].set_title(self.p.file_name_pic)

        axes[1].imshow(dark_channel, cmap='gray')
        axes[1].set_title("Dark Channel")

        axes[2].imshow(image_enhanced)
        axes[2].set_title("Image Enhanced")

        plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.2)

        # Save the results
        plt.savefig(os.path.join(self.p.dehazed_pic_path, self.p.file_name_pic[:-4] + '_Result.png'))
        plt.show()

    def Dehaze(self):
        self.Histogram()
       # cv2.waitKey()
        if self.p.median == "image":
            # Normalize the image
            img_processed_for_DarkChannel = self.BasicProcessing()

            # Get dark channel
            dark_channel_1 = DarkChannel(img_processed_for_DarkChannel)

            # Compute Atmospheric Illumination
            A = self.AtmosphericIllumination(dark_channel_1)

            # Compute transmit estimation
            transEst = self.TransmissionEstimate(A)
            tranEst_refined = self.TransmissionEstimateRefine(transEst)

            # Recover the image
            img = self.BasicProcessing()
            enhanced_img = RecoverImage(img, tranEst_refined, A)

            # Show the results
            self.ResultShow(dark_channel_1, enhanced_img)

        elif self.p.median == "video":
            # Set parameters to load video
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = int(self.raw_video.get(cv2.CAP_PROP_FPS))
            print(fps)
            out = cv2.VideoWriter(os.path.join(self.p.dehazed_video_path, self.p.file_name_video[:-4] + '_Result.mp4'),
                                  fourcc, fps, (self.frame_width, self.frame_height))

            # Start a while loop to process the video
            while self.raw_video.isOpened():
                ret, self.frame = self.raw_video.read()
                if not ret:
                    break
                frame_processed_for_DarkChannel = self.BasicProcessing()
                dark_channel_video = DarkChannel(frame_processed_for_DarkChannel)
                A = self.AtmosphericIllumination(dark_channel_video)
                transEst = self.TransmissionEstimate(A)
                tranEst_refined = self.TransmissionEstimateRefine(transEst)
                frame = self.BasicProcessing()
                enhanced_frame = RecoverImage(frame, tranEst_refined, A)
                result = (enhanced_frame * 255).astype('uint8')
                out.write(result)

            # Release the video objects
            self.raw_video.release()
            out.release()
            cv2.destroyAllWindows()
            print("Done")
