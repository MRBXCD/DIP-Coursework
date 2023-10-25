import argparse
from dehaze import DehazeProcess


def main():
    parser = argparse.ArgumentParser(description="Parameters of this code")
    parser.add_argument("--file_name_pic", type=str, default="H26.jpg", help="Which image to process")
    parser.add_argument("--file_name_video", type=str, default="train.mp4", help="Which video to process")
    parser.add_argument("--raw_pic_dir", type=str, default="./IEI2019", help="Path of raw images")
    parser.add_argument("--raw_video_dir", type=str, default="./IEV2022", help="Path of raw video")
    parser.add_argument("--dehazed_pic_path", type=str, default="./output/pic", help="Path of output image")
    parser.add_argument("--dehazed_video_path", type=str, default="./output/video", help="Path of output video")
    parser.add_argument("--median", type=str, default="image", help="Decision of processing the image or video")
    parser.add_argument("--frame_shape", type=list, default=[288, 352], help="Shape of a frame in the video")
    parser.add_argument("--dir_to_save_histograms", type=str, default='./output/histograms', help="Path to save histogram")
    parameters = parser.parse_args()

    processor = DehazeProcess(parameters)
    processor.Dehaze()

if __name__ == '__main__':
	main()

