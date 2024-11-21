from moviepy.editor import *

class VideoProcessor:
    def __init__(self, video_path):
        self.video = VideoFileClip(video_path)

    def get_resolution(self):
        return self.video.size

    def get_duration(self):
        return self.video.duration

    def speed_up(self, factor):
        self.video = self.video.speedx(factor)

    def save_video(self, output_path):
        self.video.write_videofile(output_path)