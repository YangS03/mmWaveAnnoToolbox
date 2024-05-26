from moviepy.editor import ImageSequenceClip, VideoFileClip
import glob

image_folder = './viz/single_0'
video_name = 'output_video.mp4'

images = sorted(glob.glob(f"{image_folder}/*.png"))  

clip = ImageSequenceClip(images, fps=10)

clip.write_videofile(video_name)
