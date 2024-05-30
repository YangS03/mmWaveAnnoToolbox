from moviepy.editor import ImageSequenceClip, VideoFileClip
import glob

# image_folder = 'viz/2024-05-29-22-21-29-074018/heatmap' # right
image_folder = 'viz/2024-05-29-22-22-05-443181/heatmap' # left
# image_folder = 'viz/2024-05-29-22-22-37-027792/heatmap' # T
video_name = 'pred_result.mp4'

images = sorted(glob.glob(f"{image_folder}/*.png"))  

clip = ImageSequenceClip(images, fps=10)

clip.write_videofile(video_name)
