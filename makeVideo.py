from moviepy.editor import ImageSequenceClip, VideoFileClip
import glob

# image_folder = 'viz/2024-05-26-22-39-35/heatmap'
# image_folder = 'viz/2024-05-28-20-36-52/heatmap'
image_folder = 'viz/2024-05-28-21-29-20/heatmap'
# image_folder = 'viz/2024-05-28-23-55-50-200299/heatmap'
video_name = 'pred_result.mp4'

images = sorted(glob.glob(f"{image_folder}/*.png"))  

clip = ImageSequenceClip(images, fps=10)

clip.write_videofile(video_name)
