import os
import gradio as gr
import glob
import cv2
import shutil
from PIL import Image
import json
# Path to the folder containing videos
video_folder = './videos'
user_name = None
last_frame = 0
annotation_output = []
# last_video = ""
# trimmed = False

def list_videos():
    # List all files in the directory and filter for common video file extensions
    return [file for file in os.listdir(video_folder) if file.endswith(('.mp4', '.avi', '.mov'))]

def video_to_images(video_path):
    cap = cv2.VideoCapture(video_path)
    images = []
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    success, image = cap.read()
    count = 0
    while success:
        # Save frame as JPEG file
        temp_path = f'temp_frame_{count}.jpg'
        cv2.imwrite(f"./images/{temp_path}", image)
        images.append(temp_path)
        success, image = cap.read()
        count += 1
    cap.release()
    return images, frame_rate

def display_image(image_files, index):
    image_files = eval(image_files)
    if index>=len(image_files)-1:
        output_image = Image.open(f"./images/{image_files[-1]}")
        gr.Warning(f"only {len(image_files)} frames are loaded")
        return output_image, len(image_files)-1
    if image_files:
        output_image = Image.open(f"./images/{image_files[index]}")
        return output_image, index


def set_user(inp):
    global user_name
    user_name = inp
    gr.Info(f"successfully save your name {inp}")
    return inp

def reset_annotation():
    global last_frame,annotation_output
    last_frame = 0
    annotation_output = []
    gr.Warning(f"reset last frame to {last_frame} and annotated result to {annotation_output}")
    
def add_annotation(annotation,slider,video_selector,meta_info):
    global last_frame,annotation_output
    meta_info = eval(meta_info)
    new_anno = {
            "annotation": annotation,
            "video":video_selector,
            "from":last_frame,
            "to":slider,
            "total": len(meta_info)
        }
    gr.Info(f"add annotation {new_anno}")
    last_frame = slider
    annotation_output.append(new_anno)

def write_annotation(out):
    global last_frame,annotation_output
    if out=="locked name":
        gr.Error("You MUST login")
        return
    path = f"{out}.json"
    with open(path,'w') as f:
        json.dump(annotation_output,f,indent=4)
    gr.Info(f"save annotation to {path}")

def main():
    # Get the list of video files
    video_files = list_videos()
    # Define the interface
    with gr.Blocks() as demo:
        gr.Markdown("Start typing below and then click **Run** to see the output.")
        with gr.Column():
            image_viewer = gr.Image(show_label=False)
            slider = gr.Slider(minimum=0, maximum=100, step=1, label="Frame Slider")
            with gr.Row():
                with gr.Column():
                    gr.Markdown("plz login first")
                    with gr.Row():
                        btn_to_user = gr.Button("press to lock user")
                        inp = gr.Textbox(placeholder="Register your name")
                        out = gr.Textbox(placeholder="locked name",interactive=False)
                    btn_to_add = gr.Button("add annotation")
                    btn_to_save = gr.Button("Submit your annotation")

                    annotation  =gr.Textbox(placeholder="write your annotation here and press submit!!")
                with gr.Column():
                    video_selector = gr.Dropdown(label="Select a Video", choices=video_files)
                    btn_load_video = gr.Button("load video")
                    meta_info = gr.Textbox(label="meta info", lines=3)
            
        def load_video(video_file):
            global last_frame
            last_frame = 0
            gr.Info("reset last frame")
            image_paths, frame_rate = video_to_images(f"videos/{video_file}")
            return image_paths, f"Loaded {video_file} with frame rate: {frame_rate} fps"
        btn_load_video.click(fn=load_video, inputs=video_selector, outputs=[meta_info, slider])
        slider.release(fn=display_image, inputs=[meta_info, slider], outputs=[image_viewer,slider])
        btn_to_add.click(fn=add_annotation, inputs = [annotation,slider,video_selector,meta_info])
        btn_to_user.click(fn=set_user, inputs = inp, outputs = out)
        btn_to_save.click(fn=write_annotation,inputs = out)

    # Launch the interface
    demo.launch()

if __name__ == "__main__":
    main()