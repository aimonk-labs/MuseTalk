import argparse
import cv2
import glob
import numpy as np
import os
import torch
from basicsr.utils import imwrite

from gfpgan import GFPGANer
from tqdm import tqdm

def main():
    """Inference demo for GFPGAN (for users).
    """
    parser = argparse.ArgumentParser(description='Example script using argparse.')
    parser.add_argument('--input_video_path', type=str, help='An optional argument', default='default_value')
    parser.add_argument('--folder_path', type=str, help='An optional argument', default='default_value')
    parser.add_argument('--audio_path', type=str, help='An optional argument', default='default_value')
    # parser.add_argument('--optional_arg', type=str, help='An optional argument', default='default_value')
    args = parser.parse_args()
    input_video_path=args.input_video_path   
    folder_path=args.folder_path
    audio_path=args.audio_path
    bg_upsampler = None    
    model_path="/bv3/debasish_works/MuseTalk/GFPGANv1.4.pth"
    restorer = GFPGANer(
        model_path=model_path,
        upscale=1,
        arch="clean",
        channel_multiplier=2,
        bg_upsampler=bg_upsampler)

    # ------------------------ restore ------------------------
    vid=cv2.VideoCapture(input_video_path)
    fps=vid.get(cv2.CAP_PROP_FPS)
    out_file_name=os.path.basename(input_video_path).split(".")[0]+"_gfgan.avi"
    # while True:
    # for img_path in img_list:
        # read image
    frame_width = int(vid.get(3))
    frame_height = int(vid.get(4))
    # out = cv2.VideoWriter('outpy_african30male.avi',cv2.VideoWriter_fourcc('M','J','P','G'), fps, (frame_width,frame_height))
    out = cv2.VideoWriter(os.path.join(folder_path,out_file_name),cv2.VideoWriter_fourcc('M','J','P','G'), fps, (frame_width,frame_height))


    while(tqdm(vid.isOpened())):
        ret, input_img = vid.read()
        
        
        if ret == True:
            # restore faces and background if necessary
            _, _, restored_img = restorer.enhance(input_img,has_aligned=False,only_center_face=False,
                paste_back=True)
            out.write(restored_img)
        
        else:
            out.release()
            vid.release()
            break
    out_file_name_audio=os.path.basename(input_video_path).split(".")[0]+"_gfgan_audio.mp4"
    # os.system(f"ffmpeg -i {os.path.join(folder_path,out_file_name)}  {os.path.join(folder_path,out_file_name_audio)}")
    os.system(f"ffmpeg -i {os.path.join(folder_path,out_file_name)} -i {audio_path} {os.path.join(folder_path,out_file_name_audio)}")
  
    # out.release()
    # vid.release()

if __name__ == '__main__':
    main()