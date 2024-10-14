import argparse
import os
from omegaconf import OmegaConf
import numpy as np
import cv2
import torch
import glob
import pickle
from tqdm import tqdm
import copy
import sys
from pathlib import Path
MuseTalk_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(MuseTalk_dir)
CHECKPOINTS_DIR=os.path.join(Path(MuseTalk_dir).parent,"checkpoints")
from musetalk.utils.utils import get_file_type,get_video_fps,datagen
from musetalk.utils.preprocessing import get_landmark_and_bbox,read_imgs,coord_placeholder
from musetalk.utils.blending import get_image
from musetalk.utils.utils import load_all_model
import shutil
import time
from fast_gfpgan import FAST_GFGGaner
from global_variable import FPS,BBOX_SHIFT,AVATAR_PRESAVES_DIR,BATCH_SIZE
from bgremoval_package.demo.run import matting
from bgremoval_package.demo.run import load_model as load_model_modenet

class MusetalkTritonInference:
    def __init__(self) -> None:
        self.audio_processor, self.vae, self.unet, self.pe = load_all_model()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.timesteps = torch.tensor([0], device=self.device)
        # model_path="/bv3/debasish_works/MuseTalk/GFPGANv1.4.pth"  # TODO: To be dynamic 
        model_path=os.path.join(CHECKPOINTS_DIR,"GFPGANv1.4.pth")
        self.gfpgan=FAST_GFGGaner(
            model_path=model_path,
            upscale=1,
            arch="clean",
            channel_multiplier=2,
            bg_upsampler=None,device="cuda") 
        self.modnet=load_model_modenet()

    def read_avatar_pose_video(self,video_path,input_basename,result_dir,output_basename):
        # if output_vid_name is None:
        #     output_vid_name = os.path.join(result_dir, output_basename+".mp4")
        # else:
        #     output_vid_name = os.path.join(result_dir, output_vid_name)
        ############################################## extract frames from source video ##############################################
        input_img_list=[]
        if get_file_type(video_path)=="video":
            save_dir_full = os.path.join(result_dir, input_basename)
            os.makedirs(save_dir_full,exist_ok = True)
            cmd = f"ffmpeg -v fatal -i {video_path} -start_number 0 {save_dir_full}/%08d.png"
            os.system(cmd)
            input_img_list = sorted(glob.glob(os.path.join(save_dir_full, '*.[jpJP][pnPN]*[gG]')))
            fps = get_video_fps(video_path)
        elif get_file_type(video_path)=="image":
            input_img_list = [video_path, ]
            fps = FPS
        elif os.path.isdir(video_path):  # input img folder
            input_img_list = glob.glob(os.path.join(video_path, '*.[jpJP][pnPN]*[gG]'))
            input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
            fps = FPS
        else:
            raise ValueError(f"{video_path} should be a video file, an image file or a directory of images")
        return input_img_list

    def read_audio_features(self,audio_path):
        whisper_feature = self.audio_processor.audio2feat(audio_path)
        whisper_chunks = self.audio_processor.feature2chunks(feature_array=whisper_feature,fps=FPS)
        return whisper_chunks

    def extract_coordinate(self,input_img_list):
        # if os.path.exists(crop_coord_save_path) and use_saved_coord:
        #     print("using extracted coordinates")
        #     with open(crop_coord_save_path,'rb') as f:
        #         coord_list = pickle.load(f)
        #     frame_list = read_imgs(input_img_list)
        # else:
        print("extracting landmarks...time consuming")
        coord_list, frame_list = get_landmark_and_bbox(input_img_list, BBOX_SHIFT)
        # with open(crop_coord_save_path, 'wb') as f:
        #     pickle.dump(coord_list, f)
                
        i = 0
        input_latent_list = []
        for bbox, frame in zip(coord_list, frame_list):
            if bbox == coord_placeholder:
                continue
            x1, y1, x2, y2 = bbox
            crop_frame = frame[y1:y2, x1:x2]
            crop_frame = cv2.resize(crop_frame,(256,256),interpolation = cv2.INTER_LANCZOS4)
            latents = self.vae.get_latents_for_unet(crop_frame)
            input_latent_list.append(latents)
    
        # to smooth the first and the last frame
        frame_list_cycle = frame_list + frame_list[::-1]
        coord_list_cycle = coord_list + coord_list[::-1]
        input_latent_list_cycle = input_latent_list + input_latent_list[::-1]
        return frame_list_cycle,coord_list_cycle,input_latent_list_cycle

    def renderer(self,whisper_chunks,input_latent_list_cycle,batch_size,timesteps):
        video_num = len(whisper_chunks)
        batch_size = batch_size
        gen = datagen(whisper_chunks,input_latent_list_cycle,batch_size)
        res_frame_list = []
        for i, (whisper_batch,latent_batch) in enumerate(tqdm(gen,total=int(np.ceil(float(video_num)/batch_size)))):
            audio_feature_batch = torch.from_numpy(whisper_batch)
            audio_feature_batch = audio_feature_batch.to(device=self.unet.device,
                                                        dtype=self.unet.model.dtype) # torch, B, 5*N,384
            audio_feature_batch = self.pe(audio_feature_batch)
            latent_batch = latent_batch.to(dtype=self.unet.model.dtype)
            
            pred_latents = self.unet.model(latent_batch, timesteps, encoder_hidden_states=audio_feature_batch).sample
            recon = self.vae.decode_latents(pred_latents)
            for res_frame in recon:
                res_frame_list.append(res_frame)
        return res_frame_list
    
    def blending_enhancer(self,res_frame_list,coord_list_cycle,frame_list_cycle,result_img_save_path):
        print("pad talking image to original video")
        for i, res_frame in enumerate(tqdm(res_frame_list)):
            bbox = coord_list_cycle[i%(len(coord_list_cycle))]
            ori_frame = copy.deepcopy(frame_list_cycle[i%(len(frame_list_cycle))])
            x1, y1, x2, y2 = bbox
            try:
                res_frame = cv2.resize(res_frame.astype(np.uint8),(x2-x1,y2-y1))
            except:
#                 print(bbox)
                continue
            combine_frame = get_image(ori_frame,res_frame,bbox)
            _, _, frame = self.gfpgan.enhance(combine_frame, has_aligned=False, only_center_face=False, paste_back=True) ##applied GFPGAN
            cv2.imwrite(f"{result_img_save_path}/{str(i).zfill(8)}.png",frame)

    def infer(self,video_path,audio_path,bg_removalflag,avatar_name,result_dir):
        input_basename = avatar_name
        audio_basename  = os.path.basename(audio_path).split('.')[0]
        output_basename = f"{input_basename}_{audio_basename}"
        result_img_save_path = os.path.join(result_dir, output_basename) # related to video & audio inputs
        output_vid_name=os.path.join(result_dir,f"{output_basename}_enhanced_audio.mp4")
        # crop_coord_save_path = os.path.join(result_img_save_path, input_basename+".pkl") # only related to video input
        os.makedirs(result_img_save_path,exist_ok =True)

        input_img_list=self.read_avatar_pose_video(video_path,input_basename,result_dir,output_basename)
        whisper_chunks=self.read_audio_features(audio_path)

        pkl_save_folder_path=os.path.join(AVATAR_PRESAVES_DIR,"presaved_files",input_basename)
        if os.path.exists(pkl_save_folder_path):
            pkl_flag=True
        else:
            os.makedirs(pkl_save_folder_path,exist_ok =True)
            pkl_flag=False
        
        if not pkl_flag:
            frame_list_cycle,coord_list_cycle,input_latent_list_cycle=self.extract_coordinate(input_img_list)
            frame_list_cycle_save_path=os.path.join(pkl_save_folder_path,"frame_list_cycle")
            with open(frame_list_cycle_save_path, 'wb') as f:     
                pickle.dump(frame_list_cycle, f)
            coord_list_cycle_save_path=os.path.join(pkl_save_folder_path,"coord_list_cycle")
            with open(coord_list_cycle_save_path, 'wb') as f:     
                pickle.dump(coord_list_cycle, f)
            input_latent_list_cycle_save_path=os.path.join(pkl_save_folder_path,"input_latent_list_cycle")
            with open(input_latent_list_cycle_save_path, 'wb') as f:     
                pickle.dump(input_latent_list_cycle, f)
        else:
            frame_list_cycle_save_path=os.path.join(pkl_save_folder_path,"frame_list_cycle")
            coord_list_cycle_save_path=os.path.join(pkl_save_folder_path,"coord_list_cycle")
            input_latent_list_cycle_save_path=os.path.join(pkl_save_folder_path,"input_latent_list_cycle")
            
            with open(frame_list_cycle_save_path,'rb') as flsi:
                frame_list_cycle = pickle.load(flsi)
        
            with open(coord_list_cycle_save_path,'rb') as flsi:
                coord_list_cycle = pickle.load(flsi)

            with open(input_latent_list_cycle_save_path,'rb') as flsi:
                input_latent_list_cycle = pickle.load(flsi)

    
        res_frame_list=self.renderer(whisper_chunks,input_latent_list_cycle,BATCH_SIZE,self.timesteps)

        self.blending_enhancer(res_frame_list,coord_list_cycle,frame_list_cycle,result_img_save_path)
        
        cmd_img2video = f"ffmpeg -y -v warning -r {FPS} -f image2 -i {result_img_save_path}/%08d.png -vcodec libx264 -vf format=rgb24,scale=out_color_matrix=bt709,format=yuv420p -crf 18 {result_dir}/temp.mp4"
        print(cmd_img2video)
        os.system(cmd_img2video)
        
        cmd_combine_audio = f"ffmpeg -y -v warning -i {audio_path} -i {result_dir}/temp.mp4 {output_vid_name}"
        print(cmd_combine_audio)
        os.system(cmd_combine_audio)
        
        # os.remove("temp.mp4")
        # shutil.rmtree(result_img_save_path)
        print(f"result is save to {output_vid_name}")
        if bg_removalflag:
            output_path=os.path.join(result_dir,"matts")
            os.makedirs(output_path,exist_ok=True)
            matting(output_vid_name,output_path,self.modnet)
            return output_path
        else:
            return output_vid_name


        

    