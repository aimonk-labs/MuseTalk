# import argparse
# import os
# from omegaconf import OmegaConf
# import numpy as np
# import cv2
# import torch
# import glob
# import pickle
# from tqdm import tqdm
# import copy

# from musetalk.utils.utils import get_file_type,get_video_fps,datagen
# from musetalk.utils.preprocessing import get_landmark_and_bbox,read_imgs,coord_placeholder
# from musetalk.utils.blending import get_image
# from musetalk.utils.utils import load_all_model
# import shutil
# from fast_gfpgan import FAST_GFGGaner

# # load model weights
# audio_processor, vae, unet, pe = load_all_model()
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# timesteps = torch.tensor([0], device=device)
# model_path="/bv3/debasish_works/MuseTalk/GFPGANv1.4.pth"
# gfpgan=FAST_GFGGaner(
#         model_path=model_path,
#         upscale=1,
#         arch="clean",
#         channel_multiplier=2,
#         bg_upsampler=None,device="cuda") 
# @torch.no_grad()
# def main(args):
#     global pe
#     if args.use_float16 is True:
#         pe = pe.half()
#         vae.vae = vae.vae.half()
#         unet.model = unet.model.half()
    
#     inference_config = OmegaConf.load(args.inference_config)
#     print(inference_config)
#     for task_id in inference_config:
#         video_path = inference_config[task_id]["video_path"]
#         audio_path = inference_config[task_id]["audio_path"]
#         bbox_shift = inference_config[task_id].get("bbox_shift", args.bbox_shift)

#         input_basename = os.path.basename(video_path).split('.')[0]
#         audio_basename  = os.path.basename(audio_path).split('.')[0]
#         output_basename = f"{input_basename}_{audio_basename}"
#         result_img_save_path = os.path.join(args.result_dir, output_basename) # related to video & audio inputs
#         crop_coord_save_path = os.path.join(result_img_save_path, input_basename+".pkl") # only related to video input
#         os.makedirs(result_img_save_path,exist_ok =True)
        
#         if args.output_vid_name is None:
#             output_vid_name = os.path.join(args.result_dir, output_basename+".mp4")
#         else:
#             output_vid_name = os.path.join(args.result_dir, args.output_vid_name)
#         ############################################## extract frames from source video ##############################################
#         if get_file_type(video_path)=="video":
#             save_dir_full = os.path.join(args.result_dir, input_basename)
#             os.makedirs(save_dir_full,exist_ok = True)
#             cmd = f"ffmpeg -v fatal -i {video_path} -start_number 0 {save_dir_full}/%08d.png"
#             os.system(cmd)
#             input_img_list = sorted(glob.glob(os.path.join(save_dir_full, '*.[jpJP][pnPN]*[gG]')))
#             fps = get_video_fps(video_path)
#         elif get_file_type(video_path)=="image":
#             input_img_list = [video_path, ]
#             fps = args.fps
#         elif os.path.isdir(video_path):  # input img folder
#             input_img_list = glob.glob(os.path.join(video_path, '*.[jpJP][pnPN]*[gG]'))
#             input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
#             fps = args.fps
#         else:
#             raise ValueError(f"{video_path} should be a video file, an image file or a directory of images")
#         # import pdb;pdb.set_trace()

#         #print(input_img_list)
#         ############################################## extract audio feature ##############################################
#         whisper_feature = audio_processor.audio2feat(audio_path)
#         whisper_chunks = audio_processor.feature2chunks(feature_array=whisper_feature,fps=fps)
#         ############################################## preprocess input image  ##############################################
#         if os.path.exists(crop_coord_save_path) and args.use_saved_coord:
#             print("using extracted coordinates")
#             with open(crop_coord_save_path,'rb') as f:
#                 coord_list = pickle.load(f)
#             frame_list = read_imgs(input_img_list)
#         else:
#             print("extracting landmarks...time consuming")
#             coord_list, frame_list = get_landmark_and_bbox(input_img_list, bbox_shift)
#             with open(crop_coord_save_path, 'wb') as f:
#                 pickle.dump(coord_list, f)
#         # import pdb;pdb.set_trace()       
#         i = 0
#         input_latent_list = []
#         for bbox, frame in zip(coord_list, frame_list):
#             if bbox == coord_placeholder:
#                 continue
#             x1, y1, x2, y2 = bbox
#             crop_frame = frame[y1:y2, x1:x2]
#             crop_frame = cv2.resize(crop_frame,(256,256),interpolation = cv2.INTER_LANCZOS4)
#             latents = vae.get_latents_for_unet(crop_frame)
#             input_latent_list.append(latents)
    
#         # to smooth the first and the last frame
#         frame_list_cycle = frame_list + frame_list[::-1]
#         coord_list_cycle = coord_list + coord_list[::-1]
#         input_latent_list_cycle = input_latent_list + input_latent_list[::-1]
#         ############################################## inference batch by batch ##############################################
#         print("start inference")
#         video_num = len(whisper_chunks)
#         batch_size = args.batch_size
#         gen = datagen(whisper_chunks,input_latent_list_cycle,batch_size)
#         res_frame_list = []
#         for i, (whisper_batch,latent_batch) in enumerate(tqdm(gen,total=int(np.ceil(float(video_num)/batch_size)))):
#             audio_feature_batch = torch.from_numpy(whisper_batch)
#             audio_feature_batch = audio_feature_batch.to(device=unet.device,
#                                                          dtype=unet.model.dtype) # torch, B, 5*N,384
#             audio_feature_batch = pe(audio_feature_batch)
#             latent_batch = latent_batch.to(dtype=unet.model.dtype)
            
#             pred_latents = unet.model(latent_batch, timesteps, encoder_hidden_states=audio_feature_batch).sample
#             recon = vae.decode_latents(pred_latents)
#             for res_frame in recon:
#                 res_frame_list.append(res_frame)
                
#         ############################################## pad to full image ##############################################
#         print("pad talking image to original video")
#         for i, res_frame in enumerate(tqdm(res_frame_list)):
#             bbox = coord_list_cycle[i%(len(coord_list_cycle))]
#             ori_frame = copy.deepcopy(frame_list_cycle[i%(len(frame_list_cycle))])
#             x1, y1, x2, y2 = bbox
#             try:
#                 res_frame = cv2.resize(res_frame.astype(np.uint8),(x2-x1,y2-y1))
#             except:
# #                 print(bbox)
#                 continue
            
#             # combine_frame = get_image(ori_frame,res_frame,bbox)
#             combine_frame = get_image(ori_frame,res_frame,bbox,True)

#             _, _, frame = gfpgan.enhance(combine_frame, has_aligned=False, only_center_face=False, paste_back=True)
#             cv2.imwrite(f"{result_img_save_path}/{str(i).zfill(8)}.png",frame)

#         cmd_img2video = f"ffmpeg -y -v warning -r {fps} -f image2 -i {result_img_save_path}/%08d.png -vcodec libx264 -vf format=rgb24,scale=out_color_matrix=bt709,format=yuv420p -crf 18 temp.mp4"
#         print(cmd_img2video)
#         os.system(cmd_img2video)
        
#         cmd_combine_audio = f"ffmpeg -y -v warning -i {audio_path} -i temp.mp4 {output_vid_name}"
#         print(cmd_combine_audio)
#         os.system(cmd_combine_audio)
        
#         # os.remove("temp.mp4")
#         # shutil.rmtree(result_img_save_path)
#         print(f"result is save to {output_vid_name}")

##Defining modal codebase from here
from modal import Stub,Image, method,gpu,enter,App
import modal

stub = App("vidgen-musetalk-modal-2")
volume = modal.Volume.from_name("my-test-volume")
# dockerfile_image = Image.from_dockerfile("docker/Dockerfile.modal").pip_install("fmpeg-python").copy_local_dir("./../MuseTalk", remote_path="/root")
# dockerfile_image = Image.from_dockerfile("docker/Dockerfile.modal").copy_local_dir("./../MuseTalk", remote_path="/root")
dockerfile_image = Image.from_dockerfile("docker/Dockerfile.modal")


# .copy_local_dir("./../../MuseTalk/GFPGANv1.4.pth", remote_path="/root")
    # .copy_local_dir("models/IP_LAP/1/IP_LAP", remote_path="/vidgen") \
    # .copy_local_dir("models/IP_LAP/1/checkpoints",remote_path="/checkpoints").copy_local_dir("models/IP_LAP/1/temp_results",remote_path="/temp_results/") \
    #     .copy_local_dir("models/vidgen_editor/1/IP_LAP_temp_results/audio",remote_path="/IP_LAP_temp_results/audio").copy_local_dir("models/vidgen_editor/1/mcnet_predefined_avatar_videos",remote_path="/mcnet_predefined_avatar_videos")  
        
# @stub.cls(image=dockerfile_image,gpu=gpu.A10G(count=1),timeout=360,concurrency_limit=10,volumes={"/root/musetalk": volume})
# class Model():
#     # def __init__(self):
#     #     print("Intialized")
#         # import os
        
#         # # os.system(f"pip3 install ffmpeg-python")
#         # # os.system(f"pip3 install transformers==4.33.1")
#         # import ffmpeg
        
#         # print("succesfully ffmpeg")
#         # from musetalk.utils.utils import load_all_model
#         # from fast_gfpgan import FAST_GFGGaner
#         # self.audio_processor, self.vae, self.unet, self.pe = load_all_model()
#         # import torch
#         # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         # self.timesteps = torch.tensor([0], device="cuda")
#         # model_path="./gfpgan/weights/GFPGANv1.4.pth"
#         # self.gfpgan=FAST_GFGGaner(
#         #     model_path=model_path,
#         #     upscale=1,
#         #     arch="clean",
#         #     channel_multiplier=2,
#         #     bg_upsampler=None,device="cuda")  
#         # global pe
#         ## to speed up the inference
#         # if args.use_float16 is True:
#         #     pe = pe.half()
#         #     vae.vae = vae.vae.half()
#         #     unet.model = unet.model.half()
#     @modal.enter()
#     def enter(self):
        
#         import os
        
#         # os.system(f"pip3 install ffmpeg-python")
#         # os.system(f"pip3 install transformers==4.33.1")
#         import ffmpeg
        
#         print("succesfully ffmpeg")
#         from musetalk.utils.utils import load_all_model
#         from fast_gfpgan import FAST_GFGGaner
#         self.audio_processor, self.vae, self.unet, self.pe = load_all_model()
#         import torch
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.timesteps = torch.tensor([0], device="cuda")
#         model_path="./gfpgan/weights/GFPGANv1.4.pth"
#         self.gfpgan=FAST_GFGGaner(
#             model_path=model_path,
#             upscale=1,
#             arch="clean",
#             channel_multiplier=2,
#             bg_upsampler=None,device="cuda")
    
#     @modal.method()
#     def test2(self,i):
#         import os
#         print("Path ",os.system("pwd"))
#         print("Files ",os.system("ls"))
#         print("This code is running on a remote worker!", self.gfpgan)
#         return "RAN2"
    
# import modal

# app = modal.App("my-shared-app")


@stub.cls(image=dockerfile_image,gpu=gpu.A10G(count=1),timeout=360,concurrency_limit=10,volumes={"/root/musetalk": volume})
class MyLifecycleClass:
    @modal.enter()
    def enter(self):
        self.var = "hello world"

    @modal.method()
    def foo(self):
        return self.var
    # @method()    
    # def get_batch_numbers(self,video_num,total_len,batch_size):
    #     self.MAX_BATCH_SIZE_INSTANCE=100
    #     total_len=total_len
    #     no_of_invokiations=total_len//self.MAX_BATCH_SIZE_INSTANCE
    #     left_over=total_len%self.MAX_BATCH_SIZE_INSTANCE
        
    #     # print("RAN")
    #     if left_over!=0:
    #         strt_end_tuple=[[i, i + self.MAX_BATCH_SIZE_INSTANCE] for i in range(0, self.MAX_BATCH_SIZE_INSTANCE*no_of_invokations, self.MAX_BATCH_SIZE_INSTANCE)]

    #         strt_end_tuple+=[[self.MAX_BATCH_SIZE_INSTANCE*no_of_invokations,self.MAX_BATCH_SIZE_INSTANCE*no_of_invokations+left_over]]
    #         no_of_invokations+=1
    #     else:
    #         strt_end_tuple=[[i, i + self.MAX_BATCH_SIZE_INSTANCE] for i in range(0, self.MAX_BATCH_SIZE_INSTANCE*no_of_invokations, self.MAX_BATCH_SIZE_INSTANCE)]

    #     print(strt_end_tuple)
    #     return strt_end_tuple
    #     # return int(np.ceil(float(video_num)/batch_size))
                   
    # @method()
    # def run_inference(self,strt_index,end_index):
        
    #     strt_index=strt_index
    #     end_index=end_index
       
    #     gen = datagen(whisper_chunks,input_latent_list_cycle,batch_size)
    #     self.res_frame_list = []
    #     ## take in the server
    #     final_frame=[]
    #     for i, (whisper_batch,latent_batch) in enumerate(tqdm(gen,total=int(np.ceil(float(video_num)/batch_size)))):
            
    #         if i>=strt_index and i<=end_index:
    #             audio_feature_batch = torch.from_numpy(whisper_batch)
    #             audio_feature_batch = audio_feature_batch.to(device=unet.device,
    #                                                             dtype=unet.model.dtype) # torch, B, 5*N,384
    #             audio_feature_batch = pe(audio_feature_batch)
    #             latent_batch = latent_batch.to(dtype=unet.model.dtype)
                
    #             pred_latents = unet.model(latent_batch, timesteps, encoder_hidden_states=audio_feature_batch).sample
    #             recon = vae.decode_latents(pred_latents)
    #             for res_frame in recon:
    #                 self.res_frame_list.append(res_frame)
    #             i_n=i+strt_index  
    #             # bbox = coord_list_cycle[i%(len(coord_list_cycle))]
    #             bbox = coord_list_cycle[i_n%(len(coord_list_cycle))]
    #             # ori_frame = copy.deepcopy(frame_list_cycle[i%(len(frame_list_cycle))])
    #             ori_frame = copy.deepcopy(frame_list_cycle[i_n%(len(frame_list_cycle))])

    #             x1, y1, x2, y2 = bbox
    #             try:
    #                 res_frame = cv2.resize(res_frame.astype(np.uint8),(x2-x1,y2-y1))
    #             except:
    #                 continue

    #             # combine_frame = get_image(ori_frame,res_frame,bbox)
    #             combine_frame = get_image(ori_frame,res_frame,bbox,True)

    #             _, _, frame = gfpgan.enhance(combine_frame, has_aligned=False, only_center_face=False, paste_back=True)
    #             final_frame.append(frame)  
    #     return final_frame
       
       
       
        
# @stub.local_entrypoint()
# def musetalk_entrypoint():     
#     video_path = ""
#     audio_path = ""
#     bbox_shift = 0
#     m=Model().test.remote()





    # input_basename = os.path.basename(video_path).split('.')[0]
    # audio_basename  = os.path.basename(audio_path).split('.')[0]
    # output_basename = f"{input_basename}_{audio_basename}"
    # result_img_save_path = os.path.join(args.result_dir, output_basename) # related to video & audio inputs
    # crop_coord_save_path = os.path.join(result_img_save_path, input_basename+".pkl") # only related to video input
    
    # if args.output_vid_name is None:
    #     output_vid_name = os.path.join(args.result_dir, output_basename+".mp4")
    # else:
    #     output_vid_name = os.path.join(args.result_dir, args.output_vid_name)
        
    # if get_file_type(video_path)=="video":
    #     save_dir_full = os.path.join(args.result_dir, input_basename)
    #     os.makedirs(save_dir_full,exist_ok = True)
    #     cmd = f"ffmpeg -v fatal -i {video_path} -start_number 0 {save_dir_full}/%08d.png"
    #     os.system(cmd)
    #     input_img_list = sorted(glob.glob(os.path.join(save_dir_full, '*.[jpJP][pnPN]*[gG]')))
    #     fps = get_video_fps(video_path)
    # elif get_file_type(video_path)=="image":
    #     input_img_list = [video_path, ]
    #     fps = args.fps
    # elif os.path.isdir(video_path):  # input img folder
    #     input_img_list = glob.glob(os.path.join(video_path, '*.[jpJP][pnPN]*[gG]'))
    #     input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    #     fps = args.fps
    # else:
    #     raise ValueError(f"{video_path} should be a video file, an image file or a directory of images")
    
    # whisper_feature = audio_processor.audio2feat(audio_path)
    # whisper_chunks = audio_processor.feature2chunks(feature_array=whisper_feature,fps=fps)
    
    # ############################################## preprocess input image  ##############################################
    # # if os.path.exists(crop_coord_save_path) and args.use_saved_coord:
    # #     print("using extracted coordinates")
    # #     with open(crop_coord_save_path,'rb') as f:
    # #         coord_list = pickle.load(f)
    # #     frame_list = read_imgs(input_img_list)
    # # else:
    # print("extracting landmarks...time consuming")
    # coord_list, frame_list = get_landmark_and_bbox(input_img_list, bbox_shift)
    # with open(crop_coord_save_path, 'wb') as f:
    #     pickle.dump(coord_list, f)
    # ####################################################################################################################     
      
    # i = 0
    # input_latent_list = []
    # for bbox, frame in zip(coord_list, frame_list):
    #     if bbox == coord_placeholder:
    #         continue
    #     x1, y1, x2, y2 = bbox
    #     crop_frame = frame[y1:y2, x1:x2]
    #     crop_frame = cv2.resize(crop_frame,(256,256),interpolation = cv2.INTER_LANCZOS4)
    #     latents = vae.get_latents_for_unet(crop_frame)
    #     input_latent_list.append(latents)

    # # to smooth the first and the last frame
    # frame_list_cycle = frame_list + frame_list[::-1]
    # coord_list_cycle = coord_list + coord_list[::-1]
    # input_latent_list_cycle = input_latent_list + input_latent_list[::-1]
    # ############################################## inference batch by batch ##############################################
    # print("start inference")
    # video_num = len(whisper_chunks)
    # batch_size = batch_size
    # total=int(np.ceil(float(video_num)/batch_size))
    # MAX_BATCH_SIZE_INSTANCE=100
    # total_len=total
    # no_of_invokations=total_len//MAX_BATCH_SIZE_INSTANCE
    # left_over=total_len%MAX_BATCH_SIZE_INSTANCE
        
    # # print("RAN")
    # if left_over!=0:
    #     strt_end_tuple=[[i, i + MAX_BATCH_SIZE_INSTANCE] for i in range(0, MAX_BATCH_SIZE_INSTANCE*no_of_invokations, MAX_BATCH_SIZE_INSTANCE)]

    #     strt_end_tuple+=[[MAX_BATCH_SIZE_INSTANCE*no_of_invokations,MAX_BATCH_SIZE_INSTANCE*no_of_invokations+left_over]]
    #     no_of_invokations+=1
    # else:
    #     strt_end_tuple=[[i, i + MAX_BATCH_SIZE_INSTANCE] for i in range(0, MAX_BATCH_SIZE_INSTANCE*no_of_invokations, MAX_BATCH_SIZE_INSTANCE)]

    # print(strt_end_tuple)
    # out_stream_2 = cv2.VideoWriter('{}/result_modal.avi'.format(tmp_dir), cv2.VideoWriter_fourcc(*'DIVX'), 25,
    #                             (frame_w,frame_h))   ##change the dimension 
    ##concurrency logic starts from here
    # for result in Model(i_audio,i_input_video).output_renderer.map(strt_end_tuple):  
    #     print("result no :",len(result[0])," index : ",result[1])
    #     arr.append(result)
    
    ##above probably needed to be transferred to the class init
    
    
    # gen = datagen(whisper_chunks,input_latent_list_cycle,batch_size)
    # res_frame_list = []
    
    ## take in the server
    # for i, (whisper_batch,latent_batch) in enumerate(tqdm(gen,total=int(np.ceil(float(video_num)/batch_size)))):
    #     audio_feature_batch = torch.from_numpy(whisper_batch)
    #     audio_feature_batch = audio_feature_batch.to(device=unet.device,
    #                                                     dtype=unet.model.dtype) # torch, B, 5*N,384
    #     audio_feature_batch = pe(audio_feature_batch)
    #     latent_batch = latent_batch.to(dtype=unet.model.dtype)
        
    #     pred_latents = unet.model(latent_batch, timesteps, encoder_hidden_states=audio_feature_batch).sample
    #     recon = vae.decode_latents(pred_latents)
    #     for res_frame in recon:
    #         res_frame_list.append(res_frame)
    
    ##concurrency logic starts from here
    ####################################################
    # bbox = coord_list_cycle[i%(len(coord_list_cycle))]
    # ori_frame = copy.deepcopy(frame_list_cycle[i%(len(frame_list_cycle))])
    # x1, y1, x2, y2 = bbox
    # try:
    #     res_frame = cv2.resize(res_frame.astype(np.uint8),(x2-x1,y2-y1))
    # except:
    #     continue
    
    # # combine_frame = get_image(ori_frame,res_frame,bbox)
    # combine_frame = get_image(ori_frame,res_frame,bbox,True)

    # _, _, frame = gfpgan.enhance(combine_frame, has_aligned=False, only_center_face=False, paste_back=True)
    # cv2.imwrite(f"{result_img_save_path}/{str(i).zfill(8)}.png",frame)  
        
        
        
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--inference_config", type=str, default="configs/inference/test_img.yaml")
#     parser.add_argument("--bbox_shift", type=int, default=0)
#     parser.add_argument("--result_dir", default='./results', help="path to output")

#     parser.add_argument("--fps", type=int, default=25)
#     parser.add_argument("--batch_size", type=int, default=8)
#     parser.add_argument("--output_vid_name", type=str, default=None)
#     parser.add_argument("--use_saved_coord",
#                         action="store_true",
#                         help='use saved coordinate to save time')
#     parser.add_argument("--use_float16",
#                         action="store_true",
#                         help="Whether use float16 to speed up inference",
#     )

#     args = parser.parse_args()
#     main(args)
