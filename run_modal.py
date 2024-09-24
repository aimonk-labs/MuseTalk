
from modal import Stub,Image, method,gpu,enter,App
import modal

stub = App("vidgen-musetalk-modal-3")
volume = modal.Volume.from_name("my-test-volume")
dockerfile_image = Image.from_dockerfile("docker/Dockerfile.modal")

@stub.cls(image=dockerfile_image,gpu=gpu.A100(count=1,size="40GB"),timeout=360*2,concurrency_limit=10,volumes={"/root/musetalk": volume})
class MyLifecycleClass:
    @modal.enter()
    def enter(self):
        import os
            
        # os.system(f"pip3 install ffmpeg-python")
        # os.system(f"pip3 install transformers==4.33.1")
        import ffmpeg
        
        print("succesfully ffmpeg")
        from musetalk.utils.utils import load_all_model
        from fast_gfpgan import FAST_GFGGaner
        self.audio_processor, self.vae, self.unet, self.pe = load_all_model()
        import torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.timesteps = torch.tensor([0], device="cuda")
        model_path="./gfpgan/weights/GFPGANv1.4.pth"
        self.gfpgan=FAST_GFGGaner(
            model_path=model_path,
            upscale=1,
            arch="clean",
            channel_multiplier=2,
            bg_upsampler=None,device="cuda")
        self.var = "hello world"

    @modal.method()
    def foo(self,video_path,audio_path):
        import torch
        import os
        from musetalk.utils.utils import get_file_type,get_video_fps,datagen
        from musetalk.utils.preprocessing import get_landmark_and_bbox,read_imgs,coord_placeholder
        import glob
        import cv2
        import copy
        from musetalk.utils.blending import get_image
        import pickle
        from tqdm import tqdm
        import numpy as np
        @torch.no_grad()
        def main(vp,ap):
            if False:
                # global self.pe
                video_path = vp
                audio_path = ap
                bbox_shift = 0.0
                result_dir="/root/"
                input_basename = os.path.basename(video_path).split('.')[0]
                audio_basename  = os.path.basename(audio_path).split('.')[0]
                output_basename = f"{input_basename}_{audio_basename}"
                result_img_save_path = os.path.join(result_dir, output_basename) # related to video & audio inputs
                crop_coord_save_path = os.path.join(result_img_save_path, input_basename+".pkl") # only related to video input
                os.makedirs(result_img_save_path,exist_ok =True)
                
                
                output_vid_name = os.path.join(result_dir, output_basename+".mp4")
                ############################################## extract frames from source video ##############################################
                if get_file_type(video_path)=="video":
                    save_dir_full = os.path.join(result_dir, input_basename)
                    os.makedirs(save_dir_full,exist_ok = True)
                    cmd = f"ffmpeg -v fatal -i {video_path} -start_number 0 {save_dir_full}/%08d.png"
                    os.system(cmd)
                    input_img_list = sorted(glob.glob(os.path.join(save_dir_full, '*.[jpJP][pnPN]*[gG]')))
                    fps = get_video_fps(video_path)
                # elif get_file_type(video_path)=="image":
                #     input_img_list = [video_path, ]
                #     fps = args.fps
                # elif os.path.isdir(video_path):  # input img folder
                #     input_img_list = glob.glob(os.path.join(video_path, '*.[jpJP][pnPN]*[gG]'))
                #     input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
                #     fps = args.fps
                else:
                    raise ValueError(f"{video_path} should be a video file, an image file or a directory of images")
                # import pdb;pdb.set_trace()
                
                #print(input_img_list)
                ############################################## extract audio feature ##############################################
                whisper_feature = self.audio_processor.audio2feat(audio_path)
                whisper_chunks = self.audio_processor.feature2chunks(feature_array=whisper_feature,fps=fps)
                ############################################## preprocess input image  ##############################################
                # if os.path.exists(crop_coord_save_path) and args.use_saved_coord:
                #     print("using extracted coordinates")
                #     with open(crop_coord_save_path,'rb') as f:
                #         coord_list = pickle.load(f)
                #     frame_list = read_imgs(input_img_list)
                # else:
                #     print("extracting landmarks...time consuming")
                #     coord_list, frame_list = get_landmark_and_bbox(input_img_list, bbox_shift)
                #     with open(crop_coord_save_path, 'wb') as f:
                #         pickle.dump(coord_list, f)
                print("extracting landmarks...time consuming")
                coord_list, frame_list = get_landmark_and_bbox(input_img_list, bbox_shift)
                with open(crop_coord_save_path, 'wb') as f:
                    pickle.dump(coord_list, f)
                # import pdb;pdb.set_trace()       
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
                height,width=frame_list[0].shape[:2]
            
                # to smooth the first and the last frame
                frame_list_cycle = frame_list + frame_list[::-1]
                with open("frame_list_cycle", 'wb') as f:     
                    pickle.dump(frame_list_cycle, f)
                    
                coord_list_cycle = coord_list + coord_list[::-1]
                with open("coord_list_cycle", 'wb') as f:     
                    pickle.dump(coord_list_cycle, f)
                    
                input_latent_list_cycle = input_latent_list + input_latent_list[::-1]
                with open("input_latent_list_cycle", 'wb') as f:     
                    pickle.dump(input_latent_list_cycle, f)
                    
                ############################################## inference batch by batch ##############################################
                print("start inference")
                video_num = len(whisper_chunks)
                batch_size = 8
                # import pdb;pdb.set_trace()
                gen = datagen(whisper_chunks,input_latent_list_cycle,batch_size)
                res_frame_list = []
                
                
                for i, (whisper_batch,latent_batch) in enumerate(tqdm(gen,total=int(np.ceil(float(video_num)/batch_size)))):
                    audio_feature_batch = torch.from_numpy(whisper_batch)
                    audio_feature_batch = audio_feature_batch.to(device=self.unet.device,
                                                                dtype=self.unet.model.dtype) # torch, B, 5*N,384
                    audio_feature_batch = self.pe(audio_feature_batch)
                    latent_batch = latent_batch.to(dtype=self.unet.model.dtype)
                    
                    pred_latents = self.unet.model(latent_batch, self.timesteps, encoder_hidden_states=audio_feature_batch).sample
                    recon = self.vae.decode_latents(pred_latents)
                    for res_frame in recon:
                        res_frame_list.append(res_frame)
                    
                
                with open("res_frame_list", 'wb') as f:     
                    pickle.dump(res_frame_list, f)
                f.close()
                volume.commit()
                return len(res_frame_list),height,width
            else:
                print("Running the else part")
                video_path = vp
                audio_path = ap
                bbox_shift = 0.0
                result_dir="/root/"
                input_basename = os.path.basename(video_path).split('.')[0]
                audio_basename  = os.path.basename(audio_path).split('.')[0]
                output_basename = f"{input_basename}_{audio_basename}"
                result_img_save_path = os.path.join(result_dir, output_basename) # related to video & audio inputs
                crop_coord_save_path = os.path.join(result_img_save_path, input_basename+".pkl") # only related to video input
                os.makedirs(result_img_save_path,exist_ok =True)
                fps = get_video_fps(video_path)
                
                output_vid_name = os.path.join(result_dir, output_basename+".mp4")
                # with open("res_frame_list",'rb') as flsi:
                #     res_frame_list_l = pickle.load(flsi)
            
                with open("frame_list_cycle",'rb') as flsi:
                    frame_list_cycle = pickle.load(flsi)
            
                with open("coord_list_cycle",'rb') as flsi:
                    coord_list_cycle = pickle.load(flsi)

                with open("input_latent_list_cycle",'rb') as flsi:
                    input_latent_list_cycle = pickle.load(flsi)   
                whisper_feature = self.audio_processor.audio2feat(audio_path)
                whisper_chunks = self.audio_processor.feature2chunks(feature_array=whisper_feature,fps=fps)
                print("start inference")
                video_num = len(whisper_chunks)
                batch_size = 8
                # import pdb;pdb.set_trace()
                # gen = datagen(whisper_chunks,input_latent_list_cycle,batch_size)
                gen = list(datagen(whisper_chunks,input_latent_list_cycle,batch_size))
                with open("gen", 'wb') as f:     
                    pickle.dump(gen, f)
                # res_frame_list = []
                # for i, (whisper_batch,latent_batch) in enumerate(tqdm(gen,total=int(np.ceil(float(video_num)/batch_size)))):
                #     audio_feature_batch = torch.from_numpy(whisper_batch)
                #     audio_feature_batch = audio_feature_batch.to(device=self.unet.device,
                #                                                 dtype=self.unet.model.dtype) # torch, B, 5*N,384
                #     audio_feature_batch = self.pe(audio_feature_batch)
                #     latent_batch = latent_batch.to(dtype=self.unet.model.dtype)
                    
                #     pred_latents = self.unet.model(latent_batch, self.timesteps, encoder_hidden_states=audio_feature_batch).sample
                #     recon = self.vae.decode_latents(pred_latents)
                #     for res_frame in recon:
                #         res_frame_list.append(res_frame)
                    
                
                # with open("res_frame_list", 'wb') as f:     
                #     pickle.dump(res_frame_list, f)
                # f.close()
                volume.commit()
                height,width=frame_list_cycle[0].shape[:2]
                return len(gen),height,width
                # return len(res_frame_list),height,width

        no_,height,width=main(video_path,audio_path)
        return no_,height,width
    @modal.method()
    def run_concurrency(self,ls):
        import cv2
        from musetalk.utils.blending import get_image
        import numpy as np
        import copy
        from tqdm import tqdm
        import pickle
        import os
        import torch
        print("pad talking image to original video")
        print("size: ",os.system(f"du -sh res_frame_list")," MB")
        with open("res_frame_list",'rb') as flsi:
            res_frame_list_l = pickle.load(flsi)
            
        with open("frame_list_cycle",'rb') as flsi:
            frame_list_cycle = pickle.load(flsi)
            
        with open("coord_list_cycle",'rb') as flsi:
            coord_list_cycle = pickle.load(flsi)
        with open("gen",'rb') as flsi:
            gen = pickle.load(flsi)
        # res_frame_list=self.res_frame_list[0:10]

        strt_index=ls[0]
        # end_index=ls[1]-1
        end_index=ls[1]
        print("length of res_frame_list ",len(res_frame_list_l))
        print("end index ",end_index)
        f_arr=[]
        gen=gen[strt_index:end_index]
        # res_frame_list=[]
        for i, (whisper_batch,latent_batch) in enumerate(tqdm(gen)):
            audio_feature_batch = torch.from_numpy(whisper_batch)
            audio_feature_batch = audio_feature_batch.to(device=self.unet.device,
                dtype=self.unet.model.dtype) # torch, B, 5*N,384
            audio_feature_batch = self.pe(audio_feature_batch)
            latent_batch = latent_batch.to(dtype=self.unet.model.dtype)
                    
            pred_latents = self.unet.model(latent_batch, self.timesteps, encoder_hidden_states=audio_feature_batch).sample
            recon = self.vae.decode_latents(pred_latents)
            l_recon=recon.shape[0]
            for j,res_frame in enumerate(recon):
                # res_frame_list.append(res_frame)
                # i_n=(j+strt_index)+(i*8) #0
                # i_n=((strt_index+i)*8)+j
                i_n=((strt_index+i)*l_recon)+j
                print(f"index for {strt_index} : {i_n}")
                bbox = coord_list_cycle[i_n%(len(coord_list_cycle))]
                ori_frame = copy.deepcopy(frame_list_cycle[i_n%(len(frame_list_cycle))])
                x1, y1, x2, y2 = bbox
                try:
                    res_frame = cv2.resize(res_frame.astype(np.uint8),(x2-x1,y2-y1))
                except:
                    continue
                combine_frame = get_image(ori_frame,res_frame,bbox,True)
                _, _, frame = self.gfpgan.enhance(combine_frame, has_aligned=False, only_center_face=False, paste_back=True)
                f_arr.append(frame)
        return f_arr
        # res_frame_list=res_frame_list_l[strt_index:end_index]
        # for i, res_frame in enumerate(tqdm(res_frame_list)):
        #     i_n=i+strt_index
        #     bbox = coord_list_cycle[i_n%(len(coord_list_cycle))]
        #     ori_frame = copy.deepcopy(frame_list_cycle[i_n%(len(frame_list_cycle))])
        #     x1, y1, x2, y2 = bbox
        #     try:
        #         res_frame = cv2.resize(res_frame.astype(np.uint8),(x2-x1,y2-y1))
        #     except:
        #         #                 print(bbox)
        #         continue
            
        #     # combine_frame = get_image(ori_frame,res_frame,bbox)
        #     combine_frame = get_image(ori_frame,res_frame,bbox,True)

        #     _, _, frame = self.gfpgan.enhance(combine_frame, has_aligned=False, only_center_face=False, paste_back=True)
        #     f_arr.append(frame)
            # cv2.imwrite(f"{result_img_save_path}/{str(i+index_list[0]).zfill(8)}.png",frame)

        # return f_arr
       
       






   