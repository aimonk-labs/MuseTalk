
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
    def foo(self,video_path,audio_path,bbx_shift):
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
        def main(vp,ap,bbx_shift):
            input_basename = os.path.basename(vp).split('.')[0]
            pkl_save_folder_path=os.path.join("/root/musetalk/","presaved_files",input_basename)
            if os.path.exists(pkl_save_folder_path):
                Flag=True
            else:
                Flag=False

            if Flag:
                video_path = vp
                audio_path = ap
                bbox_shift = bbx_shift
                result_dir="/root/"
                input_basename = os.path.basename(video_path).split('.')[0]
                audio_basename  = os.path.basename(audio_path).split('.')[0]
                output_basename = f"{input_basename}_{audio_basename}"                
                ############################################## extract frames from source video ##############################################
                if get_file_type(video_path)=="video":
                    save_dir_full = os.path.join(result_dir, input_basename)
                    os.makedirs(save_dir_full,exist_ok = True)
                    cmd = f"ffmpeg -v fatal -i {video_path} -start_number 0 {save_dir_full}/%08d.png"
                    os.system(cmd)
                    input_img_list = sorted(glob.glob(os.path.join(save_dir_full, '*.[jpJP][pnPN]*[gG]')))
                    fps = get_video_fps(video_path)
                else:
                    raise ValueError(f"{video_path} should be a video file, an image file or a directory of images")
                
                ############################################## extract audio feature ##############################################
                whisper_feature = self.audio_processor.audio2feat(audio_path)
                whisper_chunks = self.audio_processor.feature2chunks(feature_array=whisper_feature,fps=fps)
                ############################################## preprocess input image  ##############################################
                print("extracting landmarks...time consuming")
                coord_list, frame_list = get_landmark_and_bbox(input_img_list, bbox_shift)
     
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
                pkl_save_folder_path=os.path.join("/root/musetalk/","presaved_files",input_basename)
                os.makedirs(pkl_save_folder_path,exist_ok=True)
                # to smooth the first and the last frame

                frame_list_cycle = frame_list + frame_list[::-1]
                frame_list_cycle_save_path=os.path.join(pkl_save_folder_path,"frame_list_cycle")
                with open(frame_list_cycle_save_path, 'wb') as f:     
                    pickle.dump(frame_list_cycle, f)
                
                coord_list_cycle = coord_list + coord_list[::-1]
                coord_list_cycle_save_path=os.path.join(pkl_save_folder_path,"coord_list_cycle")
                with open(coord_list_cycle_save_path, 'wb') as f:     
                    pickle.dump(coord_list_cycle, f)
                    
                input_latent_list_cycle = input_latent_list + input_latent_list[::-1]
                input_latent_list_cycle_save_path=os.path.join(pkl_save_folder_path,"input_latent_list_cycle")
                with open(input_latent_list_cycle_save_path, 'wb') as f:     
                    pickle.dump(input_latent_list_cycle, f)
                    
                ############################################## inference batch by batch ##############################################
                print("start inference")
                # video_num = len(whisper_chunks)
                batch_size = 8

                gen = list(datagen(whisper_chunks,input_latent_list_cycle,batch_size))
                gen_save_path=os.path.join(pkl_save_folder_path,"gen")
                with open(gen_save_path, 'wb') as f:     
                    pickle.dump(gen, f)

                f.close()
                volume.commit()
                return len(gen),height,width
            else:
                print("Running the presaved part")
                video_path = vp
                audio_path = ap
                bbox_shift = bbx_shift
                result_dir="/root/"
                input_basename = os.path.basename(video_path).split('.')[0]
                audio_basename  = os.path.basename(audio_path).split('.')[0]
                fps = get_video_fps(video_path)

                pkl_save_folder_path=os.path.join("/root/musetalk/","presaved_files",input_basename)
                frame_list_cycle_save_path=os.path.join(pkl_save_folder_path,"frame_list_cycle")
                coord_list_cycle_save_path=os.path.join(pkl_save_folder_path,"coord_list_cycle")
                input_latent_list_cycle_save_path=os.path.join(pkl_save_folder_path,"input_latent_list_cycle")
                
                with open(frame_list_cycle_save_path,'rb') as flsi:
                    frame_list_cycle = pickle.load(flsi)
            
                with open(coord_list_cycle_save_path,'rb') as flsi:
                    coord_list_cycle = pickle.load(flsi)

                with open(input_latent_list_cycle_save_path,'rb') as flsi:
                    input_latent_list_cycle = pickle.load(flsi)

                whisper_feature = self.audio_processor.audio2feat(audio_path)
                whisper_chunks = self.audio_processor.feature2chunks(feature_array=whisper_feature,fps=fps)
                print("start inference")
                # video_num = len(whisper_chunks)
                batch_size = 8
                
                gen = list(datagen(whisper_chunks,input_latent_list_cycle,batch_size))
                gen_save_path=os.path.join(pkl_save_folder_path,"gen")
                with open(gen_save_path, 'wb') as f:     
                    pickle.dump(gen, f)

                volume.commit()
                height,width=frame_list_cycle[0].shape[:2]
                return len(gen),height,width,input_basename

        no_,height,width,input_basename=main(video_path,audio_path,bbx_shift)
        return no_,height,width,input_basename
    
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
        # print("size: ",os.system(f"du -sh res_frame_list")," MB")
        strt_index=ls[0]
        end_index=ls[1]
        input_basename=ls[2]

        pkl_save_folder_path=os.path.join("/root/musetalk/","presaved_files",input_basename)
        frame_list_cycle_save_path=os.path.join(pkl_save_folder_path,"frame_list_cycle")
        coord_list_cycle_save_path=os.path.join(pkl_save_folder_path,"coord_list_cycle")
        gen_save_path=os.path.join(pkl_save_folder_path,"gen")

        with open(frame_list_cycle_save_path,'rb') as flsi:
            frame_list_cycle = pickle.load(flsi)
        with open(coord_list_cycle_save_path,'rb') as flsi:
            coord_list_cycle = pickle.load(flsi)
        with open(gen_save_path,'rb') as flsi:
            gen = pickle.load(flsi)

        
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
       






   