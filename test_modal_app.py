import modal
import time
import cv2
import os
# a=modal.Cls.lookup("vidgen-musetalk-modal-2","Model")


def make_batches(no_len,input_basename):
    MAX_BATCH_SIZE_INSTANCE=15
    total_len=no_len
    no_of_invokations=total_len//MAX_BATCH_SIZE_INSTANCE
    left_over=total_len%MAX_BATCH_SIZE_INSTANCE
    if left_over!=0:
        strt_end_tuple=[[i, i + MAX_BATCH_SIZE_INSTANCE,input_basename] for i in range(0, MAX_BATCH_SIZE_INSTANCE*no_of_invokations,MAX_BATCH_SIZE_INSTANCE)]
        strt_end_tuple+=[[MAX_BATCH_SIZE_INSTANCE*no_of_invokations,MAX_BATCH_SIZE_INSTANCE*no_of_invokations+left_over,input_basename]]
        no_of_invokations+=1
    else:
        strt_end_tuple=[[i, i + MAX_BATCH_SIZE_INSTANCE,input_basename] for i in range(0, MAX_BATCH_SIZE_INSTANCE*no_of_invokations, MAX_BATCH_SIZE_INSTANCE)]
    print(strt_end_tuple)
    return strt_end_tuple


st_1=time.time()
a=modal.Cls.lookup("vidgen-musetalk-modal-3","MyLifecycleClass")
print("Define time 1 : ",time.time()-st_1)
st_2=time.time()
a_obj=a()
print("Define time 2 : ",time.time()-st_2)
video_path="data/video/asian_30_male_1_brian_cox_3min_512x512_2_1034_withaudio.mp4"
audio_path="data/audio/elevenlabs-2min-audio.wav"
print("Test4")
# a_obj.test2.remote()
st_3=time.time()
# print(a_obj.foo.remote(test_value=1))
no_len,height,width,input_basename=a_obj.foo.remote(video_path=video_path,audio_path=audio_path,bbx_shift=0.0)
fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
# width=0
# height=0
video = cv2.VideoWriter('video_check.avi', fourcc, 24, (width, height))
print("Define time 3 : ",time.time()-st_3)
str_end_batch=make_batches(no_len,input_basename)
# print(a_obj.run_concurrency.remote())
st_4=time.time()
# for i,_ in enumerate(a_obj.run_concurrency.map([1]*100)):
#     print("run ",i)
vid_arr=[]
for i,out in enumerate(a_obj.run_concurrency.map(str_end_batch)):
    vid_arr.append(out)
    print("run ",i)
print("Define time 4 : ",time.time()-st_4)

for vid_frm in vid_arr:
    for frm in vid_frm:
        video.write(frm)
video.release()
cmd_combine_audio = f"ffmpeg -y -v warning -i {audio_path} -i video_check.avi video_check.mp4"
os.system(cmd_combine_audio)

