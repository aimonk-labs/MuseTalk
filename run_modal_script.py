import os
import subprocess
import time
import cv2
import modal
import numpy as np
import pickle

# def upload_to_modal_volume(volume_name, local_path, remote_path=None):
#     """
#     Upload a local file to a Modal volume using CLI command.
    
#     Args:
#         volume_name (str): Name of the Modal volume
#         local_path (str): Local file path to upload
#         remote_path (str, optional): Destination path in the volume. Defaults to None.
    
#     Returns:
#         str: Remote path of the uploaded file
#     """
#     # Prepare the command
#     cmd = ['modal', 'volume', 'put', volume_name, local_path]
#     if remote_path:
#         cmd.append(remote_path)
    
#     try:
#         # Run the command
#         result = subprocess.run(cmd, capture_output=True, text=True, check=True)
#         print(f"Successfully uploaded {local_path} to volume {volume_name}")
        
#         # If no remote path specified, return the local filename in the volume
#         if not remote_path:
#             remote_path = os.path.basename(local_path)
        
#         return remote_path
    
#     except subprocess.CalledProcessError as e:
#         print(f"Error uploading file: {e}")
#         print(f"STDOUT: {e.stdout}")
#         print(f"STDERR: {e.stderr}")
#         raise


def upload_to_modal_volume(volume_name, local_path, remote_path=None):
    """
    Upload a local file to a Modal volume only if it doesn't already exist.
    
    Args:
        volume_name (str): Name of the Modal volume
        local_path (str): Local file path to upload
        remote_path (str, optional): Destination path in the volume
    
    Returns:
        str: Remote path of the file (uploaded or existing)
    """
    # If no remote path specified, use the original filename
    if not remote_path:
        remote_path = os.path.basename(local_path)
    
    # Check if file already exists in the volume
    try:
        # import pdb;pdb.set_trace()
        # Attempt to list the file
        check_cmd = ['modal', 'volume', 'ls', volume_name, remote_path]
        check_result = subprocess.run(check_cmd, capture_output=True, text=True,encoding='utf-8')
        
        # If file exists, return the existing path
        if check_result.returncode == 0:
            print(f"File {remote_path} already exists in volume {volume_name}. Skipping upload.")
            return remote_path
    
    except Exception as e:
        print(f"Error checking file existence: {e}")
    
    # If file doesn't exist, upload it
    try:
        upload_cmd = ['modal', 'volume', 'put', volume_name, local_path, remote_path]
        upload_result = subprocess.run(upload_cmd, capture_output=True, text=True, check=True,encoding='utf-8')
        
        # print(f"Successfully uploaded {local_path} to volume {volume_name} as {remote_path}")
        return remote_path
    
    except subprocess.CalledProcessError as e:
        print(f"Error uploading file: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        raise

# def make_batches(no_len, input_basename, max_batch_size=15):
#     """
#     Create batches for processing video frames.
    
#     Args:
#         no_len (int): Total number of frames
#         input_basename (str): Base name for input files
#         max_batch_size (int, optional): Maximum batch size. Defaults to 15.
    
#     Returns:
#         list: List of batch tuples containing start, end, and basename
#     """
#     total_len = no_len
#     no_of_invokations = total_len // max_batch_size
#     left_over = total_len % max_batch_size

#     if left_over != 0:
#         strt_end_tuple = [
#             [i, i + max_batch_size, input_basename] 
#             for i in range(0, max_batch_size * no_of_invokations, max_batch_size)
#         ]
#         strt_end_tuple += [[
#             max_batch_size * no_of_invokations, 
#             max_batch_size * no_of_invokations + left_over, 
#             input_basename
#         ]]
#         no_of_invokations += 1
#     else:
#         strt_end_tuple = [
#             [i, i + max_batch_size, input_basename] 
#             for i in range(0, max_batch_size * no_of_invokations, max_batch_size)
#         ]
    
#     return strt_end_tuple

def make_batches(no_len, input_basename, no_of_invokations=5):
    """
    Create batches for processing video frames.

    Args:
        no_len (int): Total number of frames.
        input_basename (str): Base name for input files.
        no_of_invokations (int, optional): Desired number of invocations. Defaults to 5.

    Returns:
        list: List of batch tuples containing start, end, and basename.
    """
    # Calculate max_batch_size based on no_len and no_of_invokations
    max_batch_size = no_len // no_of_invokations
    leftover = no_len % no_of_invokations

    strt_end_tuple = []

    # Create batches
    for i in range(no_of_invokations):
        start = i * max_batch_size
        end = start + max_batch_size

        # Add leftover frames to the last batch
        if i == no_of_invokations - 1:
            end += leftover

        strt_end_tuple.append([start, end, input_basename,i])

    return strt_end_tuple

def process_video_with_modal(video_path, audio_path, volume_name="take0-modal-volume", modal_class_name="take0-musetalk-modal",final_video_path="temp.mp4",result_path=None):
    """
    Process video using Modal with specified video and audio paths.
    
    Args:
        video_path (str): Path to input video
        audio_path (str): Path to input audio
        volume_name (str): Name of the Modal volume
        modal_class_name (str, optional): Name of the Modal class to use
    
    Returns:
        str: Path to the final processed video
    """
    print("----------------------------------------STARTED------------------------------------")
    # Upload files to Modal volume
    video_remote_path = upload_to_modal_volume(volume_name, video_path,os.path.join("data/video",os.path.basename(video_path)))
    audio_remote_path = upload_to_modal_volume(volume_name, audio_path,os.path.join("data/audio",os.path.basename(audio_path)))
    
    # Timing and performance tracking
    timings = {}
    
    # Lookup Modal class and instantiate
    st_1 = time.time()
    modal_class = modal.Cls.lookup(modal_class_name, "MyLifecycleClass")
    timings['define_time_1'] = time.time() - st_1
    
    st_2 = time.time()
    modal_obj = modal_class()
    timings['define_time_2'] = time.time() - st_2
    
    # Process video and audio using remote paths
    st_3 = time.time()
    no_len, height, width, input_basename = modal_obj.foo.remote(
        video_path=f"{video_remote_path}", 
        audio_path=f"{audio_remote_path}", 
        bbx_shift=0.0
    )
    timings['define_time_3'] = time.time() - st_3
    
    # Prepare video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video_path = 'video_check.mp4'
    video = cv2.VideoWriter(output_video_path, fourcc, 24.0, (width, height))
    
    # Create batches and process
    st_4 = time.time()
    str_end_batch = make_batches(no_len, input_basename)
    
    # vid_arr = []
    # for i, out in enumerate(modal_obj.run_concurrency.map(str_end_batch)):
    #     vid_arr.append(out)

    vid_arr = [None] * len(str_end_batch)  # Create an empty list of correct size
    for i, out in enumerate(modal_obj.run_concurrency.map(str_end_batch)):
        vid_arr[i] = out  # Store at the correct index
    
    timings['define_time_4'] = time.time() - st_4
    
    # Write video frames

    c=0
    for vid_frm in vid_arr:
        for frm in vid_frm:
            if frm is None:
                print(f"⚠️ Warning: Found a `None` frame at index {c}, skipping...")
                continue  # Skip invalid frame
            if not isinstance(frm, np.ndarray):
                print(f"⚠️ Warning: Frame {c} is not a NumPy array, skipping...")
                continue  # Skip invalid frame

            if frm.shape[:2] != (height, width):
                print(f"⚠️ Warning: Frame {c} has incorrect dimensions {frm.shape}, expected ({height}, {width}), skipping...")
                continue  # Skip frames with wrong resolution
            video.write(frm)
            if result_path:
                output_path = os.path.join(result_path, f'{c:04d}.png')
                cv2.imwrite(output_path, frm)
                c=c+1
    video.release()

    # Combine audio with video
    cmd_combine_audio = f"ffmpeg -y -v warning -i {audio_path} -i {output_video_path} {final_video_path}"
    os.system(cmd_combine_audio)
    
    # Print timings
    for key, value in timings.items():
        print(f"{key}: {value}")
    
    return final_video_path

def main():
    # Example usage
    volume_name = "my-test-volume"  # Replace with your actual Modal volume name
    video_path = "/vidgen/VIDGEN_AI_CMPT_MODAL/vidgen_ai_cmpt/models/vidgen_editor/1/MuseTalk/data/video/video1.mp4"
    audio_path = "/vidgen/VIDGEN_AI_CMPT_MODAL/vidgen_ai_cmpt/models/vidgen_editor/1/MuseTalk/data/audio/audio1.wav"
    
    final_video = process_video_with_modal(video_path, audio_path, volume_name)
    print(f"Processed video saved at: {final_video}")

if __name__ == "__main__":
    main()