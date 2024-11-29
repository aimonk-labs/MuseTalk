import torch
from RobustVideoMatting.model import MattingNetwork
from RobustVideoMatting.inference import convert_video

class RVMVideoMatting:
    def __init__(self, model_type="mobilenetv3", model_path="rvm_mobilenetv3.pth", device="cuda"):
        """
        Initializes the RVM model for video matting.

        Args:
            model_type (str): Type of the model architecture ('mobilenetv3' or 'resnet50').
            model_path (str): Path to the pretrained model weights.
            device (str): Device to run the model ('cuda' or 'cpu').
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model_type = model_type

        # Initialize model based on type
        if model_type == "mobilenetv3":
            self.model = MattingNetwork("mobilenetv3").eval().to(self.device)
        elif model_type == "resnet50":
            self.model = MattingNetwork("resnet50").eval().to(self.device)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        self._load_model_weights(model_path)

    def _load_model_weights(self, model_path):
        """
        Load the pretrained weights into the model.

        Args:
            model_path (str): Path to the pretrained model weights.
        """
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Model loaded successfully from {model_path}.")
        except Exception as e:
            print(f"Error loading model weights: {e}")
            raise

    def convert_video(
        self, 
        input_source, 
        output_type="video", 
        output_composition=None, 
        output_alpha=None, 
        output_foreground=None, 
        output_video_mbps=6, 
        downsample_ratio=1, 
        seq_chunk=32
    ):
        """
        Convert a video or image sequence to apply RVM matting.

        Args:
            input_source (str): Path to the input video file or image sequence directory.
            output_type (str): "video" or "png_sequence".
            output_composition (str): Path to save the composed output.
            output_alpha (str): Path to save the raw alpha predictions.
            output_foreground (str): Path to save the raw foreground predictions.
            output_video_mbps (int): Bitrate for the output video.
            downsample_ratio (float): Downsampling ratio for input frames (or None for auto).
            seq_chunk (int): Number of frames to process at a time.
        """
        try:
            convert_video(
                self.model,
                input_source=input_source,
                output_type=output_type,
                output_composition=output_composition,
                output_alpha=output_alpha,
                output_foreground=output_foreground,
                output_video_mbps=output_video_mbps,
                downsample_ratio=downsample_ratio,
                seq_chunk=seq_chunk,
                progress=True
            )
            print("Video conversion completed successfully.")
        except Exception as e:
            print(f"Error during video conversion: {e}")
            raise

# Example Usage
if __name__ == "__main__":
    # Initialize the model with resnet50
    video_matting = RVMVideoMatting(
        model_type="resnet50",  # Change model type here
        model_path="rvm_resnet50.pth",  # Provide the path to ResNet50 weights
        device="cuda"
    )

    # Convert video with matting applied
    video_matting.convert_video(
        input_source="input.mp4",
        output_type="video",
        output_composition="com.mp4",
        output_alpha="pha.mp4",
        output_foreground="fgr.mp4",
        output_video_mbps=4,
        downsample_ratio=None,
        seq_chunk=12
    )