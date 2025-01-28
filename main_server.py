from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
from scripts.inference_gen import run_musetalk

app = FastAPI()

# Allow all CORS origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods
    allow_headers=["*"],  # Allows all headers
)

class InferenceRequest(BaseModel):
    input_video_folder: str

@app.post("/run_inference")
def run_inference(request: InferenceRequest):
    input_video_folder = request.input_video_folder
    result_dir = "silence_output_3"
    inference_config = "configs/inference/test.yaml"
    print("Inference Started")
    # Validate input folder
    input_folder_path = Path(input_video_folder)
    if not input_folder_path.exists() or not input_folder_path.is_dir():
        raise HTTPException(status_code=400, detail="Invalid input_video_folder path")
    
    # Call the inference process
    result_message = run_musetalk(input_video_folder=input_video_folder, result_dir=result_dir, inference_config=inference_config)
    
    return {"message": result_message}