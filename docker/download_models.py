import os
import json
import hashlib
import argparse

import boto3
from botocore.exceptions import NoCredentialsError

calculate_md5 = lambda filepath: hashlib.md5(open(filepath, 'rb').read()).hexdigest()

def download_from_s3(bucket_name, s3_file_key, local_file_path):
    # Create an S3 client
    s3 = boto3.client('s3')

    try:
        # Download the file
        s3.download_file(bucket_name, s3_file_key, local_file_path)
        print(f"File downloaded successfully to {local_file_path}")
    except FileNotFoundError:
        print("The file was not found on S3")
    except NoCredentialsError:
        print("Credentials not available")
        
def parse_args():
    args = argparse.ArgumentParser(
        description="Download model checkpoints from S3"
    )
    args.add_argument(
        "--models", choices=['all', 'tortoise', 'nemo', 'sadtalker', 'gfpgan', 'iplap', 'musetalk'],
        default='all', help="Which models to download"
    )
    args.add_argument(
        "--skip_hash", action="store_true", help="Skip checking md5 checksum of files"
    )
    args.add_argument(
        "--json", default="docker/s3_models.json", 
        help="Json file containing s3 models link"
    )
    return args.parse_args()
        
def main():
    args = parse_args()
    
    with open(args.json, 'r') as f:
        s3_models_conf = json.load(f)
        
    if args.models == "all":
        models_to_download = list(s3_models_conf.keys())
    else:
        models_to_download = [args.models]
        
    for model_to_download in models_to_download:
        model_conf = s3_models_conf[model_to_download]
        bucket = model_conf['bucket']
        
        for file in model_conf['files']:
            print("Checking ", file['save_path'])
            if os.path.exists(file['save_path']):
                md5sum = calculate_md5(file['save_path'])
                if not args.skip_hash:
                    if md5sum == file['md5sum']:
                        print(f"File {file['save_path']} already exists, skipping")
                        continue
                    else:
                        print(f"MD5 sum of file {md5sum} does not match "
                            "{file['md5sum']}. Redownloading file.")
                else:
                    print(f"File {file['save_path']} already exists, skipping")
                    continue
            
            print("Downloading ", file['save_path'])
            try:
                download_from_s3(bucket, file['s3_key'], file['save_path'])
            except:
                print("downloading using wget")
                os.system(f"wget https://vidgendata.s3.us-west-2.amazonaws.com/{file['s3_key']} -O {file['save_path']}")
    
if __name__ == '__main__':
    main()
