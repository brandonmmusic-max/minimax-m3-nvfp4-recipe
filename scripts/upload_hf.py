#!/usr/bin/env python3
"""Upload NVFP4 checkpoint + master amax to brandonmusic/MiniMax-M3-NVFP4."""
from huggingface_hub import HfApi, create_repo
api = HfApi()
repo = "brandonmusic/MiniMax-M3-NVFP4"
create_repo(repo, repo_type="model", exist_ok=True, private=False)
api.upload_folder(folder_path="/workspace/MiniMax-M3-NVFP4", repo_id=repo,
                  ignore_patterns=["amax_checkpoint.safetensors"])
api.upload_file(path_or_fileobj="/workspace/m3_master_amax.safetensors",
                path_in_repo="calibration/m3_master_amax.safetensors", repo_id=repo)
api.upload_file(path_or_fileobj="/workspace/msa_golden/msa_sm100_golden_v1.safetensors",
                path_in_repo="msa_golden/msa_sm100_golden_v1.safetensors", repo_id=repo)
print("UPLOAD COMPLETE:", repo)
