from huggingface_hub import hf_hub_download

# 直接尝试拉 gated repo 的一个小文件
path = hf_hub_download(
    "black-forest-labs/FLUX.1-dev",
    "model_index.json",
)
print("OK, downloaded to:", path)

