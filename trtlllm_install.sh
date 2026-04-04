pip install uv
uv pip install torch==2.9.1 torchvision --index-url https://download.pytorch.org/whl/cu130
sudo apt-get -y install libopenmpi-dev aria2
CURRENT_TORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
echo "torch==$CURRENT_TORCH_VERSION" > /tmp/torch-constraint.txt
aria2c -x 16 -s 16 "https://pypi.nvidia.com/tensorrt-llm/tensorrt_llm-1.2.0-cp312-cp312-linux_x86_64.whl" -d /tmp/
uv pip install /tmp/tensorrt_llm-1.2.0-cp312-cp312-linux_x86_64.whl \
  --extra-index-url https://pypi.nvidia.com/ \
  --extra-index-url https://download.pytorch.org/whl/cu130 \
  --index-strategy unsafe-best-match \
  --prerelease=allow \
  -c /tmp/torch-constraint.txt