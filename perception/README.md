# Perception

| Folder | Content |
|---|---|
| [`sam3_detection/`](sam3_detection/) | Current top-plane-first RGB-D baseline, plus preserved SAM 3 RGB and earlier RGB-D approaches |

## SAM 3 setup

SAM 3: https://github.com/facebookresearch/sam3 (model weights: https://huggingface.co/facebook/sam3)

The setup below is the one used to produce the results in `sam3_detection/`. It used an NVIDIA RTX 4080 SUPER with a CUDA 12.8 PyTorch build. SAM 3 needs Python 3.12+, PyTorch 2.7+ and a CUDA 12.6+ GPU.

1. Create the conda environment:

   ```bash
   conda create -n sam3 python=3.12
   conda activate sam3
   ```

2. Install PyTorch with CUDA support:

   ```bash
   pip install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128
   ```

3. Clone SAM 3 and install it:

   ```bash
   git clone https://github.com/facebookresearch/sam3.git
   cd sam3
   pip install -e .
   cd ..
   ```

4. Install the packages the detection script uses. It loads SAM 3 through Hugging Face `transformers`:

   ```bash
   pip install transformers==5.20.1 opencv-python pillow numpy
   ```

5. Get access to the weights. Request access on the [facebook/sam3](https://huggingface.co/facebook/sam3) model page. Once approved, create an access token under Hugging Face → Settings → Access Tokens and log in:

   ```bash
   hf auth login
   ```

   The weights download on the first run. On this workstation, keep them on the
   larger cache volume by prefixing SAM commands with:

   ```bash
   HF_HOME=/cache/s95vkond/huggingface \
   HF_TOKEN_PATH=/home_domuser2/s95vkond/.cache/huggingface/token
   ```

6. Run the current RGB-D top-layer baseline:

   ```bash
   cd sam3_detection
   HF_HOME=/cache/s95vkond/huggingface \
   HF_TOKEN_PATH=/home_domuser2/s95vkond/.cache/huggingface/token \
   python segment_lego_rgbd_topplane.py \
     /cache/s95vkond/rgbd_data/extracted/extracted \
     --datasets data3 data4 \
     --out results_rgbd_topplane_v2
   ```

Versions used: Python 3.12.15, torch 2.10.0+cu128, torchvision 0.25.0+cu128, transformers 5.20.1, opencv-python 4.11.0, numpy 1.26.4, SAM 3 at commit `7f473cf`.
