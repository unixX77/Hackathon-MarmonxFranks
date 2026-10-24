# Perception

| Folder | Content |
|---|---|
| [`sam3_detection/`](sam3_detection/) | SAM 3 instance and colour segmentation of LEGO bricks from a single text prompt, with a good / bad label per brick, and the individual top-layer bricks of a stack from RGB-D with a brick-size prior |

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

   The weights download on the first run.

6. Run the detection:

   ```bash
   cd sam3_detection
   python segment_lego_sam3_single_prompt.py path/to/rgb --out results
   ```

Versions used: Python 3.12.15, torch 2.10.0+cu128, torchvision 0.25.0+cu128, transformers 5.20.1, opencv-python 4.11.0, numpy 1.26.4, SAM 3 at commit `7f473cf`.
