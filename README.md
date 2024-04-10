# wdv3-timm

A small example demonstrating how to use `timm` for running the WD Tagger V3 models, with added functionality for batch processing images within directories, including optional recursion through subdirectories.

## How To Use

1. Clone the repository and navigate to the directory:
```
git clone https://github.com/arielfikru/wdv3-timm.git
cd wd3-timm
```

2. Create a virtual environment and install the required Python packages.

For Linux users, a script is provided for ease of setup:
```
bash setup.sh
```

Windows users (or those preferring manual setup) can follow these instructions:
```
# Create the virtual environment
python -m venv .venv
# Activate the environment
source .venv/bin/activate
# Update pip, setuptools, and wheel
python -m pip install -U pip setuptools wheel
# Optionally, manually install PyTorch (e.g., for non-nVidia GPU users)
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
# Install the remaining requirements
python -m pip install -r requirements.txt
```

3. To run the script, select one of the 3 models and provide an image file or a directory as input. The script can now process all images within a directory and optionally includes images within subdirectories.
```
python wdv3_timm.py --model <swinv2|convnext|vit> --gen_threshold <between 0.0 to 1.0> --char_threshold <between 0.0 to 1.0> <path/to/input, can be image or dir path> --recursive=True
```

Example output for `python wdv3_timm.py --model "vit" --gen_threshold=0.3 --char_threshold 0.6 "./animuData" --recursive=True` might look like this:
```
Processing image: path/to/directory/image1.png
Saved tags to image1.txt
Processing image: path/to/directory/subdirectory/image2.jpg
Saved tags to image2.txt
...
Done!
```

The output files (`image1.txt`, `image2.txt`, etc.) contain a unique set of tags for each image, effectively demonstrating the script's capability to handle large sets of images efficiently and autonomously.
