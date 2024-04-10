from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List
from glob import glob

import numpy as np
import pandas as pd
import timm
import torch
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import HfHubHTTPError
from PIL import Image
from simple_parsing import ArgumentParser, field
from timm.data import create_transform, resolve_data_config
from torch import Tensor
from torch.nn import functional as F

torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_REPO_MAP = {
    "vit": "SmilingWolf/wd-vit-tagger-v3",
    "swinv2": "SmilingWolf/wd-swinv2-tagger-v3",
    "convnext": "SmilingWolf/wd-convnext-tagger-v3",
}

def pil_ensure_rgb(image: Image.Image) -> Image.Image:
    if image.mode not in ["RGB", "RGBA"]:
        image = image.convert("RGBA") if "transparency" in image.info else image.convert("RGB")
    if image.mode == "RGBA":
        canvas = Image.new("RGBA", image.size, (255, 255, 255))
        canvas.alpha_composite(image)
        image = canvas.convert("RGB")
    return image

def pil_pad_square(image: Image.Image) -> Image.Image:
    w, h = image.size
    px = max(image.size)
    canvas = Image.new("RGB", (px, px), (255, 255, 255))
    canvas.paste(image, ((px - w) // 2, (px - h) // 2))
    return canvas

@dataclass
class LabelData:
    names: list[str]
    rating: list[np.int64]
    general: list[np.int64]
    character: list[np.int64]

def load_labels_hf(repo_id: str, revision: Optional[str] = None, token: Optional[str] = None) -> LabelData:
    try:
        csv_path = hf_hub_download(repo_id=repo_id, filename="selected_tags.csv", revision=revision, token=token)
        csv_path = Path(csv_path).resolve()
    except HfHubHTTPError as e:
        raise FileNotFoundError(f"selected_tags.csv failed to download from {repo_id}") from e
    df: pd.DataFrame = pd.read_csv(csv_path, usecols=["name", "category"])
    tag_data = LabelData(
        names=df["name"].tolist(),
        rating=list(np.where(df["category"] == 9)[0]),
        general=list(np.where(df["category"] == 0)[0]),
        character=list(np.where(df["category"] == 4)[0]),
    )
    return tag_data

def get_tags(probs: Tensor, labels: LabelData, gen_threshold: float, char_threshold: float):
    probs = probs.cpu().numpy()
    probs = list(zip(labels.names, probs))

    rating_labels = {probs[i][0].replace("_", " "): probs[i][1] for i in labels.rating}
    gen_labels = {probs[i][0].replace("_", " "): probs[i][1] for i in labels.general if probs[i][1] > gen_threshold}
    char_labels = {probs[i][0].replace("_", " "): probs[i][1] for i in labels.character if probs[i][1] > char_threshold}

    combined_names = list(gen_labels.keys()) + list(char_labels.keys())
    combined_names = list(set(combined_names)) 

    final_caption = ", ".join(combined_names)
    
    return final_caption

@dataclass
class ScriptOptions:
    input_path: Path = field(positional=True)
    model: str = field(default="vit")
    gen_threshold: float = field(default=0.35)
    char_threshold: float = field(default=0.75)
    recursive: bool = field(default=False)

def process_image(image_path: Path, model, transform, labels, opts: ScriptOptions):
    print(f"Processing image: {image_path}")
    img_input: Image.Image = Image.open(image_path)
    img_input = pil_ensure_rgb(img_input)
    img_input = pil_pad_square(img_input)
    inputs: Tensor = transform(img_input).unsqueeze(0).to(torch_device)
    inputs = inputs[:, [2, 1, 0]]  # NCHW image RGB to BGR
    with torch.inference_mode():
        outputs = model(inputs)
        outputs = F.sigmoid(outputs)
    final_caption = get_tags(outputs.squeeze(0), labels, opts.gen_threshold, opts.char_threshold)
    txt_filename = image_path.parent / f"{image_path.stem}.txt"
    with open(txt_filename, "w") as f:
        f.write(final_caption)
    print(f"Saved tags to {txt_filename}")

def find_images(directory: Path, recursive: bool) -> List[Path]:
    patterns = ["*.jpg", "*.png", "*.jpeg"]
    images = []
    if recursive:
        for pattern in patterns:
            images.extend(directory.rglob(pattern))
    else:
        for pattern in patterns:
            images.extend(directory.glob(pattern))
    return images

def main(opts: ScriptOptions):
    input_path = Path(opts.input_path).resolve()
    if input_path.is_dir():
        image_paths = find_images(input_path, opts.recursive)
    elif input_path.is_file():
        image_paths = [input_path]
    else:
        raise FileNotFoundError(f"Input path not found: {input_path}")

    repo_id = MODEL_REPO_MAP.get(opts.model)
    print(f"Loading model '{opts.model}' from '{repo_id}'...")
    model = timm.create_model("hf-hub:" + repo_id, pretrained=True).eval().to(torch_device)

    print("Loading tag list...")
    labels = load_labels_hf(repo_id=repo_id)

    print("Creating data transform...")
    transform = create_transform(**resolve_data_config({}, model=model))

    for image_path in image_paths:
        process_image(image_path, model, transform, labels, opts)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(ScriptOptions, dest="opts")
    args = parser.parse_args()
    opts = args.opts
    if opts.model not in MODEL_REPO_MAP:
        print(f"Available models: {list(MODEL_REPO_MAP.keys())}")
        raise ValueError(f"Unknown model name '{opts.model}'")
    main(opts)
