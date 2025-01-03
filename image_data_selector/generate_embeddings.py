from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet18


class ImageDataset(Dataset):
    def __init__(self, image_dir: Path):
        extensions = [".jpg", ".png"]
        self.image_paths = []
        _ = [
            self.image_paths.extend(
                list(image_dir.rglob(f"*{ext}")) for ext in extensions
            )
        ]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx) -> Tuple[str, Image.Image]:
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        return str(image_path), image


def get_embeddings(dataloader: DataLoader, model: str = "resnet18") -> List[np.ndarray]:
    """
    Get embeddings from a model for a given dataloader.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    time_string = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    if model == "resnet18":
        model: torch.nn.Module = resnet18(pretrained=True)
    else:
        raise ValueError(f"Model '{model}' not supported")

    embeddings: Dict[str, np.ndarray] = {}
    for image_path, images in dataloader:
        images.to(device)
        embedding = model(images)
        embeddings.update(dict(zip(image_path, embedding)))

    with open(f"embeddings_{time_string}.npy", "wb") as f:
        np.save(f, embeddings)
    return embeddings
