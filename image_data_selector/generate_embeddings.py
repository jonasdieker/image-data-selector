from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import resnet18


class ImageDataset(Dataset):
    def __init__(self, image_dir: Path):
        self.image_paths = []
        extensions = [".jpg", ".png"]
        for ext in extensions:
            self.image_paths.extend(list(image_dir.rglob(f"*{ext}")))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx) -> Tuple[str, Image.Image]:
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        return str(image_path), image


def custom_collate_fn(
    batch: List[Tuple[str, Image.Image]]
) -> Tuple[List[str], torch.Tensor]:
    """
    Custom collate function for DataLoader.
    Processes a batch of (file_path, PIL.Image) pairs into tensors.
    """
    # Define a transform to convert PIL images to tensors
    image_transform = transforms.Compose(
        [
            transforms.ToTensor(),  # Converts PIL Image to Tensor and normalizes to [0, 1]
        ]
    )

    file_paths = []
    images = []

    for file_path, img in batch:
        file_paths.append(file_path)
        if isinstance(img, Image.Image):  # Check if img is a PIL Image
            images.append(image_transform(img))
        else:
            raise TypeError(f"Expected PIL.Image, got {type(img)}")

    # Stack images into a batch (B, C, H, W)
    images_tensor = torch.stack(images, dim=0)

    return file_paths, images_tensor


@torch.no_grad()
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


if __name__ == "__main__":
    image_dir = Path("data/train")
    dataset = ImageDataset(image_dir)
    dataloader = DataLoader(
        dataset, batch_size=4, collate_fn=custom_collate_fn, shuffle=False
    )
    get_embeddings(dataloader)
