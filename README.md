# Image Data Coreset Selector

*Uses ideas from [Zero-Shot Coreset Selection](https://arxiv.org/pdf/2411.15349) paper from November 2024*

In most real-world applications of deep learning in computer vision, a huge amount of image data is generated. In order to improve the performance of models, some of this raw (unlabeled) data needs to be selected, labeled and finally used to re-train the model.

However, when there are millions of raw images to choose from which do you pick in order to minimize labeling cost and time?

The high-level idea is to use existing foundation models to create embeddings for all images. These embeddings are then used to perform Monte Carlo-like sampling. This ensures that the selected subset, also called coreset, covers the embedding space well and evenly.


## Usage

1. Put raw image data into folder (can contain sub-folders)
2. Run the main script

```bash
python main.py --image_dir <path/to/your/image/dir>
```

## Install

```bash
python3 -m venv env
source env/bin/activate
pip install -e .
```

## Future Extensions

- [ ] OpenAI integration to use e.g. CLIP
- [ ] Visualization of embedding and images