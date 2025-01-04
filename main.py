import pickle
from argparse import ArgumentParser, Namespace

from torch.utils.data import DataLoader

from image_data_selector.generate_embeddings import (ImageDataset,
                                                     custom_collate_fn,
                                                     get_embeddings)
from image_data_selector.zcore import zcore_score


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Generate embeddings and z-scores for images.")
    parser.add_argument("--image_dir", type=str, default=None, help="Directory containing images.")
    parser.add_argument("--embeddings_file", type=str, default=None, help="Path to save embeddings.")
    parser.add_argument("--n_sample", type=int, default=50, help="Number of samples to generate.")
    parser.add_argument("--sample_dim", type=int, default=2, help="Dimension of each sample.")
    parser.add_argument("--redund_nn", type=int, default=4, help="Number of nearest neighbors to consider.")
    parser.add_argument("--redund_exp", type=int, default=10, help="Exponent for redundancy score.")
    return parser.parse_args()


def load_zscore_dict(zscore_file: str):
    try:
        with open(zscore_file, "rb") as f:
            zscore_dict = pickle.load(f)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Z-score file {zscore_file} not found") from e
    return zscore_dict


def main():
    args = parse_args()

    if args.embeddings_file is None:
        if args.image_dir is None:
            raise ValueError("Either embeddings_file or image_dir must be provided")
        
        dataset = ImageDataset(args.image_dir)
        dataloader = DataLoader(dataset, batch_size=4, collate_fn=custom_collate_fn, shuffle=False)
        embeddings_dict = get_embeddings(dataloader)
    else:
        try:
            with open(args.embeddings_file, "rb") as f:
                embeddings_dict = pickle.load(f)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Embeddings file {args.embeddings_file} not found") from e

    zscore_dict = zcore_score(embeddings_dict, args.n_sample, args.sample_dim, args.redund_nn, args.redund_exp)

    with open("zscore_dict.pkl", "wb") as f:
        pickle.dump(zscore_dict, f)

    # TODO: visualize images with highest and lowest z-scores


if __name__ == "__main__":
    main()