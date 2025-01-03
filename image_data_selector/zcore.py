import pickle
from pathlib import Path
from typing import Dict

import numpy as np


def zcore_score(
    embeddings_file: Path,
    n_sample: int,
    sample_dim: int,
    redund_nn: int,
    redund_exp: int,
) -> np.ndarray:

    with open(embeddings_file, "rb") as f:
        embeddings = pickle.load(f)

    image_paths = list(embeddings.keys())
    embeddings = np.array(list(embeddings.values()))

    embed_info = preprocess(embeddings)

    scores = sample_score(
        embeddings, embed_info, n_sample, sample_dim, redund_nn, redund_exp
    )

    scores = (scores - np.mean(scores)) / abs(np.max(scores) - np.min(scores))
    scores = scores.astype(np.float32)

    zscore_file = Path(*embeddings_file.parts[:-1]) / "_".join(
        ["zscores"] + embeddings_file.stem.split("_")[1:] + [".npy"]
    )
    zscore_dict = dict(zip(image_paths, scores))
    with open(zscore_file, "wb") as f:
        np.save(f, zscore_dict)

    return zscore_dict


def sample_score(
    embeddings: np.ndarray,
    embed_info: Dict,
    n_sample: int,
    sample_dim: int,
    redund_nn: int,
    redund_exp: int,
) -> np.ndarray:

    scores = np.zeros(embed_info["n"])

    for _ in range(n_sample):

        # used because metrics are not effective in high dimensions
        dim = np.random.choice(embed_info["n_dim"], sample_dim, replace=False)

        # Coverage score (s_c)
        sample = np.random.triangular(
            embed_info["min"][dim], embed_info["median"][dim], embed_info["max"][dim]
        )
        # L1 distance
        embed_dist = np.sum(abs(embeddings[:, dim] - sample), axis=1)
        idx = np.argmin(embed_dist)
        scores[idx] += 1

        # Redundancy score (s_r)
        cover_sample = embeddings[idx, dim]
        nn_dist = np.sum(abs(embeddings[:, dim] - cover_sample), axis=1)
        nn = np.argsort(nn_dist)[1:]
        if nn_dist[nn[0]] == 0:
            scores[nn[0]] -= 1
        else:
            # decrease score of nearest neighbors
            nn = nn[:redund_nn]
            dist_penalty = 1 / ((nn_dist[nn] ** redund_exp) + 1e-12)
            dist_penalty /= sum(dist_penalty)
            scores[nn] -= dist_penalty

    return scores


def preprocess(embeddings: np.ndarray):

    embed_info = {
        "n": len(embeddings),
        "n_dim": len(embeddings[0]),
        "min": np.min(embeddings, axis=0),
        "max": np.max(embeddings, axis=0),
        "median": np.median(embeddings, axis=0),
    }
    return embed_info


if __name__ == "__main__":
    embeddings_file = Path("embeddings_2025-01-03_20-07-08.npy")
    n_sample = 50
    sample_dim = 2
    redund_nn = 4
    # auto-tune by size of embedding space?
    redund_exp = 2

    zcore_score(embeddings_file, n_sample, sample_dim, redund_nn, redund_exp)
