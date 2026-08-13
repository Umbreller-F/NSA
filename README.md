# Non-Semantic Anchor

Scripts, media assets, and evaluation outputs for the non-semantic-anchor experiments.

## Layout

- `generate/`: texture and video generation scripts
- `eval/`: model evaluation scripts
- `results/`: recorded evaluation outputs (`.jsonl`)
- `textures/`: versioned source images
- `videos/`, `gpt_frames/`: locally generated assets (directories are versioned with placeholders, but generated contents are ignored)
- `visualization/`: result inspection tools

Texture images are tracked with Git LFS. Generated videos and frame images are intentionally excluded from the latest version and can be reproduced with the project scripts. The `full-dataset-v1` tag preserves the original complete dataset version. Local credentials in `api_key.txt` are intentionally excluded from version control.
