# Deep Learning - Compositional Image Retrieval

Course project for Deep Learning, UNITN 2025/2026.
This repository contains the material for the Deep Learning Assignment: DYNAMIC AND HYBRID CONDITIONING FOR COMPOSITIONAL IMAGE RETRIEVAL

## Authors

- Di Quattro Kuba
- Giacomo Vettore
- Danilo Frailis

## Project Documentation

- [Current project state and proposed solution](PROJECT_STATE_AND_SOLUTION.md)
- [Complete implementation guide](PROJECT_GUIDE.md)
- [Folder and training-flow schema](PROJECT_FOLDER_AND_TRAINING_SCHEMA.md)
- [Ground-truth dashboard](ground_truth_dashboard/README.md)
- [Steps 1-3 CLIP notebook](notebooks/01_clip_steps_1_2_3.ipynb)

Generate the notebook-compatible frozen CLIP caches locally:

```bash
python3 scripts/cache_clip_embeddings.py --device auto --splits train valid test
```

## Simple CPU CLIP Difference Experiment

Given a CelebA person identity, choose two images where an attribute changes, compute the CLIP
image-embedding difference, and classify that difference using CLIP text directions:

```bash
python3 -m pip install -r requirements.txt
python3 scripts/classify_celeba_clip.py 8692 --attribute No_Beard
```

The first run downloads the OpenAI CLIP ViT-B/32 checkpoint. The script always uses the CPU and
reports raw cosine similarities rather than softmax percentages. `without_beard` and
`clean_shaven` are accepted aliases for `No_Beard`.
