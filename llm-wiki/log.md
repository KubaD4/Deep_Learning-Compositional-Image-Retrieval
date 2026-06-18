# LLM Wiki Log

## [2026-06-09] setup | Initial project wiki and MarkItDown setup

- Inspected the project boundary at `/Users/kuba/deep_learning`.
- Identified 20 non-image/non-work source files, 202599 CelebA aligned JPG images, and 9 PDF-derived work files from a prior extraction folder.
- Installed Microsoft MarkItDown in a project-local `.venv` and recorded `markitdown[all]==0.1.6` in `requirements.txt`.
- Converted all 9 PDFs plus `Project Skeleton.ipynb` into `llm-wiki/source-markdown/`.
- Created the initial LLM wiki pages: index, project overview, source inventory, dataset page, evaluation protocol, method roadmap, and MarkItDown workflow.
- Created `AGENTS.md` so future agents know to read and update the wiki first.

## [2026-06-09] query | Clarified practical input/output

- Added `llm-wiki/practical-input-output.md` to explain concrete retrieval inputs, outputs, and file locations.
- Updated `llm-wiki/index.md` to link the new page.

## [2026-06-09] query | Clarified similarity validation

- Updated `llm-wiki/evaluation-protocol.md` and `llm-wiki/practical-input-output.md` to distinguish model similarity scores from official JSON-based validation.

## [2026-06-10] documentation | Created complete group onboarding guide

- Added root-level `PROJECT_GUIDE.md` as the canonical end-to-end project explanation for the group starting development on June 11, 2026.
- Documented the exact task, practical inputs/outputs, official similarity definition, supplied files, CelebA structure, JSON format, starter notebook cell behavior, missing implementation, training/no-training options, recommended architecture, code skeleton, no-leakage split policy, group division, first-session agenda, experiment tracking, and final deliverable checklist.
- Verified a concrete source/target example against local CelebA attributes and dataset-index-to-filename mappings.
- Linked the guide from `llm-wiki/index.md` and `llm-wiki/method-roadmap.md`.

## [2026-06-10] dataset | Clarified CelebA supervision and pair generation

- Expanded `llm-wiki/dataset-celeba.md` with the exact role of `identity_CelebA.txt`, `list_attr_celeba.txt`, and `list_eval_partition.txt`.
- Documented that automatic training tuples can be built by joining on image filename, grouping by identity, and selecting pairs where a chosen attribute flips.
- Recorded current identity-count and split-count statistics from the local dataset copy.
- Added links from the wiki index and source inventory to make the pairing logic easy to find later.

## [2026-06-11] method | Formalized learned residual composition approach

- Added `llm-wiki/training-strategy.md` to formalize the group's whiteboard proposal.
- Defined directional same-identity tuples, signed attribute deltas, textual condition encoding, residual embedding prediction, and contrastive training loss.
- Recorded that the official CelebA partitions contain disjoint identities: 8192 train, 985 validation, and 1000 test identities with zero overlap.
- Measured 3,723,370 directional same-identity pairs in the training split and documented why filtered balanced sampling is required.
- Clarified the mismatch between same-identity training and the official attribute-based ground truth, and proposed same-identity, benchmark-style, and hybrid sampling experiments.
- Updated the dataset, evaluation, project overview, method roadmap, and wiki index pages with links to the new strategy.

## [2026-06-11] explanation | Documented complete CLIP data flow

- Added `llm-wiki/clip-data-flow.md` to distinguish raw images, preprocessed pixel tensors, token tensors, projected CLIP embeddings, composition-network outputs, and gallery scores.
- Clarified that CLIP has separate image/text encoders whose final projection layers map into a shared 512-dimensional space.
- Documented tensor shapes and the exact role of source, target, and text during training, validation, test, and inference.
- Added a complete Mermaid flow diagram and practical cache-size estimates.
- Expanded the ground-truth dashboard to expose every target, navigate five at a time with the source fixed, and identify same-person targets.

## [2026-06-11] audit | Consolidated current state and proposed solution

- Re-audited all current project documentation, dataset metadata, git state, new wiki pages, and the ground-truth dashboard.
- Ran the dashboard unit suite: all four tests passed.
- Verified identity counts, identity-disjoint splits, and the 3,723,370 directional train-pair count directly from local metadata.
- Added root-level `PROJECT_STATE_AND_SOLUTION.md` as the authoritative status snapshot distinguishing completed infrastructure from the not-yet-implemented model and experiments.
- Documented the selected frozen-CLIP signed gated residual proposal, hybrid supervision strategy, baselines, losses, risks, implementation order, expected artifacts, and next milestone.
- Updated `README.md`, `AGENTS.md`, `PROJECT_GUIDE.md`, the wiki index, and source inventory to point to the new snapshot.
- Reconfirmed that `llm-wiki/` is local-only and intentionally ignored by git; added local Obsidian state and the nested MarkItDown clone to ignores.

## [2026-06-11] documentation | Expanded data dictionary and usage plan

- Expanded `PROJECT_STATE_AND_SOLUTION.md` with the exact row format, identifier semantics, and intended use of every CelebA metadata file.
- Clarified the difference between image filename, person identity ID, and PyTorch split-local dataset index.
- Added a worked filename-based metadata join and showed how joined rows become source/text/target training tuples and frozen CLIP embeddings.
- Classified bounding boxes and landmarks as optional analysis metadata, while identity, attributes, partitions, images, and the JSON form the core training/evaluation pipeline.
- Updated `llm-wiki/dataset-celeba.md` with a compact data dictionary and phase-by-phase usage table.

## [2026-06-11] analysis | Counted all same-identity image pairs

- Added `scripts/count_identity_pairs.py`, which reads `identity_CelebA.txt` without modifying it and calculates image count, unique unordered pairs, and directional pairs for every person ID.
- Executed the script over all 202599 rows: 10177 identities, 10133 with at least two images, 2320695 unique unordered pairs, and 4641390 directional pairs.
- Verified the source file SHA-256 and modification timestamp were unchanged before and after execution.
- Clarified that the complete-dataset total is descriptive; leakage-safe training uses only partition 0, with 1861685 unique unordered pairs.

## [2026-06-11] architecture | Defined project tree and training execution flow

- Added `PROJECT_FOLDER_AND_TRAINING_SCHEMA.md` with the proposed repository tree and clear status markers for existing, planned, generated, and local-only content.
- Documented example structures for raw metadata, pair manifests, CLIP caches, configs, checkpoints, logs, and result files.
- Assigned responsibilities to data, feature, model, training, retrieval, evaluation, visualization, script, notebook, and test modules.
- Clarified that condition weights have no direct labels: the condition weighter and residual MLP are optimized jointly through the final target-retrieval loss.
- Added Mermaid diagrams for the architecture, one training iteration, and the complete preprocessing/training/evaluation pipeline.
- Added exact batch tensor shapes, iteration pseudocode, epoch phases, intended CLI commands, and implementation milestones.

## [2026-06-11] explanation | Added a real full training-cycle example

- Verified directly from the read-only CelebA metadata that `000023.jpg` and `145590.jpg` both belong to person ID `1` and partition `0`.
- Recorded their real values for `Big_Nose`, `Narrow_Eyes`, `Smiling`, `Eyeglasses`, `Young`, `Male`, and `Heavy_Makeup`.
- Added root-level `PRACTICAL_TRAINING_CYCLE.md`, following the real `-Smiling` example through pair filtering, manifest creation, cached CLIP tensors, the condition-weight MLP, residual MLP, cosine and contrastive losses, backpropagation, and inference.
- Clearly separated verified metadata from illustrative untrained network values and linked the example from the wiki index, training strategy, and folder schema.

## [2026-06-12] architecture | Clarified CLAY relationship and hybrid integration

- Re-read the maintained wiki, the canonical assignment, instructor slides, the CLAY paper, and the official CLAY implementation.
- Clarified that full CLAY is not mandatory: the assignment requires an original mechanism that improves on its rigid multi-condition pre-SVD stacking, while CLIP ViT-B/32 and the frozen visual database are the required foundations.
- Documented the task mismatch between CLAY's conditional similarity focus and the assignment's signed attribute-edit retrieval.
- Confirmed that the proposed signed gated residual network already addresses the core fusion requirement.
- Added `llm-wiki/clay-integration-analysis.md` with a recommended CLAY baseline and a dynamic metric/subspace module placed after residual composition and before ranking.
- Recorded edit-space and preservation-space scoring/loss variants, risks, and implementation order.

## [2026-06-12] notebook | Implemented assignment steps 1-3

- Added `notebooks/01_clip_steps_1_2_3.ipynb` as a report-style Colab notebook with alternating Markdown and Python cells.
- Implemented CelebA test exploration, official JSON checks, frozen Hugging Face CLIP ViT-B/32 extraction, and reusable image/text caches.
- Added the required direct arithmetic baseline plus sequential, contrastive-direction sum, and contrastive-direction sequential experiments.
- Added disabled full evaluation against the official JSON with dedicated result subfolders, per-query metrics, macro summaries, and retrieval records.
- Added `scripts/cache_clip_embeddings.py` to generate notebook-compatible caches locally on CPU, MPS, or CUDA.
- Generated the local text cache for 40 attributes and the complete test image cache for 19,962 images on Apple MPS.

## [2026-06-12] implementation | Added simple CPU CLIP image classifier

- Added `scripts/classify_celeba_clip.py` for zero-shot classification of a CelebA image filename or explicit path using OpenAI CLIP ViT-B/32 on CPU.
- Implemented independent positive/negative prompt comparisons for all 40 CelebA attributes because the labels are multi-label rather than mutually exclusive.
- Added the OpenAI CLIP, PyTorch, and TorchVision dependencies and documented the command in `README.md`.
- Updated the script to look up each filename in the official CelebA attribute file, build one sentence from all active ground-truth attributes and one sentence containing their opposites, and report raw image-text and text-text cosine similarities without softmax percentages.
- Reworked the script into a person-level CLIP direction experiment: it selects a minimal-change pair for one identity, computes `target_image_embedding - source_image_embedding`, compares it with `positive_text_embedding - negative_text_embedding`, and ranks the image delta against all 40 signed CelebA attribute directions.

## [2026-06-13] cluster | Completed frozen CLIP caches and official arithmetic baselines

- Built and transferred a minimal operational `cluster/` bundle containing CelebA, annotations, official JSON, Python scripts, Slurm jobs, requirements, notebook, checkpoints, and result directories.
- Fixed cluster-specific execution issues: explicit `--gres=gpu:0` for CPU jobs, `SLURM_SUBMIT_DIR` instead of Slurm's temporary `$0`, virtual-environment dependency installation, and unbuffered Python logging.
- Generated the complete 19,962-image test cache in 2 minutes 9 seconds using `openai/clip-vit-base-patch32`; final artifacts were approximately 20 MiB for test image embeddings and 126 KiB for text embeddings, with 78 resumable chunks.
- Evaluated Direct Sum, Direct Sequential, Contrastive Sum, and Contrastive Sequential over all 14 official query entries and 33,052 source-query cases.
- Measured best macro performance with Contrastive Sequential: R@1 `0.04064`, R@5 `0.12381`, R@10 `0.18708`, and P@10 `0.02698`.
- Recorded that Contrastive Sequential improves Macro R@10 by 72.54% and Macro P@10 by 83.36% relative to Direct Sum.
- Added per-query PNG comparisons, `method_comparison.csv`, and `per_query_winners.csv`.

## [2026-06-13] experiment | Tested adaptive arithmetic and CLIP prompt ensembles

- Implemented Adaptive Tangent Sequential: a source-dependent contrastive edit using `lambda = clamp(1 - alignment, 0.25, 1.75)` and tangent-plane projection.
- Evaluated it over the official benchmark. Macro R@10 was `0.18693`, slightly below Contrastive Sequential; Micro R@10 was `0.16677`, only `0.00024` above it. The method is effectively tied and does not establish a meaningful improvement.
- Added `scripts/clip_image_text_similarity.py` for local CPU/MPS inference with absolute cosine, positive/negative margins, contrastive direction score, binary prompt preference, offline-cache mode, and repeated prompt pairs.
- Verified from `list_attr_celeba.txt` that `000366.jpg` has `Wearing_Hat`, `Smiling`, `Mustache`, `Goatee`, `Male`, `Chubby`, `Double_Chin`, `Big_Lips`, `Big_Nose`, and `Narrow_Eyes` marked present.
- For `Wearing_Hat`, three concrete lexical-opposite pairs agreed `3/3`, with ensemble margin `+0.027074`, direction score `+0.052661`, and pair preference `93.75%`.
- A negation-heavy hat ensemble agreed only `1/3` and produced `49.86%`, confirming prompt/negation sensitivity.
- The smiling ensemble agreed `1/3` with margin `-0.000389` and `49.03%`; the mustache ensemble agreed `0/3` with margin `-0.017345` and `15.00%`, despite positive CelebA labels.
- Added `llm-wiki/cluster-baselines-and-prompt-experiments.md` and updated the wiki index, CLIP data flow, evaluation protocol, method roadmap, and authoritative project-state snapshot.

## [2026-06-17] training | Finalized learned gate strategy and cluster runbook

- Finalized the first learned training plan: frozen CLIP embeddings, same-identity source-target pairs, query equal to all actually changed attributes, and query length restricted to 1-3 to match the official JSON query lengths.
- Chose prompt ensembles with 2-3 manual natural-language variants per signed CelebA attribute, saved as a versioned config and embedded into `signed_attribute_prompt_embeddings.pt`.
- Chose a gated residual architecture: independent sigmoid gate weights per signed attribute, aggregated edit vector, residual MLP, and `q = normalize(z_source + Delta_z)`.
- Clarified that softmax was avoided because it forces attributes to compete, not because it is non-differentiable.
- Documented that gate weights are learned edit strengths, not calibrated attribute-presence probabilities, and recorded a future extension using an attribute-presence probe on frozen CLIP embeddings.
- Defined masked InfoNCE with an attribute-based false-negative guard, target cosine auxiliary loss, and configurable small source-preservation regularization.
- Defined validation metrics: `exact_R@K`, `attr_success@K`, `official_like@K`, and `mean_rank_B`; selected `best_val_official_like@10` as the main checkpoint criterion.
- Added short/long training profiles, timestamped run directories, `progress.txt` with `BEST_` markers, report-oriented PNG outputs, and a six-configuration manual hpsearch plan.
- Implemented the cluster scripts and Slurm jobs for prompt embeddings, pair-index construction, learned training, hpsearch, and official JSON evaluation.
- Preserved Baldo-specific Slurm fixes: explicit `--gres=gpu:0` for CPU jobs, `SLURM_SUBMIT_DIR`, unbuffered `python3 -u`, and a pre-existing `logs/` directory.
- Added `llm-wiki/learned-gate-cluster-runbook.md` and linked it from the wiki index.

## [2026-06-18] results | Compared learned gate against official JSON baselines

- Completed the first long learned-gate hpsearch (`gate_v1`, residual-only composer). Best validation run was `ov015` with `val_official_like@10 = 0.7573`, `val_exact_R@10 = 0.5574`, `val_attr_success@10 = 0.8677`, selected at epoch 2 / step 4000.
- Evaluated `ov015` on the official `celeba_evaluation.json` protocol. It achieved Macro R@10 `0.1601`, Micro R@10 `0.1552`, Macro P@10 `0.0237`, and Micro P@10 `0.0233`.
- Confirmed that `gate_v1` did not beat the best arithmetic baselines on the official JSON: Contrastive Sequential remained best on Macro R@10 (`0.1871`) and Adaptive Tangent Sequential was best on Micro R@10 (`0.1668`), effectively tied with Contrastive Sequential.
- Diagnosed that `gate_v1` is not uniformly bad: it wins or ties on several local visual edits (`+Eyeglasses`, `-Heavy_Makeup`, `+Mustache`, `+Smiling`) but loses badly on global/correlated edits such as `+Male`, `+Black_Hair,-Wavy_Hair`, `-Male,-Mustache`, `+Chubby,-Young`, and `-Smiling,+Eyeglasses,+Wearing_Hat`.
- Identified the likely architectural issue: `gate_v1` aggregates signed text/prompt embeddings only as input to a residual MLP, so the model must learn the CLIP edit direction from scratch. The stronger arithmetic baseline directly adds contrastive text directions to the source embedding.
- Implemented `gate_v2` additive-gate composer: `q = normalize(z_source + edit_scale * sum(alpha_j * d_j) + residual_scale * delta)`, where `d_j` is the signed contrastive direction `normalize(t_positive - t_negative)` or its negative.
- Added short and long additive-gate hpsearch configs and jobs. The short smoke test completed successfully: `add_s001` reached `val_official_like@10 = 0.7207`, above the previous short best (`0.6973`) but below the best long `gate_v1` run; `add_s002` without residual (`residual_scale = 0`) dropped to `0.6797`, suggesting that a small residual correction is useful.
- Left the long additive-gate hpsearch queued on Baldo (`jobs/40_hpsearch_additive_gate_long.sh`) for the next cluster availability window.
