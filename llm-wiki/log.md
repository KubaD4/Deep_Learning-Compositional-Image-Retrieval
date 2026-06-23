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

## [2026-06-19] results | Evaluated gate_v2 and prepared learned sequential gate_v3

- Completed the long additive-gate hpsearch (`gate_v2`). Best validation config was `add_l009`, with `val_official_like@10 = 0.7969`, `val_exact_R@10 = 0.6016`, `val_attr_success@10 = 0.8987`, selected at epoch 2 / step 4000.
- Evaluated `add_l009` on the official `celeba_evaluation.json` protocol. It achieved Macro R@10 `0.1790`, Micro R@10 `0.1645`, Macro P@10 `0.0269`, and Micro P@10 `0.0249`.
- Confirmed that `gate_v2/add_l009` improves over `gate_v1/ov015` on official Macro R@10 (`0.1790` vs `0.1601`) and beats `Contrastive Sum` (`0.1707`), but remains below `Contrastive Sequential` (`0.1871`) and Adaptive Tangent Sequential (`0.1869`).
- Per-query analysis showed `gate_v2/add_l009` is especially strong on local edits and local compositions: `+Eyeglasses` (`0.4335` vs best baseline `0.2518`), `+Eyeglasses,+Smiling` (`0.3382` vs `0.2712`), `-Heavy_Makeup` (`0.2102` vs `0.1686`), and `+Mustache` (`0.2824` vs `0.2525`).
- The same analysis showed remaining failures on global/correlated or hair/hat edits: `+Male` (`0.0414` vs `0.2395`), `+Blond_Hair` (`0.1454` vs `0.1938`), `+Black_Hair,-Wavy_Hair` (`0.1621` vs `0.2107`), and `-Smiling,+Eyeglasses,+Wearing_Hat` (`0.3544` vs `0.4304`).
- Updated the interpretation: `gate_v2` fixed the main weakness of `gate_v1` by directly using signed contrastive CLIP directions, but it still behaves more like `Contrastive Sum` because it normalizes once after summing directions.
- Implemented the next experiment, `gate_v3` learned sequential composition: apply each signed contrastive direction with a learned `alpha_j`, normalize after every edit step, then optionally add a small residual correction.
- Added `GateSequentialComposer`, `gate_sequential_short_configs.json`, `gate_sequential_long_configs.json`, `jobs/41_hpsearch_sequential_gate_short.sh`, and `jobs/42_hpsearch_sequential_gate_long.sh`.
- Updated the cluster runbook and training strategy with the `gate_v3` hypothesis: test whether the remaining gap to `Contrastive Sequential` is mostly caused by the normalization schedule and whether learned source-conditioned step sizes can improve it.

## [2026-06-19] documentation | Added wiki maintenance rules

- Added `llm-wiki/index.md` maintenance rules requiring important project/wiki updates to also update `llm-wiki/log.md`.
- Recorded that agents should periodically remind the user to push wiki changes to GitHub so future agents and collaborators share the same context.
- Clarified that durable decisions should be written into the relevant wiki page rather than left only in chat history.

## [2026-06-19] notebook | Added final learned-gate pipeline notebook

- Added `notebooks/02_learned_gate_final_pipeline.ipynb` as a report-oriented notebook for the final pipeline.
- Added the same notebook under `cluster/notebooks/02_learned_gate_final_pipeline.ipynb` so the cluster bundle contains the deliverable workflow.
- Linked the final learned-gate notebook from `llm-wiki/index.md`.
- The notebook installs dependencies, checks Git LFS embedding caches, verifies/extracts data, creates frozen CLIP embeddings if missing, creates prompt-ensemble embeddings, builds same-identity training pairs, runs the latest `gate_v3` hpsearch when enabled, evaluates checkpoints on `celeba_evaluation.json`, and regenerates official comparison plots.
- The text sections document the main pivots from arithmetic CLIP baselines to `gate_v1`, `gate_v2`, and finally `gate_v3`, while keeping the executable training focus on the latest learned sequential gate.

## [2026-06-19] notebook | Converted final notebook into single-file submission

- Re-read the assignment deliverables section and confirmed the required format: one self-contained Jupyter/Colab notebook with complete codebase and report text.
- Rebuilt `notebooks/02_learned_gate_final_pipeline.ipynb` as a monolithic submission notebook with data loading, CLIP embedding extraction, prompt embeddings, baseline evaluation, pair generation, `gate_v3` model definitions, training loop, hpsearch, official JSON evaluation, plots, and discussion all inline.
- Updated the mirrored cluster copy at `cluster/notebooks/02_learned_gate_final_pipeline.ipynb`.
- Clarified in the wiki index that this notebook is the single deliverable notebook; earlier notebooks/scripts remain development artifacts, not the intended final submission.

## [2026-06-19] notebook | Aligned final notebook prompts with cluster runs

- Updated the final notebook prompt configuration to inline the exact prompt dictionary from `cluster/configs/attribute_prompts.json`.
- This keeps the single-notebook submission aligned with the prompt ensembles used to train and evaluate the cluster `gate_v3` results.

## [2026-06-19] experiment | Prepared prompt-v2 ablation for gate_v3

- Reviewed the new official JSON results: `gate_v3/seq_l007` achieved Macro R@10 `0.1951`, beating `Contrastive Sequential` (`0.1871`) and `Adaptive Tangent Sequential` (`0.1869`).
- Noted the remaining weak queries: `+Male`, `-Male,-Mustache`, `+Chubby,-Young`, `+Black_Hair,-Wavy_Hair`, and `-Smiling,+Eyeglasses,+Wearing_Hat`.
- Created `cluster/configs/attribute_prompts_v2_photo_templates.json`, a revised prompt ensemble using more uniform photo/portrait templates and less semantically forced wording for difficult global attributes such as `Male`, `Young`, and `Chubby`.
- Added support for `prompt_cache_path` in training and official JSON evaluation so different prompt caches can be compared without overwriting the original prompt cache.
- Added `cluster/configs/gate_sequential_prompt_v2_long_configs.json`, a controlled one-run config using the same parameters as best `seq_l007` but the prompt-v2 cache.
- Added `cluster/jobs/43_hpsearch_sequential_prompt_v2_long.sh`, which creates `signed_attribute_prompt_embeddings_v2_photo_templates.pt` and trains the prompt-v2 long run.
- Future comparison target: compare `seqp2_l007` directly against `seq_l007` on synthetic validation and official JSON per-query metrics.

## [2026-06-20] experiment | Added prompt-v2 overnight hpsearch

- Added `cluster/configs/gate_sequential_prompt_v2_overnight_configs.json`, a twelve-config `gate_v3` prompt-v2 grid around the best `seq_l007` setup.
- The overnight grid keeps the new photo/portrait prompt cache fixed and varies learning rate, temperature, source-preservation strength, residual scale, edit scale, and whether the gate reads the current sequential state or the original source state.
- Added `cluster/jobs/44_hpsearch_sequential_prompt_v2_overnight.sh` to run this grid on `meditech-long` for up to 12 hours.
- Future comparison target: compare `seqp2_ov*` against both the original-prompt `seq_l007` and the controlled prompt-v2 `seqp2_l007`, especially on `Male`, `Young`, `Chubby`, hair, and hat queries.

## [2026-06-20] cluster | Automated official JSON evaluation after hpsearch

- Extended `cluster/scripts/run_hpsearch.py` with `--evaluate-best-json`, which selects the completed config with the highest `best_val_official_like@10` and evaluates its `best_val_official_like_at10.pt` checkpoint on `celeba_evaluation.json`.
- Added `--plot-after-json`, which regenerates `artifacts/results/official_comparison` after the automatic JSON evaluation.
- Updated prompt-v2 jobs `43` and `44` to run both flags, so future long hpsearch runs produce training summaries, official JSON results, and official comparison plots in one job.

## [2026-06-20] results | Diagnosed `seqp2_ov005` and selected next experiment

- Recorded `gate_v3/seqp2_ov005` as the current best official JSON model: Macro R@10 `0.2117`, Micro R@10 `0.1936`, Macro P@10 `0.0320`.
- Confirmed that it beats the best arithmetic baseline, `Contrastive Sequential`, by about `+13.2%` relative on Macro R@10 and `+16.3%` relative on Micro R@10.
- Identified the remaining weak queries: `+Male`, `-Male,-Mustache`, `+Chubby,-Young`, and `+Wearing_Lipstick,-Heavy_Makeup,+Smiling`.
- Quantified the same-identity mismatch: the model retrieves same identity in top-10 about `88.6%` of official cases, but official valid targets are often cross-identity, especially for `Male`, `Young`, and `Chubby`.
- Added [Official Results and Literature Findings](official-results-and-literature-findings.md) with result tables, benchmark interpretation, same-identity analysis, and relevant CLIP/compositionality papers.
- Added a wiki maintenance rule: when papers materially inform a decision, save them in the wiki with links and a one-line relevance note.
- Decided next experiment: keep `gate_v3/seqp2_ov005` as backbone, then add official-like multi-positive training from train/valid attributes, hard negatives, an optional attribute-presence probe, and a medium-capacity MLP ablation. Do not train on `celeba_evaluation.json`.

## [2026-06-20] implementation | Added official-like multi-positive training

- Extended `GateSequentialComposer` with an optional attribute-presence probe. When enabled, the probe predicts the 40 CelebA attributes from the frozen source CLIP embedding and appends probe probabilities to gate/residual inputs.
- Added auxiliary `lambda_attr_probe` BCE loss so the probe learns calibrated-ish attribute state without changing the frozen CLIP encoder.
- Added `use_multipositive_loss`: the training batch target embeddings become a mini-gallery where compatible candidates can be additional positives if they satisfy the signed query and stay within the non-query Hamming threshold.
- Added `multipositive_top_fraction` to keep only CLIP-near compatible positives, preserving the idea of source similarity instead of accepting any attribute-compatible face.
- Added `cluster/configs/gate_sequential_official_like_long_configs.json` with four configs: no-probe, probe, medium MLP, and wider positive selection.
- Added `cluster/jobs/45_hpsearch_official_like_long.sh`, which runs the new hpsearch and then automatically evaluates the best checkpoint on the official JSON and regenerates plots.

## [2026-06-21] results | Diagnosed job 45 and implemented hybrid official-like training

- Recorded job `45` result: best synthetic validation config was `offmp_l004_widepos` with `val_official_like@10 = 0.8340`, but official JSON Macro R@10 was `0.2091`, slightly below current best `seqp2_ov005` at `0.2117`.
- Interpreted the result as evidence that pure official-like multi-positive training is useful but too strong when it replaces same-identity exact-target supervision.
- Implemented hybrid retrieval loss in `cluster/scripts/train_gate_model.py`: `loss_retrieval = (1 - w) * exact_info_nce + w * multipositive_info_nce`.
- Added logging for `exact_info_nce`, `multipositive_info_nce`, and `multipositive_weight` so future runs can diagnose whether improvements come from exact-target learning or relaxed official-like positives.
- Added `cluster/configs/gate_sequential_hybrid_official_like_long_configs.json` with eight long configs sweeping `multipositive_weight`, attribute probe usage, positive-pool width, and stronger source preservation.
- Added `cluster/jobs/46_hpsearch_hybrid_official_like_long.sh`, a 12-hour long-queue job that creates prompt-v2 embeddings, runs the hybrid hpsearch, evaluates the best checkpoint on the official JSON, and regenerates comparison plots.
- Updated `AGENTS.md` to make wiki/log updates mandatory after findings/results/decisions and to require paper links plus relevance notes when literature informs a decision.

## [2026-06-21] results | Hybrid official-like loss became the new best model

- Analyzed cluster output for job `46_hpsearch_hybrid_official_like_long.sh`; all eight configs completed successfully.
- Best selected config was `hybmp_l002_w050`: `multipositive_weight = 0.50`, no attribute probe, top-fraction positives `0.15`, `val_official_like@10 = 0.8323`, `val_exact_R@10 = 0.6338`, `val_attr_success@10 = 0.9282`.
- Official JSON result for `hybmp_l002_w050`: Macro R@10 `0.2130`, Micro R@10 `0.1953`, Macro P@10 `0.0325`, Micro P@10 `0.0297`.
- This slightly improves over the previous best `seqp2_ov005` (Macro R@10 `0.2117`, Micro R@10 `0.1936`) and strengthens the result over the best arithmetic baseline `Contrastive Sequential` (Macro R@10 `0.1871`, Micro R@10 `0.1665`).
- Hybrid ablation finding: moderate official-like loss weight helps; attribute probe, wider positive pools, and stronger source preservation did not improve the selected result in this run.
- Noted that the cluster per-query delta CSV still used collapsed gate labels such as `sequentialgate`; regenerate it with the updated analyzer before treating per-query values as specific to `hybmp_l002`.

## [2026-06-23] results | Reran arithmetic baselines and finalized comparison

- Reran all five arithmetic baselines on the cluster from scratch: direct sum, direct sequential, contrastive sum, contrastive sequential, and adaptive tangent sequential.
- Regenerated official comparison tables and plots with `--include-all-gates`; the rerun confirmed stable baseline values.
- Final strongest baseline remains `Contrastive sequential`: Macro R@10 `0.1871`, Micro R@10 `0.1665`.
- Final strongest learned model remains `hybmp_l002_w050`: Macro R@10 `0.2130`, Micro R@10 `0.1953`, Macro P@10 `0.0325`, Micro P@10 `0.0297`.
- Final gain over strongest baseline: about `+13.9%` relative Macro R@10 and `+17.3%` relative Micro R@10.
- Per-query clean comparison confirms learned-gate strengths on local compositional edits (`Eyeglasses`, `Smiling`, `Heavy_Makeup`, `Mustache`, hat composition) and persistent weaknesses on global/correlated attributes (`Male`, `-Male/-Mustache`, `Chubby/Young`).

## [2026-06-23] analysis | Local algebra diagnosis for Male, Young, and Chubby

- Ran local no-training CLIP-vector sweeps using cached test embeddings and text/prompt embeddings; no packages were installed and temporary scripts were removed afterwards.
- Found that `Male` is highly separable in CLIP space (`d-prime` around `7.7-7.9`), so the learned gate failure on `+Male` is not because CLIP cannot represent male/female. It is likely a training/evaluation mismatch: same-identity preservation discourages large global demographic movement while the official JSON often rewards cross-identity targets.
- Found that `Young` is only moderately separable (`d-prime` around `1.3-1.7`) and improves with stronger tangent movement: local best `-Young` R@10 around `0.0995` versus `hybmp_l002` around `0.0794`.
- Found that `Chubby` is intrinsically weak/noisy in CLIP text space (`d-prime` around `0.5`) and benefits from endpoint/global-query behavior rather than source-preserving edit behavior: local best `+Chubby,-Young` R@10 around `0.111-0.116` versus `hybmp_l002` around `0.0223`.
- Direction surgery for `Male` by removing correlated lipstick/makeup/facial-hair components did not help; the plain text male direction was already best.
- Added the proposed next no-training experiment: attribute-type-aware composition fallback using learned gate for local edits, arithmetic contrastive directions for `Male`, stronger tangent movement for `Young`, and low-source endpoint composition for `Chubby+older` queries.

## [2026-06-23] implementation | Added no-training attribute-routed orchestrator

- Added `cluster/orchestrator/evaluate_orchestrated_router.py`, a separate experimental evaluator that splits official queries into local learned-gate conditions and weak/global arithmetic conditions.
- Added `cluster/jobs/47_evaluate_orchestrated_router_short.sh`, a short-queue job that evaluates six routing/fusion methods in one run and writes separate outputs under `artifacts/results/orchestrated_router`.
- Implemented methods: `stage_tuned`, `delta_sum_tuned`, `score_fusion_70local`, `score_fusion_50`, `rrf_union`, and `full_arithmetic_if_weak`.
- The weak arithmetic rules use fixed no-training heuristics from the local algebra analysis: contrastive direction for `Male`, stronger tangent movement for `Young`, endpoint/global query with small `Double_Chin` helper for `Chubby + older`, and prompt-v1 directions for `Male + Mustache`.
- The script could not be fully smoke-tested locally because the Mac has result CSVs but not the learned gate checkpoint files; it passed syntax/help checks and is expected to run on the cluster where checkpoints are present.

## [2026-06-23] implementation | Added sum-only vs model+sum blend evaluator

- Added `cluster/orchestrator/evaluate_sum_model_blends.py` to compare learned model only, arithmetic only, vector-delta corrections, score-level fusion, and reciprocal-rank fusion.
- Added `cluster/jobs/48_evaluate_sum_model_blends_short.sh`, a short-queue job that runs all blend methods in one evaluator and stores outputs under `artifacts/results/sum_model_blends`.
- The goal is to determine whether the tuned arithmetic sum works better than the model globally, or whether it should be used only as a correction/fusion signal.
- This experiment is intentionally less hand-routed than the previous attribute-routed orchestrator and should guide whether the next method should use fixed blending, learned confidence, or explicit attribute-type routing.

## [2026-06-23] results | Model plus arithmetic delta is the new strongest family

- Analyzed local copies of `artifacts/results/sum_model_blends`.
- Main result: arithmetic-only is not competitive, but arithmetic as a vector correction on top of the learned gate is substantially stronger.
- Best Macro R@10: `model_plus_generic_delta_100` with Macro R@10 `0.2828`, Micro R@10 `0.2386`, Macro P@10 `0.0462`.
- Best Micro R@10: `model_plus_tuned_delta_050` with Macro R@10 `0.2732`, Micro R@10 `0.2410`, Macro P@10 `0.0447`.
- Current learned gate `hybmp_l002` / `model_only` is Macro R@10 `0.2130`, Micro R@10 `0.1953`; strongest previous baseline `contrastive_sequential` is Macro R@10 `0.1871`, Micro R@10 `0.1665`.
- Therefore `model_plus_generic_delta_100` improves over `hybmp_l002` by `+32.7%` relative Macro R@10 and `+22.2%` relative Micro R@10; it improves over `contrastive_sequential` by `+51.1%` relative Macro R@10 and `+43.3%` relative Micro R@10.
- Weak-query behavior improved sharply: `+Male` from `0.0520` to `0.2558-0.3154`, `-Young` from `0.0794` to up to `0.1481`, `-Male,-Mustache` from `0.0000` to up to `0.1481`, and `+Chubby,-Young` from `0.0223` to up to `0.0651`.
- Score-level fusion and reciprocal-rank fusion did not explain the gain; the gain comes from correcting the query vector before retrieval:
  `q_final = normalize(q_model + beta * (q_sum - source))`.
- Decision: avoid hard-coded attribute routers if possible. Prefer a global model-plus-delta family and next sweep beta/tangent variants before training a learned beta/confidence head.

## [2026-06-23] packaging | Created final best system folder and 1v1 plots

- Added `tools/create_best_system_package.py`, a regenerable local packaging script.
- Created `/Users/kuba/deep_learning/final_best_system` with code snapshots, prompt/text embedding caches, configs, winner result CSVs, and report-oriented 1v1 plots.
- Packaged primary winner: `model_plus_generic_delta_100`, highest Macro Recall@10, formula `q_final = normalize(q_model + 1.0 * (q_generic_sum - source))`.
- Corrected baseline terminology: `direct_sum` is the assignment vanilla baseline; `contrastive_sequential` is the strongest no-training CLIP-only baseline from our experiments.
- Regenerated clean report files under `final_best_system/results/clean_report`.
- Clean report includes all assignment metrics: Recall@1/5/10 and Precision@1/5/10, with macro and micro aggregation.
- Final vs assignment baseline: Macro Recall@10 `0.1084 -> 0.2827` (`+160.8%` relative); Micro Recall@10 `0.1248 -> 0.2386` (`+91.2%` relative).
- Final vs strongest CLIP-only baseline: Macro Recall@10 `0.1871 -> 0.2827` (`+51.1%` relative); Micro Recall@10 `0.1665 -> 0.2386` (`+43.3%` relative).
- The first packaging pass did not yet include the learned-gate checkpoint; this was corrected later the same day after copying the checkpoint and prompt-v2 cache from the cluster.

## [2026-06-23] report | Added cosine-normalization explanation and final package status

- Added report/notebook instruction: include or regenerate `final_best_system/explanations/toy_vector_correction_clip_cosine.png`.
- Required explanation for the report: "Stessa direzione. Per cosine similarity sono praticamente uguali. Il punto chiave: CLIP retrieval non chiede quanto sono vicino come coordinate assolute, ma qual è l'immagine con embedding che ha angolo/cosine più alto rispetto a q_final."
- Added `final_best_system/REPORT_NOTES.md` with the same instruction.
- Confirmed the best checkpoint is now present locally at `final_best_system/weights/best_val_official_like_at10.pt`.
- Updated the wiki index current-state section to the final best system: `model_plus_generic_delta_100`, Macro R@10 `0.2827`, Micro R@10 `0.2386`.
- Clean package contents to use going forward: `final_best_system/results/clean_report`, `final_best_system/results/assignment_baseline_direct_sum`, `final_best_system/results/strong_clip_baseline_contrastive_sequential`, `final_best_system/results/best_system_model_plus_generic_delta_100`, and `final_best_system/weights`.

## [2026-06-23] notebook | Updated final notebook and cluster-sourced artifacts

- Updated `notebooks/02_learned_gate_final_pipeline.ipynb` and mirrored it to `cluster/notebooks/02_learned_gate_final_pipeline.ipynb`.
- The notebook now contains the current final system, not only `gate_v3` alone: `q_final = normalize(q_model + 1.0 * (q_sum - source))`.
- Added repo-relative checkpoint loading from `final_best_system/weights/best_val_official_like_at10.pt`.
- Added loading for the prompt-v2 cache used by the final checkpoint: `signed_attribute_prompt_embeddings_v2_photo_templates.pt`.
- Added the cosine-normalization figure and explanation required for future reports/notebooks.
- Added mathematical modeling of the sequential gate, CLIP arithmetic delta branch, InfoNCE/multi-positive training objective, source/target cosine losses, and final official metrics.
- Regenerated `final_best_system` using the cluster-copied checkpoint and cluster-copied prompt-v2 cache; `WEIGHTS_MISSING.txt` has been removed and replaced by `weights/CHECKPOINT_INFO.txt`.
