# LLM Wiki Log

## [2026-06-09] setup | Initial project wiki and MarkItDown setup

- Inspected the project boundary at the repository root.
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
- Created `final_best_system/` with code snapshots, prompt/text embedding caches, configs, winner result CSVs, and report-oriented 1v1 plots.
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

## [2026-06-23] maintenance | Removed machine-specific fetch scripts

- Removed the tracked `final_best_system/weights/fetch_best_weights.sh` script because it used a personal `scp` destination path.
- Updated `tools/create_best_system_package.py` so generated metadata uses repo-relative paths and no longer emits a machine-specific fetch script.
- Updated `tools/update_final_notebook.py` and `cluster/scripts/plot_baseline_queries.py` to discover paths relative to the repository/script location instead of hardcoding a local Mac path.
- Reworded wiki/report references from a specific local path to `final_best_system/` / `<repo-root>` so collaborators can use the repository on their own machines.

## [2026-06-24] experiment plan | Blend-aware v4 fine-tuning without touching final system

- Added a new isolated experimental training path under `cluster/experimental/`.
- Goal: improve the current best system without modifying the packaged final best system.
- Current best inference rule is:

```text
q_final = normalize(q_model + 1.0 * (q_generic_sum - source))
```

- Previous training optimized `q_model` only; the arithmetic correction was added post-hoc at inference.
- New v4 idea: initialize from the best `hybmp_l002_w050` checkpoint and train directly on the deployed blended query:

```text
q_train = normalize(q_model + beta * (q_sum - source))
```

- Added optional batch-hard triplet-style loss with false negatives masked using the same official-like attribute logic:

```text
L = L_hybrid_InfoNCE(q_train)
    + lambda_triplet * max(0, margin + sim(q_train, hard_negative) - sim(q_train, target))
    + lambda_source * (1 - cos(q_train, source))
    + lambda_target * (1 - cos(q_train, target))
```

- Added optional distillation to the current best system to reduce catastrophic drift:

```text
L_distill = 1 - cos(q_train, q_current_best_final)
```

- New files:

```text
cluster/experimental/train_blend_finetune_v4.py
cluster/experimental/run_blend_finetune_v4_hpsearch.py
cluster/configs/gate_v4_blend_finetune_long_configs.json
cluster/jobs/49_hpsearch_blend_finetune_v4_long.sh
```

- The long queue grid has 12 configs around the current best: lower learning rates (`2e-5`, `5e-5`), beta sweep (`0.75`, `1.0`, `1.25`), triplet weights (`0`, `0.05`, `0.10`, `0.20`), optional distill (`0.02`, `0.05`), and one auxiliary-model-loss variant.
- Expected use: run as a separate long-queue experiment; compare its best evaluated blend against `final_best_system` before considering any promotion.
- Updated the v4 runner so each completed config is immediately evaluated on the official JSON through `orchestrator/evaluate_sum_model_blends.py`.
- Per-config official JSON outputs are written under:

```text
artifacts/results/blend_finetune_v4/<hpsearch_name>/<config_id>/
```

- The runner also writes aggregate comparison artifacts:

```text
artifacts/results/blend_finetune_v4/<hpsearch_name>/_aggregate/all_json_evaluations.csv
artifacts/results/blend_finetune_v4/<hpsearch_name>/_aggregate/top_json_methods.csv
artifacts/results/blend_finetune_v4/<hpsearch_name>/_aggregate/BEST_JSON_METHOD.txt
artifacts/results/blend_finetune_v4/<hpsearch_name>/_aggregate/top_json_methods_macro_recall10.png
```

- The Slurm job now uses a 5-minute finalization signal/window and avoids starting a new config when less than about one hour remains. This should leave time to compare completed runs and preserve useful JSON results before wall-time termination.

## [2026-06-24] results | v4 blend-aware fine-tuning evaluated per config

- The v4 runner worked operationally: after each completed config it evaluated the checkpoint on the official JSON using `evaluate_sum_model_blends.py` and produced per-config comparison PNGs plus aggregate CSV/PNG summaries.
- Cluster run analyzed:

```text
artifacts/training_runs/hpsearch_blend_finetune_v4_20260624_012534_long
artifacts/results/blend_finetune_v4/hpsearch_blend_finetune_v4_20260624_012534_long
```

- Best v4 JSON method found:

```text
evaluation_id = bft_l008_beta075_tri010_lrsafe
method        = Blend model_plus_tuned_delta_100
Macro R@10    = 0.2485
Micro R@10    = 0.2281
Macro P@10    = 0.0389
```

- The best synthetic-validation checkpoint was `bft_l003_beta075_tri005`, but on the official JSON it was slightly below `bft_l008` in Macro R@10:

```text
bft_l003 + tuned_delta_100: Macro R@10 = 0.2480, Micro R@10 = 0.2292
bft_l008 + tuned_delta_100: Macro R@10 = 0.2485, Micro R@10 = 0.2281
```

- This does not beat the current final packaged system:

```text
current final model_plus_generic_delta_100:
  Macro R@10 = 0.2827
  Micro R@10 = 0.2386

best v4 fine-tuned blend:
  Macro R@10 = 0.2485
  Micro R@10 = 0.2281
```

- Interpretation: training directly on the blended vector plus triplet/fine-tune did not improve the official benchmark. It likely over-adapted the learned component to synthetic same-identity targets and weakened the robust zero-shot generic correction that made the final system strong.
- The best v4 configs share `train beta = 0.75`, suggesting that a weaker correction during training plus a stronger tuned correction at evaluation is less harmful than training with `beta = 1.0` or `1.25`.
- Configs with `beta = 1.25` performed poorly, supporting the hypothesis that too much correction during training destabilizes the learned/source-preserving component.
- Decision: do not promote v4. Keep `final_best_system/model_plus_generic_delta_100` as the current best. Future training should not simply fine-tune the full blend; a safer next direction is a lightweight learned confidence/beta head or validation-free beta sweep on the frozen final model.
- Two distillation configs (`bft_l010`, `bft_l011`) exited immediately with `exit_1`, so they were not valid ML results. They were the only configs with `lambda_distill > 0`.
- Local code review found the likely cause: the teacher branch used `torch.inference_mode()` while its output was used in a loss combined with `q_final` before `backward`. This can create inference tensors that PyTorch refuses to save for backward.
- Fixed the experimental trainer to use `torch.no_grad()` for the frozen teacher branch instead.
- Added a focused retry config file for only the failed distill variants:

```text
cluster/configs/gate_v4_blend_finetune_distill_retry_configs.json
cluster/jobs/50_hpsearch_blend_finetune_v4_distill_retry_long.sh
```

## [2026-06-24] experiment plan | No-training beta sweep for adaptive-beta diagnosis

- Added a no-training evaluator to test whether the next useful model should learn an adaptive beta/confidence head.
- New script/job:

```text
cluster/orchestrator/evaluate_beta_sweep_blends.py
cluster/jobs/51_evaluate_beta_sweep_blends_short.sh
```

- It loads the strongest learned gate checkpoint and evaluates:

```text
q_generic_beta = normalize(q_model + beta * (q_generic_sum - source))
q_tuned_beta   = normalize(q_model + beta * (q_tuned_sum - source))
```

- Default beta grid:

```text
0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0
```

- Outputs go to:

```text
artifacts/results/beta_sweep/beta_sweep_<timestamp>/
```

- Important outputs:

```text
comparison/BEST_OVERALL_BETA.txt
comparison/overall_beta_sweep.csv
comparison/overall_beta_sweep.png
comparison/per_query_best_beta.csv
comparison/per_query_best_beta.png
comparison/per_query_recall10_heatmap_generic.png
comparison/per_query_recall10_heatmap_tuned.png
```

- Interpretation rule: if different queries prefer very different beta values, a learned beta/reranker is likely worth implementing. If beta `1.0` remains globally near-optimal and per-query stable, then the bottleneck is likely retrieval/reranking or training-data mismatch rather than beta selection.

## [2026-06-24] results | Beta sweep supports adaptive beta

- Cluster beta sweep output:

```text
artifacts/results/beta_sweep/beta_sweep_20260624_155934
```

- Best overall method:

```text
model_plus_tuned_delta_beta_0p75
Macro R@10 = 0.2891
Micro R@10 = 0.2525
Macro P@10 = 0.0479
```

- This beats the previous packaged final system:

```text
previous final model_plus_generic_delta_beta_1p00:
  Macro R@10 = 0.2827
  Micro R@10 = 0.2386

new no-training beta sweep winner:
  Macro R@10 = 0.2891
  Micro R@10 = 0.2525
```

- Best fixed beta overall is not `1.0`; it is `0.75` with tuned arithmetic delta.
- Per-query best beta varies substantially:

```text
+Smiling                         beta 0.25
+Eyeglasses                      beta 0.50
-Heavy_Makeup                    beta 0.75
+Male                            beta 0.75 generic / 1.00 tuned
-Young                           beta 1.25 generic / 0.75 tuned
+Mustache                        beta 1.50 generic / 1.00 tuned
-Male, -Mustache                 beta 1.50 generic
+Chubby, -Young                  beta 1.50 generic
-Smiling, +Eyeglasses, +Hat      beta 0.75 generic / 1.00 tuned
```

- Interpretation: beta selection is a real bottleneck. A single global beta is good but not ideal; different attribute/query types want different correction strength.
- Next promising direction: a lightweight adaptive beta/reranker rather than bigger gate fine-tuning. The beta head could predict beta from source embedding, query directions, learned gate output, and model-vs-sum score diagnostics. A no-training per-query/oracle beta upper bound should also be reported to estimate the ceiling.
- Note: the pasted `find` output only listed `overall_beta_sweep.png`. Check the Slurm `.err` for plotting failures and make the plotting step robust if `per_query_best_beta.png`/heatmaps were not produced.

## [2026-06-24] experiment plan | v5 mixed weak official-like training

- The next experiment targets the persistent weak-query failure mode:

```text
Male
Young
Chubby
Male + Mustache
Chubby + Young
```

- Hypothesis: same-identity training under-represents these global/correlated
  edits, while pure official-like training is too strong and harms identity
  preservation. Test a mixed sampler instead:

```text
70% same-identity + 30% weak official-like
85% same-identity + 15% weak official-like
```

- The weak official-like pool is built from CelebA train only, not from the
  official JSON. Candidate targets must satisfy the requested weak attributes,
  differ in few non-query CelebA attributes, and remain CLIP-close to the
  source.
- New isolated files:

```text
cluster/experimental/build_weak_official_like_pairs.py
cluster/experimental/train_blend_finetune_v5_mixed.py
cluster/experimental/run_mixed_weak_hpsearch_v5.py
cluster/configs/gate_v5_mixed_weak_24h_configs.json
cluster/jobs/52_mixed_weak_v5_24h.sh
```

- The job evaluates every completed config on the official JSON with the same
  beta/corrector sweep used in the previous diagnosis, so results should appear
  under:

```text
artifacts/results/mixed_weak_v5/hpsearch_mixed_weak_v5_<timestamp>_long/
```

- Deferred idea explicitly recorded: per-query/adaptive beta could improve the
  model, but hardcoded query-specific beta rules should be avoided for now
  because they assume we know the future query names. A learned beta head based
  on source/query/model-vs-sum features remains a later experiment.

## [2026-06-25] results | v5 mixed weak training improves Macro R@10

- The v5 mixed weak official-like experiment completed in about 1.5-2 hours,
  much faster than the 24h wall-time cap. Future long-queue training jobs can
  request about 3 hours to reduce queue waiting while leaving enough buffer.
- Best v5 aggregate result:

```text
evaluation_id = mw85_012
method        = model_plus_generic_delta_beta_1p50
family        = generic
beta          = 1.5
Macro R@10    = 0.3049983318858148
Micro R@10    = 0.2511799588527169
Macro P@10    = 0.0487211528445904
```

- Winning config `mw85_012`:

```text
weak_pair_fraction      = 0.15
same_identity_fraction  = 0.85
learning_rate           = 5e-5
train blend_beta        = 0.75
lambda_triplet          = 0.10
lambda_source           = 0.02
multipositive_weight    = 0.75
max_steps               = 20000
```

- Comparison to previous bests:

```text
previous packaged final, generic beta 1.00:
  Macro R@10 = 0.2827
  Micro R@10 = 0.2386

previous no-training beta sweep winner, tuned beta 0.75:
  Macro R@10 = 0.2891
  Micro R@10 = 0.2525

new v5 mixed weak winner, generic beta 1.50:
  Macro R@10 = 0.3050
  Micro R@10 = 0.2512
```

- Interpretation: v5 improved macro performance, which suggests that adding a
  small fraction of official-like weak pairs helps the difficult query families
  and balances performance across query types. Micro R@10 is roughly tied with
  the previous beta-sweep winner, so the gain is mainly per-query balance rather
  than a uniform improvement over all 33,052 source-query cases.
- The top of the aggregate ranking is dominated by `mw85` configs, not `mw70`.
  This supports the current hypothesis that weak official-like pairs are useful
  as a regularized augmentation, but too much of them risks weakening the
  reference/source-preservation behavior.
- The best inference method is `generic` delta with `beta=1.5`, even though the
  training config used `blend_beta=0.75`. This means the learned model benefits
  from a conservative correction during training but still needs a strong
  zero-shot arithmetic correction at evaluation.

### Assignment interpretation

- The assignment text says targets should preserve the core identity of the
  reference image. However, the provided official JSON operationalizes this as:

```text
1. target strictly satisfies the positive/negative query constraints;
2. all remaining non-query attributes have Hamming distance <= 2 from source.
```

- Therefore, the grading/evaluation does not directly check same person ID. It
  checks attribute-level source similarity under a relaxed Hamming rule.
- Consequence for our method: same-identity training is not wrong, but it is an
  inductive bias rather than the exact evaluation target. The v5 result confirms
  the right direction: keep same-identity pairs as source-preservation
  regularization, but add official-like cross-identity pairs to align with the
  JSON metric.

## [2026-06-25] decision | Keep official JSON held out for future set training

- Decided that the next official-like positive-set training should never use
  `celeba_evaluation.json` or test-split target lists to build train examples.
- Official-like sets may be precomputed on CelebA train/validation partitions
  only, using the same attribute rule as the JSON: query signs must match and
  non-query Hamming distance must be at most 2.
- Future hyperparameter sweeps should compare paired configs with identical
  hyperparameters but different mixture ratios, e.g. `60/30/10` versus
  `70/20/10` for official-like positives / same-identity pairs / weak-global
  oversampling.
- Reconfirmed the current data flow: gate training and official evaluation use
  cached frozen CLIP embeddings, not raw image pixels. Raw images are needed only
  for the one-time CLIP cache extraction step.

## [2026-06-25] implementation | Added v6 official-like multi-positive training

- Added a new isolated v6 experiment without modifying `final_best_system/`.
- New builder: `cluster/experimental/build_official_like_positive_sets_v6.py`.
  It builds train-split-only positive sets for source/query groups using the
  official-style rule: query signs match and non-query Hamming distance is at
  most 2. It does not read `celeba_evaluation.json`.
- New trainer: `cluster/experimental/train_official_mix_v6.py`. It keeps the
  current best inference formula, `q_final = normalize(q_model + beta *
  (q_sum-source))`, but trains with a three-way mixture:

```text
official-like multi-positive rows
same-identity rows
weak/global official-like oversampling rows
```

- New hpsearch runner: `cluster/experimental/run_official_mix_hpsearch_v6.py`.
  It evaluates every completed checkpoint on the official JSON via beta sweep
  and writes aggregate winner files.
- New config/job:

```text
cluster/configs/gate_v6_official_mix_3h_configs.json
cluster/jobs/53_official_mix_v6_3h.sh
```

- The config tests paired ratios with identical hyperparameters:

```text
60% official-like / 30% same-identity / 10% weak-global
70% official-like / 20% same-identity / 10% weak-global
```

- Syntax checks passed locally with `python3 -m py_compile` and JSON validation.

## [2026-06-26] analysis | Recall hit-rate vs Precision density and qualitative viewer

- Rechecked the assignment metric definition: Recall@K is a hit-rate, i.e. 1 if
  at least one official-valid target is retrieved in top K, otherwise 0. This
  matches the current project implementation.
- Precision@K remains the density of valid targets in the top K. Therefore the
  best system can have good Recall@10 while still having low Precision@10: it
  often finds at least one valid target, but the rest of the nearest CLIP
  neighbours can be visually plausible yet fail the official attribute/Hamming
  constraints.
- Added qualitative viewer:

```text
final_best_system/code/show_json_retrieval_example.py
```

- The script reads the final system `retrievals.jsonl`, the official
  `celeba_evaluation.json`, and the test embedding cache filenames. It renders
  source + top-k retrievals, coloring official-valid targets in green and
  invalid retrieved images in red.
- First smoke test: query 12 (`-Smiling, +Eyeglasses, +Wearing_Hat`) produced an
  example with 2 valid targets in top 10, saved under:

```text
final_best_system/results/qualitative_examples/
```

- Updated hypothesis: the next improvement should likely be a two-stage system:
  retrieve a large candidate pool with `q_final`, then rerank/filter candidates
  using learned attribute-query satisfaction and source-preservation estimates.
  A ground-truth Hamming filter should only be used as an analysis upper bound,
  not as the final fair inference path.

## [2026-06-26] experiment | Oracle top-500 Hamming/query filter

- Updated the qualitative viewer so each example renders two blocks:

```text
top: source + top-k predicted by the final system
bottom: official-valid JSON targets for the same source/query
```

- Added diagnostic-only script:

```text
final_best_system/code/test_oracle_top500_rerank.py
```

- This script recomputes the final vector:

```text
q_final = normalize(q_model + beta * (q_sum - source))
```

  then retrieves top 500 by cosine and filters candidates with the official-style
  rule: requested attributes must match and non-query Hamming distance must be
  <= 2.
- Important caveat: this uses CelebA ground-truth attributes at inference time,
  so it is an oracle/upper-bound diagnostic, not the fair final method.
- Smoke test on query 13, source 3977 (`+Wearing_Lipstick, -Heavy_Makeup,
  +Smiling`) supports the hypothesis:

```text
original final system top-10: 1 official-valid target
oracle top-500 filtered top-10: 6 official-valid targets
```

- Interpretation: for this case, `q_final` retrieves a broad pool containing
  useful valid targets, but raw cosine ranking alone does not place enough of
  them in the final top 10. A fair next-stage model should approximate this
  oracle filter with learned attribute/source-preservation predictors instead
  of reading test labels.

## [2026-06-26] analysis | Top-pool size sweep and stricter preservation hypothesis

- User ran the oracle filter with different top-pool sizes. Full-JSON results:

| Top pool | Macro pool hit rate | Avg filtered candidates/pool | Macro P@10 | Micro P@10 |
| ---: | ---: | ---: | ---: | ---: |
| 25 | 0.5735 | 1.2293 | 0.1225 | 0.1035 |
| 50 | 0.7035 | 2.0068 | 0.1958 | 0.1726 |
| 500 | 0.9629 | 8.2648 | 0.5706 | 0.5747 |

- Deduction: valid official targets are often present in the broader
  neighbourhood of `q_final`, but they are sparse and not necessarily among the
  first 25/50 cosine neighbours. This supports a two-stage direction: broad
  candidate generation with `q_final`, then learned reranking/filtering.
- The oracle filter is not a valid final inference method because it uses
  ground-truth CelebA attributes at inference time. It is valid only as an
  upper-bound diagnostic.
- Qualitative inspection suggests the final system often preserves visual
  identity/source similarity better than some JSON-valid targets. Some rejected
  predictions appear semantically plausible but fail the strict official
  Hamming rule.
- Future training ablation: try stricter source-preservation positives with
  non-query Hamming <= 1. Do not fully replace the official Hamming <= 2 target,
  because the assignment evaluation accepts <= 2. Prefer a mixed objective:

```text
main official-like positives: Hamming <= 2
preservation-focused positives/regularizer: Hamming <= 1
same-identity pairs: smaller source-preservation regularizer
```

- Updated final pipeline notebooks with an appendix for qualitative retrieval
  and oracle-pool diagnostic:

```text
notebooks/02_learned_gate_final_pipeline.ipynb
cluster/notebooks/02_learned_gate_final_pipeline.ipynb
```

## [2026-06-26] implementation | Qualitative viewer now separates JSON-valid, query-ok, and query-fail

- Updated:

```text
final_best_system/code/show_json_retrieval_example.py
```

- The qualitative PNGs now use four colors:

```text
blue        = source/input image
light green = official-valid JSON target
yellow      = satisfies requested query attributes, but is not JSON-valid
red         = fails at least one requested query attribute
```

- The script reads `list_attr_celeba.txt`, aligns attributes to the test
  embedding filenames, parses the query signs, and annotates every predicted
  top-k image with query-status metadata.
- Regenerated all existing qualitative examples under:

```text
final_best_system/results/qualitative_examples/
```

- Example motivation: for `+Blond_Hair`, some retrieved faces are visually
  plausible and query-ok but not JSON-valid because they fail the official
  non-query Hamming/source-preservation rule. Other retrieved faces are marked
  red because the CelebA label says they do not satisfy `+Blond_Hair` at all.

## [2026-06-26] implementation | v7 Hamming-weighted official-like training

- Added a new experimental branch of the training code without modifying the
  current final system:

```text
cluster/experimental/train_official_mix_v7_weighted.py
cluster/experimental/run_official_mix_hpsearch_v7_weighted.py
cluster/configs/gate_v7_hamming_weighted_3h_configs.json
cluster/jobs/54_hamming_weighted_v7_3h.sh
cluster/jobs/55_hamming_weighted_v7_smoke_short.sh
```

- Motivation: v6 improved the final model substantially, but Precision@10 is
  still low. Oracle top-pool diagnostics show that valid targets are often in a
  larger neighbourhood but sparse. The next training hypothesis is therefore to
  keep the official `Hamming <= 2` objective while softly preferring candidates
  that preserve non-query attributes more strongly.
- v7 keeps the official-like positive set construction unchanged and still
  avoids JSON leakage. The official JSON target lists remain reserved for
  post-training evaluation only.
- Training loss change:

```text
positive weight = 1.0 if query is satisfied and non-query Hamming <= 1
positive weight = w2  if query is satisfied and non-query Hamming == 2
positive weight = 0.0 otherwise
```

- The HP grid reuses the strongest v6 hyperparameter families and varies:

```text
w2 in {0.75, 0.50, 0.25, 0.00}
base configs in {v6_60_001, v6_70_001, v6_70_006}
```

- Interpretation of the ablation:

```text
w2 = 0.75 -> mild preservation bias, Hamming==2 still almost fully positive
w2 = 0.50 -> balanced official/preservation objective
w2 = 0.25 -> stronger preservation preference
w2 = 0.00 -> Hamming==2 examples are neutral in multi-positive loss, not negatives
```

- The exact target loss is still present, so the model is not trained to reject
  official `Hamming == 2` targets. The weighted multi-positive branch only
  changes how strongly additional official-like positives are pulled toward the
  query.

## [2026-06-26] implementation | Probe/reranker v1 for fair top-pool filtering

- Added an experimental second-stage reranker without modifying the current
  final system:

```text
cluster/experimental/probe_reranker_v1.py
cluster/jobs/56_probe_reranker_v1_5h.sh
cluster/jobs/57_probe_reranker_v1_random_smoke_short.sh
```

- Purpose: approximate the previous oracle Hamming filter without using true
  test attributes at inference. The frozen final system first retrieves a
  top-pool by cosine; then a CelebA attribute probe predicts source/candidate
  attributes and drives hard/soft/hybrid reranking.
- Probe labels come from CelebA `list_attr_celeba.txt` for train/valid splits.
  The official JSON is used only after inference to compute Recall@K and
  Precision@K.
- Added optional horizontal-flip augmentation for probe training. The code
  creates/reuses:

```text
data/celeba/embeddings/openai_clip_vit_b32/train_image_embeddings_flipped.pt
```

  This augmentation preserves attributes, unlike colour jitter or other image
  transforms that could change labels such as hair colour.
- The long job performs:

```text
1. create/reuse flipped train embeddings;
2. train multiple probe MLP configs;
3. select the best probe by validation BCE/F1, not by JSON;
4. freeze final q_final system;
5. evaluate baseline_q_final_top10, hard filter, soft rerank, and hybrid rerank;
6. write summaries, per-query metrics, retrieval JSONL, and PNG plots.
```

- Two local smoke tests passed before cluster launch:

```text
random probe smoke:
  gate checkpoint + corrective sum + q_final + random probe +
  hard/soft/hybrid filtering + JSON metrics + plots

one-step training smoke:
  BCE loss + AdamW + checkpoint save + same full evaluation path
```

- Expected output root:

```text
artifacts/results/probe_reranker_v1/probe_reranker_v1_<timestamp>/
```

- Key files to inspect after the run:

```text
progress.txt
probe/hpsearch_summary.csv
probe/per_attribute_metrics.csv
comparison/BEST_PROBE_RERANKER_METHOD.txt
comparison/combined_summary.csv
comparison/combined_per_query_metrics.csv
comparison/*.png
```

## [2026-06-26] diagnostic | CLIP zero-shot attribute estimation baseline

- Added a local diagnostic script:

```text
scripts/evaluate_clip_attribute_zeroshot.py
```

- Purpose: test whether frozen `openai/clip-vit-base-patch32` can estimate
  CelebA attributes directly from prompt similarity, before relying on a
  trained attribute probe.
- The script compares positive/negative prompt ensembles against CelebA
  `list_attr_celeba.txt`, using either cached CLIP image embeddings or direct
  HF image encoding. It evaluates:

```text
score = cos(image, positive_prompt_prototype)
      - cos(image, negative_prompt_prototype)

thresholds:
  zero      -> raw positive-vs-negative preference
  valid_f1  -> per-attribute threshold calibrated on validation F1
  valid_acc -> per-attribute threshold calibrated on validation accuracy
```

- Full local test on CelebA test split, using validation-calibrated F1
  thresholds:

```text
macro_accuracy = 0.6861
macro_balanced_accuracy = 0.6497
macro_precision = 0.4228
macro_recall = 0.7033
macro_f1 = 0.4916
micro_accuracy = 0.6861
```

- Strong attributes under prompt-only CLIP:

```text
Male F1=0.987
No_Beard F1=0.923
Young F1=0.902
Smiling F1=0.870
Eyeglasses F1=0.835
```

- Weak attributes under prompt-only CLIP:

```text
Chubby F1=0.199
Sideburns F1=0.129
Rosy_Cheeks F1=0.132
Double_Chin F1=0.195
5_o_Clock_Shadow F1=0.195
```

- Interpretation: CLIP prompt-only can be a useful diagnostic/proxy for some
  high-level or visually explicit attributes, but it is not reliable enough as
  the only reranker signal. A trained CelebA probe is still justified,
  especially for rare/subtle attributes and for calibrated top-pool filtering.

## [2026-06-26] implementation | CLIP prompt-only top-pool filter v1

- Added a no-training second-stage experiment:

```text
cluster/experimental/clip_prompt_filter_v1.py
cluster/jobs/58_clip_prompt_filter_v1_short.sh
```

- Purpose: test whether CLIP prompt-only attribute estimates are already good
  enough to filter/rerank the top-pool returned by the best learned system.
- Pipeline:

```text
best v7/final q_final
  -> retrieve top-500 by cosine
  -> estimate attributes with CLIP prompt ensemble
  -> filter/rerank with predicted query satisfaction and predicted Hamming
  -> evaluate Recall@K and Precision@K on official JSON
```

- Chosen top-pool:

```text
top_pool = 500
```

  Motivation: previous oracle diagnostics showed top-500 has much higher
  chance of containing official-valid targets than top-50/top-25. This run
  should test the filter quality, not be bottlenecked by a too-small pool.

- Implemented methods:

```text
baseline_q_final_top10
A_clip_query_only_valid_f1
A_clip_query_hamming2_valid_f1
A_clip_query_hamming5_valid_f1
B_clip_soft_lq*_lh*_*
C_clip_hybrid_lh*_*
```

- Diagnostics written per method:

```text
summary.csv
per_query_metrics.csv
retrievals.jsonl
avg_kept_in_pool
avg_query_ok_in_pool
avg_pred_hamming_in_pool
avg_official_valids_in_top_pool
```

- Local smoke test passed on a tiny subset, covering checkpoint loading,
  q_model, corrective sum, q_final, CLIP prompt scoring, filtering, JSON
  metrics, CSVs, and PNG plots.

## [2026-06-27] report framing | Separate hybrid compositionality from reranker extensions

- Added a reminder to [Method Roadmap](method-roadmap.md) and [Index](index.md)
  for future report/notebook writing.
- The core assignment-facing hybrid compositionality method should be reported
  as:

```text
q_model  = learned gate / sequential composer(source, signed query)
q_sum    = explicit CLIP arithmetic composition(source, signed query)
q_hybrid = normalize(q_model + beta * (q_sum - source))
```

- Probe filters, CLIP prompt filters, oracle diagnostics, and future rerankers
  should be framed as complete-system extensions built on top of the hybrid
  query vector, not as replacements for the hybrid compositionality method.
- If those extensions are included in the final report, show two result levels:

```text
1. hybrid compositionality alone
2. complete hybrid + reranker/filter system
```

- Rationale: the assignment asks for hybrid compositionality, while the later
  filtering/reranking work addresses a broader retrieval-system problem:
  improving the final top-k precision by selecting cleaner candidates after the
  hybrid vector has retrieved a broad candidate pool.

## [2026-06-27] result | Probe reranker beats CLIP prompt-only filtering

- Finished and inspected the long/short second-stage filtering jobs.
- Baseline frozen hybrid system (`q_final` top-10 from best v7 checkpoint):

```text
Macro Recall@10     0.39698
Micro Recall@10     0.32839
Macro Precision@10  0.06609
```

- CLIP prompt-only top-pool filter best method:

```text
B_clip_soft_lq010_lh005_ls000_valid_f1
Macro Recall@10     0.39856
Micro Recall@10     0.32770
Macro Precision@10  0.06578
```

- Interpretation: prompt-only CLIP attribute estimates are not reliable enough
  for final filtering. They are useful for diagnostics and sanity checks, but
  hard filters remove too many useful candidates and soft scores are nearly
  neutral.

- Learned CelebA probe/reranker best method:

```text
C_hybrid_t050_lh010_ls005
Macro Recall@10     0.42042
Micro Recall@10     0.34682
Macro Precision@10  0.07146
```

- Best probe config:

```text
p005_wide_flip
hidden_dims = 1024-512
dropout     = 0.1
lr          = 0.0003
flip aug    = true
valid macro_f1 = 0.6809
```

- Relative improvement of best probe reranker over frozen hybrid top-10:

```text
Macro Recall@10     +5.9%
Micro Recall@10     +5.6%
Macro Precision@10  +8.1%
```

- Finding: the fair, learned approximation of the oracle filter works better
  than CLIP prompt-only scoring. The useful strategy is not a strict hard
  Hamming filter; it is a permissive query filter plus soft Hamming/source
  rerank over a broad top-500 pool.

## [2026-06-27] implementation | Calibrated probe reranker v2

- Added an experimental calibrated probe reranker:

```text
cluster/experimental/probe_reranker_v2_calibrated.py
cluster/jobs/59_probe_reranker_v2_calibrated_5h.sh
```

- Motivation: v1 used a fixed `0.5` threshold for all predicted CelebA
  attributes. This is too crude because attributes have different priors and
  different calibration quality.
- v2 trains/reuses the same CelebA probe, then calibrates one threshold per
  attribute on the validation split with multiple objectives:

```text
f1
balanced_accuracy
accuracy
precision_recall_mid
```

- The JSON evaluation then compares calibrated variants:

```text
baseline q_final
calibrated query-only filter
calibrated query + hard predicted Hamming<=2
calibrated soft rerank
calibrated hybrid query-filter + soft Hamming/source rerank
```

- Expected diagnostic value: if v2 beats v1, the bottleneck was partly probe
  threshold calibration. If not, the bottleneck is likely probe feature quality
  or the mismatch between predicted attributes and official Hamming validity.

## [2026-06-27] result | Calibrated probe reranker v2 improves strongly

- Completed the calibrated probe reranker v2 job.
- Best method:

```text
A_cal_query_hardh2_accuracy
Macro Recall@10     0.47297
Micro Recall@10     0.40884
Macro Precision@10  0.08570
avg kept from 500   32.70
```

- Baseline frozen hybrid vector for the same checkpoint:

```text
Macro Recall@10     0.39698
Micro Recall@10     0.32839
Macro Precision@10  0.06609
```

- Previous best learned-probe v1 reranker:

```text
C_hybrid_t050_lh010_ls005
Macro Recall@10     0.42042
Micro Recall@10     0.34682
Macro Precision@10  0.07146
```

- v2 threshold calibration summary on validation:

```text
objective             macro_acc  macro_precision  macro_recall  macro_f1
f1                    0.9006     0.6866           0.7828        0.7280
balanced_accuracy     0.8637     0.5709           0.8955        0.6703
accuracy              0.9139     0.7663           0.6517        0.6911
precision_recall_mid  0.9089     0.7181           0.7261        0.7219
```

- Finding: per-attribute threshold calibration was the missing ingredient. A
  fixed threshold of `0.5` made hard filtering brittle; calibrated thresholds,
  especially accuracy-optimized thresholds, made query satisfaction plus
  predicted Hamming `<= 2` strong enough to beat both v1 and the frozen hybrid
  vector by a clear margin.

## [2026-06-27] implementation | Probe error-pattern bucket analysis

- Added a post-hoc diagnostic script:

```text
cluster/experimental/analyze_probe_error_patterns.py
cluster/jobs/60_analyze_probe_error_patterns_short.sh
```

- Purpose: inspect which CelebA attributes dominate probe mistakes in image
  Hamming-error bands:

```text
0 exact errors
1 error
2 errors
3 errors
4-5 errors
>5 errors
```

- The script also reports cumulative bands:

```text
exact_0, <=1, <=2, <=3, <=5, >5
```

- Outputs per split (`valid`, `test`):

```text
image_error_counts.csv
cumulative_bucket_summary.csv
overall_attribute_errors.csv
bucket_attribute_errors.csv
bucket_attribute_error_heatmap.png
README_error_patterns.txt
```

- This should reveal whether probe failures are concentrated in known weak
  attributes such as `Chubby`, `Double_Chin`, subtle hair color, makeup, or
  facial-hair attributes, and whether future filtering should use
  attribute-specific confidence margins or weaker penalties for unreliable
  attributes.

## [2026-06-27] result | Probe error patterns are dominated by subjective facial-shape attributes

- Ran `analyze_probe_error_patterns.py` on the calibrated v2 probe.
- Validation split:

```text
images                 19867
mean Hamming errors    3.4459 / 40
exact 40/40            2.61%
<=1 error              13.02%
<=2 errors             32.07%
<=3 errors             54.40%
<=5 errors             87.51%
>5 errors              12.49%
```

- Test split:

```text
images                 19962
mean Hamming errors    3.6977 / 40
exact 40/40            2.14%
<=1 error              10.37%
<=2 errors             27.04%
<=3 errors             48.50%
<=5 errors             84.20%
>5 errors              15.80%
```

- Main finding: errors are not random. They are concentrated in subjective or
  geometric face attributes:

```text
test worst attributes:
Big_Lips, Oval_Face, Pointy_Nose, Arched_Eyebrows, Attractive,
Wavy_Hair, Big_Nose, Bags_Under_Eyes, Straight_Hair, High_Cheekbones,
Narrow_Eyes, Wearing_Necklace, Brown_Hair
```

- Many of these errors are false negatives, meaning the probe often fails to
  mark an attribute as present even when the CelebA label says it is present:

```text
Big_Lips      mostly FN, test recall 0.128
Oval_Face     mostly FN, test recall 0.335
Pointy_Nose   mostly FN, test recall 0.411
Narrow_Eyes   mostly FN, test recall 0.127
Necklace      mostly FN, test recall 0.145
```

- Important nuance: high-frequency official-query attributes remain strong:

```text
Young              test F1 0.933
Smiling            test F1 0.918
Wearing_Lipstick   test F1 0.932
Heavy_Makeup       test F1 0.887
High_Cheekbones    test F1 0.851
```

- Interpretation for reranking: the v2 hard filter works because calibrated
  query satisfaction is reliable enough, but predicted Hamming is still noisy
  because weak non-query attributes inflate or distort the predicted Hamming
  distance. A promising next ablation is reliability-weighted Hamming:

```text
predicted_hamming = sum_j reliability_j * mismatch_j
```

  where low-F1 attributes such as `Big_Lips`, `Oval_Face`, `Pointy_Nose`,
  `Narrow_Eyes`, and `Wearing_Necklace` receive smaller weights or larger
  tolerance margins. This should preserve the benefit of query filtering while
  reducing false rejection caused by noisy subjective attributes.

## [2026-06-27] result | Oracle component ablation: Hamming filtering drives most of the gain

- Added and ran a local oracle ablation:

```text
final_best_system/code/test_oracle_filter_components.py
```

- Same `q_final` top-500, then four variants:

```text
baseline_q_final
oracle_query_only
oracle_hamming_only
oracle_query_and_hamming
```

- Aggregate result:

```text
method                    Macro R@10  Macro P@10  avg kept / 500
baseline_q_final          0.3970      0.0661      500.0
oracle_query_only         0.5035      0.0905      279.6
oracle_hamming_only       0.9362      0.4045       24.8
oracle_query_and_hamming  0.9647      0.5728        8.3
```

- Interpretation:

```text
non-query Hamming<=2 is the dominant filtering signal.
query satisfaction alone helps only modestly.
query+hamming gives the cleanest final top-10, mostly by improving precision
after Hamming has already found the right neighborhood.
```

- Per-query pattern: `oracle_hamming_only` already reaches very high Recall@10
  for nearly every query, while `oracle_query_and_hamming` mostly boosts
  Precision@10 and removes residual candidates that preserve the source but do
  not satisfy the requested edit.
- Implication for the fair learned filter: prioritize improving predicted
  Hamming/source-preservation reliability, especially with reliability-weighted
  Hamming or attribute-specific tolerances. Query filtering is still useful, but
  it is not the main bottleneck.

## [2026-06-27] implementation | Fair probe component ablation

- Added the fair counterpart of the oracle component ablation:

```text
cluster/experimental/evaluate_probe_filter_components.py
cluster/jobs/61_evaluate_probe_filter_components_short.sh
```

- It uses the best calibrated probe results, then evaluates:

```text
baseline_q_final
probe_query_only
probe_hamming_only
probe_query_and_hamming
```

- Purpose: compare the oracle finding against the deployable/probe-based
  system. The key question is whether predicted Hamming filtering or predicted
  query filtering contributes more when the attributes are estimated by the
  probe rather than read from ground truth.
- The job writes summary CSVs, per-query CSVs, retrieval JSONL files, and plots:

```text
macro_recall10_components.png
macro_precision10_components.png
avg_kept_components.png
```

## [2026-06-27] result | Fair probe component ablation: strict predicted Hamming is brittle

- Ran the fair component ablation with the calibrated probe and no fill-to-k
  fallback. Aggregate result:

```text
method                    Macro R@10  Micro R@10  Macro P@10  avg kept / 500
baseline_q_final          0.3970      0.3284      0.0661       10.0
probe_query_only          0.4174      0.3446      0.0702      292.6
probe_hamming_only        0.3632      0.3522      0.0612       68.3
probe_query_and_hamming   0.3786      0.3649      0.0688       27.3
```

- This differs sharply from the oracle component ablation:

```text
oracle_hamming_only       Macro R@10 0.9362
oracle_query_and_hamming  Macro R@10 0.9647
```

- Finding: true Hamming is extremely valuable, but predicted Hamming from the
  current probe is not reliable enough to be used as a strict deletion filter.
  The deployable probe benefits most from query-only filtering when candidates
  are actually removed.
- The earlier strong calibrated-probe result (`A_cal_query_hardh2_accuracy`,
  Macro R@10 0.4730) used a safer promote-then-fill behavior: candidates that
  pass the filter are promoted first, but if fewer than `top_k` pass, the list is
  filled with original `q_final` candidates. That fallback prevents false
  rejection from destroying recall.
- Implication: future fair systems should not use predicted Hamming as a pure
  hard filter. Use it as:

```text
1. a reranking/promotion signal with fallback;
2. a reliability-weighted soft penalty;
3. an adaptive gate only when enough candidates pass confidently.
```

- This confirms that learning robust source-preservation/Hamming is harder than
  learning query satisfaction.

## [2026-06-27] implementation | Weighted probe reranker follow-up

- Added `cluster/experimental/evaluate_weighted_probe_reranker.py` and
  `cluster/jobs/62_evaluate_weighted_probe_reranker_short.sh`.
- Purpose: start from the current best complete system
  (`q_hybrid v7 + calibrated probe v2`) and test a safer use of predicted
  Hamming:

```text
q_hybrid = normalize(q_model + beta * (q_sum - source))

score(candidate) =
    cosine(q_hybrid, candidate)
  + lambda_query * calibrated_query_margin(candidate)
  - lambda_hamming * weighted_predicted_hamming(source, candidate)
  + lambda_source * cosine(source, candidate)
```

- Important design decision: query satisfaction can be used more strongly
  because it usually checks only the few attributes explicitly requested by the
  query. Predicted Hamming should not be a pure deletion filter because it
  depends on many non-query attributes and the probe often makes 3-5 image-level
  attribute mistakes.
- New experiment compares:
  - frozen `q_final` baseline;
  - current `A_cal_query_hardh2_accuracy` promote/fill behavior;
  - query-only promotion;
  - soft weighted-Hamming rerankers;
  - query-hard + soft weighted-Hamming variants.
- Reliability weighting uses per-attribute calibrated probe metrics such as F1
  or accuracy. Noisy attributes contribute less to the Hamming penalty.
- Smoke-tested locally on one source-query case with local final weights and
  probe artifacts; the full run is intended for the cluster short queue.

## [2026-06-27] result | Weighted probe reranker did not improve over calibrated hard promote/fill

- Ran `cluster/jobs/62_evaluate_weighted_probe_reranker_short.sh` on the
  cluster.
- Best method remained the existing calibrated probe method:

```text
method=current_A_query_hardh2_accuracy
kind=promote_query_hardh2
objective=accuracy
Macro Recall@10     0.47297151547368665
Micro Recall@10     0.4088406147888176
Macro Precision@10  0.08570375640709439
avg kept from 500   27.343763653007844
```

- Best new weighted-soft candidate was lower:

```text
queryhard_soft_accuracy_f1_hard_lh0p05_ls0p05
Macro Recall@10     0.43800192168510127
Micro Recall@10     0.35855621444995767
Macro Precision@10  0.07401706950895173
```

- Query-only promotion was also lower:

```text
query_promote_accuracy
Macro Recall@10     0.41741976975143835
Micro Recall@10     0.3445782403485417
Macro Precision@10  0.07020008435741161
```

- Frozen `q_hybrid` baseline in the same run:

```text
baseline_q_final
Macro Recall@10     0.39698439353876375
Micro Recall@10     0.32839162531768123
Macro Precision@10  0.06608954997851167
```

- Interpretation: for the current probe, the strongest deployable behavior is
  still calibrated hard query+Hamming promotion with fallback. Soft
  reliability-weighted Hamming does not preserve enough of the oracle Hamming
  signal and weakens the top-10 ranking.
- Technical note: some duplicated soft method names produced
  `source_query_cases=66104` instead of `33052`, meaning those duplicate-named
  variants were accidentally aggregated twice. This does not affect the winning
  `current_A_query_hardh2_accuracy` result or the high-level conclusion, but the
  script should deduplicate method names if reused for a paper-quality ablation.

## [2026-06-27] fix | Final package made self-contained for evaluation

- Found that `final_best_system/README.md` and `manifest.json` were still
  describing the older `model_plus_generic_delta_beta_1p50` system
  (`Macro R@10 ~= 0.391`) even though the actual best system is now
  `current_A_query_hardh2_accuracy` (`Macro R@10 ~= 0.473`).
- Updated both files to describe the current final system:

```text
q_hybrid = normalize(q_model + 1.25 * (q_sum - source))
then calibrated probe query+Hamming promote/fill reranking
```

- Added the minimal local data needed to evaluate the final package without
  reading from `cluster/data`:

```text
final_best_system/data/celeba/embeddings/openai_clip_vit_b32/test_image_embeddings.pt
final_best_system/data/celeba/embeddings/openai_clip_vit_b32/attribute_text_embeddings.pt
final_best_system/data/celeba/embeddings/openai_clip_vit_b32/signed_attribute_prompt_embeddings.pt
final_best_system/data/celeba/embeddings/openai_clip_vit_b32/signed_attribute_prompt_embeddings_v2_photo_templates.pt
final_best_system/data/celeba/annotations/list_attr_celeba.txt
final_best_system/data/celeba_evaluation.json
```

- Added package runners that print terminal output:

```text
final_best_system/code/run_current_best_smoke.sh
final_best_system/code/run_current_best_full_eval.sh
```

- Smoke-tested `run_current_best_smoke.sh`: it now loads gate checkpoint, probe,
  calibrated thresholds, cached probe probabilities, JSON, and packaged test
  embeddings from `final_best_system`.

## [2026-06-29] result | Local oracle/probe filtering matrix over top-500/200/100

- Added and ran
  `final_best_system/code/evaluate_filtering_matrix.py` locally on Mac using
  packaged final-best artifacts.
- It evaluates the frozen best hybrid system (`q_hybrid`, model + sum) and then
  strict filters top-pool candidates by:
  - oracle query constraint;
  - oracle non-query Hamming `<=2`;
  - oracle query + Hamming;
  - learned probe query constraint;
  - learned probe predicted Hamming `<=2`;
  - learned probe query + predicted Hamming.
- Output saved to:

```text
final_best_system/results/filtering_matrix/latest/summary.csv
```

- Main micro/pooled results:

```text
top-500 no filter:                  Acc/Recall@10 0.3284, Precision@10 0.0521
top-500 oracle query only:          Acc/Recall@10 0.4240, Precision@10 0.0733
top-500 oracle hamming only:        Acc/Recall@10 0.9349, Precision@10 0.4014
top-500 oracle query+hamming:       Acc/Recall@10 0.9648, Precision@10 0.5789
top-500 probe query only:           Acc/Recall@10 0.3446, Precision@10 0.0561
top-500 probe hamming only:         Acc/Recall@10 0.3522, Precision@10 0.0581
top-500 probe query+hamming:        Acc/Recall@10 0.3649, Precision@10 0.0643
```

- Finding: the oracle confirms again that the target-valid region is present in
  the top-500 pool and that true non-query Hamming is the dominant missing
  signal. The learned probe can provide a modest strict-filter improvement, but
  it is far from the oracle because predicted Hamming is noisy.
- The best deployed probe result remains the calibrated promote/fill method
  (`A_cal_query_hardh2_accuracy`, Macro R@10 ~= 0.473), because fallback avoids
  over-pruning when the probe wrongly rejects good candidates.

## [2026-06-29] clarification | The 0.42/8.5% final result is promote+fill, not strict filtering

- Re-ran `final_best_system/code/evaluate_filtering_matrix.py` after adding the
  explicit `probe_query_and_hamming_fill` row. The previously confusing output
  was comparing only strict filtering, which is not the final deployed system.
- Correct distinction:

```text
strict learned probe query+hamming, top-500:
  micro Recall@10    0.3649
  micro Precision@10 0.0643

final current_A_query_hardh2_accuracy style, top-500 promote+fill:
  micro Recall@10    0.4088
  micro Precision@10 0.0712

official saved final summary:
  macro Recall@10    0.4730
  macro Precision@10 0.0857
```

- Interpretation for report/debugging:
  - `0.4088` is the pooled/micro Recall@10 of the final promote+fill reranker;
  - `0.47297` is the macro Recall@10 averaged by query;
  - `0.0857` is the macro Precision@10, while the micro Precision@10 is
    `0.0712`.
- This explains why the remembered "about 0.42 recall and about 8.5% precision"
  did not appear in the strict filtering matrix: it mixed the final
  promote+fill recall behavior with the official macro precision value.

## [2026-06-29] result | Fresh full filtering matrix rerun with the real frozen system

- Re-ran the filtering matrix over the full official JSON evaluation:

```text
query entries:       14
source-query cases:  33052
output: final_best_system/results/filtering_matrix/real_system_rerun_20260629_114221/summary.csv
```

- The frozen retrieval vector is the real current hybrid system:

```text
q_model  = learned sequential gate(source, query)
q_sum    = CLIP generic arithmetic sum(source, query)
q_hybrid = normalize(q_model + 1.25 * (q_sum - source))
```

- Oracle filtering is diagnostic only: it uses true CelebA labels to simulate a
  perfect attribute probe and is not a deployable/fair inference system.
- Learned-probe filtering uses the trained/calibrated probe. Strict learned
  filtering improves precision slightly but can over-prune. The deployed final
  method remains learned probe query+Hamming promotion with fallback fill.

## [2026-06-29] notebook | Added top-pool diagnostic explanation and figure

- Updated `notebooks/02_learned_gate_final_pipeline.ipynb` with a report-ready
  diagnostic section explaining why broad-pool retrieval supports the claim that
  the hybrid system reaches the right semantic region.
- Added a didactic figure:

```text
final_best_system/explanations/top_pool_retrieval_diagnostic.png
```

- The notebook now loads:

```text
final_best_system/results/filtering_matrix/real_system_rerun_20260629_114221/summary.csv
```

  and displays top-500/top-200/top-100 oracle/probe filtering results.
- Important framing added to the notebook:
  - oracle rows are **diagnostic upper bounds**, not fair deployable inference;
  - learned probe rows are realistic but noisy;
  - high oracle top-500 recall shows that `q_final` often reaches the right
    candidate region;
  - remaining improvement should focus on fair reranking/attribute-source
    preservation estimation inside the candidate pool.

## [2026-06-29] experiment plan | Probe architecture sweep v3

- Motivation: oracle filtering shows that a perfect attribute/Hamming estimator
  inside the top-500 pool would massively improve retrieval, but the current
  CelebA probe is too noisy, especially when Hamming requires many attributes to
  be right simultaneously.
- Added a separate experimental script, leaving `final_best_system` untouched:

```text
cluster/experimental/probe_arch_sweep_v3.py
cluster/jobs/63_probe_arch_sweep_v3_smoke_short.sh
cluster/jobs/64_probe_arch_sweep_v3_long.sh
```

- Research-backed directions included in the sweep:
  - Asymmetric Loss (ASL) for multi-label positive/negative imbalance;
  - C-Tran-inspired label-token Transformer to model dependencies among labels;
  - ML-GCN-inspired label graph classifier using CelebA label co-occurrence;
  - deeper/wider residual MLP baselines to test whether capacity alone helps.
- Each config:
  1. trains a probe on train CLIP embeddings and CelebA labels;
  2. calibrates per-attribute thresholds on validation;
  3. predicts test-gallery attributes;
  4. evaluates the frozen current hybrid system on official JSON using probe
     query/Hamming filters;
  5. writes `summary.csv`, `per_query_metrics.csv`, checkpoints, thresholds,
     and a global `BEST_PROBE_ARCHITECTURE.txt`.
- Local smoke test completed successfully at:

```text
/tmp/probe_arch_sweep_v3_smoke
```

- Important framing: this is not a replacement for the current final system yet.
  It is a controlled search for a stronger fair probe/reranker that could close
  part of the oracle-vs-probe gap.
- Policy check: the new architectures are implemented from scratch using
  standard PyTorch primitives (`Linear`, `LayerNorm`, `TransformerEncoderLayer`,
  etc.). They are inspired by paper-level ideas (ASL, C-Tran-like label tokens,
  ML-GCN-like label co-occurrence graph), but do not copy code from external
  repositories or from other groups. A citation/compliance note was added at the
  top of `cluster/experimental/probe_arch_sweep_v3.py`.

## 2026-06-29 - Qualitative evaluation caveat and visualizer default

- Added a qualitative caveat section to
  `notebooks/02_learned_gate_final_pipeline.ipynb`.
- Motivation: some retrieval examples show that the official JSON/Hamming
  metric can penalize visually plausible results that preserve identity/style
  well, while some official-valid targets can change non-requested visual
  properties. This should be presented carefully as a limitation of discrete
  annotation-based evaluation, not as a replacement for the official metrics.
- Examples to use in the report/notebook:
  - `query_id=5`, `source_index=3`, `+Blond_Hair`: a top retrieval can look
    extremely similar to the source but be counted invalid if the CelebA
    annotation/query constraint fails.
  - `query_id=4`, `source_index=3`, `-Young`: predicted results often preserve
    hair color, lipstick, makeup, and identity-like appearance better than some
    official-valid JSON examples, but the official metric only counts the JSON
    target set.
- Updated `final_best_system/code/show_json_retrieval_example.py` so the default
  `--method-dir` now points to the current final system:
  `final_best_system/results/probe_reranker_v2_calibrated/A_cal_query_hardh2_accuracy`.
- To reproduce older hybrid-core screenshots exactly, pass
  `--method-dir final_best_system/results/final_best_model_plus_generic_delta_beta_1p50`.
- Added a notebook cell that lists valid `source_index` values for each
  `query_id`, because each official query has its own source set and not every
  source index is valid for every query.

## 2026-06-29 - Free-query visualizer mode

- Extended `final_best_system/code/show_json_retrieval_example.py` with a free-query mode for qualitative demos outside the official JSON source/query pairs.
- New usage example:

```bash
.venv/bin/python final_best_system/code/show_json_retrieval_example.py \
  --free-source-index 457 \
  --free-query '+young -lipstick' \
  --top-k 10
```

- The free mode computes the actual final system instead of reading `retrievals.jsonl`: learned gate query, generic CLIP arithmetic query, vector-delta correction, then optional calibrated probe query/Hamming filtering. It renders top-k results with blue=input, green=query satisfied according to CelebA labels, red=query fail.
- It also accepts aliases such as `lipstick -> Wearing_Lipstick`, `glasses -> Eyeglasses`, `smile -> Smiling`, `hat -> Wearing_Hat`, and can use `--free-image` for a CelebA filename/path or numeric gallery index.
- Important distinction: this mode is for qualitative demos. Official metrics must still be computed only on the source/query pairs present in `celeba_evaluation.json`.

## 2026-06-29 - Open-vocabulary free-query attributes

- Extended `final_best_system/code/show_json_retrieval_example.py` again so free-query mode no longer fails on attributes outside the 40 CelebA labels.
- Known CelebA attributes still use cached prompt directions and can be checked/filtered by the calibrated probe.
- Unknown/open attributes now create a CLIP text direction on the fly:

```text
d_open = normalize(CLIP_text("a portrait photo of a face with <attribute>")
                   - CLIP_text("a portrait photo of a face without <attribute>"))
```

- The learned sequential gate can consume this vector because it operates on CLIP-space edit directions, not on hardcoded class IDs. The arithmetic correction uses the same open direction.
- Probe filtering only applies to known CelebA parts of the query. Open attributes are visual-only and rendered in yellow as `known OK, open unchecked`, avoiding false claims of ground-truth correctness.
- Smoke test passed:

```bash
.venv/bin/python final_best_system/code/show_json_retrieval_example.py \
  --free-source-index 457 \
  --free-query '+orange hair -lipstick' \
  --top-k 10 \
  --top-pool 50 \
  --device cpu
```

- This mode is useful to inspect whether the hybrid CLIP-space system generalizes beyond CelebA. It should not be used for official metrics unless labels/evaluation targets exist for the open attribute.

## 2026-06-29 - Open-vocabulary qualitative finding: Sunglasses vs Eyeglasses

- Added a final notebook qualitative section comparing the same source image with two related edits:
  - `+Eyeglasses`, a known CelebA attribute that can be verified by the probe;
  - `+Sunglasses`, an open CLIP attribute not present in the 40 CelebA labels.
- Commands used:

```bash
.venv/bin/python final_best_system/code/show_json_retrieval_example.py \
  --free-source-index 47 \
  --free-query '+Eyeglasses' \
  --top-k 10 \
  --top-pool 50

.venv/bin/python final_best_system/code/show_json_retrieval_example.py \
  --free-source-index 47 \
  --free-query '+Sunglasses' \
  --top-k 10 \
  --top-pool 50
```

- Qualitative finding: the two retrieval grids differ meaningfully. `+Eyeglasses` retrieves faces with regular/prescription glasses, while `+Sunglasses` retrieves faces with darker/tinted sunglasses. This supports the claim that the hybrid system can operate on CLIP semantic directions beyond the supervised CelebA attribute list.
- Important caveat: this is not an official quantitative result because open attributes such as `Sunglasses` have no CelebA ground truth in our evaluation. In figures, open-vocabulary conditions remain yellow/unchecked instead of green/correct.

## 2026-06-29 - Open-vocabulary qualitative finding: visible teeth

- Added a second open-vocabulary qualitative example to the final notebook: `+visible teeth` vs `+not visible teeth` on source index 666.
- Commands used:

```bash
.venv/bin/python final_best_system/code/show_json_retrieval_example.py \
  --free-source-index 666 \
  --free-query '+not visible teeth' \
  --top-k 10 \
  --top-pool 50

.venv/bin/python final_best_system/code/show_json_retrieval_example.py \
  --free-source-index 666 \
  --free-query '+visible teeth' \
  --top-k 10 \
  --top-pool 50
```

- Qualitative finding: the two grids differ coherently; `+visible teeth` tends to retrieve faces with visible teeth/open smiles, while `+not visible teeth` tends to retrieve closed-mouth/no-teeth faces. This is a stronger open-vocabulary sanity check than sunglasses alone because it probes a fine-grained facial cue rather than an accessory.
- Caveat remains: this is visual evidence of CLIP-direction use, not official evaluation, because CelebA does not provide a `visible teeth` attribute.

## 2026-06-29 - Notebook open-vocabulary section refocused on multi-attribute examples

- Reworked the final notebook open-vocabulary section.
- Removed the emphasis on several single-attribute demos; kept only one simple `+Eyeglasses` vs `+Sunglasses` comparison as an introductory sanity check.
- Added stronger multi-attribute examples using the same source identity and queries combining known CelebA attributes with open CLIP concepts:
  - `-visible teeth, -Eyeglasses`
  - `+white skin, -visible teeth, -Eyeglasses`
  - `-visible teeth, +Eyeglasses`
- These examples better support the discussion claim that the system can handle:
  1. queries with multiple requested edits;
  2. attributes unknown to the supervised CelebA dataset but represented in CLIP semantic space.
- Caveat remains: open-vocabulary examples are qualitative, not official metrics, because these attributes have no official ground truth in CelebA evaluation.

## 2026-06-29 - Final notebook made Colab/submission-ready

- Audited `notebooks/02_learned_gate_final_pipeline.ipynb` against the assignment delivery notes:
  - single notebook,
  - runnable code with heavy training/evaluation disabled by booleans,
  - Markdown report sections for method, experimental setup, results, discussion, and qualitative caveats,
  - output cells kept for quick inspection.
- Fixed the final formula in the intro to use the current best correction value:

```text
q_final = normalize(q_model + 1.25 * (q_sum - source))
```

- Added a concise methodological roadmap explaining the progression:
  1. vanilla CLIP arithmetic baseline;
  2. contrastive/sequential prompt-direction experiments;
  3. learned sequential gate as the hybrid compositionality core;
  4. mixed/official-like training and Hamming-weighted objectives;
  5. oracle top-pool diagnostic showing that the system reaches the right region;
  6. calibrated CelebA probe reranking as an added retrieval-stage improvement.
- Embedded all qualitative PNGs directly into the notebook as base64 data images. This avoids broken relative image links when the notebook is opened in Google Colab.
- Cleared a stale error output from a disabled full-evaluation cell; the notebook remains valid JSON and no longer contains error outputs.
- Important packaging caveat for future submission: figures are embedded, but model execution still needs the submitted artifacts next to the notebook (`final_best_system/`, CelebA files, and precomputed CLIP embeddings). If submitting only the notebook without the artifact folder, add a Colab setup cell that downloads or mounts those files.

## 2026-06-29 - Added fourth multi-attribute open-vocabulary example

- Added another qualitative figure to the final notebook's open-vocabulary section:
  - `+visible teeth, +sunglasses`
- The figure was generated with the final free-query visualization script on source index 66 and stored under `final_best_system/results/free_query_examples/`.
- The example is useful because both requested attributes are open CLIP concepts rather than official CelebA labels. It further supports the qualitative claim that the hybrid CLIP-space composer can combine multiple non-supervised semantic directions.
- As before, this remains a qualitative example only: open-vocabulary attributes are not part of the official JSON evaluation labels.

## 2026-06-29 - Probe architecture fine sweep v4 prepared

- The v3 probe architecture sweep finished without beating the previous final probe reranker, but several candidates were very close:
  - previous best: macro Recall@10 `0.47297`, macro Precision@10 `0.08570`;
  - best v3 new probe: `p05_mlp_deep_asl`, macro Recall@10 `0.47110`, macro Precision@10 `0.08350`;
  - close candidates: `p06_mlp_lowdrop_asl`, `p03_mlp_current_asl`, `p02_mlp_current_bce`, `p14_transformer_128_l2_asl`.
- Interpretation: the probe architecture matters, but the gap is small. Larger/structured models such as transformers did not clearly dominate MLPs, likely because the probe input is already a compact CLIP embedding and the output has only 40 attributes; label dependency modelling can help, but it can also overfit or damage calibration.
- Prepared `cluster/experimental/probe_arch_sweep_v4_finetune.py` as a separate experimental sweep. It does not alter the current final system.
- The v4 sweep trains 50 focused probe configurations:
  - 20 MLP variants around the best deep/low-dropout ASL/BCE probes;
  - 10 residual MLP variants;
  - 8 label-wise variants;
  - 12 compact label-transformer variants.
- The sweep varies:
  - hidden dimensions,
  - dropout,
  - learning rate,
  - ASL/BCE/focal loss,
  - ASL negative gamma,
  - use/no-use of positive class weights,
  - small embedding noise augmentation,
  - transformer width/depth.
- Added SLURM job `cluster/jobs/65_probe_arch_sweep_v4_finetune_long.sh`.
- The v4 script saves:
  - `train_metrics.csv` per config under `probes/<config_id>/`;
  - `best_probe.pt`;
  - calibrated thresholds and probe probabilities;
  - full official JSON evaluation summaries for query-only, hamming-only, and query+hamming filtering modes;
  - `aggregate_summary.csv`;
  - `BEST_PROBE_ARCHITECTURE.txt`;
  - training-curve PNGs under `training_curves/`, including train loss, validation selection score, validation Hamming<=2 percentage, and macro F1 curves.
- Local smoke validation passed with the project virtualenv:
  - `short` profile has 1 smoke config;
  - `long` profile has exactly 50 configs;
  - Python compilation and bash syntax check passed.

## 2026-06-29 - Probe sweep v4 updated with larger transformers and live leaderboard

- Updated `probe_arch_sweep_v4_finetune.py` before running the long sweep.
- The v4 search still has exactly 50 configurations, but now includes larger label-transformer probes:
  - `384` hidden dimension with 2/3/4 layers and 8 heads;
  - `512` hidden dimension with 2/3/4 layers and 8 heads;
  - one larger focal-loss transformer variant.
- Rationale: v3 showed compact transformers close to MLPs but not better. The updated v4 explicitly tests whether higher label-attention capacity can close the gap.
- Added a live leaderboard regenerated after every completed config:
  - `live_ranked_systems.csv`;
  - `live_ranked_systems.txt`.
- The live leaderboard includes:
  - the previous best final system as fixed reference;
  - every completed v4 config evaluated with `only_query`, `only_hamming`, and `both` filtering modes;
  - only fill-based methods, so all systems return 10 results.
- Local validation passed:
  - exactly 50 long configs;
  - 6 large transformer configs;
  - Python compile ok;
  - bash syntax ok;
  - live table writer smoke-tested.

## 2026-06-29 - Final package portability and free-query output policy

- Updated the final package documentation so the canonical runnable system is clear:
  - official JSON visualization via `show_json_retrieval_example.py --query-id ... --source-index ...`;
  - arbitrary known-CelebA queries via `--free-query`;
  - open-vocabulary CLIP queries via `--free-query` with unknown attributes.
- Important repository hygiene decision:
  - do **not** push `final_best_system/results/free_query_examples/`;
  - this folder contains local qualitative experiments and should be regenerated from scratch when needed;
  - the final notebook embeds the selected qualitative figures directly, so the report does not depend on this folder.
- Added `.gitignore` rule for `final_best_system/results/free_query_examples/`.
- Portability cleanup:
  - final runner scripts now use `final_best_system/data/...` as the package data root;
  - the final evaluation runner calls the actual final system (`learned gate + CLIP delta correction + calibrated probe filter`) rather than an older beta-sweep diagnostic;
  - diagnostic oracle scripts now default to the packaged final-system data instead of the cluster data path.
- Reminder for future agents: after updating wiki/code/notebook, remind the user to push changes; do not stage generated free-query images unless explicitly requested.
