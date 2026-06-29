# Deep Learning Project LLM Wiki

This is the maintained knowledge base for the project repository.

## How To Use This Wiki

Read this index first, then open the linked pages needed for the task. Raw source files remain in the project root and `celeba/`; converted readable derivatives live in `source-markdown/`.

## Wiki Maintenance Rules

- When updating project knowledge, experiments, architecture decisions, cluster commands, or results, also update [Log](log.md) with a short dated entry.
- When several wiki updates accumulate, remind the user to push the wiki changes to GitHub so future agents and collaborators see the same context.
- Do not rely only on chat history for important decisions. If a decision affects future work, record it in the relevant wiki page and mention it in the log.
- When a paper or external reference materially informs a decision, add it to the wiki with a link and a one-line note explaining why it matters. Do not leave literature findings only in chat.
- When producing the final notebook or any report, include or regenerate `final_best_system/explanations/toy_vector_correction_clip_cosine.png` and explain that CLIP retrieval ranks by cosine angle after normalization, not raw Euclidean coordinate distance.
- When producing the final notebook/report, keep the framing in [Method Roadmap](method-roadmap.md): report `gate + CLIP arithmetic correction` as the core hybrid-compositionality method, and describe any probe/filter/reranker as an optional system-level extension built on top of that hybrid query vector.

## Core Pages

- [Current State and Proposed Solution](../PROJECT_STATE_AND_SOLUTION.md): authoritative snapshot of completed work, task definition, selected method, repository state, risks, and remaining milestones.
- [Folder and Training Flow Schema](../PROJECT_FOLDER_AND_TRAINING_SCHEMA.md): proposed repository tree, file schemas, model modules, labels, batch calls, epoch flow, and generated artifacts.
- [Practical Full Training Cycle](../PRACTICAL_TRAINING_CYCLE.md): one verified same-person CelebA pair followed from raw `.txt` rows through attribute differences, both MLPs, cosine loss, backpropagation, and inference.
- [Complete Group Project Guide](../PROJECT_GUIDE.md): canonical onboarding document covering the full task, files, starter code, implementation architecture, group workflow, and final checklist.
- [Steps 1-3 CLIP Notebook](../notebooks/01_clip_steps_1_2_3.ipynb): data exploration, offline CLIP feature caches, four zero-shot arithmetic baselines, official metrics, and qualitative comparison.
- [Final Learned Gate Pipeline Notebook](../notebooks/02_learned_gate_final_pipeline.ipynb): single self-contained submission notebook with the complete codebase inline: CLIP caches, prompt embeddings, same-identity pairs, baselines, learned gate training, final `model_plus_generic_delta_100` loading/evaluation, cosine-geometry explanation, official metrics, comparison plots, and report text.
- [Cluster Baselines and Prompt Experiments](cluster-baselines-and-prompt-experiments.md): cluster execution, frozen CLIP caches, five arithmetic methods, official quantitative results, adaptive tangent experiment, and local binary/multi-pair prompt diagnostics.
- [Project Overview](project-overview.md): what the assignment is asking for, in plain language.
- [Practical Input and Output](practical-input-output.md): concrete examples of what enters the system, what it returns, and where files live.
- [Source Inventory](source-inventory.md): every non-image project file, the CelebA image corpus, and how files relate.
- [CelebA Dataset](dataset-celeba.md): dataset structure, attributes, identity labels, splits, and how to derive same-person attribute-edit pairs automatically.
- [Evaluation Protocol](evaluation-protocol.md): official queries, JSON structure, ground truth, Recall@K, Precision@K.
- [Method Roadmap](method-roadmap.md): practical implementation plan from baseline to improved fusion.
- [Proposed Training Strategy](training-strategy.md): formal same-identity pair construction, residual composition network, contrastive loss, risks, and experiments.
- [Learned Gate Cluster Runbook](learned-gate-cluster-runbook.md): exact Slurm commands, expected outputs, checkpoints, plots, and health checks for the first learned gated residual training run.
- [Official Results and Literature Findings](official-results-and-literature-findings.md): current best official JSON results, weak queries, benchmark mismatch diagnosis, relevant CLIP/compositionality literature, and next experiment decision.
- [CLAY Relationship and Integration Decision](clay-integration-analysis.md): whether CLAY is required, how its task differs from signed editing, and where a CLAY-inspired dynamic metric can fit the residual architecture.
- [CLIP Data Flow](clip-data-flow.md): images, pixel tensors, text tokens, shared CLIP embeddings, tensor shapes, train/test/inference flow, and caching.
- [MarkItDown Workflow](markitdown-workflow.md): when to use MarkItDown and how to link produced Markdown into the wiki.
- [Log](log.md): chronological record of wiki setup and maintenance.

## Converted Source Markdown

Generated with MarkItDown 0.1.6:

- [Project assignment - V1.2](source-markdown/Project%20assignment%20-%20V1.2.md)
- [Project assignment - V1.2-2 duplicate](source-markdown/Project%20assignment%20-%20V1.2-2.md)
- [Project - Introduction-2](source-markdown/Project%20-%20Introduction-2.md)
- [Project Skeleton](source-markdown/Project%20Skeleton.md)
- [Lab 2 - CNNs](source-markdown/Lab%202%20-%20CNNs.md)
- [CLIP](source-markdown/CLIP.md)
- [CLIP Compositionality - Davide Berasi](source-markdown/CLIP%20Compositionality%20-%20Davide%20Berasi.md)
- [CLAY](source-markdown/CLAY.md)
- [Compositionality](source-markdown/Compositionality.md)
- [VM usage](source-markdown/VM%20usage.md)

## Current High-Level Understanding

The task is to build a compositional image retrieval system. Given a reference face image plus textual attribute edits such as `+Eyeglasses` or `+Black_Hair, -Wavy_Hair`, the system must retrieve other CelebA test images that preserve the reference identity/remaining attributes while satisfying the requested positive and negative conditions.

The assignment specifically asks for a more flexible fusion mechanism than CLAY's rigid pre-SVD stacking of multiple condition embeddings. The project may be training-free or training-based, but it should be lightweight, rigorously evaluated, and reported clearly in a single Colab notebook.

As of 2026-06-23, the strongest final system is `model_plus_generic_delta_100`: a learned gate query corrected by a generic CLIP arithmetic displacement, `q_final = normalize(q_model + 1.0 * (q_sum - source))`. It reaches official JSON Macro R@10 `0.2827` and Micro R@10 `0.2386`. The assignment vanilla baseline is `direct_sum` (Macro R@10 `0.1084`), while the strongest no-training CLIP-only baseline is `contrastive_sequential` (Macro R@10 `0.1871`). The clean package for report assets is `final_best_system/`, including the best checkpoint at `final_best_system/weights/best_val_official_like_at10.pt`. See [Official Results and Literature Findings](official-results-and-literature-findings.md), [Proposed Training Strategy](training-strategy.md), and [Learned Gate Cluster Runbook](learned-gate-cluster-runbook.md).

Read [PROJECT_STATE_AND_SOLUTION.md](../PROJECT_STATE_AND_SOLUTION.md) for the current project snapshot, then use [PROJECT_GUIDE.md](../PROJECT_GUIDE.md) as the detailed onboarding and execution plan.
