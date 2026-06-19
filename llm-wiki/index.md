# Deep Learning Project LLM Wiki

This is the maintained knowledge base for the project in `/Users/kuba/deep_learning`.

## How To Use This Wiki

Read this index first, then open the linked pages needed for the task. Raw source files remain in the project root and `celeba/`; converted readable derivatives live in `source-markdown/`.

## Wiki Maintenance Rules

- When updating project knowledge, experiments, architecture decisions, cluster commands, or results, also update [Log](log.md) with a short dated entry.
- When several wiki updates accumulate, remind the user to push the wiki changes to GitHub so future agents and collaborators see the same context.
- Do not rely only on chat history for important decisions. If a decision affects future work, record it in the relevant wiki page and mention it in the log.

## Core Pages

- [Current State and Proposed Solution](../PROJECT_STATE_AND_SOLUTION.md): authoritative snapshot of completed work, task definition, selected method, repository state, risks, and remaining milestones.
- [Folder and Training Flow Schema](../PROJECT_FOLDER_AND_TRAINING_SCHEMA.md): proposed repository tree, file schemas, model modules, labels, batch calls, epoch flow, and generated artifacts.
- [Practical Full Training Cycle](../PRACTICAL_TRAINING_CYCLE.md): one verified same-person CelebA pair followed from raw `.txt` rows through attribute differences, both MLPs, cosine loss, backpropagation, and inference.
- [Complete Group Project Guide](../PROJECT_GUIDE.md): canonical onboarding document covering the full task, files, starter code, implementation architecture, group workflow, and final checklist.
- [Steps 1-3 CLIP Notebook](../notebooks/01_clip_steps_1_2_3.ipynb): data exploration, offline CLIP feature caches, four zero-shot arithmetic baselines, official metrics, and qualitative comparison.
- [Cluster Baselines and Prompt Experiments](cluster-baselines-and-prompt-experiments.md): cluster execution, frozen CLIP caches, five arithmetic methods, official quantitative results, adaptive tangent experiment, and local binary/multi-pair prompt diagnostics.
- [Project Overview](project-overview.md): what the assignment is asking for, in plain language.
- [Practical Input and Output](practical-input-output.md): concrete examples of what enters the system, what it returns, and where files live.
- [Source Inventory](source-inventory.md): every non-image project file, the CelebA image corpus, and how files relate.
- [CelebA Dataset](dataset-celeba.md): dataset structure, attributes, identity labels, splits, and how to derive same-person attribute-edit pairs automatically.
- [Evaluation Protocol](evaluation-protocol.md): official queries, JSON structure, ground truth, Recall@K, Precision@K.
- [Method Roadmap](method-roadmap.md): practical implementation plan from baseline to improved fusion.
- [Proposed Training Strategy](training-strategy.md): formal same-identity pair construction, residual composition network, contrastive loss, risks, and experiments.
- [Learned Gate Cluster Runbook](learned-gate-cluster-runbook.md): exact Slurm commands, expected outputs, checkpoints, plots, and health checks for the first learned gated residual training run.
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

As of 2026-06-18, the strongest official JSON baseline is still Contrastive Sequential with Macro R@10 `0.1871`. The first learned residual gate (`gate_v1`) trained successfully but reached Macro R@10 `0.1601` on the official JSON, so it did not beat the best arithmetic baseline. Its per-query behavior motivated the new additive-gate architecture (`gate_v2`), which directly adds gated CLIP contrastive directions and uses a small residual correction. See [Proposed Training Strategy](training-strategy.md) and [Learned Gate Cluster Runbook](learned-gate-cluster-runbook.md).

Read [PROJECT_STATE_AND_SOLUTION.md](../PROJECT_STATE_AND_SOLUTION.md) for the current project snapshot, then use [PROJECT_GUIDE.md](../PROJECT_GUIDE.md) as the detailed onboarding and execution plan.
