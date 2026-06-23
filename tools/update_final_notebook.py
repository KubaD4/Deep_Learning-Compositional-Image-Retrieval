#!/usr/bin/env python3
"""Update the final notebook with the current best system section.

The notebook is JSON, so this script edits cells in a repeatable way instead of
manual notebook surgery. It keeps the training/pipeline cells, then inserts a
report-ready section for the final model+CLIP-delta system and mirrors the
result into cluster/notebooks.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path


ROOT = Path("/Users/kuba/deep_learning")
NOTEBOOK = ROOT / "notebooks" / "02_learned_gate_final_pipeline.ipynb"
CLUSTER_NOTEBOOK = ROOT / "cluster" / "notebooks" / "02_learned_gate_final_pipeline.ipynb"

MARKERS = (
    "## Final Proposed System: Learned Gate + CLIP Arithmetic Delta",
    "## Cosine Geometry: Why the Delta Correction Can Help",
    "## Loading the Final Checkpoint From the Repo",
    "## Final System Function",
    "## Final Official Results Included in the Repo",
)


def md(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.strip("\n").split("\n")],
    }


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.strip("\n").split("\n")],
    }


FINAL_METHOD_MD = r"""
## Final Proposed System: Learned Gate + CLIP Arithmetic Delta

The trained model alone is useful, but the best final system found in the cluster experiments is a **hybrid retrieval query**:

```text
q_model = learned_gate(source, query)
q_sum   = generic CLIP arithmetic(source, query)
q_final = normalize(q_model + beta * (q_sum - source))
```

The selected report system uses `beta = 1.0` and is named `model_plus_generic_delta_100`.

### Mathematical Model

Let:

```text
z_s = normalize(CLIP_image(source_image))
d_a = normalize(CLIP_text(prompt_positive(a)) - CLIP_text(prompt_negative(a)))
```

For a signed query such as `+Smiling, +Eyeglasses, -Young`, each signed condition is:

```text
c_j = sign_j * d_attribute_j
```

The learned sequential gate applies the edits one at a time:

```text
q_0     = z_s
alpha_j = gate_theta(q_{j-1}, c_j)
q_j     = normalize(q_{j-1} + edit_scale * alpha_j * c_j)
q_model = normalize(q_n + residual_scale * residual_phi(z_s, sum_j alpha_j c_j))
```

The arithmetic branch keeps the best CLIP-only intuition:

```text
e_query = normalize(sum_j c_j)
q_sum   = normalize(z_s + e_query)
```

The final query is not interpreted as an absolute coordinate target. It is a normalized retrieval direction:

```text
q_final = normalize(q_model + beta * (q_sum - z_s))
```

### Training Objective

Training uses synthetic same-identity pairs from CelebA train split:

```text
input  = source image A + changed attributes C
target = image B of the same identity where those attributes differ as requested
```

The main contrastive loss pushes `q_i` near its target embedding and away from other batch targets:

```text
L_exact = -log exp(q_i dot z_ti / tau) / sum_b exp(q_i dot z_tb / tau)
```

To avoid punishing visually valid alternatives, false negatives are masked using the assignment-like attribute compatibility rule. In later runs we also tested a multi-positive objective:

```text
P_i = compatible targets in the batch
L_multi = -log sum_{p in P_i} exp(q_i dot z_p / tau) / sum_b exp(q_i dot z_b / tau)
```

The full training loss used by the cluster code is:

```text
L = (1 - lambda_mp) * L_exact
    + lambda_mp * L_multi
    + lambda_target * (1 - cos(q_i, z_ti))
    + lambda_source * (1 - cos(q_i, z_si))
    + lambda_probe * L_probe
```

The best final checkpoint used a sequential gate trained with multi-positive/official-like validation, then the final `model + generic delta` composition was selected on the official JSON evaluation.
"""


COSINE_MD = r"""
## Cosine Geometry: Why the Delta Correction Can Help

![Toy CLIP/cosine view](../final_best_system/explanations/toy_vector_correction_clip_cosine.png)

Stessa direzione. Per cosine similarity sono praticamente uguali.

Il punto chiave: CLIP retrieval non chiede "quanto sono vicino come coordinate assolute?", ma:

```text
qual è l'immagine con embedding che ha angolo/cosine più alto rispetto a q_final?
```

This is why adding the arithmetic delta can be useful even if it looks odd in Euclidean coordinates. After normalization, retrieval ranks gallery images by angle:

```text
score_i = cosine(q_final, image_embedding_i)
topK    = argsort(score_i, descending=True)[:K]
```
"""


LOAD_CODE = r"""
## Loading the Final Checkpoint From the Repo
"""


LOAD_CHECKPOINT_CODE = r"""
# This cell uses repo-relative paths, so it works after cloning/pulling the repo.
import os
import sys
from pathlib import Path

def find_project_root():
    cwd = Path.cwd().resolve()
    candidates = [cwd, *cwd.parents]
    for candidate in candidates:
        if (candidate / "final_best_system").is_dir() and (candidate / "cluster").is_dir():
            return candidate
    raise RuntimeError("Could not find repo root containing final_best_system/ and cluster/.")

PROJECT_ROOT = find_project_root()
CLUSTER_ROOT = PROJECT_ROOT / "cluster"
FINAL_DIR = PROJECT_ROOT / "final_best_system"

os.environ["DL_PROJECT_ROOT"] = str(CLUSTER_ROOT)
sys.path.insert(0, str(CLUSTER_ROOT / "scripts"))
sys.path.insert(0, str(CLUSTER_ROOT / "orchestrator"))

from learned_gate_core import (
    condition_embeddings as learned_condition_embeddings,
    load_model_checkpoint,
    load_prompt_embedding_cache,
)
from project_core import choose_device, load_torch, parse_query, read_attribute_table

FINAL_CHECKPOINT_PATH = FINAL_DIR / "weights" / "best_val_official_like_at10.pt"
assert FINAL_CHECKPOINT_PATH.exists(), f"Missing checkpoint: {FINAL_CHECKPOINT_PATH}"

DEVICE = choose_device("auto")
final_model, final_checkpoint = load_model_checkpoint(FINAL_CHECKPOINT_PATH, DEVICE)
final_model.eval()
final_config = final_checkpoint["config"]

prompt_cache_path = CLUSTER_ROOT / final_config["prompt_cache_path"]
if not prompt_cache_path.exists():
    prompt_cache_path = FINAL_DIR / "embeddings" / Path(final_config["prompt_cache_path"]).name
assert prompt_cache_path.exists(), f"Missing prompt cache: {prompt_cache_path}"

final_prompt_cache = load_prompt_embedding_cache(prompt_cache_path)
text_bank = load_torch(FINAL_DIR / "embeddings" / "attribute_text_embeddings.pt")
attributes, _, _ = read_attribute_table()
attribute_to_index = {name: i for i, name in enumerate(attributes)}

print("Loaded final checkpoint:", FINAL_CHECKPOINT_PATH)
print("Checkpoint config_id:", final_config.get("config_id"))
print("Prompt cache:", prompt_cache_path)
print("Device:", DEVICE)
"""


FINAL_FUNCTION_MD = r"""
## Final System Function

The function below implements the winning system:

```text
q_final = normalize(q_model + beta * (q_sum - source))
```

where `q_model` comes from the trained sequential gate and `q_sum` is generic contrastive CLIP arithmetic.
"""


FINAL_FUNCTION_CODE = r"""
def _condition_tensors_for_query(conditions, batch_size, device):
    max_len = max(1, len(conditions))
    attrs = torch.full((batch_size, max_len), -1, dtype=torch.long, device=device)
    signs = torch.zeros((batch_size, max_len), dtype=torch.int8, device=device)
    for pos, (sign, attr) in enumerate(conditions):
        attrs[:, pos] = attribute_to_index[attr]
        signs[:, pos] = int(sign)
    return attrs, signs


def learned_gate_query(source_embeddings, conditions):
    source_embeddings = F.normalize(source_embeddings.float().to(DEVICE), dim=-1)
    attrs, signs = _condition_tensors_for_query(conditions, len(source_embeddings), DEVICE)
    cond, mask = learned_condition_embeddings(
        final_prompt_cache,
        attrs,
        signs,
        DEVICE,
        str(final_config.get("condition_mode", "signed_direction")),
    )
    with torch.inference_mode():
        q_model, alpha, _ = final_model(source_embeddings, cond, mask)
    return F.normalize(q_model, dim=-1), alpha


def generic_sum_query(source_embeddings, conditions):
    source_embeddings = F.normalize(source_embeddings.float().to(DEVICE), dim=-1)
    directions = F.normalize(text_bank["directions"].float().to(DEVICE), dim=-1)
    edit = torch.zeros_like(source_embeddings)
    for sign, attr in conditions:
        edit = edit + int(sign) * directions[attribute_to_index[attr]].unsqueeze(0)
    edit = F.normalize(edit, dim=-1)
    return F.normalize(source_embeddings + edit, dim=-1)


def final_model_plus_delta_query(source_embeddings, query_text, beta=1.0):
    conditions = parse_query(query_text)
    source_embeddings = F.normalize(source_embeddings.float().to(DEVICE), dim=-1)
    q_model, alpha = learned_gate_query(source_embeddings, conditions)
    q_sum = generic_sum_query(source_embeddings, conditions)
    q_final = F.normalize(q_model + beta * (q_sum - source_embeddings), dim=-1)
    return q_final, {"conditions": conditions, "alpha": alpha.detach().cpu(), "q_model": q_model.detach().cpu(), "q_sum": q_sum.detach().cpu()}


# Minimal smoke test with a random normalized vector. The real evaluation cells
# use CLIP image embeddings from the dataset.
_dummy_source = F.normalize(torch.randn(1, 512), dim=-1)
_dummy_q, _dummy_info = final_model_plus_delta_query(_dummy_source, "+Smiling, +Eyeglasses")
print("Final query shape:", tuple(_dummy_q.shape))
print("Parsed conditions:", _dummy_info["conditions"])
print("Learned gate weights:", _dummy_info["alpha"].numpy().round(3).tolist())
"""


RESULTS_MD = r"""
## Final Official Results Included in the Repo

The official JSON evaluation is already saved in `final_best_system/results/clean_report/`. The table below compares:

1. assignment vanilla baseline: `direct_sum`;
2. strongest zero-shot CLIP baseline: `contrastive_sequential`;
3. final proposed system: `model_plus_generic_delta_100`.

The assignment metrics are `Recall@1/5/10` and `Precision@1/5/10`.
"""


RESULTS_CODE = r"""
from IPython.display import Image, display

macro_metrics = pd.read_csv(FINAL_DIR / "results" / "clean_report" / "overall_metrics_macro.csv")
micro_metrics = pd.read_csv(FINAL_DIR / "results" / "clean_report" / "overall_metrics_micro.csv")
per_query_r10 = pd.read_csv(FINAL_DIR / "results" / "clean_report" / "per_query_recall10_three_systems.csv")

print("Macro metrics: each query counts equally.")
display(macro_metrics)

print("Micro metrics: every source-query case counts equally.")
display(micro_metrics)

display(Image(filename=str(FINAL_DIR / "results" / "clean_report" / "overall_metrics_three_systems.png")))
display(Image(filename=str(FINAL_DIR / "results" / "clean_report" / "per_query_recall10_three_systems.png")))

per_query_r10
"""


DISCUSSION_MD = r"""
## Discussion: Findings and Pivots

The main finding is that CLIP arithmetic is not just a toy baseline. The assignment-style direct sum is a useful lower bound, but contrastive directions and sequential normalization make zero-shot CLIP much stronger.

Development pivots:

1. **Direct arithmetic baseline.** Required by the assignment and used as the vanilla reference.
2. **Contrastive sequential baseline.** Improved by using `t_positive - t_negative` and normalizing after each edit.
3. **`gate_v1` residual-only model.** Learned something, but had to rediscover CLIP edit directions from scratch.
4. **`gate_v2` additive gate.** Used contrastive directions with learned weights.
5. **`gate_v3` sequential gate.** Learned a source-conditioned version of the strongest sequential arithmetic geometry.
6. **Final hybrid system.** The best report system combines the learned gate with a generic CLIP arithmetic delta:

```text
q_final = normalize(q_model + 1.0 * (q_sum - source))
```

Current official JSON results:

```text
Assignment direct_sum Macro Recall@10:          0.1084
Strong contrastive_sequential Macro Recall@10:  0.1871
Final model+delta Macro Recall@10:              0.2827

Assignment direct_sum Micro Recall@10:          0.1248
Strong contrastive_sequential Micro Recall@10:  0.1665
Final model+delta Micro Recall@10:              0.2386
```

The final system improves strongly on local visual edits such as eyeglasses, smile, makeup, and mustache. The hardest remaining attributes are global/correlated ones such as `Male`, `Young`, and `Chubby`. These are difficult because CLIP directions encode broad demographic/semantic shifts, while the official target sets often require subtle attribute changes without destroying identity and other non-query attributes.
"""


def update_notebook(path: Path) -> None:
    nb = json.loads(path.read_text(encoding="utf-8"))
    cells = []
    for cell in nb["cells"]:
        src = "".join(cell.get("source", []))
        if any(src.startswith(marker) for marker in MARKERS):
            continue
        cells.append(cell)

    # Replace old final-method discussion cells.
    for cell in cells:
        src = "".join(cell.get("source", []))
        if src.startswith("## Final Method: Learned Sequential Gate"):
            cell["source"] = FINAL_METHOD_MD.strip("\n").splitlines(keepends=True)
        if src.startswith("## Discussion: Findings and Pivots"):
            cell["source"] = DISCUSSION_MD.strip("\n").splitlines(keepends=True)

    insert_at = None
    for i, cell in enumerate(cells):
        src = "".join(cell.get("source", []))
        if src.startswith("## Results Tables and Plots"):
            insert_at = i
            break
    if insert_at is None:
        insert_at = len(cells) - 1

    new_cells = [
        md(COSINE_MD),
        md(LOAD_CODE),
        code(LOAD_CHECKPOINT_CODE),
        md(FINAL_FUNCTION_MD),
        code(FINAL_FUNCTION_CODE),
        md(RESULTS_MD),
        code(RESULTS_CODE),
    ]
    cells[insert_at:insert_at] = new_cells
    nb["cells"] = cells
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    update_notebook(NOTEBOOK)
    CLUSTER_NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(NOTEBOOK, CLUSTER_NOTEBOOK)
    print(f"Updated {NOTEBOOK}")
    print(f"Mirrored {CLUSTER_NOTEBOOK}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
