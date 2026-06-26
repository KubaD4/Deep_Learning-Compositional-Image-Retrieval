# Final System Architecture Diagram

This diagram summarizes the current final-best system:

- frozen CLIP image/text encoders are used to build image embeddings and signed attribute directions;
- a learned sequential gate predicts how strongly to apply each requested edit;
- a generic CLIP arithmetic displacement is used as a correction term;
- final retrieval ranks gallery images by cosine similarity against the final query vector;
- evaluation follows the official JSON with Precision@K and Recall@K.

```mermaid
flowchart LR
  classDef data fill:#f7f3e8,stroke:#b8902f,stroke-width:1.5px,color:#1f1a10
  classDef clip fill:#eaf3ff,stroke:#2f6fb0,stroke-width:1.5px,color:#102033
  classDef train fill:#eef8ee,stroke:#3f8b45,stroke-width:1.5px,color:#102214
  classDef model fill:#f2edff,stroke:#7a56c2,stroke-width:1.5px,color:#1d1233
  classDef eval fill:#fff0f0,stroke:#c45656,stroke-width:1.5px,color:#331111
  classDef output fill:#f2f2f2,stroke:#777,stroke-width:1.5px,color:#111

  subgraph D["Dataset and Official Evaluation"]
    IMG["CelebA images"]:::data
    ATTR["CelebA attributes and identity files"]:::data
    JSON["Official evaluation JSON<br/>source image + requested edits + valid targets"]:::data
  end

  subgraph C["Frozen CLIP Feature Space"]
    CLIPI["CLIP image encoder<br/>frozen"]:::clip
    CLIPT["CLIP text encoder<br/>frozen"]:::clip
    VIMG["Normalized image embeddings<br/>v_i"]:::clip
    TPROMPT["Signed attribute prompt embeddings<br/>t(+A), t(-A)"]:::clip
    DIR["Attribute edit directions<br/>d_A = normalize(t(+A) - t(-A))"]:::clip
  end

  IMG --> CLIPI --> VIMG
  ATTR --> CLIPT
  CLIPT --> TPROMPT --> DIR

  subgraph P["Training Pair Construction"]
    SAME["Same-identity pairs<br/>A -> B where identity is shared<br/>query = changed attributes"]:::train
    WEAK["Weak/global official-like pairs<br/>focused on Male, Young, Chubby"]:::train
    OFFICIAL["v6 official-like multi-positive sets<br/>train split only, JSON excluded<br/>query attrs satisfied and other-attr Hamming <= 2"]:::train
  end

  ATTR --> SAME
  ATTR --> WEAK
  ATTR --> OFFICIAL
  VIMG --> SAME
  VIMG --> WEAK
  VIMG --> OFFICIAL

  subgraph M["Learned Gate Model"]
    SRC["Source image embedding<br/>s"]:::model
    QUERY["Query edits<br/>+A, -B, ..."]:::model
    QDIRS["Ordered signed directions<br/>d_1, d_2, ..., d_n"]:::model
    GATE["Sequential gate MLP<br/>predicts edit weights alpha_j"]:::model
    QMODEL["Learned edited query<br/>q_model = sequential_norm(s + alpha_j d_j)"]:::model
  end

  VIMG --> SRC
  DIR --> QDIRS
  QUERY --> QDIRS
  SRC --> GATE
  QDIRS --> GATE
  GATE --> QMODEL

  subgraph L["Training Objective"]
    POS["Positive targets<br/>single or multi-positive target set"]:::train
    NEG["In-batch negatives<br/>false negatives masked by attribute Hamming"]:::train
    LOSS["Loss = InfoNCE / multi-positive contrastive<br/>+ target cosine + source preservation<br/>+ optional triplet regularization"]:::train
  end

  SAME --> POS
  WEAK --> POS
  OFFICIAL --> POS
  QMODEL --> LOSS
  POS --> LOSS
  NEG --> LOSS

  subgraph F["Final Inference System"]
    QSUM["Generic CLIP arithmetic<br/>q_sum = normalize(s + sum signed d_j)"]:::eval
    DELTA["Delta corrector<br/>delta = q_sum - s"]:::eval
    QFINAL["Final retrieval vector<br/>q_final = normalize(q_model + beta * delta)<br/>best beta = 1.5"]:::eval
  end

  SRC --> QSUM
  QDIRS --> QSUM
  QSUM --> DELTA
  QMODEL --> QFINAL
  DELTA --> QFINAL

  subgraph R["Retrieval and Metrics"]
    TOPK["Rank gallery images by cosine similarity<br/>top-k = highest cos(q_final, v_i)"]:::output
    METRICS["Official metrics<br/>Precision@1/5/10 and Recall@1/5/10<br/>macro and micro averages"]:::output
    REPORT["Clean report outputs<br/>CSV tables, PNG plots, retrieval JSONL"]:::output
  end

  VIMG --> TOPK
  QFINAL --> TOPK
  TOPK --> METRICS
  JSON --> METRICS
  METRICS --> REPORT

  subgraph B["Baselines Used for Comparison"]
    BASE1["Assignment vanilla baseline<br/>direct_sum"]:::eval
    BASE2["Strong zero-shot baseline<br/>contrastive_sequential"]:::eval
  end

  SRC --> BASE1
  SRC --> BASE2
  QDIRS --> BASE1
  QDIRS --> BASE2
  BASE1 --> METRICS
  BASE2 --> METRICS
```
