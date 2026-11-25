# NewRAG · GeoEvidence-RAG

NewRAG extends [MechRAG](https://github.com/SydneyGit/MechRAG) with local geometric evidence, geometry-aware candidate reranking, budgeted evidence selection, and answer verification. The original workflows are available in the notebooks described below. The upstream paper is [MechRAG: a multimodal large language model for mechanical engineering](https://doi.org/10.1038/s44172-025-00517-z).

The [GeoEvidence-RAG manuscript draft](paper/elsevier_evidence_rag/main.tex) and [experiment summaries](paper/results/README.md) document the current implementation and evaluation. With the Elsevier `elsarticle` package installed, build the draft with `latexmk -pdf main.tex` from `paper/elsevier_evidence_rag`.

## Upstream MechRAG notebooks
MechRAG is a Large Language Model (LLM)-based augmented tool for mechanical engineering. It features capabilities of analyzing mechanical engineering data from local storage with the general knowledge obtained from the pre-training of LLMs. 


The notebooks are used for:
  1) Text tasks: MechRAG_textTask_Claude.ipynb (MechRAG_textTask_Claude_git.ipynb is the same code but processed for github rendering if MechRAG_textTask_Claude.ipynb does not work properly)
  2) Multi-modal tasks: MechRAG_MutltimodalTask_GPT.ipynb (MechRAG_MutltimodalTask_GPT_git.ipynb is the same code but processed for github rendering if MechRAG_MutltimodalTask_GPT.ipynb does not work properly)


 
To use the notebooks:
1. download notebooks and datasets. The notebooks are the easiest to run on Google Colab
2. fill in the gaps in the code for:
    1) API keys
    2) address of the datasets/current working directory/...



To quickly run the notebooks, the smaller versions of the datasets are here:
  1) vectorstore_1%perClass.npz, vectorstore_2%perClass.npz, are for MechRAG_MutltimodalTask_GPT.ipynb
  2)  vectorstore_0.5%.npz, vectorstore_1%.npz, are for MechRAG_textTask_Claude.ipynb


## Evidence-first revision

The original notebook functions are retained for reproducing the paper.  New
work should use the evidence-first cells at the end of
`MechRAG_textTask_Claude_git.ipynb` and
`MechRAG_MutltimodalTask_GPT_git.ipynb`.

The revision addresses three failure modes:

1. **Unsupported answers**: retrieval and generation now use citable local
   `EvidenceUnit` records.  The generation contract returns claim-to-evidence
   links, and `AnswerEvidenceVerifier` rejects unknown evidence, unsupported
   claims, and uncovered query conditions.
2. **Whole-image similarity hiding critical CAD differences**: whole-image
   CLIP remains only the candidate stage.  A second stage compares local
   regions and critical properties (for example hole count, blind/through-hole
   type, coaxiality and symmetry).  A small MLP can be trained on hard
   negatives while all CLIP embeddings stay frozen.
3. **Global questions depending on Top-K**: max/min/sort queries use the
   structured evidence store and scan the registered part universe.  A global
   result is marked complete only when every expected part has that property.

### Evidence record

Each retrievable record implements:

```text
X = {
  part_id,
  feature_id,
  CAD face IDs,
  local regions from multiple views,
  geometric properties,
  source provenance
}
```

Per-view annotations with the same `(part_id, feature_id)` are merged by
`deduplicate_cross_view`.  Conflicting properties are errors rather than being
silently counted as separate evidence.

### Retrieval flow

```text
question -> verifiable conditions -------------------------------+
                                                                  |
whole-image CLIP -> candidate parts -> local-evidence MLP rerank  |
                                                                  v
global structured scan (when required) -> budgeted coverage selection
                                             |
                                             v
                               evidence-cited JSON answer -> verifier
```

The selector greedily maximizes newly covered condition weight per evidence
cost and subtracts a redundancy penalty.  The prompt exposes the resulting
coverage score and prevents incomplete global scans from supporting a global
claim.

### Build a local evidence index

The annotation input is JSONL using the schema returned by
`EvidenceUnit.to_dict()`.  One CAD feature may appear on several input lines,
one per view.  See `examples/evidence_annotations.example.jsonl` for a
multi-view feature and part-level property record:

```bash
python scripts/build_evidence_index.py annotations.jsonl evidence.jsonl
```

Required multimodal annotations include normalized bounding boxes, stable CAD
face IDs or feature IDs, geometric properties, and a `family_id`.  Local CLIP
embeddings should be extracted from each crop with the same frozen encoder used
for candidate retrieval.

### Train the hard-negative reranker

`profiles.jsonl` contains one `PartEvidenceProfile.to_dict()` record per part,
including the frozen whole-image embedding and all local evidence units.

```bash
pip install -e ".[reranker]"
python scripts/train_reranker.py profiles.jsonl reranker.pt \
  --critical-property hole_count \
  --critical-property hole_type \
  --critical-property coaxial \
  --critical-property symmetry
```

The miner selects high-CLIP-similarity pairs whose critical properties differ.
Use `make_single_feature_variant` when importing controlled variants from an
editable CAD model.  `split_pairs_by_family` keeps all views and variants of a
family in the same split and drops cross-split pairs.

### Notebook APIs

- Text: call `generate_description_with_evidence` from the final cell of
  `MechRAG_textTask_Claude_git.ipynb`.  It scans all metadata PDFs for global
  operators, builds an evidence prompt, calls DeepSeek, and verifies the
  returned claim graph.
- Multimodal: build `evidence.jsonl` and `profiles.jsonl`, train `reranker.pt`,
  then call `configure_evidence_runtime` and
  `generate_description_with_evidence` in the final notebook cell.  Selected
  local crops, bounding boxes, CAD faces and provenance are sent to DeepSeek.

### DeepSeek configuration

The revised GitHub-renderable notebooks use one runtime-only environment
variable for both text and multimodal generation:

```powershell
$env:DEEPSEEK_API_KEY="your-key"
& '.\.venv\Scripts\python.exe' -m jupyter lab
```

Do not paste the key into a notebook or commit it to Git.  Text generation uses
DeepSeek's Anthropic-compatible endpoint, while multimodal generation uses its
OpenAI-compatible endpoint with `deepseek-flash`.  Local CLIP retrieval,
structured aggregation, evidence selection, and reranker training do not need
an API key.

The original precomputed `.npz` files contain whole-item embeddings and do not
contain the local CAD annotations needed by the new reranker.  They can still
be used for the first-stage candidate search, but the evidence JSONL and
profiles must be constructed before evidence-first multimodal generation.

### Tests

```bash
python -m unittest discover -s tests -v
```

The regression suite covers cross-view deduplication, property conflicts,
condition coverage, answer verification, family leakage prevention, local
feature mismatch, and exact global aggregation outside Top-K.

### Test a folder of engineering drawing PDFs

Scanned/vector drawing PDFs usually have an absent or incorrectly encoded text
layer.  The drawing evaluator renders each page, removes oversized blank canvas,
creates overlapping detail crops, asks the four standard drawing questions, and
stores every answer as a page/region-citable `EvidenceUnit`.

Preparation does not use an API key:

```powershell
& '.\.venv\Scripts\python.exe' scripts\evaluate_drawing_pdfs.py `
  '.\自建数据集' '.\data\self_dataset_eval' --prepare-only
```

After setting a newly issued DeepSeek key in the same terminal, omit
`--prepare-only` to run the vision evaluation:

```powershell
$env:DEEPSEEK_API_KEY='your-rotated-key'
& '.\.venv\Scripts\python.exe' scripts\evaluate_drawing_pdfs.py `
  '.\自建数据集' '.\data\self_dataset_eval'
```

Alternatively, use the secure interactive runner.  It hides the key while it
is entered, keeps it only for the child process, and removes it immediately
after the evaluation:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File `
  '.\scripts\run_self_dataset_eval.ps1' `
  -DatasetPath '.\自建数据集'
```

Outputs are `manifest.json`, `results.jsonl`, `evidence.jsonl`, the raw provider
responses, and `report.md`.  A `valid=true` result verifies condition coverage
and citation integrity; measuring answer accuracy still requires human ground
truth.

### Run the eight same-protocol visual baselines

The comparative experiment crosses two image inputs (global overview or five
overlapping views) with four deterministic inference strategies (direct,
rationale, decomposition, and visible-evidence verification).  All eight
systems use the image-capable `deepseek-flash` endpoint; they are controlled
system configurations, not eight independent foundation-model families.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File `
  '.\scripts\run_comparative_baselines.ps1' `
  -BenchmarkRoot '..\MechVQA' `
  -Dataset both -Workers 12 -JudgeWorkers 16
```

The runner securely prompts for the credential, resumes completed target
responses, retries target errors, evaluates all 1,185 MechVQA questions, and
scores the 26 human-annotated objective fields in the self-built dataset.
Final outputs are written under `data/comparative_baselines/`, including
`report.md`, `overall_results.csv`, `summary.json`, and the per-question JSONL.
