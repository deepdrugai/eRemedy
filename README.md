# eRemedy

Interpretable rescue of antagonistic drug combinations.

eRemedy trains a multi-modal synergy predictor on the O'Neil combination
screen, attributes each prediction to individual molecular fragments, and turns
that attribution into two concrete interventions:

- **Strategy 1 — fragment replacement.** Swap the fragment held responsible for
  lost synergy, rank replacements with a per-cell-line symbolic regressor,
  rebuild the molecule, and rescore the pair.
- **Strategy 2 — existing-drug rescue.** Replace the whole drug carrying the
  fragment with a compound that already exists, matched either by fragment
  containment or by shared ChEMBL target / ATC class.

This repository contains the source for the complete pipeline and for every
experiment reported in the accompanying manuscript. It does **not** contain
generated artefacts: embeddings, trained checkpoints, per-cell-line outputs,
aggregate tables and figures are all reproduced by the commands below.

---

## 1. Installation

```bash
conda create -n eremedy python=3.8
conda activate eremedy
pip install -r requirements.txt
```

Install `torch` matched to your CUDA version first if the pinned wheel does not
suit your driver. A CUDA-capable GPU is required for teacher training,
attribution, and rescoring; symbolic regression runs on CPU.

Strategy 1 additionally needs the **eSynth2** reconstruction container, invoked
as a Docker image named `frag-replace` (see `run_pipeline_all_celllines.sh`).
Everything except the reconstruction step runs without it.

## 2. Data

All inputs are public and must be downloaded separately:

| Source | Used for |
|---|---|
| O'Neil combination screen | training and evaluation |
| NCI-ALMANAC | external validation, vocabulary-scaling study |
| DrugCombDB | candidate compound library |
| DrugComb v1.5 | name-to-SMILES resolution |
| CCLE | cell-line multi-omics features |
| ChEMBL | target and ATC annotations |

The pipeline expects each screen as a pickle exposing `meta` and `cell_omics`,
with paths supplied via `--input` / `--dataset`.

Two curated inputs **are** included, because rebuilding them requires many
thousands of external API calls:

- `everything_drug_smiles_map.csv` — drug name to SMILES, with the source and
  match type recorded per compound. Regenerate with `build_drug_smiles_map.py`.
- `drug_moa_cache_full.csv`, `drug_moa_cache.csv` — ChEMBL targets, ATC classes
  and mechanism annotations. Regenerate with `build_moa_cache.py`.
- `everything_drug_smiles_ambiguous.csv`, `everything_drug_smiles_unresolved.csv`
  — resolution audit for compounds that matched ambiguously or not at all.

## 3. Running the pipeline

### Stage 0 — featurization (once)

```bash
python map_smiles_to_fragments.py \
  --smiles-dir dataset_EVERYTHING_SMILES --fragments-dir dataset-smi-frag \
  --output RUN/smiles_to_fragments.pkl

python drug_graph_autoencoder.py \
  --smiles-dir dataset_EVERYTHING_SMILES \
  --output RUN/molecule_smiles_embeddings.pkl \
  --checkpoint-output RUN/drug_smiles_encoder.pt

python fragment_graph_autoencoder.py \
  --input RUN/smiles_to_fragments.pkl \
  --output RUN/fragment_smiles_embeddings.pkl \
  --checkpoint-output RUN/fragment_smiles_encoder.pt
```

Fragment decomposition itself uses **eMolFrag2**, run externally; these scripts
consume its output.

### Stage 1 — teacher (once)

```bash
python synergy_prediction.py \
  --input dataset_ONEIL.pkl \
  --molecule-embeddings RUN/molecule_smiles_embeddings.pkl \
  --fragment-embeddings RUN/fragment_smiles_embeddings.pkl \
  --fragment-map RUN/smiles_to_fragments.pkl \
  --drug-smiles-map everything_drug_smiles_map.csv \
  --outdir RUN/synergy
```

Writes `model.pt` plus metrics, predictions and training curves.

### Stages 2–6 — per cell line

`run_pipeline_all_celllines.sh` chains fragment attribution, symbolic
regression, SDF annotation, eSynth2 reconstruction, rescoring, and
fragment-containment repurposing for every cell line:

```bash
./run_pipeline_all_celllines.sh            # all 34 cell lines
./run_pipeline_all_celllines.sh MCF7 A375  # a subset
```

Edit the paths at the top of the script to point at your `RUN` directory.

### Strategy 2

```bash
# mechanism-matched, driven by SHAP attribution (requires stage 2 output)
./run_flow_b_shap_moa_all.sh drug_moa_cache_full.csv clean_runs/flow_b_shap_moa 50
python finalize_flow_b.py

# teacher-scored drug swap, independent of attribution
./run_flow_b_all.sh clean_runs/flow_b vocab
```

## 4. Validation

```bash
./run_lodo_sharded.sh                      # leave-one-drug-out, sharded over GPUs
python merge_lodo.py && python diagnose_lodo.py

python validate_external_almanac.py --dataset dataset_ALMANAC.pkl
python analyze_almanac_generalization.py   # strata, per-drug PCC, similarity gate

python validate_fragments_vs_measurement.py  # attribution against measurement
python validate_flow_b_ranking.py            # within-context ranking
python validate_flow_b_ranking_lodo.py       # the same, drugs held out
python calibrate_chemotype_gate.py           # can similarity predict competence?
```

## 5. Reported experiments

`experiments/` holds the controls and ablations behind the manuscript's
validation section. Each script is self-contained and writes to
`experiments/results/`.

| Script | Question |
|---|---|
| `e0_composite_audit.py` | What does the harmfulness ranking actually sort on? |
| `e1a_pool_composition.py` | Are measurement-backed selections enriched for training rows? |
| `e2_subset_bias.py` | Is the measured subset an easier one? |
| `e3_cell_mean_baseline.py` | Does a structure-free lookup table match the model? |
| `e4_divergence_noise.py` | Is cross-cell-line sign divergence real or sampling noise? |
| `e5_grouped_split_retrain.py` | Accuracy under leakage-free grouped splits |
| `e6_gbm_baseline.py` | Gradient boosting on identical embeddings |
| `e7_statistics.py` | Bootstrap intervals and FDR correction for headline numbers |
| `e8_onehot_controls.py` | Does a one-hot drug code beat the learned representation? |
| `e9_deepsynergy_baseline.py` | DeepSynergy on matched rows and splits |
| `e11_build_ecfp_embeddings.py` | ECFP6 featurization as a drop-in replacement |
| `e12_end2end_teacher.py` | Training the graph encoders on the synergy objective |
| `e13_vocab_scaling.py` | Does the representation gap close as the vocabulary grows? |

Batch drivers (`experiments/run_*.sh`) fan these across available GPUs, for
example:

```bash
bash experiments/run_finalize_jobs.sh 8     # one-hot controls + DeepSynergy
bash experiments/run_e12_end2end.sh 8       # end-to-end encoder, 2 splits x 4 seeds
bash experiments/run_e13_scaling_perdrug.sh 8   # vocabulary scaling on ALMANAC
```

## 6. Figures

```bash
python thesis/gen_fig_loewe_distribution.py       # Loewe score distribution
python thesis/gen_figures_pancancer.py            # per-cell-line outcomes, rescue vs amplify
python thesis/gen_figures_shap_divergence.py      # channel-level SHAP, cross-line divergence
python experiments/gen_fig_generalisation_ladder.py
python experiments/e13_plot_scaling.py --prefix perdrug
python analyze_almanac_generalization.py          # external validation panels
```

## 7. Layout

```
.
├── build_drug_smiles_map.py          name -> SMILES resolution
├── map_smiles_to_fragments.py        parent -> fragment mapping
├── drug_graph_autoencoder.py         molecule embeddings
├── fragment_graph_autoencoder.py     fragment embeddings
├── build_moa_cache.py                ChEMBL target / ATC retrieval
├── synergy_prediction.py             teacher model
├── synergy_shap_analysis.py          fragment attribution
├── symbolic_regression_bioisostere_updated.py   Strategy 1 ranking
├── rescore_reconstructed_candidates.py          Strategy 1 rescoring
├── flow_b_shap_moa.py                Strategy 2, mechanism-matched
├── drug_repurposing.py               Strategy 2, fragment containment
├── finalize_flow_b.py                recommendation finalisation
├── validate_*.py, analyze_*.py       validation and competence boundary
├── run_*.sh                          orchestrators
├── experiments/                      reported controls and ablations
└── thesis/                           figure generators
```

## 8. Citation

Please cite the accompanying manuscript. Details will be added here on
publication.
