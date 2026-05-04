# ProteinFP

**End-to-end protein function prediction and drug candidate design.**

Give it a UniProt ID. Get back active sites, druggable pockets, allosteric sites, EC classification, GO terms, PPI partners, therapy modality recommendations, and, with AutoDock Vina, evolved drug candidate molecules. For any protein, any disease, any organism.

```bash
pip install proteinfp
proteinfp --uniprot P28593   # Trypanothione reductase (Chagas disease)
```

```
  Protein    : Trypanothione reductase
  Gene       : TPR
  Organism   : Trypanosoma cruzi
  Confidence : VERY HIGH

  Top function     : Trypanothione is the parasite analog of glutathione
  Enzyme           : yes — EC 1.8.1.12
  Pockets          : 10 (all druggability > 0.90)
  Therapy          : SMALL_MOLECULE → active site inhibitor
```

---

## What it does

ProteinFP runs 13+ prediction modules in sequence, fusing their outputs into a single ranked, confidence-weighted report.

| Module | What it predicts |
|--------|-----------------|
| 01 | AlphaFold structure + UniProt metadata |
| 02 | Surface charge, hydrophobicity, SASA |
| 03 | Catalytic residues and active site motifs |
| 04 | Druggable binding pockets (geometry + druggability score) |
| 05 | Allosteric sites (elastic network model) |
| 06 | Chemical environment of each site |
| 07 | Sequence homologs with known function (BLAST + InterPro) |
| 08 | ESM-2 protein language model embeddings (650M parameters) |
| 09 | GO term prediction (Molecular Function, Biological Process, Cellular Component) |
| 10 | Enzyme class prediction — ML ensemble (XGBoost + LightGBM + MLP, ~97% accuracy) |
| 11 | Structural analogs via Foldseek (finds same-fold proteins regardless of sequence) |
| 12 | Protein-protein interactions (STRING DB) |
| 13 | Consensus report — fuses all evidence into a ranked, confidence-scored output |
| 14 | Molecular dynamics — RMSF, flexibility, cryptic pockets *(needs OpenMM)* |
| 15 | De novo molecular design — evolutionary drug candidate generation *(needs Vina + RDKit)* |
| 17 | Post-translational modification sites and their functional consequences |

**GRN + SIM pipeline** (disease-aware mode — requires scRNA-seq data):

| Module | What it does |
|--------|-------------|
| GRN-01 | scRNA-seq preprocessing — HVG selection, QC filtering |
| GRN-02 | GENIE3 gene regulatory network reconstruction |
| GRN-03 | Therapy modality decision — surface vs intracellular, ADC vs small molecule |
| SIM-01 | Tumor cell environment inference from marker gene expression |
| SIM-02 | Protein conformational ensemble in that environment |
| SIM-03 | Drug distribution across cell compartments |
| SIM-04 | Binding probability under real physiological conditions |
| SIM-05 | GRN perturbation — network-level consequences of drug binding |
| SIM-06 | Pharmacological scoring — efficacy, selectivity, resistance risk, grade A–F |

---

## Installation

```bash
pip install proteinfp
```

**Core pipeline** (Modules 01–13, 17) works out of the box. Optional features:

```bash
pip install proteinfp[ml]        # ESM-2 embeddings + ML EC classifier
pip install proteinfp[structure] # SASA/DSSP surface analysis
pip install proteinfp[chem]      # De novo molecular design (RDKit)
pip install proteinfp[grn]       # GRN/scRNA-seq modules (scanpy)
pip install proteinfp[sim]       # Molecular dynamics (OpenMM)
pip install proteinfp[all]       # Everything
```

For de novo design you also need [AutoDock Vina](https://vina.scripps.edu/downloads/).

Check what's available on your machine:

```bash
proteinfp --check-deps
```

---

## Quick start

```bash
# Any protein, just a UniProt ID
proteinfp --uniprot P04637       # TP53 (human tumour suppressor)
proteinfp --uniprot P28593       # Trypanothione reductase (Chagas disease)
proteinfp --uniprot P9WGR1       # InhA (drug-resistant TB)

# Force re-run even if report already exists
proteinfp --uniprot P04637 --force

# With therapy decision + de novo molecule design
proteinfp --uniprot P28593 --therapy --denovo --vina /path/to/vina

# With molecular dynamics
proteinfp --uniprot P28593 --md

# Show all modules and their status
proteinfp --list-modules
```

Reports are saved to `data/reports/{UNIPROT}_report.json` and `_report.txt`.

---

## Therapy mode

After the core pipeline runs, `--therapy` makes modality decisions automatically:

- **Surface protein** → antibody path: ranks epitope candidates by immunogenicity and accessibility
- **Intracellular with druggable pocket** → small molecule path: triggers de novo design
- **Epigenetic regulator** → adds PROTAC degrader as secondary recommendation
- **Allosteric site only** → allosteric small molecule

```bash
proteinfp --uniprot P28593 --therapy --denovo --vina pipeline/vina.exe
```

```
  → Primary modality : SMALL_MOLECULE
  → Confidence       : HIGH
  • Intracellular with druggable pocket P1 (vol=1800Å³, drug=0.90)
  • Enzyme (EC 1.8.1.12) — active site inhibition most direct mechanism
```

---

## Disease-agnostic design

The pipeline works on any protein from any organism. To switch disease context, edit one file:

```yaml
# config/disease_config.yaml
disease:
  name: "TB"
  organism: "Mycobacterium tuberculosis"
  organism_id: 83332

data:
  scrnaseq_input: "data/grn/input/your_mtb_data.csv"

driver_genes:
  - katG   # isoniazid target
  - inhA   # isoniazid target
  - rpoB   # rifampicin target
  - gyrA   # fluoroquinolone target
```

Ready-to-use configs for LUAD, CRC, TB, and Leishmaniasis are included in the file.

---

## Project structure

```
proteinfp/
├── config/
│   ├── config.yaml            ← paths, API thresholds, tool settings
│   └── disease_config.yaml    ← switch disease/organism here
├── pipeline/                  ← Modules 01–17
├── proteinfp/                 ← CLI package (pip install proteinfp)
│   ├── cli.py                 ← proteinfp --uniprot X
│   ├── orchestrator.py        ← runs all modules gracefully
│   ├── therapy.py             ← therapy decision + epitope + de novo
│   └── deps.py                ← optional dependency checker
├── sim/                       ← SIM-01 to SIM-07 (whole-cell simulation)
├── grn/                       ← GRN-01 to GRN-03 (gene regulatory network)
├── utils/                     ← config loader, PDB parser
├── tests/                     ← test suite (pytest)
├── validation/                ← validation against known drug-protein pairs
├── train/                     ← ML model training scripts
├── models/                    ← EC classifier ensemble (metadata only in repo)
└── pyproject.toml             ← pip install configuration
```

---

## Running tests

```bash
python -m pytest tests/ -v
```

---

## Reproducibility

All outputs are deterministic given the same input. Every inference step saves a JSON to `data/intermediate/` so individual modules can be re-run or inspected without rerunning the full pipeline.

---

## Citation

If you use ProteinFP in your research, please cite this repository. A methods paper describing the pipeline is in preparation.

---

## License

MIT