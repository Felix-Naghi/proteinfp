# ProteinFP

**End-to-end protein function prediction and evolutionary drug candidate design.**

Give it a UniProt ID. Get back active sites, druggable pockets, allosteric sites,
EC classification, GO terms, PPI partners, a ranked therapy decision across 7 modalities,
and (if you want) evolved drug candidates: antibodies, ADCs, CAR-T constructs,
PROTACs, allosteric small molecules, or de novo small molecules.
Works for any protein, any disease, any organism.

```bash
pip install proteinfp
proteinfp --uniprot P04637          # TP53, full pipeline in ~60s
proteinfp --uniprot P04637 --interactive   # therapy decision + guided design
```

```
  Protein    : Cellular tumor antigen p53
  Gene       : TP53  (P04637)
  Organism   : Homo sapiens
  Confidence : HIGH

  Top function : DNA-binding transcription factor
  Enzyme       : no
  Pockets      : 3  (best: P1  vol=560A  drug=0.98)
  Allosteric   : A1  corr=0.956  confidence=HIGH

  Therapy Decision  [MEDIUM]
    0.907  protac          PPI with MDM2/MDM4, warhead anchor identified
    0.842  allosteric      ENM corr=0.956, no orthosteric competition
    0.682  small_molecule  Pocket P1: vol=560A  druggability=0.98
```

---

## What it does

ProteinFP runs up to 21 modules in sequence, fusing their outputs into a single
confidence-weighted report and triggering the right design engine for your protein.

### Core pipeline (always runs, no optional deps needed for modules 01 to 13 and 17)

| # | Module | What it predicts |
|---|--------|-----------------|
| 01 | `fetch_structure` | AlphaFold structure + UniProt metadata |
| 02 | `physicochemical` | Surface charge, hydrophobicity, SASA *(needs freesasa)* |
| 03 | `active_sites` | Catalytic residues and active site motifs |
| 04 | `binding_pockets` | Druggable pockets, geometry and druggability score |
| 05 | `allosteric` | Allosteric sites via elastic network model (ENM) |
| 06 | `chemical_env` | Chemical environment of each predicted site |
| 07 | `homology` | Sequence homologs with known function (BLAST + InterPro) |
| 08 | `esm2` | ESM-2 protein language model embeddings *(needs torch + fair-esm)* |
| 10 | `ec_prediction` | Enzyme class, ML ensemble at ~97% accuracy *(ML or rules fallback)* |
| 11 | `foldseek` | Structural analogs via Foldseek API, finds same-fold proteins |
| 12 | `ppi_network` | Protein-protein interactions (STRING DB) |
| 13 | `consensus` | Final report, fuses all evidence, confidence-weighted |
| 14 | `molecular_dyn` | MD simulation, RMSF, flexibility, cryptic pockets *(needs OpenMM)* |
| 15 | `denovo_design` | De novo small molecules, evolutionary design *(needs Vina + RDKit)* |
| 16 | `antibody_design` | De novo antibody CDR design, epitope-directed evolution |
| 17 | `ptm_analysis` | Post-translational modification sites and functional consequences |

### Evolutionary design modules (all pure Python, no Vina or RDKit needed)

| # | Module | What it designs |
|---|--------|----------------|
| 18 | `adc_design` | **Antibody-Drug Conjugate** co-evolves CDR sequences + warhead (MMAE/DM1/SN-38/PBD/calicheamicin) + linker (cleavable or non-cleavable) |
| 19 | `cart_design` | **CAR-T construct** co-evolves scFv CDR sequences + CAR generation (1st through 4th gen / TRUCK) + hinge region |
| 20 | `protac_design` | **PROTAC degrader** co-evolves POI warhead SMILES + linker + E3 ligase ligand (CRBN/VHL/IAP/MDM2) with hook-effect penalty |
| 21 | `allosteric_drug` | **Allosteric small molecule** ENM-guided evolution targeting the best allosteric site from Module 05, no Vina needed |

### GRN + SIM pipeline (disease-aware mode, needs scRNA-seq data)

| Module | What it does |
|--------|-------------|
| GRN-01 | scRNA-seq preprocessing, HVG selection, QC filtering |
| GRN-02 | GENIE3 gene regulatory network reconstruction |
| GRN-03 | Therapy modality decision with expression data |
| SIM-01 | Tumour cell environment inference from marker expression |
| SIM-02 | Protein conformational ensemble in tumour environment |
| SIM-03 | Drug distribution across cell compartments |
| SIM-04 | Binding probability under real physiological conditions |
| SIM-05 | GRN perturbation, network-level drug consequence |
| SIM-06 | Pharmacological scoring, efficacy, selectivity, resistance, grade A to F |

---

## Installation

```bash
pip install proteinfp
```

The core pipeline (Modules 01 to 13, 17, and all evolutionary design modules 16 to 21)
works out of the box with no additional installs.

**Optional features:**

```bash
pip install proteinfp[structure]  # SASA/DSSP surface analysis (Module 02)
pip install proteinfp[ml]         # ESM-2 embeddings + ML EC classifier (Modules 08, 10)
pip install proteinfp[chem]       # RDKit for de novo small molecules (Module 15)
pip install proteinfp[sim]        # OpenMM molecular dynamics (Module 14)
pip install proteinfp[grn]        # scRNA-seq / GRN modules (scanpy)
pip install proteinfp[all]        # Everything
```

For Module 15 (de novo small molecules) you also need
[AutoDock Vina](https://vina.scripps.edu/downloads/). Install it separately
and pass `--vina /path/to/vina`.

Check what is available on your machine:

```bash
proteinfp --check-deps
proteinfp --list-modules
```

---

## Quick start

```bash
# Run the core pipeline on any protein
proteinfp --uniprot P04637        # TP53 (tumour suppressor)
proteinfp --uniprot P00533        # EGFR (kinase / surface receptor)
proteinfp --uniprot O60885        # BRD4 (epigenetic regulator)
proteinfp --uniprot P28593        # Trypanothione reductase (Chagas disease)

# Force re-run even if cached report exists
proteinfp --uniprot P04637 --force

# With SASA surface analysis (recommended, improves epitope quality)
pip install proteinfp[structure]
proteinfp --uniprot P04637

# With ESM-2 and ML EC classifier
pip install proteinfp[ml]
proteinfp --uniprot P04637

# With molecular dynamics
proteinfp --uniprot P04637 --md

# With de novo small molecule design (needs Vina)
proteinfp --uniprot P04637 --denovo --vina /path/to/vina

# With antibody CDR design
proteinfp --uniprot P04637 --antibody
proteinfp --uniprot P04637 --antibody --epitope-mode ppi --ab-generations 100
```

---

## Therapy mode

### Interactive mode (recommended)

Scores all 7 therapy modalities for your protein, shows a ranked menu with
guidance, then asks you to pick one or more. Each design module is launched
with parameters pre-filled from what the therapy engine found about the protein.

```bash
# Decision + interactive picker (no Vina needed for antibody/ADC/CAR-T/PROTAC/allosteric)
proteinfp --uniprot P04637 --interactive

# Include small molecule de novo (needs Vina)
proteinfp --uniprot P04637 --interactive --vina pipeline/vina.exe
```

Example session for TP53:

```
  [1] PROTAC / Protein Degrader          Score: 0.907
       PPI with MDM2/MDM4, warhead anchor identified
       Pocket P1 vol=560A, room for warhead
       Best when: Intracellular + epigenetic or strong MDM2/VHL/CRBN PPI.

  [2] Allosteric Small Molecule          Score: 0.842
       ENM correlation 0.956, strong allosteric coupling
       Best when: High ENM correlation, especially if active site is undruggable.

  [3] Small Molecule Inhibitor           Score: 0.682
       Pocket P1 druggability 0.98, excellent target

  Enter one or more numbers: 1

  PROTAC / Protein Degrader
    Context:
      Pocket druggability 0.98, warhead binding site identified
      PPI with MDM2/MDM4, this interaction is the warhead anchor

    Suggested E3 ligase: CRBN
    Use CRBN? [Enter to confirm, or type CRBN/VHL/IAP/MDM2]: MDM2
    Generations [50]: 50

  [Module 20 runs...]

  #1  poi=0.895  e3=0.968  DC50~550pM  Dmax~96%  MDM2/MI-773  PEG3  MW~904
```

### Automatic mode (runs all viable modalities)

```bash
proteinfp --uniprot P04637 --therapy
proteinfp --uniprot P04637 --therapy --vina pipeline/vina.exe
```

### Decision only (fast, about 1 second)

```bash
python proteinfp/therapy.py --uniprot P04637 --test
```

### Modality scoring

The therapy engine scores all 7 modalities from structural evidence alone,
no GRN or expression data required:

| Modality | Key signals |
|----------|-------------|
| **ADC** | Surface confirmed + internalisation GO terms + SASA 200 to 1200 sq angstrom |
| **CAR-T** | Surface + large SASA over 600 sq angstrom + tumour antigen GO terms |
| **Naked antibody** | Surface + PPI with clinically validated partners |
| **Small molecule** | Pocket druggability + volume + enzyme/EC classification |
| **PROTAC** | Intracellular + epigenetic GO + MDM2/VHL/CRBN PPI + pocket for warhead |
| **Allosteric** | ENM correlation + coupling depth + no orthosteric pocket bonus |
| **Molecular glue** | No pocket + no allosteric site + E3 complex PPI |

---

## Running the evolutionary design modules

### All modules at once (test runner)

```bash
# Quick test, 15 generations per module (~20s total)
python test_evolutionary.py P04637

# Better results, 50 generations
python test_evolutionary.py P04637 --generations 50

# Multiple proteins
python test_evolutionary.py P04637 P00533 O60885

# Specific modules only
python test_evolutionary.py P04637 --modules protac allosteric
python test_evolutionary.py P00533 --modules antibody adc cart

# Re-run even if outputs exist
python test_evolutionary.py P04637 --force
```

Expected output for TP53 (P04637), 15 generations:

```
  Module                   Protein    Status    Score    Time
  antibody                 P04637     PASS     0.984   19.1s
  adc                      P04637     PASS     0.799    0.3s
  cart                     P04637     PASS     0.746    0.3s
  protac                   P04637     PASS     0.907    0.4s
  allosteric               P04637     PASS     0.842    0.2s
```

### Standalone module commands

**Antibody CDR design (Module 16):**
```bash
python pipeline/antibody_design.py --uniprot P04637
python pipeline/antibody_design.py --uniprot P04637 --epitope-mode ppi --generations 100
# epitope-mode options: auto, active, ppi, surface, allosteric
```

**ADC design (Module 18):**
```bash
python pipeline/adc_design.py --uniprot P04637
python pipeline/adc_design.py --uniprot P04637 --warhead MMAE --generations 80
python pipeline/adc_design.py --uniprot P00533 --epitope-mode ppi
# warhead options: MMAE, DM1, DM4, SN38, Dxd, CalicheA, PBD, MMAF
```

**CAR-T design (Module 19):**
```bash
python pipeline/cart_design.py --uniprot P00533
python pipeline/cart_design.py --uniprot P00533 --car-gen 3 --generations 80
# car-gen options: 1 (CD3z), 2 (CD28), 3 (4-1BB), 4 (CD28+4-1BB), 5 (TRUCK)
```

**PROTAC design (Module 20):**
```bash
python pipeline/protac_design.py --uniprot P04637
python pipeline/protac_design.py --uniprot P04637 --e3 MDM2 --generations 80
python pipeline/protac_design.py --uniprot O60885 --e3 CRBN --linker-type PEG3
# e3 options: CRBN, VHL, IAP, MDM2
# linker-type options: PEG2, PEG3, PEG4, Alkyl3, Alkyl4, Alkyl6, Piperaz, Mixed1, Mixed2, Rigid1
```

**Allosteric drug design (Module 21):**
```bash
python pipeline/allosteric_drug_design.py --uniprot P04637
python pipeline/allosteric_drug_design.py --uniprot P04637 --site A1 --mechanism inhibitor
python pipeline/allosteric_drug_design.py --uniprot P04637 --mechanism activator --generations 80
# mechanism options: inhibitor, activator, modulator
```

**Via the main CLI (after pipeline has run):**
```bash
proteinfp --uniprot P04637 --antibody
proteinfp --uniprot P04637 --antibody --epitope-mode ppi --ab-generations 100
proteinfp --uniprot P04637 --therapy
proteinfp --uniprot P04637 --interactive
```

---

## Python API

```python
from proteinfp import run

# Run the full core pipeline
result = run("P04637")
print(result.report_path)

# Run therapy decision
from proteinfp.therapy import run_therapy
therapy = run_therapy("P04637")
print(therapy.decision.primary_modality)
print(therapy.decision.modality_scores)

# Interactive design (useful in Jupyter notebooks)
from proteinfp.therapy import interactive_design
interactive_design("P04637")

# Run a specific evolutionary module directly
from pipeline.protac_design import run_protac_design
import json
from pathlib import Path

inter = Path("data/intermediate")
result = run_protac_design(
    uniprot_id    = "P04637",
    pocket_data   = json.loads((inter / "P04637_binding_pockets.json").read_text()),
    active_data   = json.loads((inter / "P04637_active_sites.json").read_text()),
    preferred_e3  = "MDM2",
    n_generations = 50,
)
for c in result.top_candidates[:3]:
    print(c.summary_line(1))
```

---

## Output files

All outputs are saved under `data/`:

```
data/
  structures/
    P04637.pdb
  intermediate/
    P04637_active_sites.json
    P04637_binding_pockets.json
    P04637_allosteric.json
    P04637_ppi.json
    P04637_antibody.json
    P04637_adc.json
    P04637_cart.json
    P04637_protac.json
    P04637_allosteric_drug.json
  reports/
    P04637_report.json
    P04637_report.txt
    P04637_therapy.json
    P04637_therapy.txt
```

---

## Module score interpretation

### Antibody / ADC / CAR-T (Modules 16 to 19)

| Field | Meaning |
|-------|---------|
| `affinity_score` | Predicted CDR-epitope binding complementarity (0 to 1) |
| `developability` | Antibody engineering quality: charge, pI, aggregation risk (0 to 1) |
| `cdr_h3` | CDR-H3 loop sequence, the primary antigen-contact loop |
| `pI` | Isoelectric point, 6 to 8 is optimal for most therapeutics |
| `warhead_class` | ADC payload class (MMAE/DM1/PBD etc.) |
| `dar_min/max` | Drug-antibody ratio recommendation |
| `car_arch_name` | CAR generation (2nd_gen_41BB = tisagenlecleucel model) |
| `persistence_score` | Predicted T-cell persistence, 4-1BB is better than CD28 for memory |

### PROTAC (Module 20)

| Field | Meaning |
|-------|---------|
| `poi_affinity` | Warhead binding to target protein pocket (0 to 1) |
| `e3_affinity` | E3 ligase ligand binding (0 to 1) |
| `DC50` | Predicted degradation EC50, concentration for 50% target loss |
| `Dmax` | Predicted maximum degradation % at saturating PROTAC concentration |
| `hook_penalty` | Penalty for very high-affinity warheads (hook effect risk) |
| `estimated_mw` | Total PROTAC MW in Da, real PROTACs are typically 700 to 1100 Da |

### Allosteric drug (Module 21)

| Field | Meaning |
|-------|---------|
| `site_complementarity` | Shape/charge/hydrophobicity match to allosteric site (0 to 1) |
| `communication_score` | Predicted disruption of ENM pathway from active site (0 to 1) |
| `selectivity_score` | Predicted selectivity for allosteric vs orthosteric site (0 to 1) |
| `mechanism` | Predicted mode of action: inhibitor, activator, or modulator |

---

## Choosing the right modality

| Protein type | Best first choice | Why |
|---|---|---|
| Surface receptor, internalises | **ADC** | Payload delivered intracellularly |
| Surface receptor, does not internalise | **CAR-T** or **naked mAb** | T-cell direct kill or Fc-mediated |
| Intracellular, deep hydrophobic pocket | **Small molecule** | Classic active site inhibition |
| Intracellular, MDM2/VHL/CRBN PPI | **PROTAC** | Exploit existing E3 ligase proximity |
| Intracellular, epigenetic/BET/HDAC | **PROTAC** | Remove all protein functions, not just catalytic |
| No pocket, allosteric site present | **Allosteric** | ENM-guided selectivity advantage |
| No pocket, no allosteric, E3 PPI | **Molecular glue** | No warhead binding needed |

---

## Disease-agnostic design

The pipeline works on any protein from any organism. Switch disease context by
editing one config file:

```yaml
# config/disease_config.yaml
disease:
  name: "TB"
  organism: "Mycobacterium tuberculosis"
  organism_id: 83332

driver_genes:
  - katG
  - inhA
  - rpoB
  - gyrA
```

Built-in configs: LUAD (lung), CRC (colorectal), TB (tuberculosis), Leishmaniasis.

---

## Development and testing

```bash
pip install proteinfp[dev]
pytest tests/

python test_evolutionary.py P04637
python test_evolutionary.py P04637 P00533 O60885 --generations 50
python test_evolutionary.py P04637 --modules protac allosteric

ruff check .
black .
```

---

## Project structure

```
proteinfp/
├── proteinfp/
│   ├── cli.py
│   ├── orchestrator.py
│   ├── therapy.py
│   ├── deps.py
│   └── __init__.py
├── pipeline/
│   ├── fetch_structure.py     Module 01
│   ├── physicochemical.py     Module 02
│   ├── active_sites.py        Module 03
│   ├── binding_pockets.py     Module 04
│   ├── allosteric.py          Module 05
│   ├── chemical_env.py        Module 06
│   ├── homology.py            Module 07
│   ├── esm2_embeddings.py     Module 08
│   ├── ec_model_check.py      Module 10
│   ├── foldseek.py            Module 11
│   ├── ppi_network.py         Module 12
│   ├── consensus.py           Module 13
│   ├── molecular_dynamics.py  Module 14
│   ├── denovo_design.py       Module 15
│   ├── antibody_design.py     Module 16
│   ├── ptm_analysis.py        Module 17
│   ├── adc_design.py          Module 18
│   ├── cart_design.py         Module 19
│   ├── protac_design.py       Module 20
│   └── allosteric_drug_design.py  Module 21
├── grn/
├── sim/
├── utils/
├── test_evolutionary.py
├── pyproject.toml
└── README.md
```

---

## Changelog

### v0.1.7
- **New**: Module 18, ADC design (CDR + warhead + linker co-evolution)
- **New**: Module 19, CAR-T design (scFv CDR + CAR generation + hinge co-evolution)
- **New**: Module 20, PROTAC design (warhead + linker + E3 ligase co-evolution, hook-effect penalty, realistic DC50/Dmax model)
- **New**: Module 21, Allosteric drug design (ENM-guided fragment evolution, no Vina needed)
- **New**: `--interactive` flag, ranked therapy menu with guided parameter prompts
- **New**: `therapy.py` scores all 7 modalities independently with ADC/CAR-T discrimination from structural signals
- **New**: `test_evolutionary.py`, standalone test runner for all 5 evolutionary modules
- **Fix**: PROTAC DC50 now uses a physically grounded Kd-based model
- **Fix**: Allosteric SMILES validated for bracket balance before entering hall of fame
- **Fix**: CDR length constraints corrected to match seed sequence lengths

### v0.1.1
- Antibody design (Module 16) wired into main CLI as `--antibody`
- Therapy mode triggers epitope selection and de novo design
- Surface detection improved with GO ID matching and gene blocklist

### v0.1.0
- Initial release: Modules 01 to 15, 17
- Core pipeline + GRN/SIM framework
- De novo molecular design with AutoDock Vina

---

## Publishing a release

```bash
git add README.md pyproject.toml
git add proteinfp/cli.py proteinfp/therapy.py
git add pipeline/adc_design.py pipeline/cart_design.py
git add pipeline/protac_design.py pipeline/allosteric_drug_design.py
git add test_evolutionary.py

git commit -m "v0.1.7: evolutionary design modules 18-21 + interactive therapy"

git tag v0.1.7
git push origin main --tags

python -m build
twine upload dist/*
```

---

## License

MIT. See [LICENSE](LICENSE).

## Citation

If you use ProteinFP in research, please cite:

```
ProteinFP: End-to-end protein function prediction and evolutionary drug design.
https://github.com/wowcowdowjones/proteinFP2
```