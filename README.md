# ProteinFP

Computational pipeline for end-to-end protein function prediction.
Predicts active sites, binding interfaces, allosteric sites, chemical environment,
and protein-protein interactions from structure alone + sequence evidence.

---

## Quick start (Windows, RTX 5060)

### Step 1 — Open a terminal in your project folder

```
Win + R → cmd → cd C:\Users\adria\Documents\proteinFP
```

Or right-click the `proteinFP` folder in Explorer → "Open in Terminal"

---

### Step 2 — Run the setup script (once only)

```bat
setup.bat
```

This will:
- Create a Python virtual environment in `.venv/`
- Install PyTorch with CUDA 12.1 support for your RTX 5060
- Install all pipeline dependencies
- Create the `data/` directory structure
- Run the offline unit tests to confirm everything works

---

### Step 3 — Open in VS Code

```bat
code proteinFP.code-workspace
```

VS Code will auto-detect the `.venv` interpreter and enable:
- IntelliSense for all modules
- One-click test runner
- Pre-configured launch configs (F5 to run)

---

### Step 4 — Run Module 01 on your first protein

Activate the virtual environment first (if not already active):
```bat
.venv\Scripts\activate
```

Then run:
```bat
python pipeline\01_fetch_structure.py --uniprot P04637
```

`P04637` is human TP53 — one of the most studied proteins in biology,
great for validating your pipeline because all its sites are known.

Expected output:
```
12:34:01 [pipeline.01_fetch_structure] INFO  ── Module 01: Fetching P04637 ──
12:34:02 [pipeline.01_fetch_structure] INFO    [1/4] Querying AlphaFold DB...
12:34:03 [pipeline.01_fetch_structure] INFO    [2/4] Downloading .pdb...
12:34:05 [pipeline.01_fetch_structure] INFO    [3/4] Fetching UniProt metadata...
12:34:06 [pipeline.01_fetch_structure] INFO    [4/4] Parsing structure...
12:34:06 [utils.pdb_parser] INFO  Parsing structure: P04637.pdb
12:34:06 [utils.pdb_parser] INFO    → 393 residues | mean pLDDT: 72.4 | disordered: 87 residues in 4 region(s)

────────────────────────────────────────────────────────────
  Protein    : Cellular tumor antigen p53
  Gene       : TP53
  UniProt    : P04637  (Swiss-Prot (reviewed))
  Organism   : Homo sapiens
  Length     : 393 aa
  Mean pLDDT : 72.4
  High-conf  : 68.4% of residues
  Disordered : 87 residues in 4 region(s)
  .pdb saved : C:\Users\adria\Documents\proteinFP\data\structures\P04637.pdb
────────────────────────────────────────────────────────────
```

---

### Step 5 — Run the tests

```bat
python -m pytest tests\ -v
```

To run the full integration test (requires internet):
```bat
python -m pytest tests\ -v -k "Integration"
```

---

## Project structure

```
proteinFP/
├── config/
│   └── config.yaml          ← edit API keys, thresholds, paths here
├── data/
│   ├── input/               ← put your .fasta files here
│   ├── structures/          ← .pdb files downloaded from AFDB
│   ├── intermediate/        ← per-module JSON outputs
│   └── reports/             ← final HTML + JSON reports per protein
├── pipeline/
│   ├── 01_fetch_structure.py   ← START HERE (this module)
│   ├── 02_physicochemical.py   ← next to build
│   └── ...
├── utils/
│   ├── config.py            ← config loader + logger
│   └── pdb_parser.py        ← shared PDB parsing logic
├── tests/
│   └── test_01_fetch_structure.py
├── setup.bat                ← run once to set up environment
├── requirements.txt
└── proteinFP.code-workspace ← open this in VS Code
```

---

## What each module will predict

| Module | Predicts |
|--------|----------|
| 01 | Structure fetch + pLDDT confidence map |
| 02 | Physicochemical surface (charge, hydrophobicity, SASA) |
| 03 | Active site residues (catalytic + metal-binding) |
| 04 | Binding pockets (geometry + druggability score) |
| 05 | Allosteric sites (elastic network model) |
| 06 | Chemical environment of each site (electrostatics, H-bonds) |
| 07 | Sequence homologs with known function (BLAST, HHpred) |
| 08-10 | AI function prediction (DeepFRI, ESM-2, CLEAN) |
| 11 | Structural analogs via Foldseek |
| 12 | Protein-protein interactions (STRING DB + AF-Multimer) |
| 13 | Consensus scoring + HTML/JSON report |

---

## GPU note (RTX 5060)

Your RTX 5060 is ideal for:
- ESM-2 (650M parameter model fits in ~3GB VRAM)
- DeepFRI inference (~1GB VRAM)
- AlphaFold-Multimer (for PPI) — will use most of your VRAM

Modules 01–07 and 11–13 run on CPU. GPU is only needed for 08–10 and PPI docking.

To verify your GPU is detected after setup:
```bat
python -c "import torch; print(torch.cuda.get_device_name(0))"
```
