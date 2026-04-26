"""
score_all.py
─────────────
Scores ALL reports found in data/reports/ against ground truth.

For proteins in VALIDATION_SET: full scoring (GO recall, active site recall,
enzyme classification, PPI recall).

For proteins NOT in VALIDATION_SET: partial scoring based on internal
consistency and confidence (enzyme classification confidence, GO term count,
active site count, PPI count). These get an asterisk in the output.

Usage:
    python score_all.py
    python score_all.py --reports-dir data/reports
"""

from __future__ import annotations

import json
import sys
import urllib.request
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path

# ── Residue offset detection ───────────────────────────────────────────────────

_offset_cache: dict[str, int] = {}

def _get_uniprot_offset(uniprot_id: str, report: dict) -> int:
    """
    Detect the offset between UniProt canonical residue numbers and the
    AlphaFold PDB residue numbers.

    AlphaFold numbers from 1 for the first residue in the PROCESSED sequence
    (i.e. after signal peptide / propeptide cleavage). UniProt active site
    annotations use CANONICAL numbers (full precursor).

    Strategy: fetch the UniProt 'chain' feature to find where the mature
    chain starts in the canonical sequence. The offset = chain_start - 1.
    Falls back to 0 if unavailable.
    """
    if uniprot_id in _offset_cache:
        return _offset_cache[uniprot_id]

    # Quick estimate from report sequence length vs UniProt length
    # If they match, offset is 0
    report_len = report.get("sequence_length", 0)

    try:
        url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json"
        req = urllib.request.Request(
            url,
            headers={"Accept": "application/json",
                     "User-Agent": "ProteinFP-scorer/1.0"}
        )
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.loads(r.read())

        canonical_len = data.get("sequence", {}).get("length", 0)

        # Look for chain/peptide feature to find mature chain start
        offset = 0
        for feat in data.get("features", []):
            if feat.get("type") in ("Chain", "Peptide"):
                start = feat.get("location", {}).get("start", {}).get("value", 1)
                end   = feat.get("location", {}).get("end",   {}).get("value", canonical_len)
                mature_len = end - start + 1
                # Use this chain if its length matches our structure length
                if report_len and abs(mature_len - report_len) <= 5:
                    offset = start - 1
                    break

        _offset_cache[uniprot_id] = offset
        return offset

    except Exception:
        _offset_cache[uniprot_id] = 0
        return 0

# ── Ground truth (from run_validation.py) ─────────────────────────────────────

GROUND_TRUTH: dict[str, dict] = {
    "P04637": {
        "gene": "TP53", "category": "transcription_factor", "is_enzyme": False,
        "known_go_mf": ["GO:0003677","GO:0003700","GO:0046872"],
        "known_go_bp": ["GO:0006915","GO:0006974","GO:0045944"],
        "known_go_cc": ["GO:0005634","GO:0043234"],
        "known_active_residues": [176,179,248,273],
        "known_partners": ["MDM2","MDM4","ATM","CHEK2","EP300"],
    },
    "P00533": {
        "gene": "EGFR", "category": "kinase", "is_enzyme": True, "ec_number": "2.7.10.1",
        "known_go_mf": ["GO:0004672","GO:0004714","GO:0005006"],
        "known_go_bp": ["GO:0007173","GO:0008283","GO:0018108"],
        "known_go_cc": ["GO:0005887","GO:0016020"],
        "known_active_residues": [837,855],
        "known_partners": ["GRB2","SOS1","PIK3R1","SHC1"],
    },
    "P00441": {
        "gene": "SOD1", "category": "oxidoreductase", "is_enzyme": True, "ec_number": "1.15.1.1",
        "known_go_mf": ["GO:0004784","GO:0005507","GO:0008270"],
        "known_go_bp": ["GO:0019430","GO:0006801"],
        "known_go_cc": ["GO:0005737","GO:0005634"],
        "known_active_residues": [44,46,118],
        "known_partners": ["CCS","TNFRSF1A"],
    },
    "P07900": {
        "gene": "HSP90AA1", "category": "chaperone", "is_enzyme": False,
        "known_go_mf": ["GO:0005524","GO:0051082","GO:0042623"],
        "known_go_bp": ["GO:0006457","GO:0051085"],
        "known_go_cc": ["GO:0005737","GO:0005634"],
        "known_active_residues": [35,83,183],
        "known_partners": ["CDC37","AHA1","HOP","CHIP"],
    },
    "P06213": {
        "gene": "INSR", "category": "kinase", "is_enzyme": True, "ec_number": "2.7.10.1",
        "known_go_mf": ["GO:0004672","GO:0004713","GO:0005009"],
        "known_go_bp": ["GO:0008286","GO:0046628"],
        "known_go_cc": ["GO:0005887","GO:0005615"],
        "known_active_residues": [1131,1135,1136],
        "known_partners": ["IRS1","IRS2","GRB2","SHC1"],
    },
    "P38398": {
        "gene": "BRCA1", "category": "dna_repair", "is_enzyme": False,
        "known_go_mf": ["GO:0003684","GO:0003723","GO:0004842"],
        "known_go_bp": ["GO:0006281","GO:0007131","GO:0045739"],
        "known_go_cc": ["GO:0005634","GO:0010369"],
        "known_active_residues": [1763,1836],
        "known_partners": ["BARD1","RAD51","TP53","ATM"],
    },
    "P16083": {
        "gene": "NQO2", "category": "oxidoreductase", "is_enzyme": True, "ec_number": "1.10.99.2",
        "known_go_mf": ["GO:0003955","GO:0010181"],
        "known_go_bp": ["GO:0055114","GO:0042493"],
        "known_go_cc": ["GO:0005737"],
        "known_active_residues": [103,128],
        "known_partners": ["AHR"],
    },
    "P00734": {
        "gene": "F2", "category": "serine_protease", "is_enzyme": True, "ec_number": "3.4.21.5",
        "known_go_mf": ["GO:0004252","GO:0005172"],
        "known_go_bp": ["GO:0007596","GO:0030193"],
        "known_go_cc": ["GO:0005576","GO:0072562"],
        "known_active_residues": [363,419,521],
        "known_partners": ["F5","F8","THBD"],
    },
    "P68871": {
        "gene": "HBB", "category": "oxygen_transport", "is_enzyme": False,
        "known_go_mf": ["GO:0020037","GO:0019825"],
        "known_go_bp": ["GO:0015671","GO:0019430"],
        "known_go_cc": ["GO:0005833","GO:0031838"],
        "known_active_residues": [92],
        "known_partners": ["HBA1","HBA2"],
    },
    "P00918": {
        "gene": "CA2", "category": "lyase", "is_enzyme": True, "ec_number": "4.2.1.1",
        "known_go_mf": ["GO:0004089","GO:0008270"],
        "known_go_bp": ["GO:0015701","GO:0001659"],
        "known_go_cc": ["GO:0005737","GO:0005829"],
        "known_active_residues": [94,96,119],
        "known_partners": ["SLC4A1","CA1"],
    },
    "P01116": {
        "gene": "KRAS", "category": "gtpase", "is_enzyme": False,
        "known_go_mf": ["GO:0005525","GO:0003924","GO:0019003"],
        "known_go_bp": ["GO:0007165","GO:0008283"],
        "known_go_cc": ["GO:0016020","GO:0005737"],
        "known_active_residues": [10,12,13,16],
        "known_partners": ["BRAF","RAF1","SOS1","RALGDS"],
    },
    "Q00987": {
        "gene": "MDM2", "category": "ubiquitin_ligase", "is_enzyme": True, "ec_number": "2.3.2.27",
        "known_go_mf": ["GO:0061630","GO:0042802"],
        "known_go_bp": ["GO:0043066","GO:0051726"],
        "known_go_cc": ["GO:0005634","GO:0005737"],
        "known_active_residues": [305,308,319,322],
        "known_partners": ["TP53","MDM4","USP7","RB1"],
    },
    "Q9BYF1": {
        "gene": "ACE2", "category": "metallopeptidase", "is_enzyme": True, "ec_number": "3.4.17.23",
        "known_go_mf": ["GO:0008237","GO:0008241","GO:0046872"],
        "known_go_bp": ["GO:0006508","GO:0010819"],
        "known_go_cc": ["GO:0016020","GO:0005615"],
        "known_active_residues": [374,378,402],
        "known_partners": ["TMPRSS2","AGT","SLC6A19"],
    },
    "O15151": {
        "gene": "MDM4", "category": "ubiquitin_ligase", "is_enzyme": False,
        "known_go_mf": ["GO:0061630","GO:0008270","GO:0004842"],
        "known_go_bp": ["GO:0043066","GO:0051726","GO:0006915"],
        "known_go_cc": ["GO:0005634","GO:0005737"],
        "known_active_residues": [460,463,466,469],
        "known_partners": ["MDM2","TP53","USP7"],
    },
    "P42574": {
        "gene": "CASP3", "category": "cysteine_protease", "is_enzyme": True, "ec_number": "3.4.22.56",
        "known_go_mf": ["GO:0004197","GO:0008234","GO:0008233"],
        "known_go_bp": ["GO:0006915","GO:0043525"],
        "known_go_cc": ["GO:0005737","GO:0005829"],
        "known_active_residues": [163,184],
        "known_partners": ["CASP8","CASP9","XIAP","PARP1"],
    },
    # ── New 30 ────────────────────────────────────────────────────────────────
    "P32119": {
        "gene": "PRDX2", "category": "oxidoreductase", "is_enzyme": True, "ec_number": "1.11.1.15",
        "known_go_mf": ["GO:0004601","GO:0051920","GO:0016209"],
        "known_go_bp": ["GO:0006979","GO:0045454","GO:0055114"],
        "known_go_cc": ["GO:0005737","GO:0005829"],
        "known_active_residues": [51,172],
        "known_partners": ["PRDX1","TXN","STAT3"],
    },
    "P00367": {
        "gene": "GLUD1", "category": "oxidoreductase", "is_enzyme": True, "ec_number": "1.4.1.3",
        "known_go_mf": ["GO:0004352","GO:0005524","GO:0050661"],
        "known_go_bp": ["GO:0006537","GO:0006096","GO:0009060"],
        "known_go_cc": ["GO:0005759","GO:0005744"],
        "known_active_residues": [126,166,262],
        "known_partners": ["SIRT4","GDH2"],
    },
    "P22309": {
        "gene": "UGT1A1", "category": "transferase", "is_enzyme": True, "ec_number": "2.4.1.17",
        "known_go_mf": ["GO:0035251","GO:0015020"],
        "known_go_bp": ["GO:0052696","GO:0008202","GO:0042738"],
        "known_go_cc": ["GO:0016020","GO:0005789"],
        "known_active_residues": [86,277],
        "known_partners": ["UGT1A4","UGT1A6"],
    },
    "P11413": {
        "gene": "G6PD", "category": "oxidoreductase", "is_enzyme": True, "ec_number": "1.1.1.49",
        "known_go_mf": ["GO:0004345","GO:0050661"],
        "known_go_bp": ["GO:0006006","GO:0055114","GO:0019322"],
        "known_go_cc": ["GO:0005737","GO:0005829"],
        "known_active_residues": [205,206,263],
        "known_partners": ["EPRS1","HSPA8"],
    },
    "P04899": {
        "gene": "GNAI2", "category": "g_protein", "is_enzyme": True, "ec_number": "3.6.5.1",
        "known_go_mf": ["GO:0005525","GO:0003924","GO:0031683"],
        "known_go_bp": ["GO:0007186","GO:0007165","GO:0035556"],
        "known_go_cc": ["GO:0005834","GO:0005886","GO:0031526"],
        "known_active_residues": [179,203,273],
        "known_partners": ["GNGT1","RGS4","ADCY2"],
    },
    "P07550": {
        "gene": "ADRB2", "category": "gpcr", "is_enzyme": False,
        "known_go_mf": ["GO:0004937","GO:0031693","GO:0005112"],
        "known_go_bp": ["GO:0007188","GO:0071880","GO:0008217"],
        "known_go_cc": ["GO:0016021","GO:0005886","GO:0045121"],
        "known_active_residues": [113,290,294],
        "known_partners": ["GNAI2","GNAS","ARRB1","ARRB2"],
    },
    "P35372": {
        "gene": "OPRM1", "category": "gpcr", "is_enzyme": False,
        "known_go_mf": ["GO:0004985","GO:0031836"],
        "known_go_bp": ["GO:0007193","GO:0007268","GO:0051930"],
        "known_go_cc": ["GO:0016021","GO:0005886","GO:0043025"],
        "known_active_residues": [147,236,296],
        "known_partners": ["GNAI2","GNAO1","ARRB2"],
    },
    "P21554": {
        "gene": "CNR1", "category": "gpcr", "is_enzyme": False,
        "known_go_mf": ["GO:0004949","GO:0031491"],
        "known_go_bp": ["GO:0007193","GO:0007268","GO:0010039"],
        "known_go_cc": ["GO:0016021","GO:0005886","GO:0043025"],
        "known_active_residues": [183,188,366],
        "known_partners": ["GNAI2","GNAI3","ARRB1"],
    },
    "P35228": {
        "gene": "NOS2", "category": "oxidoreductase", "is_enzyme": True, "ec_number": "1.14.13.39",
        "known_go_mf": ["GO:0004517","GO:0005506","GO:0050660"],
        "known_go_bp": ["GO:0006809","GO:0045087","GO:0042554"],
        "known_go_cc": ["GO:0005737","GO:0005829"],
        "known_active_residues": [367,371,594],
        "known_partners": ["NOS1","NOS3","HSP90AA1","CALML3"],
    },
    "P60709": {
        "gene": "ACTB", "category": "cytoskeletal", "is_enzyme": False,
        "known_go_mf": ["GO:0005524","GO:0003779","GO:0042802"],
        "known_go_bp": ["GO:0030036","GO:0007010","GO:0051014"],
        "known_go_cc": ["GO:0005856","GO:0005737","GO:0015629"],
        "known_active_residues": [14,18,158],
        "known_partners": ["ARPC2","WASL","TMSB4X","PFN1"],
    },
    "P68363": {
        "gene": "TUBA1B", "category": "cytoskeletal", "is_enzyme": False,
        "known_go_mf": ["GO:0005525","GO:0005200","GO:0015631"],
        "known_go_bp": ["GO:0007018","GO:0000226","GO:0051301"],
        "known_go_cc": ["GO:0005874","GO:0005737","GO:0015630"],
        "known_active_residues": [69,136,254],
        "known_partners": ["TUBB","TBCA","STMN1"],
    },
    "P07437": {
        "gene": "TUBB", "category": "cytoskeletal", "is_enzyme": False,
        "known_go_mf": ["GO:0005525","GO:0005200","GO:0042802"],
        "known_go_bp": ["GO:0007018","GO:0000226","GO:0051301"],
        "known_go_cc": ["GO:0005874","GO:0005737","GO:0015630"],
        "known_active_residues": [69,136,254],
        "known_partners": ["TUBA1B","STMN1","MAP2"],
    },
    "P63261": {
        "gene": "ACTG1", "category": "cytoskeletal", "is_enzyme": False,
        "known_go_mf": ["GO:0005524","GO:0003779","GO:0042802"],
        "known_go_bp": ["GO:0030036","GO:0007010","GO:0048471"],
        "known_go_cc": ["GO:0005856","GO:0015629","GO:0005737"],
        "known_active_residues": [14,18,158],
        "known_partners": ["PFN1","ARPC2","MYH9"],
    },
    "P02751": {
        "gene": "FN1", "category": "ecm", "is_enzyme": False,
        "known_go_mf": ["GO:0005178","GO:0005198","GO:0048407"],
        "known_go_bp": ["GO:0007160","GO:0030198","GO:0007229"],
        "known_go_cc": ["GO:0005578","GO:0005615","GO:0005886"],
        "known_active_residues": [1605,1612],
        "known_partners": ["ITGB1","ITGA5","ITGAV","SDC4"],
    },
    "P02787": {
        "gene": "TF", "category": "transporter", "is_enzyme": False,
        "known_go_mf": ["GO:0005506","GO:0008199","GO:0031720"],
        "known_go_bp": ["GO:0006826","GO:0055072","GO:0006879"],
        "known_go_cc": ["GO:0005615","GO:0005576"],
        "known_active_residues": [249,316,607,674],
        "known_partners": ["TFRC","LRP1"],
    },
    "P02679": {
        "gene": "FGG", "category": "coagulation", "is_enzyme": False,
        "known_go_mf": ["GO:0005198","GO:0042802","GO:0003786"],
        "known_go_bp": ["GO:0007596","GO:0072378","GO:0030168"],
        "known_go_cc": ["GO:0005576","GO:0005615","GO:0072562"],
        "known_active_residues": [295,308,336],
        "known_partners": ["FGA","FGB","ITGB3","F13A1"],
    },
    "P00748": {
        "gene": "F12", "category": "coagulation", "is_enzyme": True, "ec_number": "3.4.21.38",
        "known_go_mf": ["GO:0004252","GO:0008236"],
        "known_go_bp": ["GO:0007596","GO:0006954","GO:0002542"],
        "known_go_cc": ["GO:0005576","GO:0072562"],
        "known_active_residues": [368,393,465],
        "known_partners": ["KLKB1","SERPING1","F11"],
    },
    "P00749": {
        "gene": "PLAU", "category": "coagulation", "is_enzyme": True, "ec_number": "3.4.21.73",
        "known_go_mf": ["GO:0004252","GO:0005178"],
        "known_go_bp": ["GO:0007596","GO:0031639","GO:0001525"],
        "known_go_cc": ["GO:0005576","GO:0005615"],
        "known_active_residues": [157,202,255],
        "known_partners": ["PLAUR","SERPINE1","PLG"],
    },
    "P01031": {
        "gene": "C5", "category": "complement", "is_enzyme": False,
        "known_go_mf": ["GO:0003823","GO:0042834"],
        "known_go_bp": ["GO:0006958","GO:0006956","GO:0045087"],
        "known_go_cc": ["GO:0005576","GO:0005615","GO:0005581"],
        "known_active_residues": [751,752],
        "known_partners": ["C5AR1","CFB","CFD","CD59"],
    },
    "P08603": {
        "gene": "CFH", "category": "complement", "is_enzyme": False,
        "known_go_mf": ["GO:0001848","GO:0030449","GO:0005102"],
        "known_go_bp": ["GO:0006957","GO:0045916","GO:0006956"],
        "known_go_cc": ["GO:0005576","GO:0005615"],
        "known_active_residues": [380,426],
        "known_partners": ["C3B","C3D","ITGAM"],
    },
    "P02748": {
        "gene": "C9", "category": "complement", "is_enzyme": False,
        "known_go_mf": ["GO:0005198","GO:0003823"],
        "known_go_bp": ["GO:0006956","GO:0045087"],
        "known_go_cc": ["GO:0005576","GO:0005581","GO:0072562"],
        "known_active_residues": [359,394],
        "known_partners": ["C8A","C8B","C5B","CD59"],
    },
    "P11021": {
        "gene": "HSPA5", "category": "chaperone", "is_enzyme": True, "ec_number": "3.6.4.10",
        "known_go_mf": ["GO:0005524","GO:0051082","GO:0031072"],
        "known_go_bp": ["GO:0006457","GO:0030968","GO:0006986"],
        "known_go_cc": ["GO:0005788","GO:0005789"],
        "known_active_residues": [199,246,342],
        "known_partners": ["HSPA8","DNAJB11","HYOU1","PDIA3"],
    },
    "P38646": {
        "gene": "HSPA9", "category": "chaperone", "is_enzyme": True, "ec_number": "3.6.4.10",
        "known_go_mf": ["GO:0005524","GO:0051082","GO:0031072"],
        "known_go_bp": ["GO:0006457","GO:0001836","GO:0051087"],
        "known_go_cc": ["GO:0005759","GO:0005744","GO:0005737"],
        "known_active_residues": [199,246,342],
        "known_partners": ["TP53","TRAP1","TIMM44"],
    },
    "P62987": {
        "gene": "UBA52", "category": "ubiquitin", "is_enzyme": False,
        "known_go_mf": ["GO:0031386","GO:0000166"],
        "known_go_bp": ["GO:0016567","GO:0000209","GO:0006508"],
        "known_go_cc": ["GO:0005737","GO:0005840"],
        "known_active_residues": [48,63,76],
        "known_partners": ["UBB","UBC","UBE2D1"],
    },
    "P23588": {
        "gene": "EIF4B", "category": "translation", "is_enzyme": False,
        "known_go_mf": ["GO:0003743","GO:0003724","GO:0008135"],
        "known_go_bp": ["GO:0006413","GO:0006446","GO:0045948"],
        "known_go_cc": ["GO:0016281","GO:0005737"],
        "known_active_residues": [13,98],
        "known_partners": ["EIF4A1","EIF4H","EIF3A"],
    },
    "P62136": {
        "gene": "PPP1CA", "category": "phosphatase", "is_enzyme": True, "ec_number": "3.1.3.16",
        "known_go_mf": ["GO:0004722","GO:0046872"],
        "known_go_bp": ["GO:0006470","GO:0007596","GO:0045087"],
        "known_go_cc": ["GO:0005737","GO:0005829","GO:0000922"],
        "known_active_residues": [64,96,134],
        "known_partners": ["PPP1R3A","PPP1R7","SDS22"],
    },
    "P17706": {
        "gene": "PTPN2", "category": "phosphatase", "is_enzyme": False,
        "known_go_mf": ["GO:0004725","GO:0046872"],
        "known_go_bp": ["GO:0006470","GO:0042523","GO:0045087"],
        "known_go_cc": ["GO:0005634","GO:0005737"],
        "known_active_residues": [216,221],
        "known_partners": ["JAK1","JAK3","EGFR","INSR"],
    },
    "P28482": {
        "gene": "MAPK1", "category": "kinase", "is_enzyme": True, "ec_number": "2.7.11.24",
        "known_go_mf": ["GO:0004672","GO:0004707","GO:0005524"],
        "known_go_bp": ["GO:0007165","GO:0000187","GO:0043066"],
        "known_go_cc": ["GO:0005737","GO:0005634","GO:0005829"],
        "known_active_residues": [147,150,185],
        "known_partners": ["MAP2K1","MAP2K2","RSK1","MNK1"],
    },
    "Q16539": {
        "gene": "MAPK14", "category": "kinase", "is_enzyme": False,
        "known_go_mf": ["GO:0004672","GO:0004705","GO:0005524"],
        "known_go_bp": ["GO:0006950","GO:0043066","GO:0007165"],
        "known_go_cc": ["GO:0005737","GO:0005634","GO:0005829"],
        "known_active_residues": [147,150,169],
        "known_partners": ["MAP2K3","MAP2K6","MK2","TAB1"],
    },
    "P49841": {
        "gene": "GSK3B", "category": "kinase", "is_enzyme": True, "ec_number": "2.7.11.26",
        "known_go_mf": ["GO:0004672","GO:0004693","GO:0005524"],
        "known_go_bp": ["GO:0006468","GO:0051151","GO:0043066"],
        "known_go_cc": ["GO:0005737","GO:0005634","GO:0016023"],
        "known_active_residues": [85,181,200],
        "known_partners": ["AXIN1","APC","CTNNB1","DISC1"],
    },
}

# ── Scoring helpers ────────────────────────────────────────────────────────────

def _recall(predicted: set, known: set) -> float:
    if not known:
        return 1.0
    return len(predicted & known) / len(known)


def score_with_ground_truth(report: dict, gt: dict, uid: str) -> dict:
    pred_mf = {t["go_id"] for t in report.get("go_terms_mf", [])}
    pred_bp = {t["go_id"] for t in report.get("go_terms_bp", [])}
    pred_cc = {t["go_id"] for t in report.get("go_terms_cc", [])}

    go_mf = _recall(pred_mf, set(gt.get("known_go_mf", [])))
    go_bp = _recall(pred_bp, set(gt.get("known_go_bp", [])))
    go_cc = _recall(pred_cc, set(gt.get("known_go_cc", [])))
    go_mean = (go_mf + go_bp + go_cc) / 3

    # ── Active site with offset correction ────────────────────────────────────
    # UniProt canonical numbers may be offset from AlphaFold PDB numbers
    # (signal peptides, propeptides). Fetch the offset and adjust.
    offset = _get_uniprot_offset(uid, report)

    pred_active = {r["residue_number"] for r in report.get("active_sites", [])}
    known_active_canonical = gt.get("known_active_residues", [])

    # Convert canonical → structure numbering by subtracting offset
    known_active = {ka - offset for ka in known_active_canonical}

    if known_active:
        # ±5 tolerance to handle insertion codes and small model variations
        matched = sum(
            1 for ka in known_active
            if any(abs(pa - ka) <= 5 for pa in pred_active)
        )
        as_recall = matched / len(known_active)
    else:
        as_recall = 1.0

    pred_enzyme  = report.get("is_enzyme", False)
    known_enzyme = gt.get("is_enzyme", False)
    enzyme_correct = (pred_enzyme == known_enzyme)

    pred_partners = {p["partner_name"].upper() for p in report.get("ppi_partners", [])}
    known_partners = {p.upper() for p in gt.get("known_partners", [])}
    ppi_recall = _recall(pred_partners, known_partners)

    overall = round(
        go_mean    * 35 +
        as_recall  * 25 +
        (1.0 if enzyme_correct else 0.0) * 20 +
        ppi_recall * 20,
        1
    )

    return {
        "go_mean": go_mean,
        "as_recall": as_recall,
        "enzyme_correct": enzyme_correct,
        "ppi_recall": ppi_recall,
        "overall": overall,
        "has_ground_truth": True,
        "category": gt.get("category", "other"),
    }


def score_without_ground_truth(report: dict) -> dict:
    """
    Confidence-based scoring for proteins without ground truth.
    Uses internal report fields: overall_confidence, GO count, AS count, PPI count.
    """
    conf_map = {"VERY HIGH": 0.95, "HIGH": 0.85, "MEDIUM": 0.65, "LOW": 0.4, "VERY LOW": 0.2}
    conf_raw = report.get("overall_confidence", "MEDIUM")
    conf = conf_map.get(str(conf_raw).upper().strip(), 0.5)

    n_mf  = len(report.get("go_terms_mf", []))
    n_bp  = len(report.get("go_terms_bp", []))
    n_cc  = len(report.get("go_terms_cc", []))
    n_as  = len(report.get("active_sites", []))
    n_ppi = len(report.get("ppi_partners", []))

    # Normalise counts to 0-1 (cap at expected maximums)
    go_score  = min((n_mf + n_bp + n_cc) / 45.0, 1.0)
    as_score  = min(n_as / 10.0, 1.0)
    ppi_score = min(n_ppi / 10.0, 1.0)

    overall = round(
        go_score  * 35 * conf +
        as_score  * 25 * conf +
        conf       * 20 +       # enzyme proxy
        ppi_score * 20 * conf,
        1
    )

    return {
        "go_mean": go_score * conf,
        "as_recall": as_score * conf,
        "enzyme_correct": conf > 0.5,
        "ppi_recall": ppi_score * conf,
        "overall": overall,
        "has_ground_truth": False,
        "category": "unvalidated",
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports-dir", default="data/reports")
    args = parser.parse_args()

    reports_dir = Path(args.reports_dir)
    val_dir = reports_dir / "validation"
    val_dir.mkdir(exist_ok=True)

    rows = []
    for report_path in sorted(reports_dir.glob("*_report.json")):
        uid = report_path.stem.replace("_report", "")
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"  Could not read {uid}: {e}")
            continue

        gene = report.get("gene_name", uid)

        if uid in GROUND_TRUTH:
            s = score_with_ground_truth(report, GROUND_TRUTH[uid], uid)
        else:
            s = score_without_ground_truth(report)

        rows.append({
            "uid": uid,
            "gene": gene,
            "category": s["category"],
            "go_mean": s["go_mean"],
            "as_recall": s["as_recall"],
            "enzyme_correct": s["enzyme_correct"],
            "ppi_recall": s["ppi_recall"],
            "overall": s["overall"],
            "gt": s["has_ground_truth"],
        })

    # ── Summary stats (ground-truth proteins only) ────────────────────────────
    gt_rows = [r for r in rows if r["gt"]]
    all_rows = rows

    n_gt  = len(gt_rows)
    n_all = len(all_rows)

    def mean(vals): return sum(vals) / len(vals) if vals else 0.0

    mean_go  = mean([r["go_mean"]   for r in gt_rows])
    mean_as  = mean([r["as_recall"] for r in gt_rows])
    enz_acc  = mean([1.0 if r["enzyme_correct"] else 0.0 for r in gt_rows])
    mean_ppi = mean([r["ppi_recall"] for r in gt_rows])
    mean_ov  = mean([r["overall"]   for r in gt_rows])

    # ── Print report ──────────────────────────────────────────────────────────
    sep = "=" * 70
    dash = "─" * 70

    lines = [
        sep,
        "  ProteinFP Validation Report (full set)",
        f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"  Total reports scored : {n_all}",
        f"  With ground truth    : {n_gt}",
        f"  Without ground truth : {n_all - n_gt}  (confidence-based, marked *)",
        sep,
        "",
        "  ── Ground-truth metrics ──────────────────────────────────────────",
        f"  Mean GO term recall    : {mean_go*100:.1f}%",
        f"  Mean active site recall: {mean_as*100:.1f}%",
        f"  Enzyme classification  : {enz_acc*100:.1f}%",
        f"  PPI partner recall     : {mean_ppi*100:.1f}%",
        f"  Overall accuracy score : {mean_ov:.1f}/100",
        "",
        dash,
        "  Per-protein breakdown:",
        dash,
    ]

    for r in rows:
        marker = " " if r["gt"] else "*"
        lines.append(
            f"  {marker}{r['uid']:12s} {r['gene']:10s} "
            f"[{r['category']:22s}] "
            f"GO={r['go_mean']*100:.0f}% "
            f"AS={r['as_recall']*100:.0f}% "
            f"Enz={'Y' if r['enzyme_correct'] else 'N'} "
            f"PPI={r['ppi_recall']*100:.0f}% "
            f"[{r['overall']:.1f}]"
        )

    lines += ["", dash, "  Category breakdown (ground-truth only):", dash]

    categories: dict[str, list[float]] = {}
    for r in gt_rows:
        categories.setdefault(r["category"], []).append(r["overall"])
    for cat, scores in sorted(categories.items()):
        lines.append(
            f"  {cat:25s}: {mean(scores):.1f}/100  ({len(scores)} proteins)"
        )

    lines.append(sep)
    lines.append("  * = no ground truth; scored by internal confidence metrics")
    lines.append(sep)

    report_text = "\n".join(lines)
    print(report_text)

    # Save
    out_txt  = val_dir / "validation_report_full.txt"
    out_json = val_dir / "validation_report_full.json"
    out_txt.write_text(report_text, encoding="utf-8")
    out_json.write_text(json.dumps({
        "run_date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "n_total": n_all,
        "n_ground_truth": n_gt,
        "mean_go_recall": round(mean_go, 4),
        "mean_as_recall": round(mean_as, 4),
        "enzyme_accuracy": round(enz_acc, 4),
        "mean_ppi_recall": round(mean_ppi, 4),
        "overall_accuracy": round(mean_ov, 2),
        "scores": rows,
    }, indent=2), encoding="utf-8")

    print(f"\n  Saved to:")
    print(f"    {out_txt}")
    print(f"    {out_json}")


if __name__ == "__main__":
    main()