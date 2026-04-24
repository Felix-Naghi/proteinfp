$uids = (python batch_100.py | Select-Object -Last 1) -split " "
$total = $uids.Count
$i = 0
foreach ($uid in $uids) {
    $i++
    Write-Host "[$i/$total] $uid" -ForegroundColor Cyan
    python pipeline\fetch_structure.py --uniprot $uid 2>&1 | Out-Null
    python pipeline\physicochemical.py --uniprot $uid 2>&1 | Out-Null
    python pipeline\active_sites.py    --uniprot $uid 2>&1 | Out-Null
    python pipeline\esm2_embeddings.py --uniprot $uid 2>&1 | Out-Null
    python pipeline\homology.py        --uniprot $uid 2>&1 | Out-Null
    python pipeline\deepfri_go.py      --uniprot $uid 2>&1 | Out-Null
    python pipeline\clean_ec.py        --uniprot $uid 2>&1 | Out-Null
    python pipeline\foldseek.py        --uniprot $uid 2>&1 | Out-Null
    python pipeline\ppi_network.py     --uniprot $uid 2>&1 | Out-Null
    python pipeline\consensus.py       --uniprot $uid 2>&1 | Out-Null
    Write-Host "  done" -ForegroundColor Green
}
Write-Host "All done. Running validation..." -ForegroundColor Yellow
python validation\run_validation.py --score-only
