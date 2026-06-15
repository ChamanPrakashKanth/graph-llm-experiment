# ============================================================
# Train Mechanical Engineering Domain (1000 Chunks)
# Second-Order Differential Loss Experiment
# ============================================================
# Usage: powershell -ExecutionPolicy Bypass -File scripts/train_mechanical_engineering.ps1
# ============================================================

$ErrorActionPreference = "Stop"
$python = ".\.venv\Scripts\python.exe"

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  VLCM Mechanical Engineering Training Pipeline" -ForegroundColor Cyan
Write-Host "  1000-Chunk Dataset + Second-Order Differential Loss" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

# ----------------------------------------------------------
# Step 1: Grow the GNN concept graph from ME docs
# ----------------------------------------------------------
Write-Host "`n>>> Step 1: Growing GNN concept graph from mechanical engineering docs..." -ForegroundColor Yellow
& $python vlcm/run_vlcm.py grow-graph `
    --dataset data/mechanical_engineering_dataset.json `
    --documents data/mechanical_engineering_docs.txt `
    --output data/mechanical_engineering_graph.json `
    --png reports/mechanical_engineering_graph.png `
    --window 3

# ----------------------------------------------------------
# Step 2: Train BASELINE (no second-order loss)
# ----------------------------------------------------------
Write-Host "`n>>> Step 2: Training BASELINE model (no second-order loss)..." -ForegroundColor Yellow
& $python vlcm/run_vlcm.py train `
    --dataset data/mechanical_engineering_dataset.json `
    --checkpoint-dir checkpoints/vlcm_mech_baseline `
    --epochs 30 `
    --batch-size 8 `
    --lr 3e-4 `
    --path-length 8 `
    --concept-dim 128 `
    --hidden-size 128 `
    --graph-layers 2 `
    --second-order-weight 0.0

# ----------------------------------------------------------
# Step 3: Evaluate BASELINE
# ----------------------------------------------------------
Write-Host "`n>>> Step 3: Evaluating BASELINE..." -ForegroundColor Yellow
& $python vlcm/run_vlcm.py evaluate `
    --dataset data/mechanical_engineering_dataset.json `
    --checkpoint-dir checkpoints/vlcm_mech_baseline

# ----------------------------------------------------------
# Step 4: Train with SECOND-ORDER DIFFERENTIAL LOSS
# ----------------------------------------------------------
Write-Host "`n>>> Step 4: Training with SECOND-ORDER DIFFERENTIAL loss (weight=0.1)..." -ForegroundColor Green
& $python vlcm/run_vlcm.py train `
    --dataset data/mechanical_engineering_dataset.json `
    --checkpoint-dir checkpoints/vlcm_mech_2nd_order `
    --epochs 30 `
    --batch-size 8 `
    --lr 3e-4 `
    --path-length 8 `
    --concept-dim 128 `
    --hidden-size 128 `
    --graph-layers 2 `
    --second-order-weight 0.1

# ----------------------------------------------------------
# Step 5: Evaluate SECOND-ORDER model
# ----------------------------------------------------------
Write-Host "`n>>> Step 5: Evaluating SECOND-ORDER model..." -ForegroundColor Green
& $python vlcm/run_vlcm.py evaluate `
    --dataset data/mechanical_engineering_dataset.json `
    --checkpoint-dir checkpoints/vlcm_mech_2nd_order

# ----------------------------------------------------------
# Step 6: Run benchmarks
# ----------------------------------------------------------
Write-Host "`n>>> Step 6: Running benchmarks..." -ForegroundColor Yellow
& $python vlcm/run_vlcm.py benchmark

# ----------------------------------------------------------
# Step 7: Sample inference
# ----------------------------------------------------------
Write-Host "`n>>> Step 7: Sample inferences..." -ForegroundColor Yellow
& $python vlcm/run_vlcm.py infer `
    --dataset data/mechanical_engineering_dataset.json `
    --checkpoint-dir checkpoints/vlcm_mech_2nd_order `
    --question "Why does a column buckle under compression?"

& $python vlcm/run_vlcm.py infer `
    --dataset data/mechanical_engineering_dataset.json `
    --checkpoint-dir checkpoints/vlcm_mech_2nd_order `
    --question "How does thermal stress cause cracking?"

Write-Host "`n============================================================" -ForegroundColor Cyan
Write-Host "  Training Pipeline Complete!" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
