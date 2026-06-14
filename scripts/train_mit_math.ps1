# =============================================================================
# Train CAT V2 on MIT OCW Engineering Mathematics
# Covers: 18.01, 18.02, 18.03, 18.06
# =============================================================================

Write-Host "=============================================" -ForegroundColor Cyan
Write-Host " MIT OCW Engineering Mathematics Training" -ForegroundColor Cyan
Write-Host " 18.01 + 18.02 + 18.03 + 18.06" -ForegroundColor Cyan
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host ""

$python = ".\.venv\Scripts\python.exe"

# Step 1: Grow the GNN graph from dataset + documentation
Write-Host "[1/4] Growing GNN concept graph from dataset + docs..." -ForegroundColor Yellow
& $python run_reasoning.py grow-graph `
    --dataset data/mit_math_dataset.json `
    --documents data/mit_math_docs.txt `
    --output data/mit_math_graph.json `
    --png reports/mit_math_graph.png

Write-Host ""

# Step 2: Train the model
Write-Host "[2/4] Training CAT V2 on MIT Math domain (30 epochs)..." -ForegroundColor Yellow
& $python run_reasoning.py train `
    --dataset data/mit_math_dataset.json `
    --checkpoint-dir checkpoints/cat_v2_mit_math `
    --graph-file data/mit_math_graph.json `
    --epochs 30

Write-Host ""

# Step 3: Evaluate
Write-Host "[3/4] Evaluating trained model..." -ForegroundColor Yellow
& $python run_reasoning.py evaluate `
    --dataset data/mit_math_dataset.json `
    --checkpoint-dir checkpoints/cat_v2_mit_math `
    --graph-file data/mit_math_graph.json

Write-Host ""

# Step 4: Sample inferences
Write-Host "[4/4] Running sample inferences across all 4 courses..." -ForegroundColor Green

$questions = @(
    "Why does the chain rule decompose composite function derivatives?",
    "How does the divergence theorem relate volume and surface integrals?",
    "Why does resonance cause unbounded growth in forced oscillators?",
    "How does eigenvalue decomposition diagonalize a matrix?",
    "Why does the Fundamental Theorem of Calculus connect derivatives and integrals?",
    "How does the matrix exponential from linear algebra solve systems of differential equations?"
)

foreach ($q in $questions) {
    Write-Host ""
    Write-Host "--- QUERY ---" -ForegroundColor Magenta
    & $python run_reasoning.py infer `
        --dataset data/mit_math_dataset.json `
        --checkpoint-dir checkpoints/cat_v2_mit_math `
        --graph-file data/mit_math_graph.json `
        --question $q
    Write-Host ""
}

Write-Host "=============================================" -ForegroundColor Cyan
Write-Host " Training Complete!" -ForegroundColor Cyan
Write-Host "=============================================" -ForegroundColor Cyan
