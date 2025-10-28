# PowerShell Script for Energic Model Management

$PYTHON_PATH = "python"
$MODEL_SCRIPT = "main.py"
$TRAIN_SCRIPT = "nlp/train_ebt.py"

function Show-Help {
    Write-Host "Available commands:"
    Write-Host "  train    - Train the model"
    Write-Host "  run      - Run the main model"
    Write-Host "  clean    - Clean __pycache__ directories"
    Write-Host "  help     - Show this help message"
}

function Start-Training {
    Write-Host "Starting model training..."
    & $PYTHON_PATH $TRAIN_SCRIPT
}

function Start-Model {
    Write-Host "Running the model..."
    & $PYTHON_PATH $MODEL_SCRIPT
}

function Clear-Cache {
    Write-Host "Cleaning Python cache files..."
    Get-ChildItem -Path .\ -Include "__pycache__" -Directory -Recurse | Remove-Item -Recurse -Force
    Write-Host "Cache cleaned successfully!"
}

# Parse command line arguments
$command = $args[0]

switch ($command) {
    "train" { Start-Training }
    "run" { Start-Model }
    "clean" { Clear-Cache }
    default { Show-Help }
}