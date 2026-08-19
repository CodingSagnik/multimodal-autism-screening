# PowerShell Script to Create Multimodal Project Scaffolding
$directories = @(
    "data\raw\video",
    "data\raw\audio",
    "data\raw\text",
    "data\processed\video_landmarks",
    "data\processed\audio_features",
    "data\processed\text_embeddings",
    "src\data_processing",
    "src\models",
    "notebooks"
)

foreach ($dir in $directories) {
    New-Item -ItemType Directory -Path $dir -Force | Out-Null
}

Write-Host "Project scaffolding successfully created." -ForegroundColor Green
