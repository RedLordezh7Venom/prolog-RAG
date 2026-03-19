# PowerShell script for Phase 1 Integration Test
Write-Host "------------------------------------------------" -ForegroundColor Cyan
Write-Host "PHASE 1 INTEGRATION TEST" -ForegroundColor Cyan
Write-Host "------------------------------------------------" -ForegroundColor Cyan

$tests = @(
    @{ Name = "Basic Prolog Connectivity"; File = "test_prolog.py"; Module = $false },
    @{ Name = "Fact Extraction Logic"; File = "prolog_rag_project.core.fact_extractor"; Module = $true },
    @{ Name = "Prolog Knowledge Base & Rules"; File = "prolog_rag_project.core.prolog_kb"; Module = $true },
    @{ Name = "Query Routing"; File = "prolog_rag_project.core.query_router"; Module = $true },
    @{ Name = "Full Prolog-RAG Pipeline"; File = "prolog_rag_project.core.prolog_rag"; Module = $true }
)

$i = 1
foreach ($test in $tests) {
    Write-Host "`n[$i/5] Testing $($test.Name)..." -ForegroundColor Yellow
    if ($test.Module) {
        uv run python -m $($test.File)
    } else {
        uv run python $($test.File)
    }
    $i++
}

Write-Host "`n------------------------------------------------" -ForegroundColor Green
Write-Host "TESTS COMPLETE ✅" -ForegroundColor Green
Write-Host "------------------------------------------------" -ForegroundColor Green
