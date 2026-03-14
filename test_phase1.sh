#!/bin/bash

# Handle uv pathing on Windows/Bash
UV_BIN="uv"
if ! command -v uv &> /dev/null
then
    # Try common Windows path for uv
    UV_BIN="/c/Users/prabh/AppData/Local/Programs/Python/Python312/Scripts/uv.exe"
fi

echo "------------------------------------------------"
echo "PHASE 1 INTEGRATION TEST"
echo "------------------------------------------------"

echo ""
echo "[1/5] Testing Basic Prolog Connectivity..."
$UV_BIN run python test_prolog.py

echo ""
echo "[2/5] Testing Fact Extraction Logic..."
$UV_BIN run python -m prolog_rag_project.core.fact_extractor

echo ""
echo "[3/5] Testing Prolog Knowledge Base & Rules..."
$UV_BIN run python -m prolog_rag_project.core.prolog_kb

echo ""
echo "[4/5] Testing Query Router..."
$UV_BIN run python -m prolog_rag_project.core.query_router

echo ""
echo "[5/5] Testing Full Prolog-RAG Pipeline..."
$UV_BIN run python -m prolog_rag_project.core.prolog_rag

echo ""
echo "------------------------------------------------"
echo "TESTS COMPLETE ✅"
echo "------------------------------------------------"
