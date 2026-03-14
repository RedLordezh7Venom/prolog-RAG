#!/bin/bash

echo "------------------------------------------------"
echo "PHASE 1 INTEGRATION TEST"
echo "------------------------------------------------"

echo ""
echo "[1/5] Testing Basic Prolog Connectivity..."
python test_prolog.py

echo ""
echo "[2/5] Testing Fact Extraction Logic..."
python -m prolog_rag_project.core.fact_extractor

echo ""
echo "[3/5] Testing Prolog Knowledge Base & Rules..."
python -m prolog_rag_project.core.prolog_kb

echo ""
echo "[4/5] Testing Query Routing..."
python -m prolog_rag_project.core.query_router

echo ""
echo "[5/5] Testing Full Prolog-RAG Pipeline..."
python -m prolog_rag_project.core.prolog_rag

echo ""
echo "------------------------------------------------"
echo "TESTS COMPLETE ✅"
echo "------------------------------------------------"
