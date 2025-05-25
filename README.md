# SPDB

## Backend
    cd backend
    uv sync
    source .venv/bin/activate
    uvicorn app.main:app --reload

## Frontend
    cd frontend
    python -m http.server 5500
    http://127.0.0.1:5500/index.html

## Testy wydajności
    cd backend
    uv run -m app.test.performance.run_all

## Testy jednostkowe
    cd backend
    pytest

## Generowanie wykresów
    backend/visualisation/number_of_districts.ipynb
    backend/visualisation/number_of_nodes.ipynb
    backend/visualisation/number_of_processes.ipynb
