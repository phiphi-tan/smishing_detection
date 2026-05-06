# Smishing Detection via Fact-Checking Perspective

This repo uses an agentic approach to verify claims made in smishing (SMS phishing).

## Setup

1. Install dependencies from `uv.lock` to ensure consistent versions:

```powershell
uv sync
```

Do not run `uv init` for this cloned project; that command is for creating a new project.

## Running

Run the program inside the virtual environment:

```powershell
uv run python -m src.detection
```

To process a subset of rows:

```powershell
uv run python -m src.detection --start-index 0 --limit 1
```

To summarize existing outputs:

```powershell
uv run python src/data_processing.py
```

Each detection run writes to a new timestamped directory under `data/output/`:

```text
data/output/YYYYMMDD_HHMMSS_microseconds/
+-- claims.ndjson
+-- output.ndjson
```

To summarize a specific run:

```powershell
uv run python src/data_processing.py --run-dir data/output/YYYYMMDD_HHMMSS_microseconds
```

## Project Structure

```text
smishing_detection/
+-- src/
|   +-- tools.py
|   +-- detection.py
|   +-- agents.py
|   +-- data_processing.py
|   `-- schemas.py
+-- data/
|   +-- D2.csv
|   `-- output/<timestamp>/claims.ndjson
|   `-- output/<timestamp>/output.ndjson
+-- pyproject.toml
`-- uv.lock
```

## Results

The following results were obtained from running `src/data_processing.py` on the obtained `data/smishing_output.ndjson`.

```text
=== Message-level Stats ===
Total Messages: 50
Average claims per message: 8

Messages with at least 1 "False" claim: 29
Messages with only "Extra Evidence Needed" claims: 16
Messages with only "True" claims: 0
Other messages: 5

=== Claim-level Stats ===
Total Claims: 447

Category: Verifiable (Total claims: 163)
  False: 69
  Unsure: 14
  True: 24
  Extra evidence needed: 56

Category: Unverifiable (Total claims: 284)
  Extra evidence needed: 276
  False: 1
  True: 7
```
