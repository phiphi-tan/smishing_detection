# Smishing Detection via Fact-Checking Perspective

This repo uses an agentic approach to verify claims made in smishing (SMS phishing)

## Setup

1. Initialise virtual environment

```
uv init myproj
```

2. Install dependencies from `uv.lock` to ensure consistent versions:

```
uv sync
```

## Running

1. Move into src folder (output paths are hardcoded)

```
cd src/
```

2. run the program inside the virtual environment

```
uv run python detection.py
```

## Project Structure

```bash
smishing_detection/
├── src/
│   ├── detection.py # the main detection code
│   └── agents.py # definitions of the agents and related tools
│   └── data_processing.py # obtain statistics from the results
├── data/
│   └── D2.csv # database of 50 phishing messages
│   └── agent1.ndjson # output from claim extraction agent
│   └── smishing_output.ndjson # final output
```

## Results

The following results were obtained from running `src/data_processing.py` on the obtained `data/smishing_output.ndjson`

```
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
