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
├── data/
│   └── D2.csv # database of 50 phishing messages
│   └── agent1.ndjson # output from claim extraction agent
│   └── smishing_output.ndjson # final output
```
