You are an experienced, pragmatic software engineering AI agent. Do not over-engineer a solution when a simple one is possible. Keep edits minimal. If you want an exception to ANY rule, you MUST stop and get permission first.

## Project Overview

**bot-monitor** is a monitoring system for automated trading bots. It provides a web dashboard for viewing PnL (profit and loss), comparing theoretical vs actual positions, and executing position operations (multiply/liquidate) across multiple exchanges (Binance, Bitget, OKX).

### Technology Stack

- **Language**: Python >= 3.11
- **Package Manager**: Poetry
- **Backend API**: FastAPI + uvicorn
- **Dashboard**: Streamlit (primary), NiceGUI (legacy, commented out)
- **Data Processing**: pandas, numpy, matplotlib, plotly
- **File Watching**: watchdog
- **Web Framework**: pywebio (legacy)
- **Testing**: pytest + pytest-asyncio

### Architecture

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐
│  Trading    │────▶│   Backend    │────▶│  Dashboard   │
│  Bots       │     │  (FastAPI)   │     │ (Streamlit)  │
│  (outputs)  │◀────│  web_api.py  │◀────│  web_front   │
└─────────────┘     └──────────────┘     └──────────────┘
                         │
                         ▼
                   ┌──────────────┐
                   │ File Watcher │
                   │ (watchdog)   │
                   └──────────────┘
```

- **Backend** (`src/web_api.py`): FastAPI server exposing REST endpoints for PnL, positions, matching, and multiply operations. Runs as a background process.
- **Dashboard** (`src/web_front_sl.py`): Streamlit web app that connects to the backend API.
- **File Watcher** (`src/processors/file_watcher.py`): Monitors trading bot output files for changes and routes events to an asyncio queue.
- **Data Analyzer** (`src/data_analyzer/`): Position comparison logic (theoretical vs actual) with tolerance-based matching and dust detection.

## Reference

### Important Code Files

| File | Purpose |
|------|---------|
| `src/web_api.py` | FastAPI backend with REST endpoints (`/pose`, `/multiply`, `/status`, `/pnl`, `/aum`) |
| `src/web_front_sl.py` | Streamlit dashboard frontend (PnL, Matching, Multiply tabs) |
| `src/processors/web_processor.py` | Core processor: interfaces between web queries and strategy status/commands |
| `src/processors/file_watcher.py` | File system event handling with debouncing for bot output files |
| `src/processors/edo_processor.py` | EDO (execution) processor with killswitch support |
| `src/utils_files.py` | Utility functions: file reading, chart generation, JSONResponse class |
| `src/data_analyzer/aggregate_strategies.py` | Aggregates theoretical positions from multiple strategies |
| `src/data_analyzer/position_comparator.py` | Compares theoretical vs real positions with tolerance and dust detection |
| `src/edo_api.py` | Separate FastAPI for EDO processor (port 14040) |
| `config/web_processor.yml` | Main YAML config: sessions, log settings, pace intervals |
| `config/config_position_matching.json` | Position matching tolerance threshold |

### Important Directories

| Directory | Purpose |
|-----------|---------|
| `src/` | All source code |
| `src/processors/` | Processor modules (web, file watcher, EDO) |
| `src/data_analyzer/` | Position aggregation and comparison logic |
| `config/` | YAML and JSON configuration files |
| `output/` | Runtime log files (gitignored) |
| `temp/` | Generated chart HTML files (gitignored) |

### Project Architecture

The system follows a producer-consumer pattern:

1. **Producers**: Trading bots write state files (positions, PnL, AUM) to disk.
2. **File Watcher**: `watchdog` monitors these files for changes, debounces events, and pushes to an asyncio queue.
3. **Consumer**: `WebProcessor` consumes queue events, reads updated data, and serves it via FastAPI endpoints.
4. **Dashboard**: Streamlit app polls the FastAPI endpoints for real-time data.

Key design decisions:
- File-based communication between bots and monitor (no direct API calls to bots)
- Debouncing of file events (2s debounce, 600s delay for session files)
- Position matching uses USD amount comparison when prices are available, falls back to quantity comparison
- Dust positions: flagged when `strategy_count == 0` and USD amount < 20% of median position size

## Essential Commands

### Installation

```bash
poetry install
```

### Backend API

```bash
# Start backend (port 14440)
python src/web_api.py --config config/web_processor.yml

# Or use the launch script
LAUNCH_BACKEND=yes DASHBOARD=none ./_launch_dashboard.sh
```

### Dashboard

```bash
# Streamlit dashboard (port 8880)
streamlit run src/web_front_sl.py -- --config config/web_processor.yml --gw_port 14440

# With custom port
streamlit run src/web_front_sl.py --server.port 8501 -- --config config/web_processor.yml --gw_port 14440

# Or use the launch script
LAUNCH_BACKEND=yes DASHBOARD=streamlit ./_launch_dashboard.sh
```

### Full System

```bash
# Start backend + dashboard
./start_monitors.sh

# Stop both
./stop_monitors.sh
```

### Testing

```bash
# Run tests
poetry run pytest

# Run with verbose output
poetry run pytest -v
```

### Development

```bash
# Format (if using a formatter, add as needed)
# No formatter configured yet

# Lint (if using a linter, add as needed)
# No linter configured yet

# Clean build artifacts
rm -rf build/ dist/ *.egg-info/ __pycache__/ .pytest_cache/
```

## Patterns

### Adding a New API Endpoint

1. Add the endpoint in `src/web_api.py` using the `@app.get('/endpoint')` decorator.
2. Call the appropriate `WebProcessor` method.
3. Return `JSONResponse(report)` for consistent JSON serialization.

```python
@app.get('/new_endpoint')
async def new_endpoint(param: str = ''):
    report = processor.some_method(param)
    return JSONResponse(report)
```

### Dashboard Tab Implementation

1. Create a `create_<tab>_tab()` function in `src/web_front_sl.py`.
2. Use `st.session_state` for persistent data within the session.
3. Wrap API calls in try-except with `st.error()` for user feedback.
4. Use `st.dataframe()` for interactive tables.

### Position Comparison

- **Amount-based**: Preferred when prices are available (uses USD amounts with tolerance).
- **Quantity-based**: Fallback when prices are unavailable.
- **Dust detection**: `strategy_count == 0` AND USD amount < 20% of median position size.
- **Mismatch tracking**: Records start time of mismatches and calculates duration.

### File Watcher Pattern

```python
# File watcher maps watched files to (session, entity, file_type) tuples
watched_files = {
    file_path: (session, entity, SignalType.PNL_FILE)
}
# Events are debounced (2s) and delayed (600s for session files)
```

## Anti-patterns

- **Don't** hardcode exchange names or account keys — always use `WebSpreaderBroker.ACCOUNT_DICT`.
- **Don't** bypass the `JSONResponse` class for API responses — it ensures consistent serialization.
- **Don't** add new file watchers without debouncing — file systems generate many events.
- **Don't** modify position comparison logic without understanding the tolerance and dust detection rules.
- **Don't** use `print()` for logging — use the `LOGGER` from `logging.getLogger(__name__)`.
- **Don't** commit `poetry.lock` — it's gitignored to avoid lock conflicts.

## Code Style

- Follow PEP 8 conventions.
- Use type hints where practical.
- Use `pathlib.Path` for file operations (not `os.path`).
- Use `async/await` for async operations (not `threading`).
- Use `try-except` with specific exceptions where possible.
- Use `f-strings` for string formatting.

## Commit and Pull Request Guidelines

### Commit Messages

The project does not use conventional commits. Use descriptive but concise messages:

```
fix: handle missing price cache in position comparison
feat: add multiply endpoint to backend API
refactor: extract chart generation to utils_files
```

### Before Committing

1. Run tests: `poetry run pytest`
2. Ensure no uncommitted changes to `poetry.lock`
3. Ensure no sensitive data in config files (API keys, etc.)

### Pull Request Description

Include:
- What changed and why
- Any configuration changes (new config keys, changed defaults)
- Testing performed (manual or automated)
- Any breaking changes or migration steps
