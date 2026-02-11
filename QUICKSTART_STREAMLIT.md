# Quick Start Guide - Streamlit Dashboard Implementation

## What Was Implemented

### 1. Dark-Themed Chart Functions ✅
**File**: `src/utils_files.py`

- Updated `generate_perf_chart()` with dark background (#0d0d0d)
- Updated `generate_pnl_chart()` with dark background
- Updated `generate_daily_perf_chart()` with dark background
- Added `extract_perf_chart_data()` - JSON serialization
- Added `extract_pnl_chart_data()` - JSON serialization
- Added `extract_daily_perf_chart_data()` - JSON serialization

**Colors**:
- Background: `#0d0d0d` (dark gray)
- Perf line: `#00d4ff` (cyan)
- PnL line: `#00ff41` (bright green)
- Positive bars: `#00ff41` (bright green)
- Negative bars: `#ff4444` (bright red)

### 2. Graph Data Storage ✅
**File**: `src/processors/web_processor.py`

- Added `self.graph_data = {}` dictionary to store JSON-serializable data
- Structure: `{session: {account_key: {perf: {...}, pnl: {...}, daily: {...}, timestamp: ...}}}`
- Automatic extraction and storage during AUM updates

### 3. Data Extraction Integration ✅
**File**: `src/processors/web_processor.py` - `update_aum()` method

When 180-day graphs are generated:
- Extracts performance data via `extract_perf_chart_data(perfcum)`
- Extracts PnL data via `extract_pnl_chart_data(pnlcum)`
- Stores with timestamp for tracking

When 30-day graphs are generated:
- Extracts daily performance via `extract_daily_perf_chart_data(daily)`
- Appends to existing graph_data entry

### 4. API Endpoints ✅
**File**: `src/web_api.py`

Two new REST endpoints:

#### GET /api/available-accounts
Returns:
```json
{
  "sessions": ["binance", "okx"],
  "accounts": {"binance": ["strat1"], ...},
  "graph_data_available": {...}
}
```

#### GET /api/graph-data/{session}/{account_key}
Returns:
```json
{
  "session": "binance",
  "account_key": "binance_1",
  "timestamp": "2026-02-02T12:34:56+00:00",
  "data": {
    "perf": {"timestamps": [...], "values": [...]},
    "pnl": {"timestamps": [...], "values": [...]},
    "daily": {"timestamps": [...], "values": [...]}
  }
}
```

### 5. Streamlit Dashboard ✅
**File**: `streamlit_dashboard.py`

Interactive web dashboard featuring:
- **Session & Account Selection** (sidebar dropdowns)
- **3 Main Chart Tabs**:
  - Performance Chart (cyan line)
  - PnL Chart (green line)
  - Daily Performance (green/red bars)
- **Info Tab** with metrics
- **Auto-refresh** with configurable intervals
- **Smart Caching** (60-second TTL)
- **Dark Theme** throughout

## Getting Started

### Step 1: Start the Backend

```bash
python src/web_api.py --config config/web_processor.yml
```

Backend will:
- Load configuration
- Initialize file watchers
- Watch for AUM/PnL file changes
- Extract and cache graph data
- Serve API on `http://localhost:14440`

### Step 2: Start the Streamlit Dashboard

```bash
streamlit run streamlit_dashboard.py
```

Dashboard will:
- Open at `http://localhost:8501`
- Connect to backend API
- Display session/account selectors
- Show interactive charts with auto-refresh

### Step 3: Trigger Data Updates

The system uses a **10-minute delay per file change**. To test:

1. Monitor a live bot's AUM file
2. After 10 minutes of no changes, wait for graph generation
3. Dashboard will show newly available data
4. Click "Refresh Now" to see updates immediately

## File Changes Summary

| File | Changes | Lines |
|------|---------|-------|
| `src/utils_files.py` | Dark-themed charts + data extraction | +100 lines |
| `src/processors/web_processor.py` | Graph data storage + extraction calls | +40 lines |
| `src/web_api.py` | API endpoints for graph data | +50 lines |
| `streamlit_dashboard.py` | NEW - Complete Streamlit app | 350 lines |
| `.streamlit/secrets.toml` | NEW - Streamlit config | 2 lines |
| `STREAMLIT_DASHBOARD.md` | NEW - Full documentation | 250 lines |

## Architecture

```
File Changes (AUM/PnL)
        ↓
File Watcher (10-min delay)
        ↓
web_processor.update_aum()
        ↓
extract_*_chart_data()
        ↓
self.graph_data[session][account] = {...}
        ↓
API: /api/graph-data/{session}/{account}
        ↓
Streamlit Dashboard (plotly charts)
```

## Verification Checklist

- [ ] `utils_files.py` has dark-themed `generate_*_chart()` functions
- [ ] `utils_files.py` has new `extract_*_chart_data()` functions
- [ ] `web_processor.py` imports the extract functions
- [ ] `web_processor.py` has `self.graph_data = {}` in `__init__`
- [ ] `web_processor.py` calls extract functions in `update_aum()`
- [ ] `web_api.py` has `/api/graph-data/{session}/{account_key}` endpoint
- [ ] `web_api.py` has `/api/available-accounts` endpoint
- [ ] `streamlit_dashboard.py` exists in root directory
- [ ] `.streamlit/secrets.toml` exists with API_BASE_URL

## Testing Commands

```bash
# Test API endpoints directly
curl http://localhost:14440/api/available-accounts
curl http://localhost:14440/api/graph-data/binance/binance_1

# Test chart rendering (dark background)
# Charts are saved as HTML files in temp/ directory with dark styling

# Monitor file changes (10-minute delay applies)
# After AUM file changes, wait 10 minutes for processing
```

## Next Steps

1. **Monitor Backend Logs**
   - Look for "Stored graph data for" messages
   - Indicates successful data extraction

2. **Wait for Initial Data**
   - System needs time to accumulate 180-day and 30-day data
   - First graphs appear after these time periods

3. **Customize Dashboard** (Optional)
   - Modify chart colors in `create_*_chart()` functions
   - Add additional metrics in Info tab
   - Adjust refresh intervals

4. **Deploy** (Optional)
   - Run Streamlit on cloud server
   - Update `.streamlit/secrets.toml` with production API URL
   - Configure with reverse proxy (nginx/Apache)

## Support

For issues:
1. Check backend logs for errors
2. Verify API is responding: `curl http://localhost:14440/api/available-accounts`
3. Ensure data exists: Check `self.aum` and `self.graph_data` in web_processor logs
4. Restart both backend and dashboard if needed

---

**Implementation Complete!** ✅

The system now:
- ✅ Generates dark-themed charts for HTML frontend
- ✅ Extracts JSON data for Streamlit visualization
- ✅ Stores data with 10-minute delay per file change
- ✅ Exposes via REST API
- ✅ Provides interactive Streamlit dashboard
