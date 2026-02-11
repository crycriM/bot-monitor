# Quick Reference Card - Integrated Dashboard

## Run the Application

```bash
# Terminal 1: Backend API
python src/web_api.py --config config/web_processor.yml

# Terminal 2: Streamlit App
streamlit run src/web_front_sl.py -- --config config/web_processor.yml

# Browser
http://localhost:8880
```

## Application Tabs

| Tab | Icon | Features |
|-----|------|----------|
| PnL | 📊 | P&L metrics, cumulative returns |
| Matching | 🔄 | Position comparison (Real vs Theo) |
| **AUM & Perf** | **📈** | **NEW! Interactive graphs** |

## New Tab: AUM & Performance

### Layout
```
Session: [Dropdown]    Account: [Dropdown]

[Performance] [PnL] [Daily Performance]

[Interactive Charts with Plotly]

Data Summary:
  Perf Points: XXX  |  PnL Points: XXX  |  Daily Points: XXX
```

### Features
- **Performance Chart**: Cumulative return (cyan line)
- **PnL Chart**: Profit/loss (green line)
- **Daily Chart**: Daily returns (green/red bars)
- **Interactions**: Zoom, pan, hover tooltips
- **Metrics**: Data point counts, last update

## API Endpoints

```
GET /api/available-accounts
├─ Returns: {sessions, accounts, graph_data_available}
└─ Cache: 60 seconds

GET /api/graph-data/{session}/{account_key}
├─ Returns: {session, account_key, timestamp, data}
└─ Data: {perf: {...}, pnl: {...}, daily: {...}}
└─ Cache: 60 seconds
```

## Color Scheme

| Element | Color | Hex |
|---------|-------|-----|
| Background | Dark Gray | #0d0d0d |
| Performance Line | Cyan | #00d4ff |
| PnL Line | Bright Green | #00ff41 |
| Profit Bars | Green | #00ff41 |
| Loss Bars | Red | #ff4444 |

## File Structure

```
src/web_front_sl.py
├─ Tab 1: create_pnl_tab() [existing]
├─ Tab 2: create_matching_tab() [existing]
└─ Tab 3: create_graph_tab() [NEW]
   ├─ fetch_available_accounts_for_graphs()
   ├─ fetch_graph_data()
   ├─ create_perf_chart()
   ├─ create_pnl_chart()
   └─ create_daily_perf_chart()
```

## Key Statistics

- **Lines Added**: ~260
- **Functions Added**: 6
- **New Tabs**: 1
- **Files Modified**: 1
- **Breaking Changes**: 0
- **Backward Compatibility**: 100%

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "Unable to connect" | Backend not running? Check port 14440 |
| "No sessions available" | Backend initializing, wait 10 seconds |
| "No graph data available" | Data not generated yet, AUM files need processing |
| Charts not displaying | Ensure Plotly installed: `pip install plotly` |
| Slow performance | Increase cache TTL or wait 60 seconds |

## Environment Variables (Optional)

```bash
export BOT_MONITOR_GATEWAY_PORT=14440
export BOT_MONITOR_LOG_LEVEL=INFO
export BOT_MONITOR_CACHE_TTL=60
```

## Documentation Files

- `INTEGRATION_SUMMARY.md` - Architecture details
- `RUNNING_INTEGRATED_APP.md` - Setup & advanced usage
- `INTEGRATION_COMPLETE.md` - Full overview
- `INTEGRATION_VERIFICATION_CHECKLIST.md` - Verification details

## Next Steps

1. ✅ Start backend API
2. ✅ Start Streamlit app
3. ✅ Open http://localhost:8880
4. ✅ Click "📈 AUM & Performance" tab
5. ✅ Select session and account
6. ✅ View interactive charts

## Support Commands

```bash
# Test API endpoints
curl http://localhost:14440/api/available-accounts | jq
curl http://localhost:14440/api/graph-data/binance/binance_1 | jq

# View logs
tail -f output/web_processor.log

# Run verification
python test_streamlit_implementation.py
```

---

**Status**: ✅ Complete - Graph dashboard integrated into web_front_sl.py
