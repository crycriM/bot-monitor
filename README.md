# Streamlit Dashboard - Trading Bot Monitor

Modern interactive web dashboard built with Streamlit for monitoring trading bot positions, PnL, and executing position operations.

## Features

### 🎯 Three Main Tabs

1. **PnL Tab** (📊): View profit and loss metrics across all strategies
   - Click 'Get PnL' button to fetch current performance data
   - Display mean theoretical PnL table with all indicators
   - Pivot table summary for comparative analysis
   - Session state persists data within the session

2. **Matching Tab** (🔄): Compare theoretical vs actual positions
   - Select account from dropdown menu
   - View exposure summaries (net & gross)
   - Identify mismatched and dust positions
   - Interactive sortable tables rendered by Streamlit
   - Embedded HTML charts from temp/ folder
   - Position count summary (long vs short)

3. **Multiply Tab** (✖️): Scale positions by a factor
   - Select account from dropdown menu
   - Adjust multiplication factor (0.0 - 2.0) with interactive slider
   - Real-time factor display (0.0 = liquidate all, 1.0 = no change, 2.0 = double)
   - Execute with primary action button
   - See execution results directly in the UI

## Installation

Ensure Streamlit is installed:

```bash
poetry install
# or individually
pip install streamlit pyyaml pandas requests
```

The `pyproject.toml` includes streamlit ^1.53.

## Usage

### Basic Launch

```bash
streamlit run src/web_front_sl.py -- --config config/web_processor.yml --gw_port 14440
```

### Parameters

- `--config`: Path to YAML config file containing session configurations (required)
- `--gw_port`: Port of the backend API gateway (default: 14440)

### Port Configuration

Streamlit uses a different port configuration approach:

```bash
# Use default Streamlit port (8501)
streamlit run src/web_front_sl.py -- --config config/web_processor.yml --gw_port 14440

# Use custom port with --server.port
streamlit run src/web_front_sl.py \
    --server.port 8880 \
    -- --config config/web_processor.yml --gw_port 14440
```

Then open browser to: 
- Default: `http://localhost:8501`
- Custom: `http://localhost:8880`

### Shell Script (Optional)

You can create a shell script to simplify launching:

```bash
#!/bin/bash
streamlit run src/web_front_sl.py \
    --server.port 8501 \
    -- --config config/web_processor.yml --gw_port 14440
```

## Configuration

The dashboard reads the same config file as other web frontends:

```yaml
session:
  binance:
    config_file: 'path/to/binance_config.json'
    config_position_matching_file: 'path/to/matching_config.json'
  bitget:
    config_file: 'path/to/bitget_config.json'
    config_position_matching_file: 'path/to/matching_config.json'
```

Each session config JSON should contain:
- `strategy`: Dictionary of strategies with `active`, `account_trade`, `exchange_trade`, `send_orders` fields
- Only strategies with `active=true` and `send_orders!='dummy'` appear in account dropdowns

## API Endpoints Used

The dashboard connects to these backend endpoints:

- `GET /pnl` - Retrieve PnL data for all strategies
- `GET /matching?session=X&account_key=Y` - Get position matching comparison
- `GET /multiply?exchange=X&account=Y&factor=Z` - Execute position multiplication

Backend must be running on `--gw_port` (default 14440).

## UI Components

### Streamlit Features
- **Tabs**: Tab-based navigation for different views (native st.tabs)
- **Selectbox**: Dropdown menus for account selection
- **Slider**: Interactive range control for multiplication factor
- **DataFrames**: Interactive tables with sorting capabilities (st.dataframe)
- **Spinners**: Loading indicators during API calls (st.spinner)
- **Notifications**: Success/error messages (st.success, st.error)
- **HTML Components**: Embedded charts using st.components.v1.html
- **Dividers**: Section separators for better visual organization

### Responsive Design
- Wide layout by default for better table visibility
- Expandable sidebar with configuration info
- Mobile-friendly with responsive tables
- Auto-height adjustment for dataframes

## Differences from Original web_front.py

| Feature | Original (PyWebIO) | Streamlit |
|---------|-------------------|-----------|
| Framework | PyWebIO | Streamlit |
| Account Selection | Button grid | Dropdown select |
| Error Display | Inline text/popup | Toast notifications |
| Tables | HTML rendered | Interactive DataFrames |
| Responsiveness | Basic CSS | Built-in responsive |
| Factor Input | Text input | Slider widget |
| Session State | Manual handling | st.session_state |
| Port Configuration | `--port` argument | `--server.port` flag |
| Charts | HTML render | HTML component embed |

## Code Structure

```
web_front_sl.py
├── Globals: CONFIG, GATEWAY
├── initialize_globals() - Load config and set API gateway
├── Helper Functions
│   ├── get_any() - HTTP GET request wrapper
│   ├── dict_to_df() - Convert nested dicts to DataFrames
│   ├── replace_na_with_nan() - Clean data normalization
│   ├── multistrategy_matching_to_df() - Parse matching data into main + summary DFs
│   └── get_used_accounts() - Extract active accounts from configs
├── Tab Creation
│   ├── create_pnl_tab() - PnL performance view
│   ├── create_matching_tab() - Position matching view with charts
│   └── create_multiply_tab() - Position multiplication controls
└── main() - Root page setup and tab initialization
```

## Session State Management

Streamlit uses `st.session_state` to persist data within a user session:

- **PnL Tab**: Stores `pnl_df` and `pnl_pivot` after fetch
- **Matching Tab**: Stores `matching_main_df`, `matching_summary_df`, session, and account_key
- **Multiply Tab**: No persistent state (results shown immediately in markdown)

Data persists within the browser session but resets on page reload.

## Performance Notes

- **No auto-refresh**: Manual button clicks only (user controls when data is fetched)
- **Session-scoped state**: Data persists during session but resets on page reload
- **Efficient rendering**: Streamlit handles large dataframes with virtual scrolling
- **HTML charts**: Loaded from disk, not regenerated on each interaction
- **Responsive**: Wide layout optimized for desktop monitoring dashboards

## Troubleshooting

### Port Already in Use
```bash
# Use different port
streamlit run src/web_front_sl.py \
    --server.port 8502 \
    -- --config config/web_processor.yml
```

### No Accounts Appear
- Check config file path is correct
- Verify session configs exist at specified paths
- Ensure strategies have `active=true` and `send_orders!='dummy'`
- Check logs: `cat output/web_processor.log*`

### API Connection Errors
- Verify backend is running on `--gw_port`
- Test connection: `curl http://localhost:14440/pnl`
- Check firewall rules for port access

### Missing Dependencies
```bash
# Install all dependencies
poetry install
# or specific packages
pip install streamlit pyyaml pandas requests
```

### Streamlit Cache/Session Issues
```bash
# Clear Streamlit cache
streamlit cache clear

# Run with empty cache
streamlit run src/web_front_sl.py --logger.level=debug -- --config config/web_processor.yml
```

## Development & Testing

### Code Structure Notes
- Uses `pathlib.Path` for file operations
- All API calls wrapped in try-except with user feedback
- Session state checked before display to avoid errors
- Position filtering (dust, mismatches) done client-side

### Local Testing
```bash
# Test helper functions
python test_web_front_new.py

# Run with debug logging
streamlit run src/web_front_sl.py --logger.level=debug -- --config config/web_processor.yml
```

## Comparison with NiceGUI Version

If you were previously using [README_web_front_new.md](README_web_front_new.md) (NiceGUI version):

**Advantages of Streamlit:**
- Simpler, more lightweight
- Better table sorting and filtering
- Native session state management
- Easier deployment and scaling
- Active community and documentation
- Built-in responsive design

**Advantages of NiceGUI:**
- More fine-grained UI control
- Custom styling options
- Card-based layout system

## Performance Optimization

For better performance with large datasets:

1. **Filter before display**: Dust positions are filtered on load
2. **Limit table height**: Explicit heights set for readability
3. **Format numbers**: Use `.style.format()` for large numbers
4. **Cache config**: Load config once on startup

## Future Enhancements

- Auto-refresh toggle with configurable interval (using streamlit-autorefresh)
- Export tables to CSV/Excel
- Dark mode toggle (Streamlit theme config)
- Real-time updates via WebSocket
- Position change history/alerts
- Advanced filtering and search

## License

Same as parent project (MIT)
