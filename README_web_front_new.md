# Web Front New - NiceGUI Dashboard

Modern minimalist web dashboard built with NiceGUI for monitoring trading bot positions, PnL, and executing position operations.

## Features

### 🎯 Three Main Tabs

1. **PnL Tab**: View profit and loss metrics across all strategies
   - Click 'Get PnL' to fetch current performance data
   - Interactive sortable tables with filtering
   - Mean theoretical PnL and pivot summaries

2. **Matching Tab**: Compare theoretical vs actual positions
   - Select account from dropdown
   - View exposure summaries (net/gross)
   - See mismatched and dust positions
   - Interactive position tables with sorting
   - Embedded HTML charts from temp/ folder

3. **Multiply Tab**: Scale positions by a factor
   - Select account from dropdown
   - Adjust multiplication factor (0.0 - 2.0) with slider
   - Execute with confirmation button
   - Factor 1.0 = no change, 0.0 = liquidate all, 2.0 = double positions

## Installation

Ensure NiceGUI is installed:

```bash
poetry add nicegui
# or
pip install nicegui
```

## Usage

### Basic Launch

```bash
python src/web_front_new.py --config config/web_processor.yml --port 8880 --gw_port 14440
```

### Parameters

- `--config`: Path to YAML config file containing session configurations (required)
- `--port`: Port for web dashboard (default: 8880)
- `--gw_port`: Port of the backend API gateway (default: 14440)

### Example

```bash
# Launch dashboard on port 8880, connecting to API on port 14440
python src/web_front_new.py \
    --config config/web_processor.yml \
    --port 8880 \
    --gw_port 14440
```

Then open browser to: `http://localhost:8880`

## Configuration

The dashboard reads the same config file as the original `web_front.py`:

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
- Only strategies with `active=true` and `send_orders!='dummy'` appear in dropdowns

## API Endpoints Used

The dashboard connects to these backend endpoints:

- `GET /pnl` - Retrieve PnL data for all strategies
- `GET /matching?session=X&account_key=Y` - Get position matching comparison
- `GET /multiply?exchange=X&account=Y&factor=Z` - Execute position multiplication

Backend must be running on `--gw_port` (default 14440).

## UI Components

### Modern Features
- **Responsive Design**: Mobile-friendly tables and cards
- **AgGrid Tables**: Interactive sorting, filtering, and resizing
- **Cards**: Clean visual hierarchy with nested sections
- **Notifications**: Toast messages for success/error feedback
- **Spinners**: Loading indicators during API calls

### Color Scheme
- Primary: Blue (#1976d2)
- Secondary: Teal (#26a69a)
- Accent: Purple (#9c27b0)
- Warnings: Orange (for mismatches)

## Differences from Original web_front.py

| Feature | Original (PyWebIO) | New (NiceGUI) |
|---------|-------------------|---------------|
| Account Selection | Button grid | Dropdown select |
| Error Display | Inline text | Toast notifications |
| Tables | HTML rendered | Interactive AgGrid |
| Responsiveness | Basic CSS | Built-in responsive |
| Factor Input | Slider with inline display | Slider with label update |
| Layout | Scopes | Cards with nesting |

## Troubleshooting

### Port Already in Use
```bash
# Change the port
python src/web_front_new.py --config config/web_processor.yml --port 8881
```

### No Accounts Appear
- Check config file path is correct
- Verify session configs exist at specified paths
- Ensure strategies have `active=true` and `send_orders!='dummy'`

### API Connection Errors
- Verify backend is running on `--gw_port`
- Check GATEWAY setting in console output
- Test endpoints manually: `curl http://localhost:14440/status`

### Import Errors
```bash
# Install missing dependencies
poetry install
# or
pip install nicegui pyyaml pandas requests
```

## Development

### Testing Helper Functions

```bash
python test_web_front_new.py
```

### Code Structure

```
web_front_new.py
├── Globals: CONFIG, GATEWAY
├── initialize_globals() - Load config and set API gateway
├── Helper Functions
│   ├── get_any() - API request wrapper
│   ├── dict_to_df() - Convert nested dicts to DataFrames
│   ├── multistrategy_matching_to_df() - Parse matching data
│   └── get_used_accounts() - Extract active accounts from configs
├── Tab Creation
│   ├── create_pnl_tab() - PnL performance view
│   ├── create_matching_tab() - Position matching view
│   └── create_multiply_tab() - Position multiplication controls
└── main_app() - Root page with tab navigation
```

## Performance Notes

- No auto-refresh: Manual button clicks only (as requested)
- No session persistence: State resets on page reload (as requested)
- AgGrid handles large tables efficiently with virtual scrolling
- HTML figures loaded from disk, not regenerated

## Future Enhancements (Not Implemented)

- Auto-refresh toggle with configurable interval
- Session state persistence across tabs
- Real-time WebSocket updates
- Position history charts
- Export tables to CSV/Excel
- Dark mode toggle

## License

Same as parent project (MIT)
