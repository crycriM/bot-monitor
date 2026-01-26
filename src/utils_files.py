import os
import pandas as pd
import aiofiles
import typing
import json
from pathlib import Path
from starlette.concurrency import run_in_threadpool
from starlette.background import BackgroundTask
from datetime import datetime, timedelta, timezone
from io import StringIO
from starlette.responses import Response
from datafeed.utils_online import parse_pair, utc_ize, today_utc


def get_temp_dir():
    """Get the temp directory path from environment variable or use default.
    
    Returns:
        Path: The temp directory path
    """
    temp_dir = os.getenv('BOT_MONITOR_TEMP_DIR', 'temp')
    temp_path = Path(temp_dir)
    # Ensure directory exists
    temp_path.mkdir(parents=True, exist_ok=True)
    return temp_path


def last_modif(hearbeat_file):
    if os.path.exists(hearbeat_file):
        last_date = utc_ize(os.path.getmtime(hearbeat_file))
        now = today_utc()
        age = (now - last_date).total_seconds()
        return age
    else:
        return None

def date_parser(date_str: str):
    if '/' in date_str and '+' not in date_str:
        # Handle format like "2025/12/19 09:03:04.072418" without timezone
        return datetime.strptime(date_str, '%Y/%m/%d %H:%M:%S.%f').replace(tzinfo=timezone.utc)
    elif ':00+' in date_str:
        # Handle format like "2025-03-14 11:30:00+00:00"
        s2 = date_str.replace(' ', 'T')
    else:
        # Handle format like "2025-03-14 11:30:00.123456+00:00"
        s1 = date_str.split('+')
        s2 = s1[0].split('.')[0].replace(' ', 'T') + '+' + s1[1]
    return datetime.strptime(s2, '%Y-%m-%dT%H:%M:%S%z')

async def read_pnl_file(pnl_file):
    try:
        async with aiofiles.open(pnl_file, 'r') as myfile:
            content = await myfile.read()
        df = pd.read_csv(StringIO(content), sep=',', index_col=[0], converters={0: date_parser})
    except:
        df = pd.DataFrame()
    return df

async def read_aum_file(aum_file, exchange):
    try:
        async with aiofiles.open(aum_file, 'r') as myfile:
            content = await myfile.read()
        df = pd.read_csv(StringIO(content), sep=';', header=[0], converters={0:date_parser})
        if df.shape[1] == 2:
            df.columns = ['date', 'aum']
        elif df.shape[1] == 4:
            df.columns = ['date', 'free', 'total', 'asset']
            df['aum'] = df['total']
            if 'bitg' in exchange:
                df['aum'] += df['asset']
        else:
            df = pd.DataFrame()
    except:
        df = pd.DataFrame()

    return df

def read_pos_file(pos_filename):
    """
    Reads theo or real position file and extracts the latest position data.
    """
    with open(pos_filename, 'r') as myfile:
        lines = myfile.readlines()
        pos_str = lines[-1].strip() if lines else ''
        try:
            pos_data = {}
            if pos_str:
                parts = pos_str.split(';')
                if len(parts) < 2:
                    raise ValueError("Invalid format: missing semicolon")
                datetime_str = parts[0]
                data_str = parts[1]
                pairs = data_str.split(', ')
                for pair in pairs:
                    if ':' not in pair:
                        continue
                    else:
                        key, value = pair.replace('USDC:USDC', 'USDCUSDC').split(':')
                        key = key.strip("'").replace('USDCUSDC', 'USDC:USDC')
                    if key in ['equity', 'imbalance']:
                        continue
                    pos_data[key] = float(value)
        except Exception as e:
            raise FileNotFoundError(f'Error parsing pos file {pos_filename}: {e}')

    return pos_data

def read_pos_file_histo(pos_filename):
    """
    Reads theo or real position file and build a historical position DataFrame
    """
    with open(pos_filename, 'r') as myfile:
        lines = myfile.readlines()
        position_history = {}

        for line in lines:
            pos_str = line.strip()
            try:
                pos_data = {}
                datetime_str = ''
                if pos_str:
                    parts = pos_str.split(';')
                    if len(parts) < 2:
                        raise ValueError("Invalid format: missing semicolon")
                    datetime_str = parts[0]
                    data_str = parts[1]
                    pairs = data_str.split(', ')
                    for pair in pairs:
                        if ':' not in pair:
                            continue
                        else:
                            key, value = pair.replace('USDC:USDC', 'USDCUSDC').split(':')
                            key = key.strip("'").replace('USDCUSDC', 'USDC:USDC')
                        pos_data[key] = float(value)
                position_history[date_parser(datetime_str)] = pos_data
            except Exception as e:
                print(f'Error parsing line at {datetime_str}: {e}')

    position_history_df = pd.DataFrame.from_dict(position_history, orient='index').sort_index()

    return position_history_df

async def read_latent_file(latent_file):
    try:
        async with aiofiles.open(latent_file, 'r') as myfile:
            content = await myfile.read()
        df = pd.read_csv(StringIO(content), sep=';', header="infer", converters={0:date_parser})
        df.columns = ['ts', 'latent_return', 'latent_pnl']
    except:
        df = pd.DataFrame()

    return df

def generate_perf_chart(perfcum, session, account_key):
    """Generate cumulative performance chart and return base64 encoded HTML"""
    import matplotlib.pyplot as plt
    import base64
    from io import BytesIO

    fig, ax = plt.subplots()
    ax.plot(perfcum.index, perfcum.values)
    ax.set_xlabel('date')
    ax.set_ylabel('Cum perf')
    ax.set_title(f'{session}-{account_key}')
    ax.grid()
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)
    ax.legend()
    tmpfile = BytesIO()
    fig.savefig(tmpfile, format='png')
    encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')
    html = f'<html> <img src=\'data:image/png;base64,{encoded}\'></html>'
    plt.close()
    return html

def generate_pnl_chart(pnlcum, session, account_key):
    """Generate cumulative PnL chart and return base64 encoded HTML"""
    import matplotlib.pyplot as plt
    import base64
    from io import BytesIO

    fig, ax = plt.subplots()
    ax.plot(pnlcum.index, pnlcum.values)
    ax.set_xlabel('date')
    ax.set_ylabel('Cum pnl')
    ax.set_title(f'{session}-{account_key}')
    ax.grid()
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)
    ax.legend()
    tmpfile = BytesIO()
    fig.savefig(tmpfile, format='png')
    encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')
    html = f'<html> <img src=\'data:image/png;base64,{encoded}\'></html>'
    plt.close()
    return html

def generate_daily_perf_chart(daily, session, account_key):
    """Generate daily performance bar chart and return base64 encoded HTML"""
    import matplotlib.pyplot as plt
    import base64
    from io import BytesIO

    fig, ax = plt.subplots()
    ax.bar(x=daily.index, height=daily.values, color=daily.apply(lambda x:'red' if x<0 else 'green'))
    ax.set_xlabel('date')
    ax.set_ylabel('Daily perf')
    ax.set_title(f'{session}-{account_key}')
    ax.grid()
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)
    tmpfile = BytesIO()
    fig.savefig(tmpfile, format='png')
    encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')
    html = f'<html> <img src=\'data:image/png;base64,{encoded}\'></html>'
    plt.close()
    return html

def calculate_median_position_sizes(account_theo_pos, quotes):
    """Calculate median position size for each account based on theo positions and quotes"""
    import numpy as np
    median_sizes = {}
    for session in account_theo_pos:
        session_quotes = quotes.get(session, {})
        for account_key, theo_positions in account_theo_pos[session].items():
            position_sizes = []
            for coin, qty in theo_positions.items():
                price = session_quotes.get(coin)
                if price is not None and qty != 0:
                    amount = abs(qty * price)
                    position_sizes.append(amount)

            if len(position_sizes) > 0:
                median_sizes[account_key] = np.median(position_sizes)
            else:
                median_sizes[account_key] = 0
    return median_sizes

class JSONResponse(Response):
    media_type = "application/json"

    def __init__(
            self,
            content: typing.Any,
            status_code: int = 200,
            headers: typing.Mapping[str, str] | None = None,
            media_type: str | None = None,
            background: BackgroundTask | None = None,
    ) -> None:
        super().__init__(content, status_code, headers, media_type, background)

    def render(self, content: typing.Any) -> bytes:
        return json.dumps(
            content,
            ensure_ascii=False,
            allow_nan=True,
            indent=None,
            separators=(",", ":"),
        ).encode("utf-8")

