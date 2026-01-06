import os
import pandas as pd
import aiofiles
from datetime import datetime, timedelta, timezone
from io import StringIO
from datafeed.utils_online import parse_pair, utc_ize, today_utc


def last_modif(hearbeat_file):
    if os.path.exists(hearbeat_file):
        last_date = utc_ize(os.path.getmtime(hearbeat_file))
        now = today_utc()
        age = (now - last_date).total_seconds()
        return age
    else:
        return None

def date_parser(date_str: str):
    if ':00+' in date_str:
        s2 = date_str.replace(' ', 'T')
    else:
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

async def read_latent_file(latent_file):
    try:
        async with aiofiles.open(latent_file, 'r') as myfile:
            content = await myfile.read()
        df = pd.read_csv(StringIO(content), sep=';', header="infer", converters={0:date_parser})
        df.columns = ['ts', 'latent_return', 'latent_pnl']
    except:
        df = pd.DataFrame()

    return df
