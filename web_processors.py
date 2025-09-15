import json
import sys
import asyncio
import aiofiles
import threading
import logging
from logging.handlers import TimedRotatingFileHandler
import os
import base64
from io import BytesIO
from enum import Enum
import argparse
import traceback
import yaml
from math import isclose
import typing

from datetime import datetime, timedelta

from numba.np.math.numbers import real_divmod

try:
    from datetime import UTC
except:
    from datetime import timezone
    UTC = timezone.utc
from fastapi import FastAPI, HTTPException
from starlette.responses import Response, BackgroundTask
import uvicorn
from dotenv import load_dotenv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils_files import last_modif, read_pnl_file, read_aum_file, read_pos_file
from datafeed.utils_online import NpEncoder, parse_pair, today_utc
from datafeed.broker_handler import BrokerHandler, TRADED_ACCOUNT_DICT
from web_broker import WebSpreaderBroker
from reporting.bot_reporting import TGMessenger
# from data_analyzer.position_comparator import compare_positions

global app

SignalType = Enum('SignalType', [('STOP', 'stop'),
                                 ('REFRESH', 'refresh'),
                                 ('MATCHING', 'check_matching'),
                                 ('CHECKALL', 'check_all'),
                                 ('FILE', 'update_file'),
                                 ('PRICE_UPDATE', 'update_prices')])

"""
UPI:
/status
/pnl
/pose?exchange=x&account=n
/matching?exchange=bybit&strat=pairs1
/multiply?exchange=binance&account=cm1&factor=1.2
/multistrategy_position_details?(session: str = 'binance', account_key: str = 'bitget_2')
/multistrat_summary?(account_or_strategy, timeframe}
"""


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


class Processor:
    '''
    Responsibility: interface between web query and strat status and commands
    trigger a refresh of pairs (todo)
    get status of each strat
    get pos
    '''

    def __init__(self, config):
        self.processor_config = config['session']
        self.config_files = {session: self.processor_config[session]['config_file'] for session in self.processor_config}
        self.config_position_matching_files = {session: self.processor_config[session]['config_position_matching_file'] for session in self.processor_config}
        fmt = logging.Formatter('{asctime}:{levelname}:{name}:{message}', style='{')
        handler = TimedRotatingFileHandler(filename='output/web_processor.log',
                                           when="midnight", interval=1, backupCount=7)
        handler.setFormatter(fmt)
        logging.getLogger().setLevel(logging.INFO)
        logging.getLogger().addHandler(handler)

        self.session_configs = {}
        for session, config_file in self.config_files.items():
            if os.path.exists(config_file):
                logging.info(f'Reading session config {config_file}')
                with open(config_file, 'r') as myfile:
                    params = json.load(myfile)
                    self.session_configs[session] = params
            else:
                logging.warning(f'No config file for {session} with name {config_file}')

        self.session_matching_configs={}
        for session, config_matching_file in self.config_position_matching_files.items():
            if os.path.exists(config_matching_file):
                with open(config_matching_file, 'r') as myfile:
                    params = json.load(myfile)
                    self.session_matching_configs[session] = params
            else:
                logging.warning(f'No matching config file for {session} with name {config_matching_file}')

        self.aum = {}
        self.strategy_state = {}
        self.summaries = {}
        self.pnl = {}
        self.matching = {}
        self.dashboard = {}
        self.last_message_count = 0
        self.last_message = []
        self.used_accounts = {}  # key is session name, value is key=strat_name, value=account used
        self.account_theo_pos = {}
        self.account_real_pos = {}
        self.perimeters = {}
        self.quotes = {}
        self._init_accounts_by_strat()

    def _init_accounts_by_strat(self):
        """
        populate exchange accounts list
        """
        try:
            for session, session_params in self.session_configs.items():
                strategies = session_params['strategy']
                logging.info(f'Adding strategies for {session} session')

                for strategy_name, strategy_param in strategies.items():
                    logging.info(f'Adding strategy {strategy_name}')
                    strat_account = strategy_param['account_trade']
                    strat_exchange = strategy_param.get('exchange_trade', '')
                    active = strategy_param.get('active', False)
                    destination = strategy_param.get('send_orders', 'dummy')

                    if not active or destination == 'dummy':
                        continue
                    if session not in self.used_accounts:
                        self.used_accounts[session] = {}
                    self.used_accounts[session][strategy_name] = (strat_exchange, strat_account)
        except Exception as e:
            logging.error(f'Error updating accounts by strat: {e.args[0]}')
            logging.error(traceback.format_exc())
            self.used_accounts = {}
        logging.info(f'Built account dict {self.used_accounts}')

    def update_account_multi(self):
        """
        Update account theo, aggregated theo and real positions from bot for all sessions and strategies.
        Retrieve account positions form live file and from exchange independently
        """
        account_list = {}
        strategy_positions = {}

        for session, accounts in self.used_accounts.items():
            logging.info(f'Updating account positions for {session}')
            # populate exchange account list for the exchange session
            if session not in account_list:
                account_list[session] = []
            self.perimeters[session] = set()
            for strategy_name, (trade_exchange, account) in accounts.items():
                if (trade_exchange, account) not in account_list[session]:
                    account_list[session].append((trade_exchange, account))

                # get strategy theo positions from bot (already loaded by get_summary)
                strat_theo_pos = self.summaries[session][strategy_name]['theo']
            strategy_positions[session] = strategy_positions.get(session, {})

            # get account positions from live file
            if session not in self.account_theo_pos:
                self.account_theo_pos[session] = {}
                self.account_real_pos[session] = {}
            for (trade_exchange, account) in account_list[session]:
                key = '_'.join((trade_exchange, account))
                logging.info(f'Getting account positions for {key}')
                working_directory = self.session_configs[session]['working_directory']
                trade_account_dir = os.path.join(working_directory, key)
                theo_pos_file = os.path.join(trade_account_dir, 'current_state_theo.pos')
                real_pos_file = os.path.join(trade_account_dir, 'current_state.pos')

                if os.path.exists(theo_pos_file):
                    theo_pos_data = read_pos_file(theo_pos_file)
                    logging.info(f'Found {len(theo_pos_data)} theo positions for {session} {key}')
                else:
                    logging.warning(f'No theo pos file {theo_pos_file} for {session} {key}')
                    theo_pos_data = {}
                if os.path.exists(real_pos_file):
                    real_pos_data = read_pos_file(real_pos_file)
                    logging.info(f'Found {len(real_pos_data)} real positions for {session} {key}')
                else:
                    logging.warning(f'No real pos file {real_pos_file} for {session} {key}')
                    real_pos_data = {}
                self.account_theo_pos[session][key] = theo_pos_data.copy()
                self.account_real_pos[session][key] = real_pos_data.copy()
                # # get account positions from exchange
                # positions = await self.get_account_position(trade_exchange, account)
                #
                # if session not in self.account_positions:
                #     self.account_positions[session] = {}
                # self.account_positions[session][key] = positions
                self.perimeters[session].update(set(theo_pos_data.keys()))

                self.perimeters[session].update(set(real_pos_data.keys()))
        return

    def _do_one_matching(self, pos_data, pos_data_theo, quotes):
        match = pd.concat([pd.Series(pos_data), pd.Series(pos_data_theo), pd.Series(quotes)], axis=1).rename(
            columns={0: 'real', 1: 'theo', 2: 'price'})
        match = match.dropna(subset=['real', 'theo'], how='all').fillna(0)
        difference_qty = match['real'] - match['theo']
        difference = difference_qty * match['price']
        match['delta_qty'] = difference_qty
        match['delta_amn'] = difference
        real_amnt = match['real'] * match['price']

        def very_nonzero(x, tolerance):
            return not isclose(x, 0, abs_tol=tolerance)

        not_dust = real_amnt.apply(very_nonzero, tolerance=10)
        is_mismatch = difference.apply(very_nonzero, tolerance=10)
        match['is_mismatch'] = is_mismatch
        match['dust'] = ~not_dust

        return match

    def do_matching(self, session):
        """
        Perform matching of theo positions with real positions for all sessions and accounts.
        Each session account matching DataFrame has columns ['real', 'theo', 'price', 'delta_qty', 'delta_amn', 'significant', 'dust'].
        """
        all_pos_data = self.account_real_pos.get(session, {})
        all_pos_data_theo = self.account_theo_pos.get(session, {})
        quotes = self.quotes.get(session, {})
        if session not in self.matching:
            self.matching[session] = {}

        for key, pos_data in all_pos_data.items():
            pos_data_theo = all_pos_data_theo.get(key, {})
            matching_data = self._do_one_matching(pos_data, pos_data_theo, quotes)
            self.matching[session][key] = matching_data
            if key == 'bitget_1':
                matching_data.to_csv('web_matching.csv')

    async def fetch_quotes(self, session):
        """
        Fetch quotes for all pairs in the session.
        """
        logging.info(f'Fetching quotes for session {session}')
        exchange_name = session
        account = ''
        params = {
            'exchange_trade': exchange_name,
            'account_trade': account
        }
        quotes = {}
        end_point = BrokerHandler.build_end_point(exchange_name)
        bh = BrokerHandler(market_watch=exchange_name, end_point_trade=end_point, strategy_param=params,
                           logger_name='default')
        for coin in self.perimeters[session]:
            ticker, _ = bh.symbol_to_market_with_factor(coin, universal=True)
            symbol, _ = bh.symbol_to_market_with_factor(coin, universal=False)
            last = await end_point._exchange_async.fetch_ticker(ticker)
            if 'last' in last:
                price = last['last']
            else:
                logging.info(f'Missing {ticker} in fetch_ticker result')
                price = None
            quotes[symbol] = price
            await asyncio.sleep(0.2)  # to avoid rate limit

        await end_point._exchange_async.close()
        await bh.close_exchange_async()
        logging.info(f'Quotes ready for session {session}')

        self.quotes[session] = quotes

    async def update_config(self, session, config_file):
        if os.path.exists(config_file):
            logging.info(f'Updating {session} config {config_file}')
            async with aiofiles.open(config_file, 'r') as myfile:
                try:
                    content = await myfile.read()
                    params = json.loads(content)
                    self.session_configs[session] = params
                except Exception as e:
                    logging.error(f'Unreadable config file {config_file}')
        else:
            logging.warning(f'No config file to update for {session} with name {config_file}')

    async def update_aum(self, session):
        logging.info('updating aum for %s', session)
        now = today_utc()
        accounts = self.used_accounts[session]
        working_directory = self.session_configs[session]['working_directory']
        back_days = [1, 2, 7, 30, 90, 180]
        last_date = {day: now - timedelta(days=day) for day in back_days}
        real_pnl_dict = {}
        account_list = []
        for strategy_name, (trade_exchange, account_number) in accounts.items():
            if (trade_exchange, account_number) not in account_list:
                account_list.append((trade_exchange, account_number))
        for account in account_list:
            account_key = '_'.join(account)
            aum_file = os.path.join(working_directory, account_key, '_aum.csv')
            if os.path.exists(aum_file):
                try:
                    aum = await read_aum_file(aum_file, session)
                    aum['deltat'] = (aum['date'] - aum['date'].shift()).apply(lambda x: x.total_seconds())
                    aum.loc[0, 'deltat'] = np.nan
                    aum['diff'] = aum['aum'].diff()
                    aum['ref'] = np.nan
                    for index in aum[aum['deltat'] == 0].index:
                        aum.loc[index, 'ref'] = aum.loc[index, 'diff']
                    for index in aum[aum['deltat'].isna()].index:
                        aum.loc[index, 'ref'] = aum.loc[index, 'aum']
                    aum['ref'] = aum['ref'].fillna(0).cumsum()
                    aum['pnl'] = aum['diff'] - aum['ref'].diff()
                    aum['perf'] = (aum['diff'] - aum['ref'].diff()) / aum['ref']
                    aum.set_index('date', inplace=True)  # copie
                    aum = aum.loc[~aum.index.duplicated(keep='last')]
                    daily = aum['perf'].resample('1d').agg('sum').fillna(0)
                    hourly = aum['perf'].resample('1H').agg('sum').fillna(0)
                    is_live = hourly.loc[hourly.index > last_date[back_days[0]]].std() > 0
                    if is_live:
                        logging.info(f'session {session} account {account} is live')
                        real_pnl_dict.update({'vol': {},
                                         'apr': {},
                                         'perfcum': {},
                                         'pnlcum': {},
                                         'drawdawn': {}
                                         })
                except Exception as e:
                    is_live = False
                    aum = pd.DataFrame()
                    daily = pd.Series()
                    logging.error(f'Error in aum file {aum_file} for account {account_key}: {e.args[0]}')
                    logging.error(traceback.format_exc())
                if is_live:
                    for day in back_days:
                        try:
                            last_aum = aum.loc[aum.index > last_date[day]]
                            vol = daily.loc[daily.index > last_date[day]].std() * np.sqrt(365)
                            if np.isnan(vol):
                                vol = 0
                            perfcum = last_aum['perf'].cumsum()
                            pnlcum = last_aum['pnl'].cumsum()
                            xp_max = perfcum.expanding().max()
                            uw = perfcum - xp_max
                            drawdown = uw.expanding().min()
                            if len(perfcum) > 0:
                                days = (last_aum.index[-1] - last_aum.index[0]).days
                                if days > 0:
                                    apr = perfcum.iloc[-1] / days * 365
                                else:
                                    apr = 0
                                pc = perfcum.iloc[-1]
                                dd = drawdown.iloc[-1]
                                pnc = pnlcum.iloc[-1]
                            else:
                                apr = 0
                                pc = 0
                                dd = 0
                                pnc = 0
                            real_pnl_dict['pnlcum'].update({f'{day:03d}d': pnc})
                            real_pnl_dict['vol'].update({f'{day:03d}d': vol})
                            real_pnl_dict['apr'].update({f'{day:03d}d': apr})
                            real_pnl_dict['perfcum'].update({f'{day:03d}d': pc})
                            real_pnl_dict['drawdawn'].update({f'{day:03d}d': dd})
                        except:
                            real_pnl_dict['pnlcum'].update({f'{day:03d}d': 0})
                            real_pnl_dict['vol'].update({f'{day:03d}d': 0})
                            real_pnl_dict['apr'].update({f'{day:03d}d': 0})
                            real_pnl_dict['perfcum'].update({f'{day:03d}d': 0})
                            real_pnl_dict['drawdawn'].update({f'{day:03d}d': 0})
                            logging.error(f'Error in aum data for strat {account_key} for day {day}')

                        if day == 180:
                            logging.info(f'Generating graph for {session}-{account_key}')
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
                            filename = f'temp/{session}_{account_key}_fig1.html'

                            with open(filename, 'w') as f:
                                f.write(html)

                            plt.close()

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
                            filename = f'temp/{session}_{account_key}_fig2.html'

                            with open(filename, 'w') as f:
                                f.write(html)
                            plt.close()

                        elif day == 30:
                            daily = last_aum['perf'].resample('1d').sum()
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
                            filename = f'temp/{session}_{account_key}_fig3.html'

                            with open(filename, 'w') as f:
                                f.write(html)
                            plt.close()
                    if session not in self.aum:
                        self.aum[session] = {}
                    self.aum[session][account_key] = real_pnl_dict

    async def update_pnl(self, session, working_directory, strategy_name, strategy_param):
        logging.info('updating pnl for %s, %s', session, strategy_name)
        now = today_utc()
        strategy_directory = os.path.join(working_directory, strategy_name)
        pnl_file = os.path.join(strategy_directory, 'pnl.csv')

        days = [1, 2, 7, 30, 90, 180]
        last_date = {day: now - timedelta(days=day) for day in days}
        pnl_dict = {}

        if os.path.exists(pnl_file):
            try:
                pnl = await read_pnl_file(pnl_file)
                alloc_label = [col for col in pnl.columns if 'alloc' in col]
                if len(alloc_label) > 0:
                    label = alloc_label[0]
                else:
                    label = 'allocation'
                    pnl['allocation'] = 10
                pnl[label] = pnl[label].bfill()
                last = {day: round(pnl.loc[pnl.index > last_date[day], 'pnl_theo'].mean() * 1e4, 0) for day in days}

                def select_pnl(day):
                    val = pnl.loc[pnl.index > last_date[day], 'pnl_theo']
                    expo = pnl.loc[pnl.index > last_date[day], label]
                    return (val / expo).replace([np.inf, -np.inf], np.nan).sum(skipna=True).sum()
                last_cum = {day: round(select_pnl(day), 4) for day in days}
                for day, value in last.items():
                    if np.isnan(value) or np.isinf(value):
                        last[day] = 0
                for day, value in last_cum.items():
                    if np.isnan(value) or np.isinf(value):
                        last_cum[day] = 0

                pnl_dict.update({'mean_theo': {f'{day:03d}d': last[day] for day in days},
                            'sum_theo': {f'{day:03d}d': last_cum[day] for day in days}})
            except:
                pnl_dict.update({'mean_theo': {f'{day:03d}d': 0 for day in days},
                            'sum_theo': {f'{day:03d}d': 0 for day in days}})
                logging.error(f'Error in pnl file {pnl_file} for strat {strategy_name}')

            try:
                JSONResponse(pnl_dict)
                if session not in self.pnl:
                    self.pnl[session] = {strategy_name: pnl_dict}
                else:
                    self.pnl[session].update({strategy_name: pnl_dict})
                # filename = f'temp/pnldict.json'
                #
                # with open(filename, 'w') as myfile:
                #     j = json.dumps(self.pnl, indent=4, cls=NpEncoder)
                #     print(j, file=myfile)
            except:
                logging.error(f'Error in pnl dict for strat {session}:{strategy_name}')
                self.pnl[session] = {strategy_name: {}}

    async def update_summary(self, session, working_directory, strategy_name, strategy_param):
        try:
            strategy_directory = os.path.join(working_directory, strategy_name)
            persistence_file = os.path.join(strategy_directory, strategy_param['persistence_file'])
            items = ['exchange_trade',
                     'account_trade',
                     'type',
                     'send_orders',
                     'max_total_expo',
                     'nb_short',
                     'nb_long',
                     'pos_matching',
                     'liquidate_unmatched',
                     'set_leverage',
                     'monitor_exec',
                     'use_aum',
                     'allocation',
                     'leverage',
                     'entry',
                     'exit',
                     'lookback',
                     'lookback_short',
                     'signal_vote_size',
                     'signal_vote_majority',
                     ]
            self.summaries[session][strategy_name] = {name: strategy_param.get(name, '') for name in items
                                                      if name in strategy_param}

            if os.path.exists(persistence_file):
                async with aiofiles.open(persistence_file, mode='r') as myfile:
                    content = await myfile.read()
                state = json.loads(content)
            else:
                state = {}
            summary_pairs = {}
            summary_coins = {}
            if 'current_pair_info' in state and len(state['current_pair_info']) > 0:
                for pair_name, pair_info in state['current_pair_info'].items():
                    s1, s2 = parse_pair(pair_name)
                    if 'position' not in pair_info:
                        continue
                    if 'in_execution' in pair_info and pair_info['in_execution'] and pair_info['position'] != 0:
                        continue
                    if 'in_execution' in pair_info:
                        if pair_info['in_execution']:
                            if 'target_qty' in pair_info and 'target_price' in pair_info:
                                summary_pairs[pair_name] = {
                                    'in_execution': pair_info['in_execution'],
                                    'target_qty': pair_info['target_qty'],
                                    'target_price': pair_info['target_price']
                                }
                        else:
                            if 'entry_data' in pair_info and 'quantity' in pair_info:
                                if pair_info["entry_data"][2] is None or np.isnan(pair_info["entry_data"][2]):
                                    entry_ts = datetime.fromtimestamp(0)
                                else:
                                    ts = pair_info["entry_data"][2] / 1e9
                                    entry_ts = datetime.fromtimestamp(int(ts), tz=UTC)
                                info1 = {
                                    'ref_price': pair_info['entry_data'][0],
                                    'quantity': pair_info['quantity'][0],
                                    'amount': pair_info['quantity'][0] * pair_info['entry_data'][0],
                                    'entry_ts': f'{entry_ts}'
                                }
                                info2 = {
                                    'ref_price': pair_info['entry_data'][1],
                                    'quantity': pair_info['quantity'][1],
                                    'amount': pair_info['quantity'][1] * pair_info['entry_data'][1],
                                    'entry_ts': f'{entry_ts}'
                                }
                                summary_pairs[pair_name] = {
                                    'in_execution': pair_info['in_execution'],
                                    s1: info1,
                                    s2: info2
                                }
                                summary_coins[s1] = info1
                                summary_coins[s2] = info2
            elif 'current_coin_info' in state:
                for coin, coin_info in state['current_coin_info'].items():
                    if 'position' not in coin_info:
                        continue
                    in_exec = coin_info.get('in_execution', False)
                    quantity = coin_info.get('quantity', 0)
                    position = coin_info.get('position', 0)
                    entry_data = coin_info.get('entry_data', [0, np.nan])
                    summary_coins[coin] = {
                        'in_execution': in_exec}
                    if quantity != 0:
                        if entry_data[1] is None or np.isnan(entry_data[1]):
                            entry_ts = datetime.fromtimestamp(0)
                        else:
                            ts = entry_data[1] / 1e9
                            entry_ts = datetime.fromtimestamp(int(ts), tz=UTC)
                        summary_coins[coin].update({
                            'position': quantity * position,
                            'amount': quantity * position * entry_data[0],
                            'entry_ts': f'{entry_ts}'
                        })
                    else:
                        target_qty = coin_info.get('target_qty', 0)
                        summary_coins[coin].update({'target_position': target_qty,
                                                    'position': 0.0,
                                                    'amount': 0.0,
                                                    'entry_ts': ''
                                                    })

            self.summaries[session][strategy_name]['theo'] = {
                'pairs': summary_pairs,
                'coins': summary_coins}
        except Exception as e:
            logging.warning(f'Exception {e.args[0]} during update_summary of account {session}.{strategy_name}')

    async def update_account(self, session, strategy_name, strategy_param):
        destination = strategy_param['send_orders']
        logging.info('updating account for %s, %s', session, strategy_name)
        if session not in self.matching:
            self.matching[session] = {}
        account = strategy_param['account_trade']

        trade_exchange = strategy_param['exchange_trade']
        try:
            if destination != 'dummy':
                all_positions = await self.get_account_position(trade_exchange, account)
                if 'pose' not in all_positions:
                    logging.info('empty account for %s, %s', session, strategy_name)
                    return
                positions = all_positions['pose']
                theo = {coin: pose for coin, pose in self.summaries[session][strategy_name]['theo']['coins'].items() if
                        pose != 0}
                current = pd.DataFrame({coin: value['amount'] for coin, value in positions.items()}, index=['current']).T
                theo_pos = pd.DataFrame({coin: value['amount'] for coin, value in theo.items()}, index=['theo']).T
                current_ts = pd.DataFrame({coin: value.get('entry_ts', np.nan) for coin, value in positions.items()},
                                          index=['current_entry_ts']).T
                theo_ts = pd.DataFrame({coin: value['entry_ts'] for coin, value in theo.items()}, index=['theo_ts']).T

                matching = pd.concat([current, theo_pos], axis=1).fillna(0)
                seuil_current = matching['current'].apply(np.abs).max() / 5
                seuil_theo = matching['theo'].apply(np.abs).max() / 5
                logging.info('Thresholds for  %s, %s: %f, %f', session, strategy_name, seuil_current, seuil_theo)
                significant = matching[
                    (matching['theo'].apply(np.abs) > seuil_theo) | (matching['current'].apply(np.abs) > seuil_theo)]
                matching = matching.loc[significant.index]
                matching.loc['Total'] = matching.sum(axis=0)
                nLong = current[(current.apply(np.abs) > seuil_current) & (current > 0)].count()['current']
                nShort = current[(current.apply(np.abs) > seuil_current) & (current < 0)].count()['current']
                matching.loc['nLong', 'current'] = nLong
                matching.loc['nShort', 'current'] = nShort
                nLong = theo_pos[(theo_pos.apply(np.abs) > seuil_theo) & (theo_pos > 0)].count()['theo']
                nShort = theo_pos[(theo_pos.apply(np.abs) > seuil_theo) & (theo_pos < 0)].count()['theo']
                matching.loc['nLong', 'theo'] = nLong
                matching.loc['nShort', 'theo'] = nShort

                matching['current_ts'] = current_ts
                matching['theo_ts'] = theo_ts
                if 'USDT total' in all_positions:
                    balance = all_positions['USDT total']
                else:
                    balance = np.nan
                matching.loc['USDT total'] = balance, np.nan, '', ''

                self.matching[session].update({strategy_name: matching})
        except Exception as e:
            logging.info(f'Exception {e.args[0]} for account {session}.{strategy_name}')


    async def check_running(self):
        message = []
        # checking heartbeat
        for exchange, param_dict in self.processor_config.items():
            heartbeat_file = param_dict['session_file']
            age_seconds = last_modif(heartbeat_file)

            if age_seconds is not None:
                if age_seconds > 180:
                    if self.processor_config[exchange].get('check_running'):
                        message += [
                        f'Heartbeat {heartbeat_file} of {exchange} unchanged for {int(age_seconds / 60)} minutes']
            else:
                if self.processor_config[exchange].get('check_running'):
                    message += [f'No file {heartbeat_file} ']

        return message

    async def check_all(self):
        message = []

        for session, params in self.session_configs.items():
            strategies = params['strategy']

            for strategy_name in strategies:
                strategy_param = strategies[strategy_name]
                if strategy_param['active'] and session in self.pnl and strategy_name in self.pnl[session]:
                    pnl_dict = self.pnl[session][strategy_name]
                    pnl_2d = pnl_dict.get('perfcum', {}).get('002d', 0.0)
                    if pnl_2d < -0.05 and self.processor_config[session].get('check_pnl'):
                        message += [f'2day PnL < -5% for {strategy_name}@{session}']

        for session, matching_dict in self.matching.items():
            for account, matching in matching_dict.items():
                logging.info(f'checking {session}:{account}')
                try:
                    # keep non dust



                    if n_short != n_long and 'spot' not in session:
                        message += [f'{strat}@{session}: Theo pos imbalance']

                    if not self.processor_config[session].get('check_realpose'):
                        continue

                    if np.abs(matching.loc['Total', 'current']) > (seuil_theo * total_factor):
                        message += [f'{strat}@{session}: Residual theo expo too large']

                    # checking positions mismatch
                    if significant_theo != significant_current:
                        d = significant_theo.difference(significant_current)
                        if len(d) > 0:
                            message += [f'Discrepancy {strat}@{session}: {d} have no pose in exchange account but should']
                        d = significant_current.difference(significant_theo)
                        if len(d) > 0:
                            message += [f'Discrepancy {strat}@{session}: {d} have pose in account but not in DB']

                    # test de l'expo
                    if np.abs(matching.loc['Total', 'current']) > (seuil_current * total_factor):
                        message += [f'{strat}@{session}: Residual expo too large']
                except Exception as e:
                    logging.error(f'Exception {e.args[0]} during check of {strat}@{session}')

        return message

    async def refresh_quotes(self):
        tasks = []
        logging.info('refreshing quotes')
        for session in self.perimeters:
            if session not in self.quotes:
                self.quotes[session] = {}
                tasks.append(asyncio.create_task(self.fetch_quotes(session)))
        await asyncio.gather(*tasks)
        return

    async def refresh(self, with_matching=True):
        logging.info('refreshing')

        for session, params in self.session_configs.items():
            working_directory = params['working_directory']
            strategies = params['strategy']
            self.summaries[session] = {'exchange': params['exchange']}

            for strategy_name in strategies:
                strategy_param = strategies[strategy_name]
                if strategy_param['active']:
                    try:
                        await self.update_summary(session, working_directory, strategy_name, strategy_param)
                        await self.update_pnl(session, working_directory, strategy_name, strategy_param)
                    except Exception as e:
                        logging.error(f'exception {e.args[0]} for {session}/{strategy_name} in refresh')
                        logging.error(traceback.format_exc())
            try:
                await self.update_aum(session)
            except Exception as e:
                logging.error(f'exception {e.args[0]} for {session} in update_aum')
                logging.error(traceback.format_exc())
        self.update_account_multi()

        if with_matching:
            for session in self.session_configs:
                self.do_matching(session)
        return


    def get_status(self):
        return self.summaries

    # def get_dashboard(self, session):
    #     return self.dashboard

    def get_pnl(self):
        return self.pnl

    def get_aum(self):
        return self.aum

    def get_matching(self, session, account_key):
        if session not in self.processor_config:
                return f'Exchange not found: try {list(self.processor_config.keys())} and account key'
        if self.matching is None:
            return {}
        if session in self.matching and account_key in self.matching[session]:
            return self.matching[session][account_key]
        else:
            return {}

    async def get_account_position(self, exchange, account):
        if 'ok' in exchange:
            exchange_name = 'okexfut'
        elif 'bin' in exchange and 'fut' in exchange:
            exchange_name = 'binancefut'
        elif 'bin' in exchange:
            exchange_name = 'binance'
        elif 'byb' in exchange:
            exchange_name = 'bybit'
        elif 'bitget' in exchange:
            exchange_name = 'bitget'
        else:
            return {}
        params = {
            'exchange_trade': exchange_name,
            'account_trade': account
        }
        end_point = BrokerHandler.build_end_point(exchange_name, account)
        bh = BrokerHandler(market_watch=exchange_name, end_point_trade=end_point, strategy_param=params, logger_name='default')
        try:
            positions = await end_point.get_positions_async()
        except Exception as e:
            logging.warning(f'exchange/account {exchange_name}/{account} sent exception {e.args}')
            positions = {}

        quotes = {}

        if end_point._exchange_async.has.get('fetchTickers', False):
            tickers = []
            for coin in positions:
                ticker, _ = bh.symbol_to_market_with_factor(coin, universal=True)
                tickers.append(ticker)
            result = await end_point._exchange_async.fetchTickers(tickers)

            for ticker, info in result.items():
                symbol, _ = bh.symbol_to_market_with_factor(ticker, universal=False)
                quotes[symbol] = info.get('last', None)
        else:
            for coin in positions:
                symbol = bh.symbol_to_market_with_factor(coin)[0]
                ticker = await end_point._exchange_async.fetch_ticker(symbol)
                if 'last' in ticker:
                    price = ticker['last']
                else:
                    price = None
                quotes[symbol] = price
                await asyncio.sleep(0.2)

        cash = await end_point.get_cash_async(['USDT', 'BTC'])

        await end_point._exchange_async.close()
        await bh.close_exchange_async()

        response = {}
        for coin, info in positions.items():
            symbol, _ = bh.symbol_to_market_with_factor(coin, universal=False)
            price = info[2]
            if price is None or price == np.nan:
                price = quotes.get(symbol)
            amount = info[1]
            if amount is None or amount == np.nan:
                amount = info[0] * price

            response[symbol] = {
                'entry_ts': f'{info[3]}',
                'quantity': info[0],
                'ref_price': price,
                'amount': amount
            }
        response_df = pd.DataFrame(response).T
        if len(response_df) > 0:
            response_df.sort_values(by='amount', inplace=True)
        response = {'pose': response_df.T.to_dict()}
        response.update({'USDT free': cash[0], 'USDT total': cash[1]})

        return response

    async def multiply(self, exchange, account, factor):
        positions = await self.get_account_position(exchange, account)
        if 'fut' in exchange and 'bin' in exchange:
            exchange_trade = 'binancefut'
            exclude = []
        elif 'bitget' in exchange:
            exchange_trade = 'bitget'
            exclude = []
        else:
            exchange_trade = 'binance'
            exclude = ['BNB', 'BNBUSDT']

        params = {
            'exchange_trade': exchange_trade,
            'account_trade': account
        }
        end_point = BrokerHandler.build_end_point(exchange_trade, account)
        bh = BrokerHandler(market_watch=exchange_trade, end_point_trade=end_point, strategy_param=params, logger_name='default')
        broker = WebSpreaderBroker(market=exchange, account=account, broker_handler=bh)
        response = ''
        if np.abs(factor - 1) < 0.05:
            return True

        for coin, info in positions['pose'].items():
            if coin in exclude:
                continue
            order_id = broker.get_id
            quantity = info['quantity']

            if quantity == 0:
                continue

            target = (factor - 1) * quantity
            nature = 'n' if factor > 1 else 'x'
            action = -1 if target < 0 else 1
            comment = 'liquidation' if factor < 0.1 else 'multiplication'
            target = np.abs(target)

            response = await broker.send_simple_order(order_id, coin=coin, action=action, price=None, target_quantity=target,
                                       comment=comment, nature=nature,
                                       translate_qty_incontracts=False, use_algo=True)
            await asyncio.sleep(5)

        return response

def runner(event, processor, pace):
    async def run_web_processor():
        uvicorn_error = logging.getLogger("uvicorn.error")
        uvicorn_error.disabled = True
        uvicorn_access = logging.getLogger("uvicorn.access")
        uvicorn_access.disabled = True
        global app
        app = FastAPI()

        # app.mount('/static', StaticFiles(directory='static', html=True), name='static')

        @app.get('/pose')
        async def read_position(exchange: str = 'okx', account: str = '1'):
            if exchange not in WebSpreaderBroker.ACCOUNT_DICT:
                raise HTTPException(status_code=404, detail=f'Exchange not found: '
                                                            f'try {list(WebSpreaderBroker.ACCOUNT_DICT.keys())}')
            if account not in WebSpreaderBroker.ACCOUNT_DICT[exchange]:
                raise HTTPException(status_code=404, detail=f'Account not found: '
                                                            f'try {list(WebSpreaderBroker.ACCOUNT_DICT[exchange].keys())}')
            report = await processor.get_account_position(exchange, account)
            return JSONResponse(report)

        @app.get('/multiply')
        async def multiply(exchange: str = '', account: str = '', factor=1.0):
            if exchange not in WebSpreaderBroker.ACCOUNT_DICT:
                raise HTTPException(status_code=404, detail=f'Exchange not found: '
                                                            f'try {list(WebSpreaderBroker.ACCOUNT_DICT.keys())}')
            if len(account) == 1:
                account = int(account)
            if account not in WebSpreaderBroker.ACCOUNT_DICT[exchange]:
                raise HTTPException(status_code=404, detail=f'Account not found: '
                                                            f'try {list(WebSpreaderBroker.ACCOUNT_DICT[exchange].keys())}')
            factor = float(factor)
            if factor < 0 or factor > 2:
                raise HTTPException(status_code=403, detail='Forbidden value: try between 0 and 2')
            report = await processor.multiply(exchange, account, factor)
            return JSONResponse(report)

        @app.get('/dashboard')
        async def read_dashboard(session: str = 'bitget'):
            report = processor.get_dashboard(session=session)
            return JSONResponse(report)

        @app.get('/status')
        async def read_status():
            report = processor.get_status()
            return JSONResponse(report)

        @app.get('/pnl')
        async def read_pnl():
            report = processor.get_pnl()
            return JSONResponse(report)

        @app.get('/aum')
        async def read_aum():
            report = processor.get_aum()
            return JSONResponse(report)

        @app.get('/matching')
        async def read_matching(session: str = 'binance', account_key: str = 'bitget_2'):
            report = processor.get_matching(session, account_key)
            return JSONResponse(report.to_dict(orient='index'))
            # if isinstance(report, pd.DataFrame):
            #     return HTMLResponse(report.to_html(formatters={'entry': lambda x: x.strftime('%d-%m-%Y %H:%M'),
            #                                                    'theo': lambda x: f'{x:.0f}',
            #                                                    'real': lambda x: f'{x:.0f}',
            #                                                    }))
            # else:
            #     if report is not None:
            #     else:
            #         return HTMLResponse('N/A')

        config = uvicorn.Config(app, port=14440, host='0.0.0.0', lifespan='on')
        server = uvicorn.Server(config)
        await server.serve()

    async def heartbeat(queue, pace, action):
        while True:
            await queue.put(action)
            await asyncio.sleep(pace)

    async def watch_file_modifications(queue):
        last_mod_times = {session: os.path.getmtime(file_path) for session, file_path in processor.config_files.items()}
        while True:
            await asyncio.sleep(10)
            for session, file_path in processor.config_files.items():
                current_mod_time = os.path.getmtime(file_path)
                if current_mod_time != last_mod_times[session]:
                    last_mod_times[session] = current_mod_time
                    await queue.put((SignalType.FILE, session, file_path))

    async def refresh(with_matching):
        await processor.refresh(with_matching)

    async def refresh_quotes():
        await processor.refresh_quotes()

    async def send_alert(message):
        if message is not None:
            for error in message:
                TGMessenger.send(error, 'CM')
            logging.info(f'Sent {len(message)} msg')

    async def check(checking_coro):
        messages = await checking_coro

        await send_alert(messages)

    async def main():
        event.loop = asyncio.get_event_loop()
        event.queue = asyncio.Queue()
        task_queue = asyncio.Queue()
        await refresh(with_matching=False)  # first we initialize perimeters
        await refresh_quotes()
        event.set()
        web_runner = asyncio.create_task(run_web_processor())
        heart_runner = asyncio.create_task(heartbeat(task_queue, pace['REFRESH'], SignalType.REFRESH))
        running_runner = asyncio.create_task(heartbeat(task_queue, pace['CHECK'], SignalType.CHECKALL))
        match_runner = asyncio.create_task(heartbeat(task_queue, pace['MATCHING'], SignalType.MATCHING))
        quote_runner = asyncio.create_task(heartbeat(task_queue, pace['PRICE_UPDATE'], SignalType.PRICE_UPDATE))
        file_watcher = asyncio.create_task(watch_file_modifications(task_queue))

        while True:
            item = await task_queue.get()
            if item == SignalType.STOP:
                task_queue.task_done()
                break
            if item == SignalType.REFRESH:
                task = asyncio.create_task(refresh(with_matching=True))
                task.add_done_callback(lambda _: task_queue.task_done())
            # elif item == SignalType.CHECKALL:
            #     task = asyncio.create_task(check(processor.check_all()))
            #     task.add_done_callback(lambda _: task_queue.task_done())
            # elif item == SignalType.MATCHING:
            #     task = asyncio.create_task(check(processor.check_matching()))
            #     task.add_done_callback(lambda _: task_queue.task_done())
            elif item == SignalType.PRICE_UPDATE:
                task = asyncio.create_task(check(processor.refresh_quotes()))
                task.add_done_callback(lambda _: task_queue.task_done())
            elif isinstance(item, tuple) and item[0] == SignalType.FILE:
                await processor.update_config(item[1], item[2])
                task_queue.task_done()
        web_runner.cancel()
        heart_runner.cancel()
        match_runner.cancel()
        quote_runner.cancel()
        running_runner.cancel()
        file_watcher.cancel()
        await task_queue.join()

    asyncio.run(main())


if __name__ == '__main__':
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="input file", default='')
    args = parser.parse_args()
    config_file = args.config
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    if config_file != '':
        with open(config_file, 'r') as myfile:
            config = yaml.load(myfile, Loader=yaml.FullLoader)
    else:
        config = {}

    pace = config.get('pace', {'REFRESH': 180, 'MATCHING': 60, 'PRICE_UPDATE': 600, 'RUNNING': 300})
    started = threading.Event()
    processor = Processor(config)
    th = threading.Thread(target=runner, args=(started, processor, pace,))
    logging.info('Starting')
    th.start()
    started.wait()
    logging.info('Started')
    th.join()
    logging.info('Stopped')
