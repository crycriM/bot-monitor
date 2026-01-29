import json
import os
from datetime import datetime
from pathlib import Path
import logging
import aiofiles
import asyncio
import pandas as pd
import numpy as np
from math import isclose
from datetime import timedelta
try:
    from datetime import UTC
except ImportError:
    from datetime import timezone
    UTC = timezone.utc
import traceback

from utils_files import (
    last_modif, read_pnl_file, read_aum_file, read_pos_file, read_latent_file,
    generate_perf_chart, generate_pnl_chart, generate_daily_perf_chart,
    calculate_median_position_sizes, JSONResponse, get_temp_dir
)
from .file_watcher import FileWatcherManager
from shared_utils.online import parse_pair, today_utc
from datafeed.broker_handler import BrokerHandler
from trading_bot.web_broker import WebSpreaderBroker

LOGGER = logging.getLogger('web_processor')

class WebProcessor:
    '''
    Responsibility: interface between web query and strat status and commands
    trigger a refresh of pairs (todo)
    get status of each strat
    get pos
    '''

    def __init__(self, config):
        self.processor_config = config['session']
        self.config_files = {session: Path(self.processor_config[session]['config_file']).expanduser() for session in self.processor_config}
        self.config_position_matching_files = {session: Path(self.processor_config[session]['config_position_matching_file']).expanduser() for session in self.processor_config}

        self.session_configs = {}
        for session, config_file in self.config_files.items():
            if os.path.exists(config_file):
                LOGGER.info(f'Reading session config {config_file}')
                with open(config_file, 'r') as myfile:
                    params = json.load(myfile)
                    params['working_directory'] = Path(params['working_directory']).expanduser()
                    self.session_configs[session] = params
            else:
                LOGGER.warning(f'No config file for {session} with name {config_file}')

        self.session_matching_configs={}
        for session, config_matching_file in self.config_position_matching_files.items():
            if os.path.exists(config_matching_file):
                with open(config_matching_file, 'r') as myfile:
                    params = json.load(myfile)
                    self.session_matching_configs[session] = params
            else:
                LOGGER.warning(f'No matching config file for {session} with name {config_matching_file}')
        self.aum = {}
        self.strategy_state = {}
        self.strategy_types = {}
        self.summaries = {}
        self.pnl = {}
        self.latent = {}
        self.matching = {}
        self.dashboard = {}
        self.last_message_count = 0
        self.last_message = []
        self.used_accounts = {}  # key is session name, value is key=strat_name, value=account used
        self.account_theo_pos = {}
        self.account_real_pos = {}
        self.perimeters = {}
        self.quotes = {}

        # Attributes for position_comparator
        self.price_cache = {}
        self.median_position_sizes = {}
        self.mismatch_start_times = {}

        # File watching infrastructure - use FileWatcherManager
        self.file_watcher = FileWatcherManager()

        self._init_accounts_by_strat()

    def _init_accounts_by_strat(self):
        """
        populate exchange accounts list
        """
        try:
            for session, session_params in self.session_configs.items():
                strategies = session_params['strategy']
                LOGGER.info(f'Adding strategies for {session} session')

                for strategy_name, strategy_param in strategies.items():
                    LOGGER.info(f'Adding strategy {strategy_name}')
                    strat_account = strategy_param['account_trade']
                    strat_exchange = strategy_param.get('exchange_trade', '')
                    active = strategy_param.get('active', False)
                    destination = strategy_param.get('send_orders', 'dummy')

                    if not active:
                        continue
                    if session not in self.used_accounts:
                        self.used_accounts[session] = {}
                        self.strategy_types[session] = {}
                    if destination == 'dummy':
                        self.used_accounts[session][strategy_name] = ('dummy', strat_account)
                    else:
                        self.used_accounts[session][strategy_name] = (strat_exchange, strat_account)
                    self.strategy_types[session][strategy_name] = strategy_param.get('type', 'other')
        except Exception as e:
            LOGGER.error(f'Error updating accounts by strat: {e.args[0]}')
            LOGGER.error(traceback.format_exc())
            self.used_accounts = {}
        LOGGER.info(f'Built account dict {self.used_accounts}')

    def build_watched_files_registry(self):
        """Delegate to FileWatcherManager"""
        return self.file_watcher.build_watched_files_registry(
            self.session_configs,
            self.processor_config,
            self.used_accounts
        )

    def start_file_watchers(self, loop, task_queue):
        """Delegate to FileWatcherManager"""
        self.file_watcher.start_file_watchers(loop, task_queue)

    def stop_file_watchers(self):
        """Delegate to FileWatcherManager"""
        self.file_watcher.stop_file_watchers()

    def validate_file_watchers(self):
        """Delegate to FileWatcherManager"""
        return self.file_watcher.validate_file_watchers()

    def update_account_multi(self):
        """
        Update account theo, aggregated theo and real positions from bot for all sessions and strategies.
        Retrieve account positions form live file and from exchange independently
        """
        account_list = {}
        strategy_positions = {}

        for session, accounts in self.used_accounts.items():
            LOGGER.info(f'Updating account positions for {session}')
            # populate exchange account list for the exchange session
            if session not in account_list:
                account_list[session] = []
            self.perimeters[session] = set()
            for strategy_name, (trade_exchange, account) in accounts.items():
                if (trade_exchange, account) not in account_list[session]:
                    account_list[session].append((trade_exchange, account))

                # get strategy theo positions from bot (already loaded by get_summary)
                # strat_theo_pos = self.summaries[session][strategy_name]['theo']
            strategy_positions[session] = strategy_positions.get(session, {})

            # get account positions from live file
            if session not in self.account_theo_pos:
                self.account_theo_pos[session] = {}
                self.account_real_pos[session] = {}
            for (trade_exchange, account) in account_list[session]:
                key = '_'.join((trade_exchange, account))
                LOGGER.info(f'Getting account positions for {key}')
                working_directory = self.session_configs[session]['working_directory']
                trade_account_dir = working_directory / key
                theo_pos_file = trade_account_dir / 'current_state_theo.pos'

                if os.path.exists(theo_pos_file):
                    theo_pos_data = read_pos_file(theo_pos_file)
                    LOGGER.info(f'Found {len(theo_pos_data)} theo positions for {session} {key}')
                else:
                    LOGGER.warning(f'No theo pos file {theo_pos_file} for {session} {key}')
                    theo_pos_data = {}
                real_pos_data = {}
                if trade_exchange != 'dummy':
                    real_pos_file = trade_account_dir / 'current_state.pos'
                    if os.path.exists(real_pos_file):
                        real_pos_data = read_pos_file(real_pos_file)
                        LOGGER.info(f'Found {len(real_pos_data)} real positions for {session} {key}')
                    else:
                        LOGGER.warning(f'No real pos file {real_pos_file} for {session} {key}')
                    self.account_real_pos[session][key] = real_pos_data.copy()
                self.account_theo_pos[session][key] = theo_pos_data.copy()
                # # get account positions from exchange
                # positions = await self.get_account_position(trade_exchange, account)
                #
                # if session not in self.account_positions:
                #     self.account_positions[session] = {}
                # self.account_positions[session][key] = positions
                tickers = [name for name in set(theo_pos_data.keys()) if 'USD' in name]
                self.perimeters[session].update(tickers)
                LOGGER.info(f'Added {len(theo_pos_data)} coins from theo pos file to perimeter for {session} {key}')
                tickers = [name for name in set(real_pos_data.keys()) if 'USD' in name]
                self.perimeters[session].update(tickers)
                LOGGER.info(f'Updated with real pos, perimeter is now  {len(self.perimeters[session])} coins for {session} {key}')

        # Calculate median position sizes for each account (needed by position_comparator)
        self.median_position_sizes = calculate_median_position_sizes(
            self.account_theo_pos, self.quotes)
        for account_key, size in self.median_position_sizes.items():
            LOGGER.info(f'Median position size for {account_key}: {size:.2f}')
        return

    def _do_one_matching(self, pos_data, pos_data_theo, quotes):
        match = pd.concat([pd.Series(pos_data), pd.Series(pos_data_theo), pd.Series(quotes)], axis=1).rename(
            columns={0: 'real', 1: 'theo', 2: 'price'})

        match = match.dropna(subset=['real', 'theo'], how='all')
        match[['real', 'theo']] = match[['real', 'theo']].fillna(0)
        real_amnt = match['real'] * match['price']
        theo_amnt = match['theo'] * match['price']
        difference_qty = match['real'] - match['theo']
        difference = real_amnt - theo_amnt
        rel_difference = difference_qty / (0.5 * (match['real'] + match['theo']))
        match['delta_qty'] = difference_qty
        match['delta_amn'] = difference
        match['rel_delta'] = rel_difference

        def very_nonzero(x, tolerance):
            return not isclose(x, 0, abs_tol=tolerance)

        not_dust = real_amnt.apply(very_nonzero, tolerance=50)
        is_mismatch = rel_difference.apply(very_nonzero, tolerance=0.05)
        match['is_mismatch'] = is_mismatch & not_dust
        match['dust'] = (~not_dust) & is_mismatch

        return match.fillna('N/A')

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
                matching_data.to_csv('output/web_matching.csv')

    async def _fetch_all_tickers(self, bh, end_point, perimeter):
        quotes = {}

        if end_point._exchange_async.has.get('fetchTickers', False):
            LOGGER.info(f'entering fetchTickers')
            tickers = []
            for coin in perimeter:
                ticker, _ = bh.symbol_to_market_with_factor(coin, universal=True)
                tickers.append(ticker)
            result = await end_point._exchange_async.fetchTickers(tickers)

            for coin in perimeter:
                ticker, _ = bh.symbol_to_market_with_factor(coin, universal=True)
                symbol, _ = bh.symbol_to_market_with_factor(ticker, universal=False)
                if ticker in result:
                    info = result[ticker]
                else:
                    info = {}
                quotes[symbol] = info.get('last', None)
        else:
            LOGGER.info(f'entering fetch Ticker loop')
            for coin in perimeter:
                symbol = bh.symbol_to_market_with_factor(coin)[0]
                ticker = await end_point._exchange_async.fetch_ticker(symbol)
                if 'last' in ticker:
                    price = ticker['last']
                else:
                    price = None
                quotes[symbol] = price
                await asyncio.sleep(0.2)
        return quotes

    async def fetch_quotes(self, session):
        """
        Fetch quotes for all pairs in the session.
        """
        LOGGER.info(f'Fetching quotes for session {session}')
        exchange_name = session
        account = ''
        params = {
            'exchange_trade': exchange_name,
            'account_trade': account
        }
        end_point = None
        bh = None
        quotes = {}
        try:
            end_point = BrokerHandler.build_end_point(exchange_name)
            bh = BrokerHandler(market_watch=exchange_name, end_point_trade=end_point, strategy_param=params,
                               logger_name='broker_handler')
            LOGGER.info(f'Fetching {len(self.perimeters.get(session, []))} quotes for session {session}')
            quotes = await self._fetch_all_tickers(bh, end_point, self.perimeters.get(session, set()))
            try:
                with open('output/web_quotes.txt', 'w') as myfile:
                    print(quotes, file=myfile)
            except Exception:
                LOGGER.debug('Could not write output/web_quotes.txt')

            for coin in self.perimeters.get(session, set()):
                if coin not in quotes:
                    LOGGER.info(f'Missing {coin} in fetch_ticker result')

            LOGGER.info(f'{len(quotes)} quotes ready for session {session}')

            self.quotes[session] = quotes

            # Populate price_cache for position_comparator
            if session not in self.price_cache:
                self.price_cache[session] = {}
            for coin, price in quotes.items():
                if price is not None:
                    self.price_cache[session][coin] = {'price': price}

        except Exception as e:
            self.quotes[session] = {}
            LOGGER.error(f'Error fetching quotes for session {session}: {str(e)}')
            LOGGER.debug(traceback.format_exc())
        finally:
            # Ensure resources are closed to avoid unclosed client sessions
            try:
                if end_point is not None and hasattr(end_point, '_exchange_async') and end_point._exchange_async is not None:
                    try:
                        await end_point._exchange_async.close()
                    except Exception:
                        LOGGER.debug('Error closing end_point._exchange_async', exc_info=True)
            except Exception:
                LOGGER.debug('Error while attempting to close end_point', exc_info=True)

            try:
                if bh is not None:
                    try:
                        await bh.close_exchange_async()
                    except Exception:
                        LOGGER.debug('Error closing broker handler async session', exc_info=True)
            except Exception:
                LOGGER.debug('Error while attempting to close broker handler', exc_info=True)

    async def update_config(self, session, config_file):
        if os.path.exists(config_file):
            LOGGER.info(f'Updating {session} config {config_file}')
            async with aiofiles.open(config_file, 'r') as myfile:
                try:
                    content = await myfile.read()
                    params = json.loads(content)
                    self.session_configs[session] = params
                except Exception as e:
                    LOGGER.error(f'Unreadable config file {config_file}')
        else:
            LOGGER.warning(f'No config file to update for {session} with name {config_file}')
    async def update_aum(self, session):
        LOGGER.info('updating aum for %s', session)
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
            aum_file = working_directory / account_key / '_aum.csv'
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
                    hourly = aum['perf'].resample('1h').agg('sum').fillna(0)
                    extract = hourly.loc[hourly.index > last_date[back_days[0]]]
                    is_live = extract.std() > 0
                    if is_live:
                        LOGGER.info(f'session {session} account {account} is live')
                        real_pnl_dict.update({'vol': {},
                                              'apr': {},
                                              'perfcum': {},
                                              'pnlcum': {},
                                              'drawdawn': {}
                                              })
                    else:
                        LOGGER.info(f'session {session} account {account} is not live with data {extract}')
                except Exception as e:
                    is_live = False
                    aum = pd.DataFrame()
                    daily = pd.Series()
                    LOGGER.error(f'Error in aum file {aum_file} for account {account_key}: {e.args[0]}')
                    LOGGER.error(traceback.format_exc())
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
                            no_graph = False
                        except:
                            real_pnl_dict['pnlcum'].update({f'{day:03d}d': 0})
                            real_pnl_dict['vol'].update({f'{day:03d}d': 0})
                            real_pnl_dict['apr'].update({f'{day:03d}d': 0})
                            real_pnl_dict['perfcum'].update({f'{day:03d}d': 0})
                            real_pnl_dict['drawdawn'].update({f'{day:03d}d': 0})
                            LOGGER.error(f'Error in aum data for strat {account_key} for day {day}')
                            no_graph = True

                        if day == 180 and not no_graph:
                            LOGGER.info(f'Generating graph for {session}-{account_key}')
                            html = generate_perf_chart(perfcum, session, account_key)
                            temp_dir = get_temp_dir()
                            filename = temp_dir / f'{session}_{account_key}_fig1.html'

                            with open(filename, 'w') as f:
                                f.write(html)

                            html = generate_pnl_chart(pnlcum, session, account_key)
                            filename = temp_dir / f'{session}_{account_key}_fig2.html'

                            with open(filename, 'w') as f:
                                f.write(html)

                        elif day == 30 and not no_graph:
                            daily = last_aum['perf'].resample('1d').sum()
                            html = generate_daily_perf_chart(daily, session, account_key)
                            temp_dir = get_temp_dir()
                            filename = temp_dir / f'{session}_{account_key}_fig3.html'
                            with open(filename, 'w') as f:
                                f.write(html)
                    if session not in self.aum:
                        self.aum[session] = {}
                    self.aum[session][account_key] = real_pnl_dict
            else:
                LOGGER.info(f'No {aum_file} file')

    async def update_pnl(self, session, working_directory, strategy_name, strategy_param):
        LOGGER.info('updating pnl for %s, %s', session, strategy_name)
        now = today_utc()
        strategy_directory = working_directory / strategy_name
        pnl_file = strategy_directory / 'pnl.csv'
        latent_file = strategy_directory / 'latent_profit.csv'
        strategy_type = self.strategy_types[session][strategy_name]

        days = [1, 2, 7, 30, 90, 180]
        last_date = {day: now - timedelta(days=day) for day in days}
        pnl_dict = {'mean_theo': {}, 'sum_theo': {}}

        if os.path.exists(latent_file):
            try:
                LOGGER.info('reading latent file for %s, %s', session, strategy_name)
                latent_result = await read_latent_file(latent_file)
                pnl_dict['sum_theo']['latent_theo'] = latent_result['latent_return'].iloc[-1]
            except:
                latent_result = pd.DataFrame()
                LOGGER.error(f'Error in latent file {latent_file} for strat {strategy_name}')
                pnl_dict['sum_theo']['latent_theo'] = 0
            if session not in self.latent:
                self.latent[session] = {strategy_name: latent_result}
            else:
                self.latent[session].update({strategy_name: latent_result})

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
                pnl_dict['mean_theo']['strategy_type'] = strategy_type
                pnl_dict['sum_theo']['strategy_type'] = strategy_type
            except:
                pnl_dict.update({'mean_theo': {f'{day:03d}d': 0 for day in days},
                                 'sum_theo': {f'{day:03d}d': 0 for day in days}})
                pnl_dict['mean_theo']['strategy_type'] = strategy_type
                pnl_dict['sum_theo']['strategy_type'] = strategy_type
                LOGGER.error(f'Error in pnl file {pnl_file} for strat {strategy_name}')

            try:
                JSONResponse(pnl_dict)
                if session not in self.pnl:
                    self.pnl[session] = {strategy_name: pnl_dict}
                else:
                    self.pnl[session].update({strategy_name: pnl_dict})
                # temp_dir = get_temp_dir()
                # filename = temp_dir / 'pnldict.json'
                #
                # with open(filename, 'w') as myfile:
                #     j = json.dumps(self.pnl, indent=4, cls=NpEncoder)
                #     print(j, file=myfile)
            except:
                LOGGER.error(f'Error in pnl dict for strat {session}:{strategy_name}')
                self.pnl[session] = {strategy_name: {}}

    async def update_summary(self, session, working_directory, strategy_name, strategy_param):
        try:
            strategy_directory = working_directory / strategy_name
            persistence_file = strategy_directory / strategy_param['persistence_file']
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
            LOGGER.warning(f'Exception {e.args[0]} during update_summary of account {session}.{strategy_name}')

    async def update_account(self, session, strategy_name, strategy_param):
        destination = strategy_param['send_orders']
        LOGGER.info('updating account for %s, %s', session, strategy_name)
        if session not in self.matching:
            self.matching[session] = {}
        account = strategy_param['account_trade']

        trade_exchange = strategy_param['exchange_trade']
        try:
            if destination != 'dummy':
                all_positions = await self.get_account_position(trade_exchange, account)
                if 'pose' not in all_positions:
                    LOGGER.info('empty account for %s, %s', session, strategy_name)
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
                # Alternative threshold using median position sizes
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
            LOGGER.error(f'Exception {e.args[0]} for account {session}.{strategy_name}')

    async def check_latent(self):
        message = []
        for session, params in self.session_configs.items():
            strategies = params['strategy']
            for strategy_name in strategies:
                strategy_param = strategies[strategy_name]
                if strategy_param['active'] and session in self.latent and strategy_name in self.latent[session]:
                    latent_df = self.latent[session][strategy_name]
                    if len(latent_df) > 0:
                        latent_return = latent_df['latent_return'].iloc[-1]
                        latent_pnl = latent_df['latent_pnl'].iloc[-1]
                        if np.isnan(latent_return) or np.isinf(latent_return):
                            latent_return = 0
                        if np.isnan(latent_pnl) or np.isinf(latent_pnl):
                            latent_pnl = 0
                        LOGGER.info(f'checking latent for {session}:{strategy_name}, found {latent_return}, {latent_pnl}')
                        if latent_return < -0.04 and self.processor_config[session].get('check_latent'):
                            message += [f'Latent return < -4% for {strategy_name}@{session}']
        return message

    async def check_running(self):
        message = []
        # checking heartbeat
        for exchange, param_dict in self.processor_config.items():
            heartbeat_file = Path(param_dict['session_file']).expanduser()
            age_seconds = last_modif(heartbeat_file)

            if age_seconds is not None:
                if age_seconds > 180:
                    if self.processor_config[exchange].get('check_running'):
                        message += [
                            f'Heartbeat {heartbeat_file} of {exchange} unchanged for {int(age_seconds / 60)} minutes']
            else:
                if self.processor_config[exchange].get('check_running'):
                    message += [f'No file {heartbeat_file} ']

        # Generate killswitch signal
        for session, params in self.session_configs.items():
            strategies = params['strategy']
            killswitch_str = self.processor_config[session].get('kill_switch', '')
            killswitchfile = Path(killswitch_str).expanduser() if killswitch_str else ''
            killswitch = {}

            for strategy_name in strategies:
                strategy_param = strategies[strategy_name]
                if strategy_param['active'] and session in self.pnl and strategy_name in self.pnl[session]:
                    pnl_dict = self.pnl[session][strategy_name]
                    theopnl_1d = pnl_dict.get('sum_theo', {}).get('001d', 0.0)
                    theopnl_2d = pnl_dict.get('sum_theo', {}).get('002d', 0.0)
                    latent_theo = pnl_dict.get('sum_theo', {}).get('latent_theo', 0.0)
                    realpnl_1d = pnl_dict.get('perfcum', {}).get('001d', 0.0)
                    realpnl_2d = pnl_dict.get('perfcum', {}).get('002d', 0.0)

                    killswitch[strategy_name] = {'theopnl_1d': theopnl_1d,
                                                 'theopnl_2d': theopnl_2d,
                                                 'realpnl_1d': realpnl_1d,
                                                 'realpnl_2d': realpnl_2d,
                                                 'latent_theo': latent_theo}

            if killswitchfile:
                with open(killswitchfile, 'w') as myfile:
                    if len(killswitch) > 0:
                        LOGGER.info(f'Writing kill switch {killswitch} to {killswitchfile}')
                        json.dump(killswitch, myfile)
                    else:
                        LOGGER.info(f'No kill switch for {killswitchfile}')
                        myfile.write('{}')

        return message

    async def check_all(self):
        message = []

        # checking pnl
        for session, params in self.session_configs.items():
            strategies = params['strategy']

            for strategy_name in strategies:
                strategy_param = strategies[strategy_name]
                if strategy_param['active'] and session in self.pnl and strategy_name in self.pnl[session]:
                    pnl_dict = self.pnl[session][strategy_name]
                    pnl_2d = pnl_dict.get('perfcum', {}).get('002d', 0.0)
                    if pnl_2d < -0.05 and self.processor_config[session].get('check_pnl'):
                        message += [f'2day PnL < -5% for {strategy_name}@{session}']
        # checking matching using precomputed DataFrame
        for session, matching_dict in self.matching.items():
            for account_key, matching_df in matching_dict.items():
                LOGGER.info(f'checking {session}:{account_key}')
                try:
                    if matching_df.empty:
                        continue

                    # Calculate thresholds based on significant positions
                    theo_positions = matching_df['theo'].dropna()
                    real_positions = matching_df['real'].dropna()

                    if len(theo_positions) == 0 and len(real_positions) == 0:
                        continue

                    seuil_theo = theo_positions.abs().max() / 5 if len(theo_positions) > 0 else 0
                    seuil_current = real_positions.abs().max() / 5 if len(real_positions) > 0 else 0
                    # Alternative threshold using median position sizes
                    median_size = self.median_position_sizes.get(account_key, 0)
                    seuil_median = median_size * 0.1 if median_size > 0 else 0
                    seuil_theo = max(seuil_theo, seuil_median)
                    seuil_current = max(seuil_current, seuil_median)
                    total_factor = 3  # Default factor for threshold multiplication
                    LOGGER.info(f'{account_key}@{session}: Calculated thresholds - '
                                f'seuil_theo: {seuil_theo:.2f}, seuil_current: {seuil_current:.2f}, '
                                f'using median: {seuil_median:.2f}')

                    # Filter significant positions
                    significant_theo = set(matching_df[matching_df['theo'].abs() > seuil_theo].index)
                    significant_current = set(matching_df[matching_df['real'].abs() > seuil_current].index)
                    LOGGER.info(f'{account_key}@{session}: Significant theo positions: {len(significant_theo)},'
                                f' current: {len(significant_current)}')


                    # Count long/short positions for balance check
                    n_long_theo = len(matching_df[(matching_df['theo'].abs() > seuil_theo) & (matching_df['theo'] > 0)])
                    n_short_theo = len(matching_df[(matching_df['theo'].abs() > seuil_theo) & (matching_df['theo'] < 0)])
                    LOGGER.info(f'{account_key}@{session}: Theo balance - Long: {n_long_theo}, Short: {n_short_theo}')


                    # Check position balance (skip for spot sessions)
                    if n_short_theo != n_long_theo and 'spot' not in session:
                        message += [f'{account_key}@{session}: Theo pos imbalance (Long: {n_long_theo}, Short: {n_short_theo})']

                    if not self.processor_config[session].get('check_realpose', True):
                        continue


                    total_theo_expo = matching_df['theo'].sum()
                    if abs(total_theo_expo) > (seuil_theo * total_factor):
                        message += [f'{account_key}@{session}: Residual theo expo too large ({total_theo_expo:.2f})']

                    # Check residual account exposure
                    total_current_expo = matching_df['real'].sum()
                    if abs(total_current_expo) > (seuil_current * total_factor):
                        message += [f'{account_key}@{session}: Residual account expo too large ({total_current_expo:.2f})']

                    # Check residual theo exposure
                    LOGGER.info(f'{account_key}@{session}: Total theo expo: {total_theo_expo:.2f}, current expo: {total_current_expo:.2f}')

                    # Check position mismatches for significant positions
                    mismatched_positions = matching_df[matching_df['is_mismatch'] == True]
                    LOGGER.info(f'{account_key}@{session}: Mismatched positions count: {len(mismatched_positions)}')

                    if len(mismatched_positions) > 0:
                        LOGGER.info(f'Found {len(mismatched_positions)} mismatched positions for {account_key}@{session}')
                        for coin in mismatched_positions.index:
                            real_pos = mismatched_positions.loc[coin, 'real']
                            theo_pos = mismatched_positions.loc[coin, 'theo']
                            rel_delta = mismatched_positions.loc[coin, 'rel_delta']
                            message += [f'{account_key}@{session}: Position mismatch for {coin} - Real: {real_pos:.2f},'
                                        f'Theo: {theo_pos:.2f}, Rel delta: {rel_delta:.2%}']


        # Check for positions that should exist but don't
                    theo_only = significant_theo.difference(significant_current)
                    if len(theo_only) > 0:
                        message += [f'Discrepancy {account_key}@{session}: {theo_only} have no position in exchange account but should']

                    # Check for positions that exist but shouldn't
                    current_only = significant_current.difference(significant_theo)
                    if len(current_only) > 0:
                        message += [f'Discrepancy {account_key}@{session}: {current_only} have position in account but not in theo']

                except Exception as e:
                    LOGGER.error(f'Exception {e.args[0]} during check of {account_key}@{session}')
                    LOGGER.error(traceback.format_exc())

        return message

    async def refresh_quotes(self):
        tasks = []
        LOGGER.info('refreshing quotes')
        for session in self.perimeters:
            tasks.append(asyncio.create_task(self.fetch_quotes(session)))
        await asyncio.gather(*tasks)
        return

    async def refresh(self, with_matching=True):
        LOGGER.info('refreshing')

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
                        LOGGER.error(f'exception {e.args[0]} for {session}/{strategy_name} in refresh')
                        LOGGER.error(traceback.format_exc())
            try:
                await self.update_aum(session)
            except Exception as e:
                LOGGER.error(f'exception {e.args[0]} for {session} in update_aum')
                LOGGER.error(traceback.format_exc())
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
        elif exchange == 'hyperliquid':
            exchange_name = 'hyperliquid'
        else:
            return {}
        params = {
            'exchange_trade': exchange_name,
            'account_trade': account
        }
        end_point = None
        bh = None
        positions = {}
        cash = (None, None)
        try:
            end_point = BrokerHandler.build_end_point(exchange_name, account)
            bh = BrokerHandler(market_watch=exchange_name, end_point_trade=end_point, strategy_param=params, logger_name='default')
            try:
                positions = await end_point.get_positions_async()
            except Exception as e:
                LOGGER.warning(f'exchange/account {exchange_name}/{account} sent exception {e}')
                positions = {}

            quotes = await self._fetch_all_tickers(bh, end_point, positions)

            try:
                cash = await end_point.get_cash_async(['USDT', 'BTC'])
            except Exception:
                cash = (None, None)

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
        finally:
            # Ensure we always close async exchange clients
            try:
                if end_point is not None and hasattr(end_point, '_exchange_async') and end_point._exchange_async is not None:
                    try:
                        await end_point._exchange_async.close()
                    except Exception:
                        LOGGER.debug('Error closing end_point._exchange_async', exc_info=True)
            except Exception:
                LOGGER.debug('Error while attempting to close end_point', exc_info=True)

            try:
                if bh is not None:
                    try:
                        await bh.close_exchange_async()
                    except Exception:
                        LOGGER.debug('Error closing broker handler async session', exc_info=True)
            except Exception:
                LOGGER.debug('Error while attempting to close broker handler', exc_info=True)

    async def multiply(self, exchange, account, factor):
        positions = await self.get_account_position(exchange, account)
        if 'fut' in exchange and 'bin' in exchange:
            exchange_trade = 'binancefut'
            exclude = []
        elif 'bitget' in exchange:
            exchange_trade = 'bitget'
            exclude = []
        elif exchange == 'hyperliquid':
            exchange_trade = 'hyperliquid'
        else:
            exchange_trade = 'binance'
            exclude = ['BNB', 'BNBUSDT']

        params = {
            'exchange_trade': exchange_trade,
            'account_trade': account
        }
        end_point = None
        bh = None
        broker = None
        try:
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
        finally:
            try:
                if end_point is not None and hasattr(end_point, '_exchange_async') and end_point._exchange_async is not None:
                    try:
                        await end_point._exchange_async.close()
                    except Exception:
                        LOGGER.debug('Error closing end_point._exchange_async in multiply', exc_info=True)
            except Exception:
                LOGGER.debug('Error while attempting to close end_point in multiply', exc_info=True)

            try:
                if bh is not None:
                    try:
                        await bh.close_exchange_async()
                    except Exception:
                        LOGGER.debug('Error closing broker handler async session in multiply', exc_info=True)
            except Exception:
                LOGGER.debug('Error while attempting to close broker handler in multiply', exc_info=True)

