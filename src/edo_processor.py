import json
import asyncio
import threading
import logging
import time
import os
import argparse
from datetime import datetime
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from starlette.responses import JSONResponse
import uvicorn

from shared_utils.online import parse_pair, utc_ize

global app
STOP = 'stop'
DATEFMT = '%Y/%m/%d %H:%M:%S'
LOGGER = logging.getLogger('edo_processor')

class Processor:
    '''
    Responsibility: interface between web query and strat status and commands
    trigger a refresh of pairs
    get status of each strat
    stop/start them
    get pos
    '''
    def __init__(self, params):
        self.params = params
        self.state = None
        self.summary = {}
        self.portfolio = {}
        self.killswitch = {}
        killswitch_state_file = self.params['state_file']
        if os.path.exists(killswitch_state_file):
            try:
                with open(killswitch_state_file, 'r') as myfile:
                    self.killswitch_state = json.load(myfile)
            except json.JSONDecodeError as e:
                LOGGER.error(f'Error reading killswitch file {killswitch_state_file}: {e}')
                self.killswitch_state = {}
        else:
            self.killswitch_state = {}
            with open(killswitch_state_file, 'w') as myfile:
                j = json.dumps(self.killswitch_state, indent=4)
                print(j, file=myfile)
        self.refresh()

    def refresh(self):
        self.summary = {}
        working_directory = self.params['working_directory']
        hearbeat_file = self.params['heartbeat']
        strategies = self.params['strategies']

        last_modif = None
        killswitchfile = self.params['killswitch']
        if os.path.exists(killswitchfile):
            try:
                with open(killswitchfile, 'r') as myfile:
                    self.killswitch = json.load(myfile)
            except json.JSONDecodeError as e:
                LOGGER.error(f'Error reading killswitch file {killswitchfile}: {e}')
                self.killswitch = {}

        if os.path.exists(hearbeat_file):
            last_modif = utc_ize(os.path.getmtime(hearbeat_file))
        self.summary = {'last_update': str(last_modif)}

        LOGGER.info(f'Filling data for all strat')
        need_save = False
        for strategy_code, strategy_param in strategies.items():
            strategy_name = strategy_param['name']

            action = self.get_risk_action(strategy_name)
            if action != '':
                LOGGER.info(f'Action: {action} for strategy {strategy_name}')
                self.do_action(strategy_name, action)
                need_save = True

            if strategy_name in self.killswitch_state and self.killswitch_state[strategy_name] in ['stop', 'hold']:
                LOGGER.info(f'Strategy {strategy_name} is stopped or on hold, skipping')
                self.summary[strategy_code] = {'last_update': self.summary.get('last_update', '')}
                continue
            strategy_directory = os.path.join(working_directory, strategy_name)
            persistence_file = os.path.join(strategy_directory, strategy_param['persistence_file'])
            if os.path.exists(persistence_file):
                with open(persistence_file, 'r') as myfile:
                    self.state = json.load(myfile)
            strat_summary = {}

            if strategy_param['type'] == 'pair':
                for pair_name, pair_info in self.state['current_pair_info'].items():
                    s1, s2 = parse_pair(pair_name)
                    new_pair_name = pair_name.replace('-USDT-SWAP', 'USDT')
                    new_s1 = s1.replace('-USDT-SWAP', 'USDT')
                    new_s2 = s2.replace('-USDT-SWAP', 'USDT')

                    if 'position' not in pair_info:
                        continue
                    if 'in_execution' in pair_info and pair_info['in_execution'] and pair_info['position'] != 0:
                        continue
                    if 'in_execution' in pair_info:
                        if pair_info['in_execution']:
                            if 'target_qty' in pair_info and 'target_price' in pair_info:
                                strat_summary[new_pair_name] = {
                                    'in_execution': pair_info['in_execution'],
                                    'target_qty': pair_info['target_qty'],
                                    'target_price': pair_info['target_price']
                                }
                        else:
                            if 'entry_data' in pair_info and 'quantity' in pair_info:
                                strat_summary[new_pair_name] = {
                                    'in_execution': pair_info['in_execution'],
                                    'entry_ts': f'{datetime.fromtimestamp(pair_info["entry_data"][2] / 1e9)}',
                                    new_s1: {
                                        'ref_price': pair_info['entry_data'][0],
                                        'quantity': pair_info['quantity'][0]},
                                    new_s2: {
                                        'ref_price': pair_info['entry_data'][1],
                                        'quantity': pair_info['quantity'][1]}
                                }
            else:
                for s1, coin_info in self.state['current_coin_info'].items():
                    new_s1 = s1.replace('-USDT-SWAP', 'USDT')

                    if 'in_execution' in coin_info and coin_info['in_execution']:
                        continue
                    if coin_info.get('position', 0) != 0 and 'entry_data' in coin_info and 'quantity' in coin_info:
                        strat_summary[new_s1] = {
                            'in_execution': coin_info.get('in_execution', False),
                            'entry_ts': f'{datetime.fromtimestamp(coin_info["entry_data"][1] / 1e9)}',
                            new_s1: {
                                'ref_price': coin_info['entry_data'][0],
                                'quantity': coin_info['quantity'] * coin_info['position']}
                        }

                    # if 'in_execution' in coin_info:
                    #     if coin_info['in_execution']:
                    #         if 'target_qty' in coin_info and 'target_price' in coin_info:
                    #             strat_summary[new_s1] = {
                    #                 'in_execution': coin_info['in_execution'],
                    #                 'target_qty': coin_info['target_qty'],
                    #                 'target_price': coin_info['target_price']
                    #             }
                    #     else:
                    #         if 'entry_data' in coin_info and 'quantity' in coin_info:
                    #             strat_summary[new_s1] = {
                    #                 'in_execution': coin_info['in_execution'],
                    #                 'entry_ts': f'{datetime.fromtimestamp(coin_info["entry_data"][1] / 1e9)}',
                    #                 new_s1: {
                    #                     'ref_price': coin_info['entry_data'][0],
                    #                     'quantity': coin_info['quantity'] * coin_info['position']}
                    #             }
            self.summary[strategy_code] = strat_summary
            self.summary[strategy_code].update({'last_update': self.summary.get('last_update', '')})

        if need_save:
            killswitch_state_file = self.params['state_file']
            LOGGER.info(f'Saving killswitch state {self.killswitch_state} to {killswitch_state_file}')
            with open(killswitch_state_file, 'w') as myfile:
                j = json.dumps(self.killswitch_state, indent=4)
                print(j, file=myfile)

    def get_pose(self, strategy):
        return self.summary.get(strategy, {})

    def get_portfolio(self):
        combined_positions = {'last_update': self.summary.get('last_update', '')}
        combined = {}
        portfolios = {}
        ratios = {}
        for strategy, positions in self.summary.items():
            current_ratio = self.params['strategies'].get(strategy, {}).get('ratio', 0)

            if current_ratio > 0 and isinstance(positions, dict):

                ratios[strategy] = current_ratio
                portfolios[strategy] = {}
                for name, value in positions.items():
                    if isinstance(value, dict):
                        for coin, info in value.items():
                            if 'USD' not in coin:
                                continue
                            price = info['ref_price']
                            qty = info['quantity']
                            amount = portfolios[strategy].get(coin, 0.0) + price * qty
                            portfolios[strategy][coin] = amount
                long = 0.0
                short = 0.0
                for name, amount in portfolios[strategy].items():
                    if amount > 0:
                        long = long + amount
                    if amount < 0:
                        short = short + amount
                portfolios[strategy]['long_amount'] = long
                portfolios[strategy]['short_amount'] = short

        for strategy, portfolio in portfolios.items():
            long  = portfolio['long_amount']
            short = portfolio['short_amount']
            if long == 0 and short == 0:
                continue
            pf_ratio = ratios.get(strategy, 0)
            for key, info in portfolio.items():
                if 'USD' in key and isinstance(info, float):
                    coin_ratio = info / (long if info > 0 else -short)
                    combined_ratio = pf_ratio * coin_ratio + combined.get(key, 0)
                    combined[key] = combined_ratio
        combined_positions.update({'positions': combined})

        return combined_positions

    def get_risk_action(self, strategy_name):
        '''
        Get the risk action for a strategy
        '''
        if strategy_name not in self.killswitch:
            LOGGER.info(f'{strategy_name} not found in killswitch, skipping')
            return ''

        params = self.params['strategies']

        for strat_code, strat_param in params.items():
            if strat_param['name'] == strategy_name:
                indicators = self.killswitch[strategy_name]
                state = self.killswitch_state.get(strategy_name, 'on')  # or hold or stopped
                stop_loss_real = strat_param.get('stop_loss_real', -1)
                restart_theo_1d = strat_param.get('restart_theo_1d', 1)
                restart_theo_2d = strat_param.get('restart_theo_2d', 1)
                theopnl_1d = indicators.get('theopnl_1d', 0.0)
                theopnl_2d = indicators.get('theopnl_2d', 0.0)
                realpnl_1d = indicators.get('realpnl_1d', 0.0)
                realpnl_2d = indicators.get('realpnl_2d', 0.0)
                latent_theo = indicators.get('latent_theo', 0.0)

                desired_start = False
                desired_stop = False

                theo_1d = theopnl_1d + latent_theo
                theo_2d = theopnl_2d+ latent_theo

                if theo_1d > restart_theo_1d or theo_2d > restart_theo_2d:
                    LOGGER.warning(f'Start desired for {strategy_name} with theo_1d {theo_1d} and'
                                   f' theo_2d {theo_2d}')
                    desired_start = True
                if realpnl_1d < stop_loss_real or realpnl_2d < stop_loss_real:
                    LOGGER.warning(f'Stop desired for {strategy_name} with realpnl_1d {realpnl_1d} and'
                                   f' realpnl_2d {realpnl_2d}')
                    desired_stop = True

                if desired_start and desired_stop:
                    return 'conflict'
                if state == 'hold' and desired_start:
                    return 'resume'
                if state == 'on' and desired_stop:
                    return 'hold'
        return ''

    def do_action(self, strategy_name, action):
        if action != '':
            if action == 'hold':
                LOGGER.warning(
                    f'Stop requested for {strategy_name} with indicators {self.killswitch[strategy_name]}')
                self.killswitch_state[strategy_name] = 'hold'
            elif action == 'resume':
                LOGGER.warning(
                    f'Restart requested for {strategy_name} with indicators {self.killswitch[strategy_name]}')
                self.killswitch_state[strategy_name] = 'on'
            elif action == 'conflict':
                LOGGER.error('Conflict detected, cannot hold and resume at the same time')

def runner(event, processor):
    async def run_web_processor():
        logger = LOGGER
        handler = logging.FileHandler(filename='output_edo/web_processor.log')
        utc_formatter = logging.Formatter('{asctime}.{msecs:.0f};{levelname};{name};{message}', style='{',
                                          datefmt=DATEFMT)
        utc_formatter.converter = time.gmtime
        handler.setFormatter(utc_formatter)
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)

        global app
        app = FastAPI()

        app.mount("/static", StaticFiles(directory="output", html=False), name="output")

        @app.get('/pose')
        async def read_pose(strategy: int=1):
            report = processor.get_pose(str(strategy))
            return JSONResponse(report)

        @app.get('/portfolio')
        async def read_portfolio(strategy: int=1):
            report = processor.get_portfolio()
            return JSONResponse(report)

        config = uvicorn.Config(app, port=14040, host='0.0.0.0', lifespan='on', log_config=None)
        server = uvicorn.Server(config)
        await server.serve()

    async def heartbeat(queue, pace, action):
        while True:
            await asyncio.sleep(pace)
            await queue.put(action)

    async def refresh():
        processor.refresh()

    async def main():
        event.loop = asyncio.get_event_loop()
        event.queue = asyncio.Queue()
        task_queue = asyncio.Queue()
        event.set()
        web_runner = asyncio.create_task(run_web_processor())
        heart_runner = asyncio.create_task(heartbeat(task_queue, 30, 'refresh'))

        while True:
            item = await task_queue.get()
            if item == STOP:
                task_queue.task_done()
                break
            task = asyncio.create_task(refresh())
            task.add_done_callback(lambda _: task_queue.task_done())
        web_runner.cancel()
        heart_runner.cancel()
        await task_queue.join()

    asyncio.run(main())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="input file", required=True)
    args = parser.parse_args()
    config_file = args.config

    with open(config_file, 'r') as myfile:
        params = json.load(myfile)

    started = threading.Event()
    th = threading.Thread(target=runner, args=(started, Processor(params),))
    th.start()
    started.wait()
    print('Started')
    th.join()


