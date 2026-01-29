import sys
import asyncio
import threading
import logging
import argparse
import traceback
import yaml
from pathlib import Path
try:
    from datetime import UTC
except:
    from datetime import timezone
    UTC = timezone.utc
from fastapi import FastAPI, HTTPException
import uvicorn
from dotenv import load_dotenv

from utils_files import calculate_median_position_sizes, JSONResponse
from processors.file_watcher import SignalType
from processors.web_processor import WebProcessor
from trading_bot.web_broker import WebSpreaderBroker
from shared_utils.bot_reporting import TGMessenger

app = None

LOGGER = logging.getLogger(__name__)


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
            try:
                # Check if report is a DataFrame
                if hasattr(report, 'to_dict'):
                    return JSONResponse(report.to_dict(orient='index'))
                else:
                    # Already a dict or string
                    return JSONResponse(report if isinstance(report, dict) else {'message': report})
            except Exception as e:
                return JSONResponse({'session': session,
                                     'account_key': account_key,
                                     'error': str(e)})
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

    async def heartbeat_price_update(queue, pace):
        """Only heartbeat for periodic quote refreshing"""
        while True:
            await queue.put((SignalType.PRICE_UPDATE, None, None, None))
            await asyncio.sleep(pace)

    async def heartbeat_validation(processor, pace, task_queue):
        """Periodic validation that file watchers are still alive"""
        while True:
            await asyncio.sleep(pace)
            try:
                if not processor.validate_file_watchers():
                    LOGGER.error('File watcher validation failed, attempting restart')
                    processor.stop_file_watchers()
                    loop = asyncio.get_event_loop()
                    processor.start_file_watchers(loop, task_queue)
            except Exception as e:
                LOGGER.error(f'Error in watcher validation: {e}')
                LOGGER.error(traceback.format_exc())

    async def refresh(with_matching):
        await processor.refresh(with_matching)

    async def refresh_quotes():
        await processor.refresh_quotes()
        # Recalculate median position sizes with fresh prices
        processor.median_position_sizes = calculate_median_position_sizes(
            processor.account_theo_pos, processor.quotes)

    async def send_alert(message):
        if message is not None:
            for error in message:
                TGMessenger.send(error, 'CM')
            LOGGER.info(f'Sent {len(message)} msg')

    async def check(checking_coro):
        messages = await checking_coro
        await send_alert(messages)

    async def handle_file_event(file_type, session, entity, file_path, task_queue):
        """Handle specific file change events"""
        try:
            if file_type == SignalType.SESSION_FILE:
                LOGGER.info(f'Session file changed: {file_path}')
                await check(processor.check_running())

            elif file_type == SignalType.CONFIG_FILE:
                LOGGER.info(f'Config file changed: {file_path}')
                await processor.update_config(session, file_path)
                # Rebuild file watchers with new config
                processor.stop_file_watchers()
                processor.build_watched_files_registry()
                loop = asyncio.get_event_loop()
                processor.start_file_watchers(loop, task_queue)

            elif file_type == SignalType.PNL_FILE or file_type == SignalType.LATENT_FILE:
                LOGGER.info(f'PnL/Latent file changed: {file_path}')
                strategy_name = entity
                if session in processor.session_configs:
                    working_directory = processor.session_configs[session].get('working_directory', '')
                    strategies = processor.session_configs[session].get('strategy', {})
                    if strategy_name in strategies:
                        strategy_param = strategies[strategy_name]
                        await processor.update_pnl(session, working_directory, strategy_name, strategy_param)
                        if file_type == SignalType.LATENT_FILE:
                            await check(processor.check_latent())

            elif file_type == SignalType.AUM_FILE:
                LOGGER.info(f'AUM file changed: {file_path}')
                await processor.update_aum(session)

            elif file_type == SignalType.POS_FILE:
                LOGGER.info(f'Position file changed: {file_path}')
                # Refresh quotes first to ensure matching uses fresh prices
                await processor.refresh_quotes()
                processor.update_account_multi()
                for sess in processor.session_configs:
                    processor.do_matching(sess)
                await check(processor.check_all())

            elif file_type == SignalType.STATE_FILE:
                LOGGER.info(f'State file changed: {file_path}')
                strategy_name = entity
                if session in processor.session_configs:
                    working_directory = processor.session_configs[session].get('working_directory', '')
                    strategies = processor.session_configs[session].get('strategy', {})
                    if strategy_name in strategies:
                        strategy_param = strategies[strategy_name]
                        await processor.update_summary(session, working_directory, strategy_name, strategy_param)

        except Exception as e:
            LOGGER.error(f'Error handling file event for {file_path}: {e}')
            LOGGER.error(traceback.format_exc())

    async def main():
        event.loop = asyncio.get_event_loop()
        task_queue = asyncio.Queue()

        # Initialize: load all data once, build file registry, start watchers
        LOGGER.info('Performing initial data load')
        await refresh(with_matching=True)  # Enable matching to detect initial mismatches

        processor.build_watched_files_registry()
        processor.start_file_watchers(event.loop, task_queue)

        event.set()

        # Start background tasks
        web_runner = asyncio.create_task(run_web_processor())
        quote_runner = asyncio.create_task(heartbeat_price_update(task_queue, pace['PRICE_UPDATE']))
        validation_runner = asyncio.create_task(heartbeat_validation(processor, pace['CHECK'], task_queue))  # 10 min validation
        # Delay first quote fetch so API starts immediately
        initial_quote_runner = asyncio.create_task(refresh_quotes())

        LOGGER.info('Event loop started, processing file events')

        while True:
            item = await task_queue.get()

            if isinstance(item, tuple) and len(item) == 4:
                file_type, session, entity, file_path = item

                if file_type == SignalType.PRICE_UPDATE:
                    task = asyncio.create_task(refresh_quotes())
                    task.add_done_callback(lambda _: task_queue.task_done())
                else:
                    # Handle file change event
                    task = asyncio.create_task(handle_file_event(file_type, session, entity, file_path, task_queue))
                    task.add_done_callback(lambda _: task_queue.task_done())

            elif item == SignalType.STOP:
                task_queue.task_done()
                break
            else:
                # Unknown event type
                LOGGER.warning(f'Unknown event type: {item}')
                task_queue.task_done()

        # Cleanup
        LOGGER.info('Stopping background tasks')
        web_runner.cancel()
        quote_runner.cancel()
        validation_runner.cancel()
        processor.stop_file_watchers()
        await task_queue.join()

    asyncio.run(main())


if __name__ == '__main__':
    load_dotenv()
    LOGGER = logging.getLogger('web_processor')
        
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
    fmt = logging.Formatter('{asctime}:{levelname}:{name}:{message}', style='{')
    filename = config.get('logs', {}).get('file', '~/logs/web_processor.log')
    filename = Path(filename).expanduser()
    filename = filename.parent / 'backend.log'
    fh = logging.FileHandler(filename)
    fh.setFormatter(fmt)
    level = config.get('logs', {}).get('level', 'INFO')
    LOGGER.setLevel(level)
    LOGGER.addHandler(fh)

    started = threading.Event()
    processor = WebProcessor(config)
    th = threading.Thread(target=runner, args=(started, processor, pace,))

    LOGGER.info('Starting')
    th.start()
    started.wait()
    LOGGER.info('Started')
    th.join()
    LOGGER.info('Stopped')