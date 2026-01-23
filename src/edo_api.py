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

from processors.edo_processor import EdoProcessor
from shared_utils.online import parse_pair, utc_ize

global app
STOP = 'stop'
DATEFMT = '%Y/%m/%d %H:%M:%S'
LOGGER = logging.getLogger('edo_processor')

def runner(event, processor):
    async def run_processor():
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
        web_runner = asyncio.create_task(run_processor())
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
    th = threading.Thread(target=runner, args=(started, EdoProcessor(params),))
    th.start()
    started.wait()
    print('Started')
    th.join()


