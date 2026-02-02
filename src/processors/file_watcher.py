"""
File watching infrastructure for bot monitoring.
Handles file system events and routes them to asyncio queue.
"""

import os
import asyncio
import logging
from datetime import datetime
from enum import Enum
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


SignalType = Enum('SignalType', [
    ('STOP', 'stop'),
    ('PRICE_UPDATE', 'update_prices'),
    ('SESSION_FILE', 'session_file'),
    ('PNL_FILE', 'pnl_file'),
    ('LATENT_FILE', 'latent_file'),
    ('AUM_FILE', 'aum_file'),
    ('POS_FILE', 'pos_file'),
    ('STATE_FILE', 'state_file'),
    ('CONFIG_FILE', 'config_file')
])


class FileChangeHandler(FileSystemEventHandler):
    """Handler for file system events that filters and routes to asyncio queue"""

    def __init__(self, watched_files, queue, loop, debounce_seconds=2.0, delay_seconds=600):
        super().__init__()
        self.watched_files = watched_files  # dict: file_path -> (session, entity, file_type)
        self.queue = queue
        self.loop = loop
        self.last_event_time = {}  # Debouncing: file_path -> datetime
        self.debounce_seconds = debounce_seconds
        self.delay_seconds = delay_seconds

    def on_modified(self, event):
        if event.is_directory:
            return

        file_path = os.path.abspath(event.src_path)

        # Check if this file is in our watch list
        if file_path not in self.watched_files:
            return

        # Debouncing: skip if we processed this file recently
        now = datetime.now()
        if file_path in self.last_event_time:
            elapsed = (now - self.last_event_time[file_path]).total_seconds()
            if elapsed < self.debounce_seconds:
                return

        self.last_event_time[file_path] = now

        # Get metadata about this file
        session, entity, file_type = self.watched_files[file_path]

        logging.info(f'File change detected: {file_path} (type={file_type}, session={session}, entity={entity})')

        # Push to asyncio queue
        asyncio.run_coroutine_threadsafe(
            self._delayed_put(file_type, session, entity, file_path),
            self.loop
        )

    async def _delayed_put(self, file_type, session, entity, file_path):
        await asyncio.sleep(self.delay_seconds)
        await self.queue.put((file_type, session, entity, file_path))


class FileWatcherManager:
    """Manages watchdog observers for file monitoring"""

    def __init__(self):
        self.observer = None
        self.file_handler = None
        self.watched_files = {}

    def build_watched_files_registry(self, session_configs, processor_config, used_accounts):
        """
        Build a comprehensive registry of all files to watch based on active strategies and accounts.
        Registry maps file_path -> (session, entity, file_type)
        """
        self.watched_files = {}

        # Add session heartbeat files
        for session, param_dict in processor_config.items():
            if 'session_file' in param_dict:
                session_file = param_dict['session_file']
                if os.path.exists(session_file):
                    abs_path = os.path.abspath(session_file)
                    self.watched_files[abs_path] = (session, session, SignalType.SESSION_FILE)
                    logging.info(f'Watching session file: {abs_path}')

        # Add config files
        config_files = {session: processor_config[session]['config_file'] for session in processor_config}
        config_position_matching_files = {session: processor_config[session]['config_position_matching_file'] for session in processor_config}

        for session, config_file in config_files.items():
            if os.path.exists(config_file):
                abs_path = os.path.abspath(config_file)
                self.watched_files[abs_path] = (session, session, SignalType.CONFIG_FILE)
                logging.info(f'Watching config file: {abs_path}')

        for session, config_file in config_position_matching_files.items():
            if os.path.exists(config_file):
                abs_path = os.path.abspath(config_file)
                self.watched_files[abs_path] = (session, session, SignalType.CONFIG_FILE)
                logging.info(f'Watching matching config file: {abs_path}')

        # Add per-strategy files
        for session, session_params in session_configs.items():
            strategies = session_params.get('strategy', {})
            working_directory = session_params.get('working_directory', '')

            for strategy_name, strategy_param in strategies.items():
                if not strategy_param.get('active', False):
                    continue

                strategy_dir = os.path.join(working_directory, strategy_name)

                # PnL file
                pnl_file = os.path.join(strategy_dir, 'pnl.csv')
                if os.path.exists(pnl_file):
                    abs_path = os.path.abspath(pnl_file)
                    self.watched_files[abs_path] = (session, strategy_name, SignalType.PNL_FILE)
                    logging.info(f'Watching PnL file: {abs_path}')

                # Latent profit file
                latent_file = os.path.join(strategy_dir, 'latent_profit.csv')
                if os.path.exists(latent_file):
                    abs_path = os.path.abspath(latent_file)
                    self.watched_files[abs_path] = (session, strategy_name, SignalType.LATENT_FILE)
                    logging.info(f'Watching latent file: {abs_path}')

                # Current state file
                persistence_file = os.path.join(strategy_dir, strategy_param.get('persistence_file', 'current_state.json'))
                if os.path.exists(persistence_file):
                    abs_path = os.path.abspath(persistence_file)
                    self.watched_files[abs_path] = (session, strategy_name, SignalType.STATE_FILE)
                    logging.info(f'Watching state file: {abs_path}')

        # Add per-account files
        for session, accounts in used_accounts.items():
            working_directory = session_configs[session].get('working_directory', '')

            account_set = set()
            for strategy_name, (trade_exchange, account) in accounts.items():
                account_set.add((trade_exchange, account))

            for trade_exchange, account in account_set:
                account_key = f'{trade_exchange}_{account}'
                account_dir = os.path.join(working_directory, account_key)

                # AUM file
                aum_file = os.path.join(account_dir, '_aum.csv')
                if os.path.exists(aum_file):
                    abs_path = os.path.abspath(aum_file)
                    self.watched_files[abs_path] = (session, account_key, SignalType.AUM_FILE)
                    logging.info(f'Watching AUM file: {abs_path}')

                # Theo position file
                theo_pos_file = os.path.join(account_dir, 'current_state_theo.pos')
                if os.path.exists(theo_pos_file):
                    abs_path = os.path.abspath(theo_pos_file)
                    self.watched_files[abs_path] = (session, account_key, SignalType.POS_FILE)
                    logging.info(f'Watching theo pos file: {abs_path}')

                # Real position file
                if trade_exchange != 'dummy':
                    real_pos_file = os.path.join(account_dir, 'current_state.pos')
                    if os.path.exists(real_pos_file):
                        abs_path = os.path.abspath(real_pos_file)
                        self.watched_files[abs_path] = (session, account_key, SignalType.POS_FILE)
                        logging.info(f'Watching real pos file: {abs_path}')

        logging.info(f'Built watched files registry with {len(self.watched_files)} files')
        return self.watched_files

    def start_file_watchers(self, loop, task_queue):
        """
        Start watchdog observers to monitor all registered files.
        Uses a single observer for all directories.
        """
        if not self.watched_files:
            logging.warning('No files to watch, skipping file watcher initialization')
            return

        # Collect all unique directories to watch
        directories = set()
        for file_path in self.watched_files.keys():
            directories.add(os.path.dirname(file_path))

        logging.info(f'Starting file watchers for {len(directories)} directories')

        # Create event handler and observer with correct queue
        self.file_handler = FileChangeHandler(self.watched_files, task_queue, loop)
        self.observer = Observer()

        # Schedule observer for each directory
        for directory in directories:
            if os.path.exists(directory):
                self.observer.schedule(self.file_handler, directory, recursive=False)
                logging.info(f'Watching directory: {directory}')
            else:
                logging.warning(f'Directory does not exist: {directory}')

        try:
            self.observer.start()
            logging.info('File watchers started successfully')
        except Exception as e:
            logging.error(f'Failed to start file watchers: {e}')
            import traceback
            logging.error(traceback.format_exc())

    def stop_file_watchers(self):
        """Stop all file watchers gracefully"""
        if self.observer:
            try:
                self.observer.stop()
                self.observer.join(timeout=5)
                logging.info('File watchers stopped successfully')
            except Exception as e:
                logging.error(f'Error stopping file watchers: {e}')
                import traceback
                logging.error(traceback.format_exc())

    def validate_file_watchers(self):
        """
        Validate that file watchers are still running.
        Returns True if healthy, False if restart needed.
        """
        if not self.observer:
            logging.error('Observer not initialized')
            return False

        if not self.observer.is_alive():
            logging.error('Observer thread is not alive')
            return False

        logging.debug('File watchers validation passed')
        return True
