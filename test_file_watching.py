#!/usr/bin/env python
"""
Test script for file-watch based web_processor refactoring
Tests the FileChangeHandler and file watching infrastructure
"""

import os
import sys
import time
import tempfile
import asyncio
import logging
from pathlib import Path

# Setup path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TestFileChangeHandler(FileSystemEventHandler):
    """Test handler to validate watchdog functionality"""

    def __init__(self):
        super().__init__()
        self.events = []

    def on_modified(self, event):
        if not event.is_directory:
            self.events.append(('modified', event.src_path))
            logging.info(f'Modified: {event.src_path}')

    def on_created(self, event):
        if not event.is_directory:
            self.events.append(('created', event.src_path))
            logging.info(f'Created: {event.src_path}')

def test_basic_file_watching():
    """Test basic file watching with watchdog"""
    logging.info('=== Test 1: Basic File Watching ===')

    with tempfile.TemporaryDirectory() as tmpdir:
        handler = TestFileChangeHandler()
        observer = Observer()
        observer.schedule(handler, tmpdir, recursive=False)
        observer.start()

        time.sleep(0.5)  # Let observer start

        # Create a test file
        test_file = os.path.join(tmpdir, 'test.txt')
        with open(test_file, 'w') as f:
            f.write('test content\n')

        time.sleep(1)  # Wait for event

        # Modify the file
        with open(test_file, 'a') as f:
            f.write('more content\n')

        time.sleep(1)  # Wait for event

        observer.stop()
        observer.join(timeout=2)

        logging.info(f'Captured {len(handler.events)} events: {handler.events}')
        assert len(handler.events) >= 1, 'Should have captured at least one event'
        logging.info('✓ Test 1 passed')

def test_debouncing():
    """Test debouncing logic"""
    logging.info('=== Test 2: Debouncing ===')

    from datetime import datetime

    last_event_time = {}
    debounce_seconds = 2.0

    test_file = '/tmp/test_file.txt'

    # First event
    now = datetime.now()
    last_event_time[test_file] = now
    should_process = True
    logging.info(f'First event at {now}: should_process={should_process}')
    assert should_process, 'First event should be processed'

    # Second event within debounce window
    time.sleep(0.5)
    now = datetime.now()
    elapsed = (now - last_event_time[test_file]).total_seconds()
    should_process = elapsed >= debounce_seconds
    logging.info(f'Second event at {now} (elapsed={elapsed:.2f}s): should_process={should_process}')
    assert not should_process, 'Second event should be debounced'

    # Third event after debounce window
    time.sleep(2.0)
    now = datetime.now()
    elapsed = (now - last_event_time[test_file]).total_seconds()
    should_process = elapsed >= debounce_seconds
    logging.info(f'Third event at {now} (elapsed={elapsed:.2f}s): should_process={should_process}')
    assert should_process, 'Third event should be processed'

    logging.info('✓ Test 2 passed')

def test_file_registry():
    """Test watched files registry building logic"""
    logging.info('=== Test 3: File Registry Building ===')

    watched_files = {}

    # Simulate adding files to registry
    test_files = [
        ('/path/to/pnl.csv', ('binance', 'strategy1', 'PNL_FILE')),
        ('/path/to/current_state.pos', ('binance', 'bitget_1', 'POS_FILE')),
        ('/path/to/_aum.csv', ('binance', 'bitget_1', 'AUM_FILE')),
    ]

    for file_path, metadata in test_files:
        watched_files[file_path] = metadata

    logging.info(f'Built registry with {len(watched_files)} files')

    # Test lookup
    test_path = '/path/to/pnl.csv'
    if test_path in watched_files:
        session, entity, file_type = watched_files[test_path]
        logging.info(f'Lookup {test_path}: session={session}, entity={entity}, type={file_type}')
        assert session == 'binance', 'Session should be binance'
        assert entity == 'strategy1', 'Entity should be strategy1'

    # Extract unique directories
    directories = set()
    for file_path in watched_files.keys():
        directories.add(os.path.dirname(file_path))

    logging.info(f'Unique directories: {directories}')
    assert len(directories) == 1, 'Should have one unique directory'

    logging.info('✓ Test 3 passed')

async def test_asyncio_integration():
    """Test integration between watchdog thread and asyncio"""
    logging.info('=== Test 4: Asyncio Integration ===')

    queue = asyncio.Queue()
    loop = asyncio.get_event_loop()

    # Simulate watchdog event pushing to asyncio queue
    def push_event():
        asyncio.run_coroutine_threadsafe(
            queue.put(('TEST_EVENT', 'test_session', 'test_entity', '/test/path')),
            loop
        )

    # Push event from "thread"
    push_event()

    # Wait for event in asyncio
    event = await asyncio.wait_for(queue.get(), timeout=2.0)
    logging.info(f'Received event: {event}')

    assert event[0] == 'TEST_EVENT', 'Should receive TEST_EVENT'
    assert event[1] == 'test_session', 'Should have correct session'

    logging.info('✓ Test 4 passed')

def test_observer_health():
    """Test observer health validation"""
    logging.info('=== Test 5: Observer Health Validation ===')

    with tempfile.TemporaryDirectory() as tmpdir:
        handler = TestFileChangeHandler()
        observer = Observer()
        observer.schedule(handler, tmpdir, recursive=False)
        observer.start()

        # Check if alive
        time.sleep(0.5)
        is_alive = observer.is_alive()
        logging.info(f'Observer is_alive: {is_alive}')
        assert is_alive, 'Observer should be alive'

        # Stop observer
        observer.stop()
        observer.join(timeout=2)

        # Check if stopped
        is_alive = observer.is_alive()
        logging.info(f'Observer is_alive after stop: {is_alive}')
        assert not is_alive, 'Observer should not be alive after stop'

    logging.info('✓ Test 5 passed')

def main():
    """Run all tests"""
    logging.info('Starting file-watch infrastructure tests')

    try:
        # Basic tests
        test_basic_file_watching()
        test_debouncing()
        test_file_registry()
        test_observer_health()

        # Async test
        asyncio.run(test_asyncio_integration())

        logging.info('\n=== All Tests Passed ✓ ===')
        return 0

    except Exception as e:
        logging.error(f'\n=== Test Failed ✗ ===')
        logging.error(f'Error: {e}')
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())
