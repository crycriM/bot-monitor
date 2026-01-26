import pytest
import yaml
import asyncio
import tempfile
import json
import os
from pathlib import Path
from datetime import datetime
try:
    from datetime import UTC
except ImportError:
    from datetime import timezone
    UTC = timezone.utc
from src.processors.web_processor import WebProcessor


@pytest.fixture
def temp_config_dir():
    """Create temporary directory structure for test configs and files"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create working directory structure
        working_dir = Path(tmpdir) / 'working'
        working_dir.mkdir()

        # Create session output directory
        output_dir = Path(tmpdir) / 'output_test'
        output_dir.mkdir()

        # Create strategy directory
        strategy_dir = working_dir / 'test_strategy'
        strategy_dir.mkdir()

        # Create account directory
        account_dir = working_dir / 'binance_1'
        account_dir.mkdir()

        yield {
            'root': tmpdir,
            'working': str(working_dir),
            'output': str(output_dir),
            'strategy': str(strategy_dir),
            'account': str(account_dir)
        }


@pytest.fixture
def test_config(temp_config_dir):
    """Create test YAML configuration"""
    config_file = Path(temp_config_dir['root']) / 'test_config.json'
    matching_file = Path(temp_config_dir['root']) / 'test_matching.json'
    session_file = Path(temp_config_dir['output']) / 'session.log'

    # Create session config
    session_config = {
        'exchange': 'binance',
        'working_directory': temp_config_dir['working'],
        'strategy': {
            'test_strategy': {
                'active': True,
                'account_trade': '1',
                'exchange_trade': 'binance',
                'type': 'spreader',
                'send_orders': 'dummy',
                'max_total_expo': 10000,
                'nb_short': 5,
                'nb_long': 5,
                'persistence_file': 'state.json',
                'allocation': 1000,
                'leverage': 1
            }
        }
    }

    with open(config_file, 'w') as f:
        json.dump(session_config, f)

    # Create matching config
    matching_config = {}
    with open(matching_file, 'w') as f:
        json.dump(matching_config, f)

    # Create session log file
    session_file.touch()

    # Create YAML config
    yaml_config = {
        'working_directory': temp_config_dir['working'],
        'pace': {
            'REFRESH': 180,
            'RUNNING': 600,
            'CHECK': 600,
            'MATCHING': 60,
            'PRICE_UPDATE': 600
        },
        'session': {
            'binance': {
                'config_file': str(config_file),
                'config_position_matching_file': str(matching_file),
                'check_running': True,
                'check_realpose': True,
                'check_pnl': True,
                'session_file': str(session_file)
            }
        }
    }

    return yaml_config, temp_config_dir


@pytest.fixture
def create_test_files(temp_config_dir):
    """Create test data files"""
    strategy_dir = Path(temp_config_dir['strategy'])
    account_dir = Path(temp_config_dir['account'])

    # Create state file
    state_file = strategy_dir / 'state.json'
    state_data = {
        'current_coin_info': {
            'BTCUSDT': {
                'position': 1,
                'quantity': 0.5,
                'in_execution': False,
                'entry_data': [50000.0, datetime.now(UTC).timestamp() * 1e9]
            }
        }
    }
    with open(state_file, 'w') as f:
        json.dump(state_data, f)

    # Create PnL file
    pnl_file = strategy_dir / 'pnl.csv'
    pnl_content = 'date,pnl_theo,allocation\n'
    pnl_content += f'{datetime.now(UTC).isoformat()},100,1000\n'
    with open(pnl_file, 'w') as f:
        f.write(pnl_content)

    # Create latent file
    latent_file = strategy_dir / 'latent_profit.csv'
    latent_content = 'date,latent_return,latent_pnl\n'
    latent_content += f'{datetime.now(UTC).isoformat()},0.01,50\n'
    with open(latent_file, 'w') as f:
        f.write(latent_content)

    # Create AUM file
    aum_file = account_dir / '_aum.csv'
    aum_content = 'date,aum\n'
    aum_content += f'{datetime.now(UTC).isoformat()},10000\n'
    with open(aum_file, 'w') as f:
        f.write(aum_content)

    # Create position files
    theo_pos_file = account_dir / 'current_state_theo.pos'
    with open(theo_pos_file, 'w') as f:
        json.dump({'BTCUSDT': 0.5, 'ETHUSDT': 1.0}, f)

    real_pos_file = account_dir / 'current_state.pos'
    with open(real_pos_file, 'w') as f:
        json.dump({'BTCUSDT': 0.5, 'ETHUSDT': 0.9}, f)


def test_web_processor_initialization(test_config):
    """Test WebProcessor initialization with YAML config"""
    yaml_config, temp_dirs = test_config

    processor = WebProcessor(yaml_config)

    # Verify session configs loaded
    assert 'binance' in processor.session_configs
    assert processor.session_configs['binance']['exchange'] == 'binance'

    # Verify strategy configs loaded
    assert 'test_strategy' in processor.session_configs['binance']['strategy']
    assert processor.session_configs['binance']['strategy']['test_strategy']['active']

    # Verify used_accounts populated
    assert 'binance' in processor.used_accounts
    assert 'test_strategy' in processor.used_accounts['binance']
    assert processor.used_accounts['binance']['test_strategy'] == ('dummy', '1')


@pytest.mark.asyncio
async def test_web_processor_refresh(test_config, create_test_files):
    """Test WebProcessor refresh functionality"""
    yaml_config, temp_dirs = test_config

    processor = WebProcessor(yaml_config)

    # Perform refresh
    await processor.refresh(with_matching=False)

    # Verify summaries updated
    assert 'binance' in processor.summaries
    assert 'test_strategy' in processor.summaries['binance']

    # Verify PnL updated
    assert 'binance' in processor.pnl
    assert 'test_strategy' in processor.pnl['binance']
    assert 'mean_theo' in processor.pnl['binance']['test_strategy']

    # Verify latent updated
    assert 'binance' in processor.latent
    assert 'test_strategy' in processor.latent['binance']


@pytest.mark.asyncio
async def test_web_processor_matching(test_config, create_test_files):
    """Test position matching functionality"""
    yaml_config, temp_dirs = test_config

    processor = WebProcessor(yaml_config)
    processor.quotes['binance'] = {'BTCUSDT': 50000.0, 'ETHUSDT': 3000.0}

    # Update account positions
    processor.update_account_multi()

    # Perform matching
    processor.do_matching('binance')

    # Verify matching results
    assert 'binance' in processor.matching
    assert 'binance_1' in processor.matching['binance']

    matching_df = processor.matching['binance']['binance_1']
    assert 'BTCUSDT' in matching_df.index
    assert 'ETHUSDT' in matching_df.index
    assert 'delta_qty' in matching_df.columns
    assert 'is_mismatch' in matching_df.columns


def test_file_watcher_registry(test_config):
    """Test file watcher registry building"""
    yaml_config, temp_dirs = test_config

    processor = WebProcessor(yaml_config)
    registry = processor.build_watched_files_registry()

    # Verify registry structure
    assert 'binance' in registry
    assert 'session_file' in registry['binance']
    assert 'config_file' in registry['binance']
    assert 'strategies' in registry['binance']
    assert 'test_strategy' in registry['binance']['strategies']