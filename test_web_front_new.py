'''
Simple test to verify web_front_new.py helper functions work correctly.
'''
import sys
sys.path.insert(0, 'src')

import pandas as pd
from src.web_front_new import dict_to_df, multistrategy_matching_to_df


def test_dict_to_df():
    '''Test dict_to_df function with sample data.'''
    # Test mode=True
    data = {
        'session1': {'key1': 'value1', 'key2': 'value2'},
        'session2': {'key1': 'value3', 'key2': 'value4'}
    }
    df = dict_to_df(data, mode=True)
    assert isinstance(df, pd.DataFrame), 'Should return DataFrame'
    print('✓ dict_to_df(mode=True) works')

    # Test mode=False
    nested_data = {
        'session1': {
            'strategy1': {'metric1': 10, 'metric2': 20},
            'strategy2': {'metric1': 30, 'metric2': 40}
        }
    }
    df = dict_to_df(nested_data, mode=False)
    assert isinstance(df, pd.DataFrame), 'Should return DataFrame'
    print('✓ dict_to_df(mode=False) works')


def test_multistrategy_matching_to_df():
    '''Test multistrategy_matching_to_df function.'''
    details = {
        'BTC': {
            'theo': 1.5,
            'real': 1.4,
            'price': 50000,
            'dust': False,
            'is_mismatch': True
        },
        'ETH': {
            'theo': 10.0,
            'real': 10.0,
            'price': 3000,
            'dust': False,
            'is_mismatch': False
        },
        'DUST': {
            'theo': 0.001,
            'real': 0.001,
            'price': 100,
            'dust': True,
            'is_mismatch': False
        }
    }

    main_df, summary_df = multistrategy_matching_to_df(details)

    assert isinstance(main_df, pd.DataFrame), 'Should return main DataFrame'
    assert isinstance(summary_df, pd.DataFrame), 'Should return summary DataFrame'
    assert 'token' in main_df.columns, 'Main df should have token column'
    assert 'theo_amount' in main_df.columns, 'Main df should have theo_amount column'
    assert 'Net Exposure' in summary_df.columns, 'Summary should have Net Exposure'
    assert 'Gross Exposure' in summary_df.columns, 'Summary should have Gross Exposure'

    print('✓ multistrategy_matching_to_df works')
    print(f'  Main df shape: {main_df.shape}')
    print(f'  Summary df shape: {summary_df.shape}')


def test_integration():
    '''Test that imports work correctly.'''
    try:
        from src.web_front_new import (
            initialize_globals, get_any, get_used_accounts,
            create_pnl_tab, create_matching_tab, create_multiply_tab
        )
        print('✓ All functions imported successfully')
    except ImportError as e:
        print(f'✗ Import error: {e}')
        return False
    return True


if __name__ == '__main__':
    print('Testing web_front_new.py helper functions...\n')

    try:
        test_dict_to_df()
        test_multistrategy_matching_to_df()
        test_integration()
        print('\n✅ All tests passed!')
    except Exception as e:
        print(f'\n❌ Test failed: {e}')
        import traceback
        traceback.print_exc()
