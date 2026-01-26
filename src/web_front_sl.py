import os
import argparse
import yaml
import json
import requests
import numpy as np
import pandas as pd
from pathlib import Path
import streamlit as st
from streamlit import session_state as ss

from utils_files import get_temp_dir

CONFIG = None
GATEWAY = None


def initialize_globals(config_file: str, gw_port: int):
    '''Initialize global configuration variables.'''
    global CONFIG, GATEWAY

    if not config_file:
        raise ValueError('No config file provided. Use --config to specify the config file path.')

    with open(config_file, 'r') as myfile:
        CONFIG = yaml.load(myfile, Loader=yaml.FullLoader)

    GATEWAY = f'http://localhost:{gw_port}'


def get_any(uri, params):
    '''Make GET request to API endpoint.'''
    if GATEWAY is None:
        raise RuntimeError('GATEWAY not initialized. Call initialize_globals() first.')
    response = requests.get(uri, params=params)
    return response


def dict_to_df(data_dict, mode=True):
    '''Convert nested dictionary to pandas DataFrame.'''
    try:
        if mode:
            df = pd.DataFrame.from_dict(data_dict, orient='index').stack().to_frame()
            df = pd.DataFrame(df[0].values.tolist(), index=df.index)
        else:
            dic_new = {(outerKey, innerKey, secondKey): values for outerKey, innerDict in data_dict.items()
                       for innerKey, secondDict in innerDict.items() for secondKey, values in secondDict.items()}
            df = pd.DataFrame(dic_new).T
            df.sort_index(axis=1, inplace=True)
    except Exception as e:
        print(f'Error in dict_to_df: {e}')
        df = pd.DataFrame()
    return df


def replace_na_with_nan(df):
    '''Replace string 'N/A' with NaN in DataFrame.'''
    return df.replace('N/A', np.nan)


def multistrategy_matching_to_df(details_dict):
    '''
    Convert multistrategy position details JSON to a main DataFrame and a summary DataFrame for exposures.
    Excludes positions marked as dust (is_dust: True) from the main DataFrame.
    '''
    try:
        # Create main DataFrame
        main_df = pd.DataFrame.from_dict(details_dict, orient='index')
        main_df.reset_index(inplace=True)
        main_df.rename(columns={'index': 'token'}, inplace=True)
        main_df = main_df[['token', 'theo', 'real', 'price', 'dust', 'is_mismatch']]
        main_df = replace_na_with_nan(main_df)

        main_df['ref_price'] = main_df['price'].fillna(0.0)
        main_df['theo_qty'] = main_df['theo'].fillna(0)
        main_df['theo_amount'] = main_df['theo_qty'] * main_df['ref_price']
        main_df['real_qty'] = main_df['real'].fillna(0)
        main_df['real_amount'] = main_df['real_qty'] * main_df['ref_price']
        main_df['is_dust'] = main_df['dust'].fillna(False)
        main_df['is_mismatch'] = main_df['is_mismatch'].fillna(False)

        # Create summary DataFrame (includes all positions, even dust)
        summary_df = pd.DataFrame({
            'Net Exposure': [main_df['theo_amount'].sum(), main_df['real_amount'].sum()],
            'Gross Exposure': [int(main_df['theo_amount'].abs().sum()), int(main_df['real_amount'].abs().sum())]
        }, index=['Theo', 'Real'])

        main_df = main_df[['token', 'theo_amount', 'real_amount', 'is_mismatch', 'is_dust']]

        return main_df, summary_df
    except Exception as e:
        print(f'Error converting multistrategy matching data: {e}')
        main_df = pd.DataFrame(columns=['token', 'theo_amount', 'real_amount', 'is_mismatch', 'is_dust'])
        summary_df = pd.DataFrame({
            'Net Exposure': [0, 0],
            'Gross Exposure': [0, 0]
        }, index=['Theo', 'Real'])
        return main_df, summary_df


def get_used_accounts():
    '''Parse config files to extract active accounts.'''
    if CONFIG is None:
        raise RuntimeError('CONFIG not initialized. Call initialize_globals() first.')

    session_config = CONFIG['session']
    config_files = {exchange: session_config[exchange]['config_file'] for exchange in session_config}

    session_configs = {}
    used_accounts = {}

    for session, config_file in config_files.items():
        filename = Path(config_file).expanduser()
        if filename.exists():
            with open(filename, 'r') as myfile:
                params = json.load(myfile)
                session_configs[session] = params

    for session, session_params in session_configs.items():
        strategies = session_params['strategy']

        for strategy_name, strategy_param in strategies.items():
            strat_account = strategy_param['account_trade']
            strat_exchange = strategy_param.get('exchange_trade', '')
            active = strategy_param.get('active', False)
            destination = strategy_param.get('send_orders', 'dummy')

            if not active or destination == 'dummy':
                continue
            if session not in used_accounts:
                used_accounts[session] = set()
            used_accounts[session].add((strat_exchange, strat_account))

    account_list = []
    for session, accounts in used_accounts.items():
        for exchange, account in accounts:
            account_list.append(f'{session}:{exchange}_{account}')

    return sorted(account_list)


def create_pnl_tab():
    '''Create the PnL tab with Get PnL button and results display.'''
    st.header('PnL Performance')
    
    if st.button('Get PnL', key='fetch_pnl'):
        with st.spinner('Fetching PnL data...'):
            try:
                uri = GATEWAY + '/pnl'
                response = get_any(uri, params={})

                if response.ok:
                    pnl_dict = json.loads(response.content.decode())
                    df = dict_to_df(pnl_dict, False)
                    df.index.names = ('session', 'strat', 'indicator')

                    pnl_table = df.xs('mean_theo', level='indicator')
                    pnl_pivot = pnl_table.pivot_table(
                        index=('session', 'strategy_type'),
                        values=[cols for cols in pnl_table.columns if cols != 'strategy_type'],
                        aggfunc='mean'
                    )

                    # Store in session state
                    ss['pnl_df'] = df
                    ss['pnl_pivot'] = pnl_pivot
                    st.success('PnL data loaded successfully')
                else:
                    st.error(f'Error: PnL endpoint returned status {response.status_code}')
            except Exception as e:
                st.error(f'Error fetching PnL: {str(e)}')

    # Display results if available
    if 'pnl_df' in ss:
        st.subheader('Mean Theoretical PnL')
        df_display = replace_na_with_nan(ss['pnl_df'].reset_index())
        st.dataframe(df_display, width='stretch', height=400)

        st.divider()

        st.subheader('Pivot Summary')
        pivot_display = replace_na_with_nan(ss['pnl_pivot'].reset_index())
        st.dataframe(pivot_display, width='stretch', height=400)


def create_matching_tab():
    '''Create the Matching tab with account selection and position comparison.'''
    st.header('Position Matching')
    
    accounts = get_used_accounts()

    if not accounts:
        st.error('No active accounts found in configuration')
        return

    account_str = st.selectbox('Select Account', accounts, key='matching_account')

    if st.button('Fetch Matching Data', key='fetch_matching'):
        with st.spinner('Fetching matching data...'):
            try:
                session, account = account_str.split(':')
                account_key = account

                uri_details = GATEWAY + '/matching'
                params_details = {'session': session, 'account_key': account_key}
                response_details = get_any(uri_details, params=params_details)

                if not response_details.ok:
                    st.error(f'Error: Matching endpoint returned status {response_details.status_code}')
                    return

                details_dict = json.loads(response_details.content.decode())
                main_df, summary_df = multistrategy_matching_to_df(details_dict)
                main_df = main_df.sort_values(by='theo_amount', ascending=False)

                dust_count = len(details_dict) - len(main_df)
                print(f'Filtered {dust_count} dust positions for {session}:{account_key}')

                # Store in session state
                ss['matching_main_df'] = main_df
                ss['matching_summary_df'] = summary_df
                ss['matching_session'] = session
                ss['matching_account_key'] = account_key
                
                st.success('Matching data loaded successfully')

            except Exception as e:
                st.error(f'Error fetching matching data: {str(e)}')
                print(f'Error in fetch_matching: {e}')

    # Display results if available
    if 'matching_main_df' in ss:
        main_df = ss['matching_main_df']
        summary_df = ss['matching_summary_df']
        session = ss['matching_session']
        account_key = ss['matching_account_key']

        # Exposure Summary
        st.subheader('Exposure Summary')
        st.dataframe(summary_df, width='stretch', height=400)

        st.divider()

        # Position counts
        meaningful = main_df[main_df['theo_amount'] != 0][['token', 'theo_amount', 'real_amount']]
        count_pos = (meaningful['real_amount'] > 0).sum()
        count_neg = (meaningful['real_amount'] < 0).sum()
        count_pos_th = (meaningful['theo_amount'] > 0).sum()
        count_neg_th = (meaningful['theo_amount'] < 0).sum()

        st.subheader('Theo Positions Matched')
        tc = pd.DataFrame({
            'Theo Positions': [count_pos_th, count_neg_th],
            'Real Positions': [count_pos, count_neg],
        }, index=['long#', 'short#'])
        st.dataframe(tc, width='stretch', height=200)

        st.divider()

        # Mismatched positions
        mismatch = main_df[main_df['is_mismatch']][['token', 'theo_amount', 'real_amount']]
        if not mismatch.empty:
            st.subheader('⚠️ Mismatched Positions')
            st.dataframe(mismatch.style.format({'theo_amount': '{:.0f}', 'real_amount': '{:.0f}'}), 
                        width='stretch', height=400)
            st.divider()

        # Dust positions
        dust = main_df[main_df['is_dust']][['token', 'real_amount']]
        if not dust.empty:
            st.subheader('Dust Positions')
            st.dataframe(dust.style.format({'real_amount': '{:.1f}'}), width='stretch', height=400)
            st.divider()

        # Load HTML figures from temp folder
        prefix = f'{session}_{account_key}'
        temp_dir = get_temp_dir()
        for i in range(1, 4):
            figname = temp_dir / f'{prefix}_fig{i}.html'
            if figname.exists():
                with open(figname, 'r') as figfile:
                    figure = figfile.read()
                    st.components.v1.html(figure, height=480, scrolling=True)

        # All positions table
        st.subheader('All Positions')
        positions_df = main_df[['token', 'theo_amount', 'real_amount']]
        st.dataframe(positions_df.style.format({'theo_amount': '{:.0f}', 'real_amount': '{:.0f}'}), 
                    width='stretch', height=400)


def create_multiply_tab():
    '''Create the Multiply tab with account selection and factor slider.'''
    st.header('Multiply Positions')
    
    accounts = get_used_accounts()

    if not accounts:
        st.error('No active accounts found in configuration')
        return

    account_str = st.selectbox('Select Account', accounts, key='multiply_account')

    st.subheader('Multiplication Factor')
    factor = st.slider('Factor', min_value=0.0, max_value=2.0, value=1.0, step=0.1)
    st.info(f'Current factor: {factor:.1f}')

    if st.button('Execute Multiply', type='primary', key='execute_multiply'):
        if factor < 0 or factor > 2:
            st.error('Factor must be between 0 and 2')
            return

        with st.spinner('Executing multiply...'):
            try:
                session, account = account_str.split(':')
                exchange, account_id = account.split('_', 1)

                params = {'exchange': exchange, 'account': account_id, 'factor': factor}
                uri = GATEWAY + '/multiply'
                response = get_any(uri, params)

                if response.ok:
                    result_text = response.content.decode()
                    st.subheader('Multiply Result')
                    st.markdown(result_text, unsafe_allow_html=True)
                    st.success(f'Multiply executed successfully with factor {factor:.1f}')
                else:
                    st.error(f'Error: Multiply endpoint returned status {response.status_code}: {response.text}')

            except Exception as e:
                st.error(f'Error executing multiply: {str(e)}')
                print(f'Error in execute_multiply: {e}')


def main():
    '''Main application setup.'''
    # Page configuration
    st.set_page_config(
        page_title='Tartineur furtif',
        layout='wide',
        initial_sidebar_state='expanded'
    )

    # Title
    st.title('🎯 Tartineur furtif')

    # Create tabs
    tab1, tab2, tab3 = st.tabs(['📊 PnL', '🔄 Matching', '✖️ Multiply'])

    with tab1:
        create_pnl_tab()

    with tab2:
        create_matching_tab()

    with tab3:
        create_multiply_tab()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='input file', default='')
    parser.add_argument('--port', help='port', default=8880)
    parser.add_argument('--gw_port', help='gateway port', default=14440)
    args = parser.parse_args()

    initialize_globals(args.config, int(args.gw_port))

    # Note: Streamlit port is set via command line when running
    # Run with: streamlit run web_front_sl.py --server.port 8880 -- --config=config/web_processor.yml --gw_port=14440
    main()