import os
import argparse
import yaml
import json
import requests
import pandas as pd
from nicegui import ui

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

        main_df['ref_price'] = main_df['price'].fillna(0.0)
        main_df['theo_qty'] = main_df['theo'].fillna(0)
        main_df['theo_amount'] = (main_df['theo_qty'] * main_df['ref_price']).astype(int)
        main_df['real_qty'] = main_df['real'].fillna(0)
        main_df['real_amount'] = (main_df['real_qty'] * main_df['ref_price']).astype(int)
        main_df['is_dust'] = main_df['dust'].fillna(False)
        main_df['is_mismatch'] = main_df['is_mismatch'].fillna(False)

        # Create summary DataFrame (includes all positions, even dust)
        summary_df = pd.DataFrame({
            'Net Exposure': [int(main_df['theo_amount'].sum()), int(main_df['real_amount'].sum())],
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

    # Load all config files and extract active strategies
    for session, config_file in config_files.items():
        if os.path.exists(config_file):
            with open(config_file, 'r') as myfile:
                params = json.load(myfile)
                session_configs[session] = params

    # Populate accounts
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

    # Convert to list of session:exchange_account strings
    account_list = []
    for session, accounts in used_accounts.items():
        for exchange, account in accounts:
            account_list.append(f'{session}:{exchange}_{account}')

    return sorted(account_list)


def create_pnl_tab():
    '''Create the PnL tab with Get PnL button and results display.'''
    with ui.card().classes('w-full'):
        ui.label('PnL Performance').classes('text-h5')
        ui.separator()

        pnl_button = ui.button('Get PnL', on_click=lambda: fetch_pnl())
        pnl_container = ui.column().classes('w-full')

    async def fetch_pnl():
        pnl_container.clear()
        with pnl_container:
            ui.spinner(size='lg')

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

                pnl_container.clear()
                with pnl_container:
                    with ui.card().classes('w-full'):
                        ui.label('Mean Theoretical PnL').classes('text-h6')
                        # Convert DataFrame to dict for aggrid
                        df_records = df.reset_index().to_dict('records')
                        ui.aggrid({
                            'columnDefs': [
                                {'field': col, 'sortable': True, 'filter': True}
                                for col in df.reset_index().columns
                            ],
                            'rowData': df_records,
                            'defaultColDef': {'resizable': True, 'flex': 1}
                        }).classes('w-full')

                    ui.space()

                    with ui.card().classes('w-full'):
                        ui.label('Pivot Summary').classes('text-h6')
                        pivot_records = pnl_pivot.reset_index().to_dict('records')
                        ui.aggrid({
                            'columnDefs': [
                                {'field': col, 'sortable': True, 'filter': True}
                                for col in pnl_pivot.reset_index().columns
                            ],
                            'rowData': pivot_records,
                            'defaultColDef': {'resizable': True, 'flex': 1}
                        }).classes('w-full')

                ui.notify('PnL data loaded successfully', type='positive')
            else:
                pnl_container.clear()
                ui.notify(f'Error: PnL endpoint returned status {response.status_code}', type='negative')
        except Exception as e:
            pnl_container.clear()
            ui.notify(f'Error fetching PnL: {str(e)}', type='negative')


def create_matching_tab():
    '''Create the Matching tab with account selection and position comparison.'''
    accounts = get_used_accounts()

    with ui.card().classes('w-full'):
        ui.label('Position Matching').classes('text-h5')
        ui.separator()

        if not accounts:
            ui.label('No active accounts found in configuration').classes('text-red')
            return

        account_select = ui.select(
            label='Select Account',
            options=accounts,
            value=accounts[0] if accounts else None
        ).classes('w-64')

        fetch_button = ui.button('Fetch Matching Data', on_click=lambda: fetch_matching(account_select.value))
        matching_container = ui.column().classes('w-full')

    async def fetch_matching(account_str):
        if not account_str:
            ui.notify('Please select an account', type='warning')
            return

        matching_container.clear()
        with matching_container:
            ui.spinner(size='lg')

        try:
            session, account = account_str.split(':')
            account_key = account

            uri_details = GATEWAY + '/matching'
            params_details = {'session': session, 'account_key': account_key}
            response_details = get_any(uri_details, params=params_details)

            if not response_details.ok:
                matching_container.clear()
                ui.notify(f'Error: Matching endpoint returned status {response_details.status_code}', type='negative')
                return

            details_dict = json.loads(response_details.content.decode())
            main_df, summary_df = multistrategy_matching_to_df(details_dict)
            main_df = main_df.sort_values(by='theo_amount', ascending=False)

            dust_count = len(details_dict) - len(main_df)
            print(f'Filtered {dust_count} dust positions for {session}:{account_key}')

            matching_container.clear()
            with matching_container:
                # Summary card
                with ui.card().classes('w-full'):
                    ui.label('Exposure Summary').classes('text-h6')
                    summary_records = summary_df.reset_index().to_dict('records')
                    ui.aggrid({
                        'columnDefs': [
                            {'field': 'index', 'headerName': '', 'sortable': False},
                            {'field': 'Net Exposure', 'sortable': True, 'valueFormatter': 'value.toFixed(0)'},
                            {'field': 'Gross Exposure', 'sortable': True, 'valueFormatter': 'value.toFixed(0)'}
                        ],
                        'rowData': summary_records,
                        'defaultColDef': {'resizable': True, 'flex': 1},
                        'domLayout': 'autoHeight'
                    }).classes('w-full')

                ui.space()

                # Position counts
                meaningful = main_df[main_df['theo_amount'] != 0][['token', 'theo_amount', 'real_amount']]
                count_pos = (meaningful['real_amount'] > 0).sum()
                count_neg = (meaningful['real_amount'] < 0).sum()
                count_pos_th = (meaningful['theo_amount'] > 0).sum()
                count_neg_th = (meaningful['theo_amount'] < 0).sum()

                with ui.card().classes('w-full'):
                    ui.label('Theo Positions Matched').classes('text-h6')
                    tc = pd.DataFrame({
                        'theo_pos': [count_pos_th, count_neg_th],
                        'real_pos': [count_pos, count_neg],
                    }, index=['long#', 'short#'])
                    tc_records = tc.reset_index().to_dict('records')
                    ui.aggrid({
                        'columnDefs': [
                            {'field': 'index', 'headerName': 'Type', 'sortable': False},
                            {'field': 'theo_pos', 'headerName': 'Theo Positions', 'sortable': True},
                            {'field': 'real_pos', 'headerName': 'Real Positions', 'sortable': True}
                        ],
                        'rowData': tc_records,
                        'defaultColDef': {'resizable': True, 'flex': 1},
                        'domLayout': 'autoHeight'
                    }).classes('w-full')

                ui.space()

                # Mismatched positions
                mismatch = main_df[main_df['is_mismatch']][['token', 'theo_amount', 'real_amount']]
                if not mismatch.empty:
                    with ui.card().classes('w-full'):
                        ui.label('Mismatched Positions').classes('text-h6 text-orange')
                        mismatch_records = mismatch.to_dict('records')
                        ui.aggrid({
                            'columnDefs': [
                                {'field': 'token', 'headerName': 'Token', 'sortable': True, 'filter': True},
                                {'field': 'theo_amount', 'headerName': 'Theo Amount', 'sortable': True, 'valueFormatter': 'value.toFixed(0)'},
                                {'field': 'real_amount', 'headerName': 'Real Amount', 'sortable': True, 'valueFormatter': 'value.toFixed(0)'}
                            ],
                            'rowData': mismatch_records,
                            'defaultColDef': {'resizable': True, 'flex': 1},
                            'domLayout': 'autoHeight'
                        }).classes('w-full')

                    ui.space()

                # Dust positions
                dust = main_df[main_df['is_dust']][['token', 'real_amount']]
                if not dust.empty:
                    with ui.card().classes('w-full'):
                        ui.label('Dust Positions').classes('text-h6')
                        dust_records = dust.to_dict('records')
                        ui.aggrid({
                            'columnDefs': [
                                {'field': 'token', 'headerName': 'Token', 'sortable': True, 'filter': True},
                                {'field': 'real_amount', 'headerName': 'Real Amount', 'sortable': True, 'valueFormatter': 'value.toFixed(1)'}
                            ],
                            'rowData': dust_records,
                            'defaultColDef': {'resizable': True, 'flex': 1},
                            'domLayout': 'autoHeight'
                        }).classes('w-full')

                    ui.space()

                # Load HTML figures from temp folder
                prefix = f'{session}_{account_key}'
                for i in range(1, 4):
                    figname = f'temp/{prefix}_fig{i}.html'
                    if os.path.exists(figname):
                        with open(figname, 'r') as figfile:
                            figure = figfile.read()
                            with ui.card().classes('w-full').style('min-height: 400px'):
                                ui.html(figure)
                        ui.space()

                # All positions table
                with ui.card().classes('w-full'):
                    ui.label('All Positions').classes('text-h6')
                    positions_df = main_df[['token', 'theo_amount', 'real_amount']]
                    positions_records = positions_df.to_dict('records')
                    ui.aggrid({
                        'columnDefs': [
                            {'field': 'token', 'headerName': 'Token', 'sortable': True, 'filter': True, 'pinned': 'left'},
                            {'field': 'theo_amount', 'headerName': 'Theo Amount', 'sortable': True, 'valueFormatter': 'value.toFixed(0)'},
                            {'field': 'real_amount', 'headerName': 'Real Amount', 'sortable': True, 'valueFormatter': 'value.toFixed(0)'}
                        ],
                        'rowData': positions_records,
                        'defaultColDef': {'resizable': True, 'flex': 1}
                    }).classes('w-full h-96')

            ui.notify('Matching data loaded successfully', type='positive')

        except Exception as e:
            matching_container.clear()
            ui.notify(f'Error fetching matching data: {str(e)}', type='negative')
            print(f'Error in fetch_matching: {e}')


def create_multiply_tab():
    '''Create the Multiply tab with account selection and factor slider.'''
    accounts = get_used_accounts()

    with ui.card().classes('w-full'):
        ui.label('Multiply Positions').classes('text-h5')
        ui.separator()

        if not accounts:
            ui.label('No active accounts found in configuration').classes('text-red')
            return

        account_select = ui.select(
            label='Select Account',
            options=accounts,
            value=accounts[0] if accounts else None
        ).classes('w-64')

        ui.space()

        with ui.card().classes('w-full'):
            ui.label('Multiplication Factor').classes('text-h6')
            factor_label = ui.label('Factor: 1.0')
            factor_slider = ui.slider(min=0, max=2, step=0.1, value=1.0).props('label-always').classes('w-full')

            def update_factor_label():
                factor_label.text = f'Factor: {factor_slider.value:.1f}'

            factor_slider.on_value_change(update_factor_label)

            ui.space()

            ui.button('Execute Multiply', on_click=lambda: execute_multiply(account_select.value, factor_slider.value)).props('color=warning')

        result_container = ui.column().classes('w-full')

    async def execute_multiply(account_str, factor):
        if not account_str:
            ui.notify('Please select an account', type='warning')
            return

        if factor < 0 or factor > 2:
            ui.notify('Factor must be between 0 and 2', type='negative')
            return

        result_container.clear()
        with result_container:
            ui.spinner(size='lg')

        try:
            session, account = account_str.split(':')
            exchange, account_id = account.split('_', 1)

            params = {'exchange': exchange, 'account': account_id, 'factor': factor}
            uri = GATEWAY + '/multiply'
            response = get_any(uri, params)

            result_container.clear()

            if response.ok:
                result_text = response.content.decode()
                with result_container:
                    with ui.card().classes('w-full'):
                        ui.label('Multiply Result').classes('text-h6')
                        ui.html(result_text)
                ui.notify(f'Multiply executed successfully with factor {factor:.1f}', type='positive')
            else:
                ui.notify(f'Error: Multiply endpoint returned status {response.status_code}: {response.text}', type='negative')

        except Exception as e:
            result_container.clear()
            ui.notify(f'Error executing multiply: {str(e)}', type='negative')
            print(f'Error in execute_multiply: {e}')


@ui.page('/')
def main_app():
    '''Main application setup with tabs.'''
    ui.colors(primary='#1976d2', secondary='#26a69a', accent='#9c27b0')

    with ui.header().classes('items-center justify-between'):
        ui.label('Tartineur furtif').classes('text-h4')

    with ui.tabs().classes('w-full') as tabs:
        pnl_tab = ui.tab('PnL')
        matching_tab = ui.tab('Matching')
        multiply_tab = ui.tab('Multiply')

    with ui.tab_panels(tabs, value=pnl_tab).classes('w-full'):
        with ui.tab_panel(pnl_tab):
            create_pnl_tab()

        with ui.tab_panel(matching_tab):
            create_matching_tab()

        with ui.tab_panel(multiply_tab):
            create_multiply_tab()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='input file', default='')
    parser.add_argument('--port', help='port', default=8880)
    parser.add_argument('--gw_port', help='gateway port', default=14440)
    args = parser.parse_args()

    initialize_globals(args.config, int(args.gw_port))

    ui.run(
        port=int(args.port),
        title='Tartineur furtif',
        reload=False,
        show=False
    )
