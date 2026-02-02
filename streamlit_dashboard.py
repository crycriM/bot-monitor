#!/usr/bin/env python
"""
Streamlit Dashboard for Trading Bot Monitor
Visualizes AUM, PnL, and Performance metrics with interactive charts
"""

import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title='Trading Bot Monitor - Streamlit',
    page_icon='📊',
    layout='wide',
    initial_sidebar_state='expanded'
)

# Custom CSS for dark theme
st.markdown("""
<style>
    :root {
        --primary-color: #1f77b4;
        --background-color: #0d0d0d;
        --secondary-background-color: #1a1a1a;
        --text-color: #ffffff;
    }
    .metric-card {
        background-color: #1a1a1a;
        border-radius: 10px;
        padding: 20px;
        border-left: 4px solid #00d4ff;
        color: #ffffff;
    }
    .positive {
        color: #00ff41;
    }
    .negative {
        color: #ff4444;
    }
    .chart-container {
        background-color: #1a1a1a;
        border-radius: 10px;
        padding: 20px;
    }
</style>
""", unsafe_allow_html=True)

# API Configuration
API_BASE_URL = st.secrets.get('API_BASE_URL', 'http://localhost:14440')

@st.cache_data(ttl=60)
def fetch_available_accounts():
    '''Fetch available sessions and accounts from API'''
    try:
        response = requests.get(f'{API_BASE_URL}/api/available-accounts', timeout=5)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f'Error fetching available accounts: {e}')
        st.error(f'Failed to connect to backend: {e}')
        return None

@st.cache_data(ttl=60)
def fetch_graph_data(session: str, account_key: str):
    '''Fetch graph data from API for specific session and account'''
    try:
        response = requests.get(
            f'{API_BASE_URL}/api/graph-data/{session}/{account_key}',
            timeout=5
        )
        response.raise_for_status()
        data = response.json()
        if 'error' in data:
            logger.warning(f'API error: {data.get("error")}')
            return None
        return data
    except Exception as e:
        logger.error(f'Error fetching graph data: {e}')
        return None

def create_perf_chart(timestamps, values, title):
    '''Create cumulative performance chart with Plotly'''
    df = pd.DataFrame({
        'timestamp': pd.to_datetime(timestamps),
        'value': values
    })

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['value'],
        mode='lines',
        name='Cumulative Performance',
        line=dict(color='#00d4ff', width=2),
        hovertemplate='<b>%{x|%Y-%m-%d %H:%M}</b><br>Perf: %{y:.4f}<extra></extra>'
    ))

    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Cumulative Performance',
        hovermode='x unified',
        template='plotly_dark',
        paper_bgcolor='#0d0d0d',
        plot_bgcolor='#1a1a1a',
        font=dict(color='#ffffff', size=11),
        margin=dict(l=50, r=50, t=80, b=50),
        height=500
    )

    return fig

def create_pnl_chart(timestamps, values, title):
    '''Create cumulative PnL chart with Plotly'''
    df = pd.DataFrame({
        'timestamp': pd.to_datetime(timestamps),
        'value': values
    })

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['value'],
        mode='lines',
        name='Cumulative PnL',
        line=dict(color='#00ff41', width=2),
        hovertemplate='<b>%{x|%Y-%m-%d %H:%M}</b><br>PnL: %{y:.4f}<extra></extra>'
    ))

    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Cumulative PnL',
        hovermode='x unified',
        template='plotly_dark',
        paper_bgcolor='#0d0d0d',
        plot_bgcolor='#1a1a1a',
        font=dict(color='#ffffff', size=11),
        margin=dict(l=50, r=50, t=80, b=50),
        height=500
    )

    return fig

def create_daily_perf_chart(timestamps, values, title):
    '''Create daily performance bar chart with Plotly'''
    df = pd.DataFrame({
        'timestamp': pd.to_datetime(timestamps),
        'value': values
    })

    # Color bars based on positive/negative
    colors = ['#00ff41' if v >= 0 else '#ff4444' for v in df['value']]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df['timestamp'],
        y=df['value'],
        name='Daily Performance',
        marker=dict(color=colors),
        hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Perf: %{y:.4f}<extra></extra>'
    ))

    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Daily Performance',
        hovermode='x unified',
        template='plotly_dark',
        paper_bgcolor='#0d0d0d',
        plot_bgcolor='#1a1a1a',
        font=dict(color='#ffffff', size=11),
        margin=dict(l=50, r=50, t=80, b=50),
        height=500,
        showlegend=False
    )

    return fig

def main():
    '''Main Streamlit app'''
    # Header
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title('📊 Trading Bot Monitor')
        st.markdown('*Interactive dashboard for real-time AUM and performance tracking*')
    with col2:
        if st.button('🔄 Refresh', key='refresh_main'):
            st.cache_data.clear()
            st.rerun()

    st.divider()

    # Fetch available accounts
    accounts_data = fetch_available_accounts()

    if accounts_data is None:
        st.error('⚠️ Unable to connect to backend API. Make sure the web API is running on http://localhost:14440')
        st.info('Start the backend with: `python src/web_api.py --config config/web_processor.yml`')
        return

    sessions = accounts_data.get('sessions', [])

    if not sessions:
        st.warning('No sessions available. Waiting for data...')
        return

    # Sidebar configuration
    with st.sidebar:
        st.header('⚙️ Configuration')
        selected_session = st.selectbox('Select Session', sessions)

        # Get accounts for selected session
        available_accounts = accounts_data.get('accounts', {}).get(selected_session, [])

        if available_accounts:
            selected_account = st.selectbox('Select Account', available_accounts)
        else:
            st.warning('No accounts available for this session')
            return

        # Get graph data availability
        graph_available = accounts_data.get('graph_data_available', {}).get(selected_session, [])

        if selected_account not in graph_available:
            st.info(f'⏳ Waiting for graph data for {selected_account}...')
            st.caption('Graph data is generated when AUM file is updated')
            return

        st.success(f'✓ Data available for {selected_account}')

        # Refresh rate
        refresh_rate = st.slider('Auto-refresh interval (seconds)', 10, 300, 60)

    # Fetch and display graph data
    graph_data = fetch_graph_data(selected_session, selected_account)

    if graph_data is None or not graph_data.get('data'):
        st.warning(f'No graph data available for {selected_session} - {selected_account}')
        return

    data = graph_data.get('data', {})
    timestamp = graph_data.get('timestamp', 'N/A')

    # Display timestamp
    st.caption(f'Last updated: {timestamp}')

    # Display tabs for different chart types
    tab1, tab2, tab3, tab4 = st.tabs(['📈 Performance', '💰 PnL', '📊 Daily Perf', 'ℹ️ Info'])

    with tab1:
        if 'perf' in data:
            perf_data = data['perf']
            fig = create_perf_chart(
                perf_data.get('timestamps', []),
                perf_data.get('values', []),
                f'Cumulative Performance - {selected_session}/{selected_account}'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info('Performance data not available yet')

    with tab2:
        if 'pnl' in data:
            pnl_data = data['pnl']
            fig = create_pnl_chart(
                pnl_data.get('timestamps', []),
                pnl_data.get('values', []),
                f'Cumulative PnL - {selected_session}/{selected_account}'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info('PnL data not available yet')

    with tab3:
        if 'daily' in data:
            daily_data = data['daily']
            fig = create_daily_perf_chart(
                daily_data.get('timestamps', []),
                daily_data.get('values', []),
                f'Daily Performance - {selected_session}/{selected_account}'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info('Daily performance data not available yet')

    with tab4:
        col1, col2 = st.columns(2)

        with col1:
            st.metric('Session', selected_session)
            st.metric('Account', selected_account)

        with col2:
            st.metric('Data Updated', timestamp.split('T')[0] if timestamp != 'N/A' else 'N/A')
            if 'perf' in data and data['perf'].get('values'):
                latest_perf = data['perf']['values'][-1]
                st.metric('Latest Perf', f'{latest_perf:.6f}',
                         delta_color='normal' if latest_perf >= 0 else 'inverse')

        st.divider()
        st.subheader('Available Data')
        cols = st.columns(3)
        cols[0].metric('Perf Data Points', len(data.get('perf', {}).get('values', [])))
        cols[1].metric('PnL Data Points', len(data.get('pnl', {}).get('values', [])))
        cols[2].metric('Daily Data Points', len(data.get('daily', {}).get('values', [])))

    # Auto-refresh
    if refresh_rate > 0:
        placeholder = st.empty()
        with placeholder.container():
            col1, col2 = st.columns([3, 1])
            with col2:
                if st.button('🔄 Refresh Now'):
                    st.cache_data.clear()
                    st.rerun()

        time.sleep(refresh_rate)
        st.rerun()

if __name__ == '__main__':
    main()
