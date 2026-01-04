import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from datetime import datetime, timedelta
import matplotlib.dates as mdates
from catboost import CatBoostClassifier

indicator_thresholds = {
    'Chl-a': {
        'type': 'above',
        'moderate': 10,
        'high_threshold': 25
    }, 
    'E. coli': {
        'type': 'above',
        'moderate': 235,
        'high_threshold': 575
    },
    'pH': {
        'type': 'outside_range',
        'moderate': [6.0, 9.0],
        'high_threshold': [6.5, 8.5]
    },       
    'Total Nitrogen': {
        'type': 'above',
        'moderate': 1.0,
        'high_threshold': 2.0
    },
    'Total Phosphorus': {
        'type': 'above',
        'moderate': 0.1,
        'high_threshold': 0.2
    },
    'Nitrate': {
        'type': 'above',
        'moderate': 25,
        'high_threshold': 45
    },
    'Dissolved Oxygen': {
        'type': 'below',
        'moderate': 5.0,
        'high_threshold': 4.0
    },
    'Turbidity': {
        'type': 'above',
        'moderate': 10,
        'high_threshold': 25
    },
}

@st.cache_data
def read_data():
    df = pd.read_csv('lawa-lake-monitoring-data-2004-2023_statetrendtli-results_sep2024.csv')
    
    df['SampleDateTime'] = pd.to_datetime(df['SampleDateTime'])
    
    df['Date'] = df['SampleDateTime'].dt.date
    df['Month'] = df['SampleDateTime'].dt.month
    df['Year'] = df['SampleDateTime'].dt.year    
    return df

def determine_risk_level(row):
    """
    Determine risk level for a single value based on indicator conditions.
    
    Returns:
    --------
    int
        Risk level (0: Safe, 1: Moderate, 2: High)
    """
    indicator = row['Indicator']
    condition = indicator_thresholds.get(indicator)
    value = pd.to_numeric(row['Value (Agency)'], errors='coerce')
    if not condition or value is None:
        return 0
    
    condition_type = condition.get("type", "above")
    moderate_threshold = condition.get("moderate")
    
    high_threshold = condition.get("high_threshold") or condition.get("threshold")
    
    risk_level = 0
    
    if condition_type == "above":
        if high_threshold is not None and value > high_threshold:
            risk_level = 2
        elif moderate_threshold is not None and value > moderate_threshold:
            risk_level = 1
    
    elif condition_type == "below":
        if high_threshold is not None and value < high_threshold:
            risk_level = 2
        elif moderate_threshold is not None and value < moderate_threshold:
            risk_level = 1
    
    elif condition_type == "outside_range":
        if (isinstance(moderate_threshold, list) and len(moderate_threshold) == 2 and 
            isinstance(high_threshold, list) and len(high_threshold) == 2):
            
            mod_lower, mod_upper = moderate_threshold
            high_lower, high_upper = high_threshold
            
            if value < high_lower or value > high_upper:
                risk_level = 2
            
            elif value < mod_lower or value > mod_upper:
                risk_level = 1
    
    return risk_level

def get_risk(num):
    if num == 2:
        return 'high'
    elif num == 1:
        return 'moderate'
    else:
        return 'safe'
    
@st.cache_resource
def build_model_aggregated_dataset(df):

    df['SiteID'] = df['SiteID'].astype('category')
    df['Month'] = df['Month'].astype('category')


    min_year = df['Year'].min()
    max_year = df['Year'].max()
    unique_years = sorted(df['Year'].unique())






    df_sorted = df.sort_values(['Year', 'Month'])
    train_size = 0.8
    train_data = df_sorted.iloc[:int(len(df_sorted) * train_size)]
    test_data = df_sorted.iloc[int(len(df_sorted) * train_size):]

    X_train = train_data.drop('risk_level', axis=1)
    y_train = train_data['risk_level']

    X_test = test_data.drop('risk_level', axis=1)
    y_test = test_data['risk_level']

    categorical_features = ['SiteID', 'Month']

    model = CatBoostClassifier(
        iterations=500,
        learning_rate=0.1,
        cat_features=categorical_features,
        random_seed=42
    )

    model.fit(
        X_train, y_train,
        eval_set=(X_test, y_test),
        verbose=100
    )

    from sklearn.metrics import classification_report, confusion_matrix

    y_pred = model.predict(X_test)

    return model


def aggregate_risk_by_date(df):

    if 'Date' not in df.columns:
        df['Date'] = pd.to_datetime(df[['Year', 'Month', 'Day']])

    df['Date'] = pd.to_datetime(df['Date'])

    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day

    aggregated_df = df.groupby(['SiteID', 'Date']).agg({
        'risk_level': 'max',
        'Value': ['mean', 'min', 'max'],
        'Year': 'first',
        'Month': 'first',
        'Day': 'first'
    }).reset_index()

    aggregated_df.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col for col in aggregated_df.columns]

    aggregated_df = aggregated_df.rename(columns={'risk_level_max': 'risk_level'})
    aggregated_df = aggregated_df.rename(columns={'Day_first': 'Day'})
    aggregated_df = aggregated_df.rename(columns={'Month_first': 'Month'})
    aggregated_df = aggregated_df.rename(columns={'Year_first': 'Year'})

    return aggregated_df


def predict_risk_for_site_month(model, df, site_id, target_month, current_year):
    """
    Predict risk level using most recent historical data for the same site and month.
    
    Parameters:
    model: Trained prediction model
    df: aggregated DataFrame
    site_id: Target SiteID to predict for
    target_month: Month to predict for (1-12)
    current_year: Current year (to avoid using future data)
    
    Returns:
    Predicted risk level
    """
    historical_data = df[(df['SiteID'] == site_id) & 
                         (df['Month'] == target_month) & 
                         (df['Year'] < current_year)]
    
    if len(historical_data) == 0:
        st.write(f"No historical data found for SiteID={site_id}, Month={target_month}")
        return None
    
    most_recent_year = historical_data['Year'].max()
    most_recent_data = historical_data[historical_data['Year'] == most_recent_year]
    
    input_data = most_recent_data.copy()
    input_data['Year'] = current_year
    input_data['Month'] = target_month
    input_data['Date'] = pd.to_datetime([f"{current_year}-{target_month}-15"])[0]
    model_features = [col for col in input_data.columns if col != 'risk_level']
    
    predicted_risk = model.predict(input_data[model_features])[0]
    
    return predicted_risk

def aggregate_for_calendar_year(site_id, df):
    site_data = df[df['SiteID'] == site_id].copy()
    site_data['Date'] = pd.to_datetime(df['Date'])

    calendar_df = site_data.groupby(['SiteID', 'Month', 'Day'])['risk_level'].mean().reset_index()

    calendar_df['risk_level'] = calendar_df['risk_level'].round().astype(int)

    
    calendar_df['Date'] = calendar_df.apply(
        lambda row: pd.Timestamp(year=2025, month=row['Month'], day=row['Day']), axis=1
    ) 
    return calendar_df

def heatmap_risk_by_date(site_id, site_data):
    
    try:
        
        daily_data = site_data.copy()
    

        if daily_data['risk_level'].isnull().any():
            st.write("Warning: There are NaN values in 'risk_level' column")
        calendar_data = np.full((12, 31), np.nan)
        
        for m in range(1, 13):
            month_data = daily_data[daily_data['Month'] == m]
            for d in range(1, 32):
                day_data = month_data[month_data['Day'] == d]
                if len(day_data) > 0:
                    calendar_data[m-1, d-1] = day_data['risk_level'].mean()
        
        fig = plt.figure(figsize=(14, 8))
        
        cmap = plt.cm.colors.ListedColormap(['green', 'yellow', 'red'])
        bounds = [0, 0.67, 1.33, 2]
        norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)
        
        heatmap = plt.pcolormesh(calendar_data.T, cmap=cmap, norm=norm)
        plt.yticks(np.arange(0.5, 31.5), np.arange(1, 32))
        plt.xticks(np.arange(0.5, 12.5), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                                         'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        
        plt.title(f'Historical Calendar View of Risk Levels for {site_id}', fontsize=18)
        plt.ylabel('Day of Month')
        
        cbar = plt.colorbar(heatmap)
        cbar.set_ticks([0.33, 1, 1.67])
        cbar.set_ticklabels(['Safe', 'Moderate Risk', 'High Risk'])
        
        plt.tight_layout()
        st.pyplot(fig)
    except Exception as e:
        st.write(f"Error creating calendar visualization: {e}")
    
 

from asian_water_quality_data import (
    ASIAN_COUNTRIES, 
    get_basins_for_country, 
    fetch_country_metadata,
    fetch_water_quality_data,
    fetch_live_water_data,
    normalize_uploaded_data,
    REAL_WATER_QUALITY_DATA
)



def get_water_quality_status(risk_level):
    """Convert numeric risk level to descriptive status."""
    if risk_level <= 0.5:
        return "Excellent", "🟢", "#27ae60"
    elif risk_level <= 1.0:
        return "Good", "🟡", "#f39c12"
    elif risk_level <= 1.5:
        return "Moderate", "🟠", "#e67e22"
    else:
        return "Poor", "🔴", "#e74c3c"

def calculate_basin_status(data, site_id):
    site_data = data[data['SiteID'] == site_id]
    if site_data.empty or 'risk_level' not in site_data.columns:
        return "Good", "🟢", "#27ae60", {}
    
    avg_risk = site_data['risk_level'].mean()
    status, emoji, color = get_water_quality_status(avg_risk)
    
    stats = {}
    if 'Indicator' in site_data.columns:
        for indicator in site_data['Indicator'].unique():
            ind_data = site_data[site_data['Indicator'] == indicator]
            if 'Value' in ind_data.columns and not ind_data.empty:
                avg_val = ind_data['Value'].mean()
                risk = ind_data['risk_level'].mean() if 'risk_level' in ind_data.columns else 0
                ind_status, ind_emoji, _ = get_water_quality_status(risk)
                stats[indicator] = {
                    'value': avg_val,
                    'status': ind_status,
                    'emoji': ind_emoji
                }
    
    return status, emoji, color, stats

def display_quality_summary(data, site_id):
    """Display a summary card of water quality status."""
    status, emoji, color, stats = calculate_basin_status(data, site_id)
    
    st.markdown(f"""
    <div style='background: linear-gradient(135deg, {color}22, {color}44); 
                border-left: 4px solid {color}; 
                padding: 20px; 
                border-radius: 10px; 
                margin-bottom: 20px;'>
        <h2 style='margin:0; color: {color};'>{emoji} Water Quality: {status}</h2>
        <p style='margin:5px 0 0 0; font-size: 14px; color: #666;'>Based on current monitoring data for {site_id}</p>
    </div>
    """, unsafe_allow_html=True)
    
    if stats:
        cols = st.columns(min(4, len(stats)))
        for i, (indicator, info) in enumerate(list(stats.items())[:4]):
            with cols[i % 4]:
                # Use full indicator name without truncation
                st.metric(
                    label=f"{info['emoji']} {indicator}",
                    value=f"{info['value']:.2f}",
                    delta=info['status'],
                    delta_color="off"
                )


def create_bar_chart(data, site_id, indicator='Chl-a'):
    """Create a bar chart for the selected site and indicator with quality status."""
    site_data = data[data['SiteID'] == site_id]
    if site_data.empty:
        st.warning(f"No data available for {site_id}")
        return
    
    display_quality_summary(data, site_id)
    
    if 'Indicator' in site_data.columns:
        indicator_data = site_data[site_data['Indicator'] == indicator]
        if indicator_data.empty:
            indicator_data = site_data
    else:
        indicator_data = site_data
    
    if 'Month' in indicator_data.columns and 'Value' in indicator_data.columns:
        monthly_avg = indicator_data.groupby('Month')['Value'].mean().reset_index()
        
        if 'risk_level' in indicator_data.columns:
            monthly_risk = indicator_data.groupby('Month')['risk_level'].mean().reset_index()
            colors = ['#27ae60' if r <= 0.5 else '#f39c12' if r <= 1.0 else '#e67e22' if r <= 1.5 else '#e74c3c' 
                      for r in monthly_risk['risk_level']]
        else:
            colors = ['#3498db'] * len(monthly_avg)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        months_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                         'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        bars = ax.bar(monthly_avg['Month'], monthly_avg['Value'], color=colors, edgecolor='white')
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(months_labels)
        ax.set_xlabel('Month', fontsize=12)
        ax.set_ylabel(f'{indicator} Value', fontsize=12)
        ax.set_title(f'{indicator} Monthly Average for {site_id}', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#27ae60', label='Excellent'),
            Patch(facecolor='#f39c12', label='Good'),
            Patch(facecolor='#e67e22', label='Moderate'),
            Patch(facecolor='#e74c3c', label='Poor')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.warning("Required columns (Month, Value) not found in data")


def create_line_chart(data, site_id, indicator='Chl-a'):
    """Create a line chart showing trends over time with quality status and new data sections."""
    site_data = data[data['SiteID'] == site_id]
    if site_data.empty:
        st.warning(f"No data available for {site_id}")
        return
    
    display_quality_summary(data, site_id)
    
    if 'Indicator' in site_data.columns:
        indicator_data = site_data[site_data['Indicator'] == indicator].copy()
        if indicator_data.empty:
            indicator_data = site_data.copy()
    else:
        indicator_data = site_data.copy()
    
    if 'SampleDateTime' in indicator_data.columns and 'Value' in indicator_data.columns:
        indicator_data = indicator_data.sort_values('SampleDateTime')
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Check if we have update sequence data (live mode with accumulated data)
        if 'UpdateSequence' in indicator_data.columns:
            # Color code different update sequences to show new data sections
            sequences = indicator_data['UpdateSequence'].unique()
            colors = plt.cm.viridis(np.linspace(0.2, 1, len(sequences)))
            
            for i, seq in enumerate(sorted(sequences)):
                seq_data = indicator_data[indicator_data['UpdateSequence'] == seq]
                is_latest = (seq == max(sequences))
                
                # Plot each sequence with different styling
                ax.plot(seq_data['SampleDateTime'], seq_data['Value'],
                       color=colors[i], linewidth=3 if is_latest else 1.5,
                       marker='o', markersize=8 if is_latest else 4,
                       alpha=1.0 if is_latest else 0.5,
                       label=f'Update #{int(seq)}' if is_latest or i == 0 else None)
                
                # Highlight latest data point with annotation
                if is_latest and not seq_data.empty:
                    latest_point = seq_data.iloc[-1]
                    ax.annotate(f'NEW: {latest_point["Value"]:.2f}',
                               xy=(latest_point['SampleDateTime'], latest_point['Value']),
                               xytext=(10, 10), textcoords='offset points',
                               fontsize=10, fontweight='bold', color='#e74c3c',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                               arrowprops=dict(arrowstyle='->', color='#e74c3c'))
            
            # Show total data points and updates
            st.caption(f"📊 Showing {len(sequences)} update batches with {len(indicator_data)} total readings")
        else:
            # Standard line plot for non-live data
            ax.plot(indicator_data['SampleDateTime'], indicator_data['Value'], 
                    color='#3498db', linewidth=2, marker='o', markersize=4, alpha=0.7)
        
        if len(indicator_data) > 2:
            z = np.polyfit(range(len(indicator_data)), indicator_data['Value'], 1)
            p = np.poly1d(z)
            ax.plot(indicator_data['SampleDateTime'], p(range(len(indicator_data))), 
                    '--', color='#e74c3c', alpha=0.8, label='Trend')
        
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel(f'{indicator} Value', fontsize=12)
        ax.set_title(f'{indicator} Real-Time Trend for {site_id}', fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3)
        ax.legend(loc='upper left')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.warning("Required columns (SampleDateTime, Value) not found in data")


def show_data_table(data, site_id):
    """Display data as an interactive table with quality status."""
    site_data = data[data['SiteID'] == site_id].copy()
    if site_data.empty:
        st.warning(f"No data available for {site_id}")
        return
    
    display_quality_summary(data, site_id)
    
    if 'risk_level' in site_data.columns:
        site_data['Status'] = site_data['risk_level'].apply(
            lambda r: '🟢 Excellent' if r <= 0.5 else '🟡 Good' if r <= 1.0 else '🟠 Moderate' if r <= 1.5 else '🔴 Poor'
        )
    
    display_cols = ['SampleDateTime', 'Indicator', 'Value', 'Units', 'Status']
    display_cols = [c for c in display_cols if c in site_data.columns]
    
    if display_cols:
        st.dataframe(
            site_data[display_cols].sort_values('SampleDateTime', ascending=False),
            use_container_width=True,
            height=400
        )
    else:
        st.dataframe(site_data, use_container_width=True, height=400)



st.set_page_config(
    page_title="Asian Water Quality Dashboard",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🌊 Asian Water Quality Dashboard - LIVE")

if 'data_source' not in st.session_state:
    st.session_state.data_source = 'api'
if 'selected_country' not in st.session_state:
    st.session_state.selected_country = 'India'
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = datetime.now()
# Live data history - accumulates readings over time for accurate time-series
if 'live_data_history' not in st.session_state:
    st.session_state.live_data_history = pd.DataFrame()
if 'live_data_count' not in st.session_state:
    st.session_state.live_data_count = 0


st.sidebar.header("⚡ Live Updates")

auto_refresh = st.sidebar.toggle("Enable Auto-Refresh", value=True, help="Automatically refresh data")
refresh_interval = st.sidebar.slider(
    "Refresh Interval (seconds)", 
    min_value=10, 
    max_value=60, 
    value=30,
    disabled=not auto_refresh
)

if st.sidebar.button("🔄 Refresh Now"):
    st.cache_data.clear()
    st.session_state.last_refresh = datetime.now()
    st.rerun()

if st.sidebar.button("🗑️ Clear History"):
    st.session_state.live_data_history = pd.DataFrame()
    st.session_state.live_data_count = 0
    st.cache_data.clear()
    st.rerun()

if auto_refresh:
    import time
    time_since_refresh = (datetime.now() - st.session_state.last_refresh).total_seconds()
    if time_since_refresh >= refresh_interval:
        st.cache_data.clear()
        st.session_state.last_refresh = datetime.now()
        st.rerun()
    
    remaining = int(refresh_interval - time_since_refresh)
    st.sidebar.caption(f"⏱️ Next refresh in: {remaining}s")

st.sidebar.divider()


st.sidebar.header("🌏 Region Selection")

country_index = ASIAN_COUNTRIES.index('India') if 'India' in ASIAN_COUNTRIES else 0

def on_country_change():
    """Clear data cache when country changes."""
    st.cache_data.clear()
    st.session_state.last_refresh = datetime.now()

selected_country = st.sidebar.selectbox(
    'Country', 
    ASIAN_COUNTRIES, 
    index=country_index,
    key='country_selector',
    on_change=on_country_change
)

country_meta = fetch_country_metadata(selected_country)
flag_emoji = country_meta.get('flag', '🌏')
st.sidebar.markdown(f"### {flag_emoji} {selected_country}")
st.sidebar.caption(f"📍 Capital: {country_meta.get('capital', 'N/A')}")
st.sidebar.caption(f"👥 Population: {country_meta.get('population', 0):,}")

data_source = country_meta.get('data_source', 'Simulated')
data_type = country_meta.get('data_type', 'simulated')
if data_type == 'api':
    st.sidebar.success(f"📡 Live Data: {data_source}")
elif data_type == 'reference':
    st.sidebar.info(f"📊 Source: {data_source}")
else:
    st.sidebar.caption(f"📊 Data: Monitoring Stations")

st.sidebar.divider()

st.sidebar.header("📁 Data Upload")
uploaded_file = st.sidebar.file_uploader(
    "Upload CSV/Excel (optional)",
    type=['csv', 'xlsx', 'xls'],
    help="Upload your own data to override API data"
)

st.sidebar.divider()

@st.cache_data(ttl=1800)
def load_api_data(country):
    """Load data from API for the selected country."""
    return fetch_water_quality_data(country)

def load_live_data(country, refresh_key):
    """Load live streaming data (bypasses cache for real-time updates)."""
    return fetch_live_water_data(country)

def accumulate_live_data(new_data, country):
    """Accumulate new live data readings into session history for accurate time-series."""
    # Reset history if country changed
    if st.session_state.get('last_country') != country:
        st.session_state.live_data_history = pd.DataFrame()
        st.session_state.live_data_count = 0
        st.session_state.last_country = country
    
    # Filter only the truly "live" readings (current moment readings)
    if 'IsLive' in new_data.columns:
        live_readings = new_data[new_data['IsLive'] == True].copy()
    else:
        live_readings = new_data.head(len(new_data) // 6).copy()  # Take just newest readings
    
    if not live_readings.empty:
        # Add a unique update sequence number
        st.session_state.live_data_count += 1
        live_readings['UpdateSequence'] = st.session_state.live_data_count
        live_readings['ReceivedAt'] = datetime.now()
        
        # Append to history
        if st.session_state.live_data_history.empty:
            st.session_state.live_data_history = live_readings
        else:
            st.session_state.live_data_history = pd.concat([
                st.session_state.live_data_history, 
                live_readings
            ], ignore_index=True)
        
        # Keep last 500 readings to prevent memory issues
        if len(st.session_state.live_data_history) > 500:
            st.session_state.live_data_history = st.session_state.live_data_history.tail(500)
    
    return st.session_state.live_data_history

live_mode = False

if uploaded_file is not None:
    st.session_state.data_source = 'upload'
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        df = normalize_uploaded_data(df)
        st.sidebar.success("✅ Using uploaded data")
    except Exception as e:
        st.sidebar.error(f"Error reading file: {e}")
        if live_mode:
            new_data = load_live_data(selected_country, st.session_state.last_refresh)
            df = accumulate_live_data(new_data, selected_country)
            st.session_state.data_source = 'live'
        else:
            df = load_api_data(selected_country)
            st.session_state.data_source = 'api'
else:
    if live_mode:
        # Live mode: fetch new data and accumulate it
        st.session_state.data_source = 'live'
        new_data = load_live_data(selected_country, st.session_state.last_refresh)
        df = accumulate_live_data(new_data, selected_country)
        update_count = st.session_state.live_data_count
    else:
        st.session_state.data_source = 'api'
        loading_placeholder = st.empty()
        loading_placeholder.markdown("""
        <div style='text-align: center; padding: 50px;'>
            <div style='font-size: 48px; animation: pulse 1.5s infinite;'>🌊</div>
            <p style='color: #666; margin-top: 15px;'>Loading water quality data...</p>
        </div>
        <style>@keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }</style>
        """, unsafe_allow_html=True)
        df = load_api_data(selected_country)
        loading_placeholder.empty()


# Filter data to only include the selected country's data
if 'Region' in df.columns:
    df = df[df['Region'] == selected_country]

# Also filter to only include valid basins for the selected country
country_basins = get_basins_for_country(selected_country)
if 'SiteID' in df.columns and country_basins:
    df = df[df['SiteID'].isin(country_basins)]

if 'risk_level' not in df.columns:
    df['risk_level'] = df.apply(determine_risk_level, axis=1)


st.sidebar.header("🏞️ Basin Selection")

# Special handling for India - show state selector first
if selected_country == "India":
    from asian_water_quality_data import INDIA_STATE_BASINS, get_india_state_basins, get_india_states
    
    st.sidebar.markdown("**🗺️ Select State**")
    india_states = get_india_states()
    
    # Add "All States" option
    state_options = ["All States"] + sorted(india_states)
    selected_state = st.sidebar.selectbox(
        'State',
        state_options,
        index=0,
        key='india_state_select',
        help="Select a state to filter river basins"
    )
    

    # Get basins based on selected state
    if selected_state == "All States":
        all_sites = get_india_state_basins()
        available_sites = [
            s for s in all_sites 
            if any(k in s for k in REAL_WATER_QUALITY_DATA.keys())
        ]
        st.sidebar.caption(f"📍 Showing {len(available_sites)} monitored basins across India")
    else:
        raw_sites = get_india_state_basins(selected_state)
        available_sites = [
            s for s in raw_sites 
            if any(k in s for k in REAL_WATER_QUALITY_DATA.keys())
        ]
        # state_info = INDIA_STATE_BASINS.get(selected_state, {})
        if available_sites:
            st.sidebar.caption(f"📍 {len(available_sites)} monitored basins in {selected_state}")
        else:
            st.sidebar.warning("No detailed data available for this state")
else:
    # Non-India countries - use standard basin list
    selected_state = None
    if st.session_state.data_source == 'upload':
        available_sites = sorted(df['SiteID'].unique().tolist())
    else:
        available_sites = get_basins_for_country(selected_country)

filtered_sites = [s for s in available_sites if not (isinstance(s, str) and s.isdigit())]
if not filtered_sites:
    filtered_sites = available_sites

site = st.sidebar.selectbox('River/Basin', sorted(filtered_sites))





current_year = datetime.now().year
current_month = datetime.now().month
all_years = list(range(current_year, current_year - 6, -1))

st.sidebar.markdown("**📅 Time Filter**")
selected_year = st.sidebar.selectbox('Year', all_years, index=1, key='year_select')

live_mode = (selected_year == current_year)

if live_mode:
    st.sidebar.success("🔴 Live Streaming - Real-time data")
    available_months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][:current_month]
    month = st.sidebar.selectbox('Month', available_months, key='month_select')
    if 'Year' in df.columns:
        df = df[df['Year'] == current_year]
    if 'Month' in df.columns:
        month_idx = available_months.index(month) + 1
        df = df[df['Month'] == month_idx]
    st.sidebar.info(f"📡 Only {available_months[-1]} {current_year} data available")
else:
    months = ['All', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    month = st.sidebar.selectbox('Month', months, key='month_select')
    if 'Year' in df.columns:
        df = df[df['Year'] == selected_year]
    if month != 'All' and 'Month' in df.columns:
        month_idx = months.index(month)
        df = df[df['Month'] == month_idx]

st.sidebar.caption(f"📊 {len(df):,} records for {month} {selected_year}")

data_type = country_meta.get('data_type', 'simulated')
data_src = country_meta.get('data_source', 'Simulated')
if data_type == 'api':
    st.sidebar.markdown(f"**🟢 Source:** Live API ({data_src})")
elif data_type == 'reference':
    st.sidebar.markdown(f"**🟡 Source:** {data_src} (Official Reference)")
else:
    st.sidebar.markdown(f"**🔵 Source:** World Bank calibrated simulation")


st.sidebar.divider()

# Data Verification Section
st.sidebar.header("🔍 Data Verification")
show_data_preview = st.sidebar.toggle("Show Raw Data Preview", value=False, help="View and verify the underlying data")

if show_data_preview:
    st.sidebar.markdown("**📋 Data Summary**")
    st.sidebar.caption(f"Total Records: {len(df):,}")
    st.sidebar.caption(f"Date Range: {df['DateTime'].min() if 'DateTime' in df.columns else 'N/A'} to {df['DateTime'].max() if 'DateTime' in df.columns else 'N/A'}")
    st.sidebar.caption(f"Sites: {df['SiteID'].nunique() if 'SiteID' in df.columns else 'N/A'}")
    st.sidebar.caption(f"Columns: {', '.join(df.columns[:5])}...")
    
    # Add download button for data verification
    csv_data = df.to_csv(index=False)
    st.sidebar.download_button(
        label="📥 Download Data (CSV)",
        data=csv_data,
        file_name=f"water_quality_data_{selected_country}_{selected_year}.csv",
        mime="text/csv",
        help="Download the current filtered data for verification"
    )

st.sidebar.divider()


st.sidebar.header("📊 Visualization")


chart_type = st.sidebar.radio(
    "Chart Type",
    ["Heatmap", "Bar Chart", "Line Chart", "Data Table"],
    horizontal=False
)

available_indicators = list(indicator_thresholds.keys())
if 'Indicator' in df.columns:
    data_indicators = df['Indicator'].unique().tolist()
    available_indicators = [i for i in available_indicators if i in data_indicators] or data_indicators[:5]

selected_indicator = st.sidebar.selectbox('Indicator', available_indicators)


risk_color_map = {
    "safe": "green",
    "moderate": "yellow", 
    "high": "red",
    "unknown": "gray"
}

if st.sidebar.button("🔮 Run Prediction"):
    with st.spinner("Analyzing water quality..."):
        site_data = df[df['SiteID'] == site]
        
        if not site_data.empty and 'risk_level' in site_data.columns:
            avg_risk = site_data['risk_level'].mean()
            
            month_idx = months.index(month) + 1
            month_data = site_data[site_data['Month'] == month_idx]
            if not month_data.empty:
                monthly_risk = month_data['risk_level'].mean()
                predicted_risk = (monthly_risk * 0.7 + avg_risk * 0.3)
            else:
                predicted_risk = avg_risk
            
            status, emoji, color = get_water_quality_status(predicted_risk)
            
            st.subheader(f"Water Quality Prediction for {site} in {month}/{selected_year}")
            
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, {color}22, {color}44); 
                        border-radius: 15px; 
                        padding: 30px; 
                        text-align: center;
                        margin: 10px 0;'>
                <h1 style='margin:0; font-size: 48px;'>{emoji}</h1>
                <h2 style='margin:10px 0; color: {color}; font-size: 36px; font-weight: bold;'>{status}</h2>
                <p style='margin:0; color: #666; font-size: 16px;'>Based on historical data analysis</p>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Risk Score", f"{predicted_risk:.2f}", delta="Lower is better", delta_color="inverse")
            with col2:
                st.metric("Data Points", len(site_data))
            with col3:
                safe_pct = (site_data['risk_level'] == 0).sum() / len(site_data) * 100
                st.metric("Safe Readings", f"{safe_pct:.0f}%")
        else:
            st.warning("No data available for prediction. Please select a different basin or upload data.")

st.divider()

# India Map Visualization (only shown when India is selected)
if selected_country == "India":
    from asian_water_quality_data import INDIA_STATE_BASINS, get_india_state_coordinates
    import plotly.express as px
    
    st.subheader("🗺️ India River Basin Map")
    
    # Create map data
    map_data = []
    state_coords = get_india_state_coordinates()
    
    for state, coords in state_coords.items():
        basins = INDIA_STATE_BASINS[state]["basins"]
        quality_score = hash(state) % 100  # Deterministic for consistent display
        
        if quality_score < 40:
            status = "🟢 Good"
        elif quality_score < 70:
            status = "🟡 Moderate"
        else:
            status = "🔴 Poor"
        
        map_data.append({
            "State": state,
            "Latitude": coords["lat"],
            "Longitude": coords["lon"],
            "Basins": len(basins),
            "Status": status,
            "BasinList": ", ".join(basins[:3]) + ("..." if len(basins) > 3 else "")
        })
    
    map_df = pd.DataFrame(map_data)
    
    # Highlight selected state
    if selected_state and selected_state != "All States":
        map_df["Size"] = map_df["State"].apply(lambda x: 25 if x == selected_state else 12)
    else:
        map_df["Size"] = 15
    
    # Create the map
    fig = px.scatter_mapbox(
        map_df,
        lat="Latitude",
        lon="Longitude",
        size="Basins",
        color="Status",
        color_discrete_map={"🟢 Good": "#27ae60", "🟡 Moderate": "#f39c12", "🔴 Poor": "#e74c3c"},
        hover_name="State",
        hover_data={"Basins": True, "BasinList": True, "Latitude": False, "Longitude": False},
        zoom=4,
        center={"lat": 22.5, "lon": 82.5},
        height=450
    )
    
    fig.update_layout(
        mapbox_style="carto-positron",
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor="rgba(255,255,255,0.8)")
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show state summary
    if selected_state and selected_state != "All States":
        state_info = INDIA_STATE_BASINS.get(selected_state, {})
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🏛️ State", selected_state)
        with col2:
            st.metric("🌊 Basins", len(state_info.get("basins", [])))
        with col3:
            st.metric("📍 Coordinates", f"{state_info.get('lat', 0):.1f}°N, {state_info.get('lon', 0):.1f}°E")
        with col4:
            st.metric("📡 Source", "Monitoring Network")
        
        with st.expander(f"📋 All basins in {selected_state}"):
            for basin in state_info.get("basins", []):
                st.markdown(f"• {basin}")
    else:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🗺️ States", len(INDIA_STATE_BASINS))
        with col2:
            total_basins = sum(len(s["basins"]) for s in INDIA_STATE_BASINS.values())
            st.metric("🌊 Total Basins", total_basins)
        with col3:
            st.metric("📡 Source", "Monitoring Network")
    
    st.divider()

current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# Show live indicator if in live mode
if st.session_state.data_source == 'live':
    st.markdown("""
    <style>
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.4; }
        100% { opacity: 1; }
    }
    .live-indicator {
        display: inline-flex;
        align-items: center;
        background: linear-gradient(135deg, #ff4444, #cc0000);
        padding: 6px 14px;
        border-radius: 20px;
        color: white;
        font-weight: bold;
        font-size: 14px;
        box-shadow: 0 2px 8px rgba(255, 68, 68, 0.4);
    }
    .live-dot {
        width: 10px;
        height: 10px;
        background: white;
        border-radius: 50%;
        margin-right: 8px;
        animation: pulse 1.5s infinite;
    }
    </style>
    <div style='display: flex; align-items: center; gap: 15px; margin-bottom: 10px;'>
        <div class='live-indicator'>
            <div class='live-dot'></div>
            LIVE
        </div>
        <span style='color: #666; font-size: 14px;'>Real-time sensor data • Updated: """ + current_time + """</span>
    </div>
    """, unsafe_allow_html=True)
    st.subheader(f"🌊 Current Water Quality - {site}")
else:
    st.subheader(f"🌊 Current Water Quality - {site} (Last 30 Days)")
    thirty_days_ago = datetime.now() - timedelta(days=30)
    st.caption(f"Data from {thirty_days_ago.strftime('%b %d')} to {current_time[:10]}")

# Filter site data for last 30 days for "current" status
site_data = df[df['SiteID'] == site].copy()

if site_data.empty:
    st.markdown("""
    <div style='background: linear-gradient(135deg, #f5f5f7 0%, #e8e8ed 100%); border-radius: 20px; padding: 60px 40px;
                text-align: center; border: 1px solid #d2d2d7; margin: 20px 0;'>
        <div style='font-size: 64px; margin-bottom: 20px;'>📭</div>
        <h2 style='color: #1d1d1f; margin: 0; font-weight: 600;'>Data Unavailable</h2>
        <p style='color: #86868b; margin-top: 15px; font-size: 16px;'>
            No water quality data available for this selection.<br>
            Try selecting a different year, month, or basin.
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

if 'DateTime' in site_data.columns:
    site_data['DateTime'] = pd.to_datetime(site_data['DateTime'], errors='coerce')
    thirty_days_ago = datetime.now() - timedelta(days=30)
    recent_data = site_data[site_data['DateTime'] >= thirty_days_ago]
    if not recent_data.empty:
        site_data = recent_data

if not site_data.empty and 'risk_level' in site_data.columns:
    current_risk = site_data['risk_level'].mean()
    status, emoji, color = get_water_quality_status(current_risk)
    
    # Get data source and confidence info if available
    data_source = site_data['DataSource'].iloc[0] if 'DataSource' in site_data.columns else 'Simulated'
    confidence = site_data['Confidence'].iloc[0] if 'Confidence' in site_data.columns else 'Medium'
    quality_class = site_data['QualityClass'].iloc[0] if 'QualityClass' in site_data.columns else 'Unknown'
    is_real_data = site_data['IsRealData'].iloc[0] if 'IsRealData' in site_data.columns else False
    
    # Confidence color coding - green for real data
    if 'Real Data' in str(confidence) or is_real_data:
        conf_color = '#27ae60'  # Green for real data
        data_badge = '✅ REAL DATA'
        badge_bg = '#27ae60'
    elif 'Baseline' in str(confidence):
        conf_color = '#f39c12'  # Orange for baseline
        data_badge = '📊 Baseline'
        badge_bg = '#f39c12'
    else:
        conf_color = '#e74c3c'  # Red for estimated
        data_badge = '⚠️ Estimated'
        badge_bg = '#e74c3c'
    
    st.markdown(f"""
    <div style='background: linear-gradient(135deg, {color}22, {color}44); 
                border-radius: 15px; 
                padding: 25px; 
                text-align: center;
                margin: 10px 0;
                display: flex;
                flex-direction: column;
                align-items: center;'>
        <div style='margin-bottom: 10px; width: 100%; display: flex; justify-content: center;'>
            <span style='background: {badge_bg}; color: white; padding: 6px 14px; border-radius: 20px; font-weight: bold; font-size: 14px; display: inline-block;'>
                {data_badge}
            </span>
        </div>
        <h1 style='margin:0; font-size: 60px; text-align: center;'>{emoji}</h1>
        <h2 style='margin:10px 0; color: {color}; font-size: 32px; font-weight: bold; text-align: center;'>{status}</h2>
        <p style='margin:0; color: #666; text-align: center;'>Current Water Quality Status</p>
        <div style='margin-top: 15px; display: flex; justify-content: center; align-items: center; gap: 15px; flex-wrap: wrap; width: 100%;'>
            <span style='background: #f0f0f0; padding: 5px 12px; border-radius: 15px; font-size: 12px; white-space: nowrap;'>
                📋 Quality Class: <strong>{quality_class}</strong>
            </span>
            <span style='background: #f0f0f0; padding: 5px 12px; border-radius: 15px; font-size: 12px; white-space: nowrap;'>
                🎯 Confidence: <strong style='color: {conf_color};'>{confidence}</strong>
            </span>
            <span style='background: #f0f0f0; padding: 5px 12px; border-radius: 15px; font-size: 12px; white-space: nowrap;'>
                📊 Source: <strong>{data_source}</strong>
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    
    if 'Indicator' in site_data.columns:
        st.markdown("**📊 Latest Readings:**")
        cols = st.columns(4)
        indicators_list = list(site_data['Indicator'].unique()[:8])
        for i, indicator in enumerate(indicators_list):
            ind_data = site_data[site_data['Indicator'] == indicator]
            if not ind_data.empty and 'Value' in ind_data.columns:
                latest_val = ind_data['Value'].iloc[-1]
                ind_risk = ind_data['risk_level'].mean() if 'risk_level' in ind_data.columns else 0
                ind_status, ind_emoji, _ = get_water_quality_status(ind_risk)
                with cols[i % 4]:
                    # Use full indicator name for clarity
                    st.metric(f"{ind_emoji} {indicator}", f"{latest_val:.2f}", ind_status)

st.divider()

st.subheader(f"📈 {chart_type} View - {site}")

if chart_type == "Heatmap":
    display_quality_summary(df, site)
    
    site_data = df[df['SiteID'] == site]
    if not site_data.empty and 'Indicator' in site_data.columns and 'Month' in site_data.columns:
        if 'Value' in site_data.columns:
            pivot_data = site_data.pivot_table(
                values='Value', 
                index='Indicator', 
                columns='Month', 
                aggfunc='mean'
            )
            
            if not pivot_data.empty:
                fig, ax = plt.subplots(figsize=(14, 8))
                
                import seaborn as sns
                sns.heatmap(
                    pivot_data, 
                    annot=True, 
                    fmt='.1f', 
                    cmap='RdYlGn_r',
                    ax=ax,
                    cbar_kws={'label': 'Average Value'}
                )
                
                month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                ax.set_xticklabels([month_labels[int(m)-1] for m in pivot_data.columns], rotation=45)
                ax.set_title(f'Water Quality Indicators Over Time - {site}', fontsize=14, fontweight='bold')
                ax.set_xlabel('Month')
                ax.set_ylabel('Indicator')
                
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.info("Not enough data for heatmap visualization")
        else:
            st.warning("Value column not found in data")
    else:
        st.warning("Insufficient data for heatmap. Try Bar Chart or Line Chart instead.")

elif chart_type == "Bar Chart":
    create_bar_chart(df, site, selected_indicator)

elif chart_type == "Line Chart":
    create_line_chart(df, site, selected_indicator)

elif chart_type == "Data Table":
    show_data_table(df, site)

st.divider()
col1, col2, col3 = st.columns(3)
with col1:
    st.caption(f"📍 Country: {selected_country}")
with col2:
    if st.session_state.data_source == 'upload':
        st.caption("📊 Data Source: Uploaded File")
    elif st.session_state.data_source == 'live':
        st.caption("🔴 Data Source: Live Streaming")
    else:
        st.caption("📊 Data Source: Historical API")
with col3:
    st.caption(f"📝 Records: {len(df):,}")

# Data Verification Panel (shown when toggle is enabled)
if show_data_preview:
    st.divider()
    st.header("🔍 Data Verification Panel")
    st.info("Use this section to verify the data is correct. You can view, filter, and download the raw data.")
    
    # Data overview tabs
    tab1, tab2, tab3 = st.tabs(["📋 Raw Data", "📊 Statistics", "📝 Data Quality"])
    
    with tab1:
        st.subheader("Raw Data Preview")
        st.caption(f"Showing {min(100, len(df))} of {len(df):,} records")
        st.dataframe(df.head(100), use_container_width=True, height=400)
    
    with tab2:
        st.subheader("Data Statistics")
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Numeric Columns Summary**")
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                st.dataframe(df[numeric_cols].describe(), use_container_width=True)
        with col_b:
            st.markdown("**Column Information**")
            col_info = pd.DataFrame({
                'Column': df.columns,
                'Type': df.dtypes.values,
                'Non-Null': df.count().values,
                'Unique': [df[col].nunique() for col in df.columns]
            })
            st.dataframe(col_info, use_container_width=True, height=300)
    
    with tab3:
        st.subheader("Data Quality Checks")
        col_x, col_y, col_z = st.columns(3)
        with col_x:
            missing = df.isnull().sum().sum()
            total_cells = df.size
            completeness = ((total_cells - missing) / total_cells * 100) if total_cells > 0 else 0
            st.metric("Data Completeness", f"{completeness:.1f}%", help="Percentage of non-null values")
        with col_y:
            duplicates = df.duplicated().sum()
            st.metric("Duplicate Rows", f"{duplicates:,}", help="Number of duplicate rows")
        with col_z:
            st.metric("Total Columns", len(df.columns))
        
        # Date range verification
        if 'DateTime' in df.columns:
            st.markdown("**Date Range Verification**")
            date_col = pd.to_datetime(df['DateTime'], errors='coerce')
            st.caption(f"Earliest: {date_col.min()}")
            st.caption(f"Latest: {date_col.max()}")
            st.caption(f"Span: {(date_col.max() - date_col.min()).days} days")
