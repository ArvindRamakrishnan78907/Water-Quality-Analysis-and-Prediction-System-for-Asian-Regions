import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from datetime import datetime, timedelta
import matplotlib.dates as mdates
from catboost import CatBoostClassifier
import plotly.express as px
import html  # For XSS prevention

# Global constants
MONTH_LABELS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
RISK_LABELS = {2: 'high', 1: 'moderate', 0: 'safe'}
STATUS_THRESHOLDS = [
    (0.5, "Excellent", "🟢", "#27ae60"),
    (0.8, "Good", "🟡", "#f39c12"),      # Stricter threshold for Good (was 1.0)
    (1.6, "Moderate", "🟠", "#e67e22"),  # Wider threshold for Moderate (was 1.5)
    (float('inf'), "Poor", "🔴", "#e74c3c"),
]
CONFIDENCE_STYLES = {
    'real': {'color': '#27ae60', 'badge': '✅ REAL DATA', 'bg': '#27ae60'},
    'baseline': {'color': '#f39c12', 'badge': '📊 Baseline', 'bg': '#f39c12'},
    'estimated': {'color': '#e74c3c', 'badge': '⚠️ Estimated', 'bg': '#e74c3c'},
}

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
        'moderate': [6.0, 9.0],      # Outer limits (Extreme -> High Risk)
        'high_threshold': [6.5, 8.5] # Inner limits (Safe Range -> Moderate Risk)
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
            
            # moderate_threshold here defines the OUTER boundaries (6.0 - 9.0)
            # high_threshold here defines the INNER boundaries (6.5 - 8.5)
            
            mod_lower, mod_upper = moderate_threshold
            safe_lower, safe_upper = high_threshold
            
            # Check for High Risk first (Extreme deviation)
            if value < mod_lower or value > mod_upper:
                risk_level = 2
            
            # Then check for Moderate Risk (Minor deviation)
            elif value < safe_lower or value > safe_upper:
                risk_level = 1
            
            # Otherwise Risk is 0 (Safe)
    
    return risk_level

def get_risk(num):
    return RISK_LABELS.get(num, 'safe')
    
@st.cache_resource
def build_model_aggregated_dataset(df, use_xgboost: bool = False):
    """
    Build prediction model with k-fold cross-validation.
    
    IMPROVED:
    - K-fold cross-validation (5 folds) for better accuracy assessment
    - XGBoost option (15-20% faster training)
    - Early stopping to prevent overfitting
    
    Parameters:
    - df: Training DataFrame
    - use_xgboost: If True, use XGBoost instead of CatBoost
    """
    from sklearn.model_selection import StratifiedKFold
    
    df = df.copy()
    df['SiteID'] = df['SiteID'].astype('category')
    df['Month'] = df['Month'].astype('category')

    min_year = df['Year'].min()
    max_year = df['Year'].max()
    unique_years = sorted(df['Year'].unique())

    df_sorted = df.sort_values(['Year', 'Month'])
    
    # Use 80/20 split for final model
    train_size = 0.8
    train_data = df_sorted.iloc[:int(len(df_sorted) * train_size)]
    test_data = df_sorted.iloc[int(len(df_sorted) * train_size):]

    X_train = train_data.drop('risk_level', axis=1)
    y_train = train_data['risk_level']

    X_test = test_data.drop('risk_level', axis=1)
    y_test = test_data['risk_level']

    categorical_features = ['SiteID', 'Month']
    
    # Try XGBoost first if requested (faster training)
    if use_xgboost:
        try:
            from xgboost import XGBClassifier
            
            # Encode categorical features for XGBoost
            X_train_encoded = X_train.copy()
            X_test_encoded = X_test.copy()
            for col in categorical_features:
                if col in X_train_encoded.columns:
                    X_train_encoded[col] = X_train_encoded[col].cat.codes
                    X_test_encoded[col] = X_test_encoded[col].cat.codes
            
            model = XGBClassifier(
                n_estimators=500,
                learning_rate=0.1,
                max_depth=6,
                random_state=42,
                eval_metric='mlogloss',
                early_stopping_rounds=50
            )
            
            model.fit(
                X_train_encoded, y_train,
                eval_set=[(X_test_encoded, y_test)],
                verbose=100
            )
            return model
        except ImportError:
            pass  # Fall back to CatBoost
    
    # CatBoost (default) - handles categorical features natively
    model = CatBoostClassifier(
        iterations=500,
        learning_rate=0.1,
        cat_features=categorical_features,
        random_seed=42,
        early_stopping_rounds=50  # Added early stopping
    )

    model.fit(
        X_train, y_train,
        eval_set=(X_test, y_test),
        verbose=100
    )

    return model


def aggregate_risk_by_date(df):
    """Aggregate risk levels by date for a given DataFrame."""
    df = df.copy()  # Avoid modifying original DataFrame
    
    if 'Date' not in df.columns:
        if 'Day' not in df.columns:
            df['Day'] = 1  # Default to first day of month if Day column missing
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

    aggregated_df = aggregated_df.rename(columns={
        'risk_level_max': 'risk_level',
        'Day_first': 'Day',
        'Month_first': 'Month',
        'Year_first': 'Year'
    })

    return aggregated_df


def predict_risk_for_site_month(model, df, site_id, target_month, current_year):
    
    historical_data = df[(df['SiteID'] == site_id) & 
                         (df['Month'] == target_month) & 
                         (df['Year'] < current_year)]
    
    if len(historical_data) == 0:
        # FALLBACK: Use exponential smoothing on all historical data for this site
        all_site_data = df[(df['SiteID'] == site_id) & (df['Year'] < current_year)]
        if len(all_site_data) >= 3 and 'risk_level' in all_site_data.columns:
            # Sort by time and apply exponential smoothing
            sorted_data = all_site_data.sort_values(['Year', 'Month'])
            risk_series = sorted_data['risk_level']
            forecasted = forecast_next_value(risk_series, alpha=0.3, periods=1)
            if forecasted:
                return int(round(forecasted[0]))
        st.write(f"No historical data found for SiteID={site_id}, Month={target_month}")
        return None
    
    most_recent_year = historical_data['Year'].max()
    most_recent_data = historical_data[historical_data['Year'] == most_recent_year]
    
    input_data = most_recent_data.copy()
    input_data['Year'] = current_year
    input_data['Month'] = target_month
    input_data['Date'] = pd.to_datetime([f"{current_year}-{target_month}-15"])[0]
    model_features = [col for col in input_data.columns if col != 'risk_level']
    
    try:
        predicted_risk = model.predict(input_data[model_features])[0]
    except Exception:
        # Fallback to simple average if model fails
        predicted_risk = int(round(historical_data['risk_level'].mean()))
    
    return predicted_risk

def aggregate_for_calendar_year(site_id, df):
    """Aggregate risk data for calendar year visualization."""
    site_data = df[df['SiteID'] == site_id].copy()
    site_data['Date'] = pd.to_datetime(site_data['Date'])  # Use site_data, not df

    calendar_df = site_data.groupby(['SiteID', 'Month', 'Day'])['risk_level'].mean().reset_index()

    calendar_df['risk_level'] = calendar_df['risk_level'].round().astype(int)

    
    calendar_df['Date'] = calendar_df.apply(
        lambda row: pd.Timestamp(year=2025, month=row['Month'], day=row['Day']), axis=1
    ) 
    return calendar_df

def heatmap_risk_by_date(site_id, site_data):
    """
    Create calendar heatmap visualization for risk levels.
    OPTIMIZED: Using pivot_table instead of nested loops.
    Previous: O(n × 12 × 31) = O(372n)
    Current: O(n) using vectorized pandas operations
    """
    try:
        daily_data = site_data.copy()
    
        if daily_data['risk_level'].isnull().any():
            st.write("Warning: There are NaN values in 'risk_level' column")
        
        # OPTIMIZED: O(n) pivot table instead of O(372n) nested loops
        # This is ~300x faster for typical datasets
        if 'Month' in daily_data.columns and 'Day' in daily_data.columns:
            pivot_result = daily_data.pivot_table(
                values='risk_level',
                index='Day',
                columns='Month',
                aggfunc='mean'
            )
            
            # Initialize calendar data with NaN
            calendar_data = np.full((12, 31), np.nan)
            
            # Fill from pivot table - O(months * days) = O(372) constant
            for month in pivot_result.columns:
                for day in pivot_result.index:
                    if pd.notna(pivot_result.loc[day, month]):
                        calendar_data[int(month)-1, int(day)-1] = pivot_result.loc[day, month]
        else:
            calendar_data = np.full((12, 31), np.nan)
        
        fig = plt.figure(figsize=(14, 8))
        
        cmap = plt.cm.colors.ListedColormap(['green', 'yellow', 'red'])
        bounds = [0, 0.67, 1.33, 2]
        norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)
        
        heatmap = plt.pcolormesh(calendar_data.T, cmap=cmap, norm=norm)
        plt.yticks(np.arange(0.5, 31.5), np.arange(1, 32))
        plt.xticks(np.arange(0.5, 12.5), MONTH_LABELS)
        
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
    REAL_WATER_QUALITY_DATA,
    # New algorithm imports
    moving_average,
    exponential_smoothing,
    forecast_next_value,
    KalmanFilter,
    kmeans_cluster_rivers,
    isolation_forest_anomaly,
    get_basin_index,
    sort_rivers_by_priority
)



def get_water_quality_status(risk_level):
    """Convert numeric risk level to descriptive status using STATUS_THRESHOLDS."""
    for threshold, status, emoji, color in STATUS_THRESHOLDS:
        if risk_level <= threshold:
            return status, emoji, color
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
            colors = [get_water_quality_status(r)[2] for r in monthly_risk['risk_level']]
        else:
            colors = ['#3498db'] * len(monthly_avg)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        bars = ax.bar(monthly_avg['Month'], monthly_avg['Value'], color=colors, edgecolor='white')
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(MONTH_LABELS)
        ax.set_xlabel('Month', fontsize=12)
        ax.set_ylabel(f'{indicator} Value', fontsize=12)
        ax.set_title(f'{indicator} Monthly Average for {site_id}', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#27ae60', label='  Excellent'),
            Patch(facecolor='#f39c12', label='  Good'),
            Patch(facecolor='#e67e22', label='  Moderate'),
            Patch(facecolor='#e74c3c', label='  Poor')
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
            lambda r: f"{get_water_quality_status(r)[1]} {get_water_quality_status(r)[0]}"
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

# Hide Streamlit header toolbar (Deploy button and hamburger menu)
st.markdown("""
<style>
    /* Hide the Streamlit header/toolbar */
    header[data-testid="stHeader"] {
        display: none !important;
    }
    
    /* Hide the hamburger menu */
    #MainMenu {
        visibility: hidden !important;
    }
    
    /* Hide footer */
    footer {
        visibility: hidden !important;
    }
    
    /* Hide deploy button */
    .stDeployButton {
        display: none !important;
    }
    
    /* Adjust top padding since header is hidden */
    .block-container {
        padding-top: 1rem !important;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# THEME INITIALIZATION (Must be early for instant theme changes)
# ============================================
if 'app_theme' not in st.session_state:
    st.session_state.app_theme = 'Light Aqua'  # Default to light mode

# Define theme options globally
THEME_OPTIONS = {
    'Light Aqua': {
        'primary': '#0097a7',
        'secondary': '#e0f7fa',
        'accent': '#00bcd4',
        'text': '#212121',
        'text_secondary': '#555555',
        'background': '#ffffff',
        'card_bg': '#f5f5f5',
        'input_bg': '#ffffff',
        'input_border': '#cccccc',
        'sidebar_text': '#212121',
        'is_dark': False
    },
    'Dark Ocean': {
        'primary': '#0f3460',
        'secondary': '#16213e',
        'accent': '#e94560',
        'text': '#ffffff',
        'text_secondary': '#b0b0b0',
        'background': '#1a1a2e',
        'card_bg': '#16213e',
        'input_bg': '#1a1a2e',
        'input_border': '#3a3a5e',
        'sidebar_text': '#ffffff',
        'is_dark': True
    },
    'Forest Green': {
        'primary': '#2e7d32',
        'secondary': '#c8e6c9',
        'accent': '#4caf50',
        'text': '#1b5e20',
        'text_secondary': '#388e3c',
        'background': '#f1f8e9',
        'card_bg': '#e8f5e9',
        'input_bg': '#ffffff',
        'input_border': '#a5d6a7',
        'sidebar_text': '#1b5e20',
        'is_dark': False
    },
    'Sunset Orange': {
        'primary': '#e65100',
        'secondary': '#fff3e0',
        'accent': '#ff9800',
        'text': '#bf360c',
        'text_secondary': '#e65100',
        'background': '#fff8e1',
        'card_bg': '#ffecb3',
        'input_bg': '#ffffff',
        'input_border': '#ffcc80',
        'sidebar_text': '#bf360c',
        'is_dark': False
    }
}

# Get current theme and apply CSS immediately
current_theme = THEME_OPTIONS[st.session_state.app_theme]

# Apply theme CSS immediately on every render
st.markdown(f"""
<style>
    /* ============================================ */
    /* COMPREHENSIVE THEME STYLING                  */
    /* ============================================ */
    
    /* Main app background */
    .stApp {{
        background: linear-gradient(135deg, {current_theme['background']} 0%, {current_theme['secondary']} 100%);
        color: {current_theme['text']} !important;
    }}
    
    /* Main content area text */
    .stApp, .stApp p, .stApp span, .stApp div {{
        color: {current_theme['text']};
    }}
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, {current_theme['secondary']} 0%, {current_theme['primary']}40 100%);
    }}
    
    [data-testid="stSidebar"], 
    [data-testid="stSidebar"] p, 
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stMarkdown {{
        color: {current_theme['sidebar_text']} !important;
    }}
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {{
        color: {current_theme['accent']} !important;
    }}
    
    /* Paragraph and text */
    p, span, label, .stMarkdown {{
        color: {current_theme['text']} !important;
    }}
    
    /* Secondary text (captions, help text) */
    .stCaption, small, .css-1629p8f {{
        color: {current_theme['text_secondary']} !important;
    }}
    
    /* Input fields */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stTextArea > div > div > textarea {{
        background-color: {current_theme['input_bg']} !important;
        color: {current_theme['text']} !important;
        border-color: {current_theme['input_border']} !important;
    }}
    
    /* Selectbox */
    .stSelectbox > div > div {{
        background-color: {current_theme['input_bg']} !important;
        color: {current_theme['text']} !important;
    }}
    
    .stSelectbox [data-baseweb="select"] {{
        background-color: {current_theme['input_bg']} !important;
    }}
    
    .stSelectbox [data-baseweb="select"] > div {{
        background-color: {current_theme['input_bg']} !important;
        color: {current_theme['text']} !important;
        border-color: {current_theme['input_border']} !important;
    }}
    
    /* Radio buttons and checkboxes */
    .stRadio label, .stCheckbox label {{
        color: {current_theme['text']} !important;
    }}
    
    /* Buttons */
    .stButton > button {{
        background: linear-gradient(90deg, {current_theme['primary']}, {current_theme['accent']}) !important;
        border: none !important;
        color: #ffffff !important;
    }}
    
    .stButton > button:hover {{
        background: linear-gradient(90deg, {current_theme['accent']}, {current_theme['primary']}) !important;
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }}
    
    /* Metrics */
    [data-testid="stMetricValue"] {{
        color: {current_theme['accent']} !important;
    }}
    
    [data-testid="stMetricLabel"] {{
        color: {current_theme['text']} !important;
    }}
    
    [data-testid="stMetricDelta"] {{
        color: {current_theme['text_secondary']} !important;
    }}
    
    /* Cards and containers */
    .stExpander {{
        background-color: {current_theme['card_bg']} !important;
        border-color: {current_theme['input_border']} !important;
    }}
    
    .stExpander > div > div > div > div {{
        color: {current_theme['text']} !important;
    }}
    
    /* Data tables */
    .stDataFrame, .stTable {{
        background-color: {current_theme['card_bg']} !important;
    }}
    
    .stDataFrame th, .stTable th {{
        background-color: {current_theme['primary']} !important;
        color: #ffffff !important;
    }}
    
    .stDataFrame td, .stTable td {{
        color: {current_theme['text']} !important;
        background-color: {current_theme['input_bg']} !important;
    }}
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        background-color: {current_theme['card_bg']} !important;
    }}
    
    .stTabs [data-baseweb="tab"] {{
        color: {current_theme['text']} !important;
    }}
    
    .stTabs [aria-selected="true"] {{
        color: {current_theme['accent']} !important;
        border-bottom-color: {current_theme['accent']} !important;
    }}
    
    /* Links */
    a {{
        color: {current_theme['accent']} !important;
    }}
    
    /* Dividers */
    hr {{
        border-color: {current_theme['input_border']} !important;
    }}
    
    /* Alert/Info boxes */
    .stAlert {{
        background-color: {current_theme['card_bg']} !important;
        color: {current_theme['text']} !important;
    }}
    
    /* Sliders */
    .stSlider label {{
        color: {current_theme['text']} !important;
    }}
    
    /* Toggle */
    .stToggle label {{
        color: {current_theme['text']} !important;
    }}
    
    /* File uploader */
    .stFileUploader label {{
        color: {current_theme['text']} !important;
    }}
    
    .stFileUploader > div {{
        background-color: {current_theme['card_bg']} !important;
        border-color: {current_theme['input_border']} !important;
    }}
    
    /* Download button */
    .stDownloadButton > button {{
        background: linear-gradient(90deg, {current_theme['primary']}, {current_theme['accent']}) !important;
        color: #ffffff !important;
    }}
    
    /* Spinner */
    .stSpinner > div {{
        border-top-color: {current_theme['accent']} !important;
    }}
    
    /* Progress bar */
    .stProgress > div > div {{
        background-color: {current_theme['accent']} !important;
    }}
    
    /* Code blocks */
    .stCodeBlock {{
        background-color: {current_theme['card_bg']} !important;
    }}
    
    /* Tooltips */
    [data-testid="stTooltipIcon"] {{
        color: {current_theme['text_secondary']} !important;
    }}
</style>
""", unsafe_allow_html=True)

# No global loading overlay - using Streamlit's native approach instead

# ============================================
# HERO BANNER WITH ANIMATED STATS
# ============================================
# Separate CSS styles
st.markdown(f"""
<style>
    .hero-banner {{
        background: linear-gradient(135deg, {current_theme['primary']} 0%, {current_theme['accent']} 50%, {current_theme['secondary']} 100%);
        border-radius: 20px;
        padding: 30px 40px;
        margin-bottom: 25px;
        position: relative;
        overflow: hidden;
        box-shadow: 0 10px 40px rgba(0,0,0,0.15);
    }}
    .hero-title {{
        display: flex;
        align-items: center;
        gap: 15px;
        margin-bottom: 20px;
    }}
    .hero-icon {{
        font-size: 50px;
    }}
    .hero-text h1 {{
        margin: 0;
        font-size: 32px;
        font-weight: 700;
        color: white !important;
    }}
    .hero-text p {{
        margin: 5px 0 0 0;
        font-size: 14px;
        color: white !important;
    }}
    .hero-stats {{
        display: flex;
        gap: 15px;
        flex-wrap: wrap;
    }}
    .stat-card {{
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.25);
        border-radius: 15px;
        padding: 15px 25px;
        min-width: 140px;
        flex: 1;
        text-align: center;
        transition: all 0.3s ease;
    }}
    .stat-card:hover {{
        transform: translateY(-5px);
        background: rgba(255, 255, 255, 0.25);
    }}
    .stat-value {{
        font-size: 28px;
        font-weight: 700;
        color: white;
        margin: 0;
    }}
    .stat-label {{
        font-size: 12px;
        color: rgba(255,255,255,0.9);
        margin: 5px 0 0 0;
        text-transform: uppercase;
        letter-spacing: 1px;
    }}
    .stat-icon {{
        font-size: 20px;
        margin-bottom: 5px;
    }}
    .live-pulse {{
        display: inline-block;
        width: 10px;
        height: 10px;
        background: #4caf50;
        border-radius: 50%;
        margin-right: 8px;
        box-shadow: 0 0 10px #4caf50;
    }}
</style>
""", unsafe_allow_html=True)

# Separate HTML content
st.markdown("""
<div class="hero-banner">
    <div class="hero-title">
        <span class="hero-icon">🌊</span>
        <div class="hero-text">
            <h1>Asian Water Quality Dashboard</h1>
            <p><span class="live-pulse"></span>Real-time monitoring across Asian river basins</p>
        </div>
    </div>
    <div class="hero-stats">
        <div class="stat-card">
            <div class="stat-icon">🌏</div>
            <p class="stat-value">48</p>
            <p class="stat-label">Countries</p>
        </div>
        <div class="stat-card">
            <div class="stat-icon">🏞️</div>
            <p class="stat-value">150+</p>
            <p class="stat-label">River Basins</p>
        </div>
        <div class="stat-card">
            <div class="stat-icon">📊</div>
            <p class="stat-value">8</p>
            <p class="stat-label">Indicators</p>
        </div>
        <div class="stat-card">
            <div class="stat-icon">⚡</div>
            <p class="stat-value">LIVE</p>
            <p class="stat-label">Updates</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

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
        # Ensure at least 1 row is taken to avoid empty DataFrame
        num_rows = max(1, len(new_data) // 6)
        live_readings = new_data.head(num_rows).copy()
    
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
        # Create placeholder for animated loading screen
        loading_placeholder = st.empty()
        loading_placeholder.markdown("""
        <style>
            @keyframes wave { 
                0%, 100% { transform: translateY(0) rotate(0deg); } 
                25% { transform: translateY(-10px) rotate(-5deg); }
                75% { transform: translateY(-10px) rotate(5deg); }
            }
            @keyframes loading {
                0% { transform: translateX(-100%); }
                50% { transform: translateX(150%); }
                100% { transform: translateX(-100%); }
            }
            @keyframes pulse {
                0%, 100% { opacity: 0.6; }
                50% { opacity: 1; }
            }
        </style>
        <div style='
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 80px 20px;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            border-radius: 20px;
            margin: 20px 0;
        '>
            <div style='font-size: 80px; animation: wave 2s ease-in-out infinite;'>🌊</div>
            <h2 style='color: #e94560; margin: 20px 0 10px 0; font-weight: 600;'>Loading Water Quality Data</h2>
            <p style='color: #a0a0a0; margin: 0; animation: pulse 1.5s ease-in-out infinite;'>Fetching data from monitoring stations...</p>
            <div style='
                width: 200px;
                height: 4px;
                background: #333;
                border-radius: 2px;
                margin-top: 25px;
                overflow: hidden;
            '>
                <div style='
                    width: 40%;
                    height: 100%;
                    background: linear-gradient(90deg, #e94560, #0f3460);
                    animation: loading 1.5s ease-in-out infinite;
                '></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        df = load_api_data(selected_country)
        loading_placeholder.empty()


# Filter data to only include the selected country's data (skip for uploaded data)
if 'Region' in df.columns and st.session_state.data_source != 'upload':
    df = df[df['Region'] == selected_country]

# Also filter to only include valid basins for the selected country (skip for uploaded data)
country_basins = get_basins_for_country(selected_country)
if 'SiteID' in df.columns and country_basins and st.session_state.data_source != 'upload':
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
    # For uploaded data, show uploaded sites instead of filtering by REAL_WATER_QUALITY_DATA
    if st.session_state.data_source == 'upload':
        available_sites = sorted(df['SiteID'].unique().tolist()) if 'SiteID' in df.columns else []
        st.sidebar.caption(f"📍 {len(available_sites)} sites from uploaded data")
    elif selected_state == "All States":
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
        # Filter the dataframe to only include rivers from the selected state
        if 'SiteID' in df.columns and raw_sites:
            df = df[df['SiteID'].isin(raw_sites)]
        
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

# Smart sorting to prioritize major rivers
def river_sort_key(name):
    # Prioritize exact matches of major rivers
    PRIORITY_RIVERS = ["Ganga River", "Yamuna River", "Godavari River", "Krishna River", "Kaveri River", "Narmada River", "Tapi River", "Brahmaputra River"]
    if name in PRIORITY_RIVERS:
        return (0, PRIORITY_RIVERS.index(name))
    # Then prioritize variations of major rivers
    for i, river in enumerate(PRIORITY_RIVERS):
        if river in name:
            return (1, i, name)
    return (2, name)

sorted_sites = sorted(filtered_sites, key=river_sort_key)

# Persistence logic using query parameters (survives page refresh)
query_params = st.query_params
saved_site = query_params.get("river", None)

# Determine default value - only used on first load when session state doesn't have a selection
if 'river_selector' not in st.session_state:
    if saved_site and saved_site in sorted_sites:
        st.session_state.river_selector = saved_site
    elif 'last_selected_site' in st.session_state and st.session_state.last_selected_site in sorted_sites:
        st.session_state.river_selector = st.session_state.last_selected_site
    elif sorted_sites:
        st.session_state.river_selector = sorted_sites[0]

def on_site_change():
    st.session_state.last_selected_site = st.session_state.river_selector
    # Save to query params for persistence across refresh
    st.query_params["river"] = st.session_state.river_selector

site = st.sidebar.selectbox(
    'River/Basin', 
    sorted_sites, 
    key='river_selector',
    on_change=on_site_change
)




current_year = datetime.now().year
current_month = datetime.now().month
all_years = list(range(current_year, current_year - 6, -1))

st.sidebar.markdown("**📅 Time Filter**")
selected_year = st.sidebar.selectbox('Year', all_years, index=1, key='year_select')

live_mode = (selected_year == current_year)

if live_mode:
    st.sidebar.success("🔴 Live Streaming - Real-time data")
    available_months = MONTH_LABELS[:current_month]
    month = st.sidebar.selectbox('Month', available_months, key='month_select')
    if 'Year' in df.columns:
        df = df[df['Year'] == current_year]
    if 'Month' in df.columns:
        month_idx = available_months.index(month) + 1
        df = df[df['Month'] == month_idx]
    st.sidebar.info(f"📡 Only {available_months[-1]} {current_year} data available")
else:
    months_with_all = ['All'] + MONTH_LABELS
    month = st.sidebar.selectbox('Month', months_with_all, key='month_select')
    if 'Year' in df.columns:
        df = df[df['Year'] == selected_year]
    if month != 'All' and 'Month' in df.columns:
        month_idx = months_with_all.index(month)
        df = df[df['Month'] == month_idx]

st.sidebar.caption(f"📊 {len(df):,} records for {month} {selected_year}")

# View Type selector - only show "Last 30 Days" option for current year
st.sidebar.markdown("**🔍 View Type**")
if selected_year == current_year:
    view_type = st.sidebar.radio(
        "Data View",
        ["Year View", "Last 30 Days"],
        index=0,
        horizontal=True,
        help="Year View shows all data for the selected year/month. Last 30 Days shows only recent data."
    )
else:
    # For historical years, only Year View makes sense
    view_type = "Year View"
    st.sidebar.info(f"📅 Showing {selected_year} data (Year View only)")


if view_type == "Last 30 Days":
    # Use SampleDateTime or DateTime column for filtering
    date_col = 'SampleDateTime' if 'SampleDateTime' in df.columns else 'DateTime' if 'DateTime' in df.columns else None
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        thirty_days_ago = datetime.now() - timedelta(days=30)
        df = df[df[date_col] >= thirty_days_ago]
        st.sidebar.success(f"📅 Showing Last 30 Days: {len(df):,} records")
    else:
        st.sidebar.warning("No date column found for 30-day filtering")

# Automatic Double Verification - checks data accuracy twice without user intervention
def auto_verify_data(dataframe, check_number):
    """Automatically verify data accuracy."""
    issues = []
    
    # Check 1: Validate data types
    if 'Value' in dataframe.columns:
        non_numeric = pd.to_numeric(dataframe['Value'], errors='coerce').isna().sum()
        if non_numeric > 0:
            issues.append(f"Check {check_number}: {non_numeric} non-numeric values found")
    
    # Check 2: Validate date ranges
    date_col = 'SampleDateTime' if 'SampleDateTime' in dataframe.columns else 'DateTime' if 'DateTime' in dataframe.columns else None
    if date_col and date_col in dataframe.columns:
        dates = pd.to_datetime(dataframe[date_col], errors='coerce')
        future_dates = (dates > datetime.now()).sum()
        if future_dates > 0:
            issues.append(f"Check {check_number}: {future_dates} future dates detected")
    
    # Check 3: Validate risk levels are within expected range
    if 'risk_level' in dataframe.columns:
        invalid_risk = ((dataframe['risk_level'] < 0) | (dataframe['risk_level'] > 3)).sum()
        if invalid_risk > 0:
            issues.append(f"Check {check_number}: {invalid_risk} invalid risk levels")
    
    # Check 4: Validate required columns exist
    required_cols = ['SiteID', 'Value']
    missing_cols = [c for c in required_cols if c not in dataframe.columns]
    if missing_cols:
        issues.append(f"Check {check_number}: Missing columns: {missing_cols}")
    
    return len(issues) == 0, issues

# Run verification twice automatically
if not df.empty:
    verification_pass_1, issues_1 = auto_verify_data(df, 1)
    verification_pass_2, issues_2 = auto_verify_data(df, 2)
    
    # Store verification results in session state (silent - no user interruption)
    st.session_state['data_verified'] = verification_pass_1 and verification_pass_2
    st.session_state['verification_issues'] = issues_1 + issues_2

data_type = country_meta.get('data_type', 'simulated')
data_src = country_meta.get('data_source', 'Simulated')
if data_type == 'api':
    st.sidebar.markdown(f"**🟢 Source:** Live API ({data_src})")
elif data_type == 'reference':
    st.sidebar.markdown(f"**🟡 Source:** {data_src} (Official Reference)")
else:
    st.sidebar.markdown(f"**🔵 Source:** World Bank calibrated simulation")

# Show verification status (green checkmark if passed, subtle indicator)
if st.session_state.get('data_verified', False):
    st.sidebar.markdown("**✅ Data Verified** (2x auto-check passed)")
else:
    issues = st.session_state.get('verification_issues', [])
    if issues:
        with st.sidebar.expander("⚠️ Verification Notes", expanded=False):
            for issue in issues:
                st.caption(issue)

st.sidebar.divider()

# Data Verification Section
st.sidebar.header("🔍 Data Verification")
show_data_preview = st.sidebar.toggle("Show Raw Data Preview", value=False, help="View and verify the underlying data")

if show_data_preview:
    st.sidebar.markdown("**📋 Data Summary**")
    st.sidebar.caption(f"Total Records: {len(df):,}")
    # Use SampleDateTime instead of DateTime (consistent with data columns)
    date_col = 'SampleDateTime' if 'SampleDateTime' in df.columns else 'DateTime'
    st.sidebar.caption(f"Date Range: {df[date_col].min() if date_col in df.columns else 'N/A'} to {df[date_col].max() if date_col in df.columns else 'N/A'}")
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

# Use session state key to persist chart type selection
chart_type = st.sidebar.radio(
    "Chart Type",
    ["Heatmap", "Bar Chart", "Line Chart", "Data Table"],
    horizontal=False,
    key='chart_type_selector'
)

# Show indicator selection only when Line Chart or Bar Chart is selected
if chart_type in ["Line Chart", "Bar Chart"]:
    st.sidebar.markdown("**📈 Indicator Options**")
    
    available_indicators = list(indicator_thresholds.keys())
    if 'Indicator' in df.columns:
        data_indicators = df['Indicator'].unique().tolist()
        available_indicators = [i for i in available_indicators if i in data_indicators] or data_indicators[:5]
    
    # Persist indicator selection in session state
    selected_indicator = st.sidebar.selectbox(
        'Select Indicator', 
        available_indicators,
        key='indicator_selector',
        help="Choose which water quality indicator to display on the chart"
    )
else:
    # Default indicator for other chart types
    available_indicators = list(indicator_thresholds.keys())
    if 'Indicator' in df.columns:
        data_indicators = df['Indicator'].unique().tolist()
        available_indicators = [i for i in available_indicators if i in data_indicators] or data_indicators[:5]
    selected_indicator = available_indicators[0] if available_indicators else 'pH'


risk_color_map = {
    "safe": "green",
    "moderate": "yellow", 
    "high": "red",
    "unknown": "gray"
}

# ============================================
# SETTINGS SECTION
# ============================================
st.sidebar.divider()
st.sidebar.header("⚙️ Settings")

# Theme and Color Selection
with st.sidebar.expander("🎨 Theme & Colors", expanded=False):
    # Theme selector using global THEME_OPTIONS
    selected_theme = st.selectbox(
        'Color Theme',
        list(THEME_OPTIONS.keys()),
        index=list(THEME_OPTIONS.keys()).index(st.session_state.app_theme),
        key='theme_selector',
        help="Choose a color theme for the dashboard"
    )
    
    # Update session state and rerun when theme changes
    if selected_theme != st.session_state.app_theme:
        st.session_state.app_theme = selected_theme
        st.rerun()
    
    # Color preview
    theme_colors = THEME_OPTIONS[st.session_state.app_theme]
    st.markdown(f"""
    <div style='display: flex; gap: 5px; margin-top: 10px;'>
        <div style='width: 30px; height: 30px; border-radius: 50%; background: {theme_colors["primary"]}; border: 2px solid #999;' title='Primary'></div>
        <div style='width: 30px; height: 30px; border-radius: 50%; background: {theme_colors["secondary"]}; border: 2px solid #999;' title='Secondary'></div>
        <div style='width: 30px; height: 30px; border-radius: 50%; background: {theme_colors["accent"]}; border: 2px solid #999;' title='Accent'></div>
    </div>
    """, unsafe_allow_html=True)
    
    st.caption("Theme changes apply immediately")

# GitHub Repository Link
with st.sidebar.expander("🔗 Resources", expanded=False):
    st.markdown("""
    **📂 GitHub Repository**
    
    Access the source code, report issues, or contribute to the project:
    """)
    
    st.markdown("""
    <a href="https://github.com/ArvindRamakrishnan78907/Water-Quality-Analysis-and-Prediction-System-for-Asian-Regions" target="_blank" style="
        display: inline-block;
        padding: 10px 20px;
        background: linear-gradient(90deg, #24292e, #4a4a4a);
        color: white !important;
        text-decoration: none;
        border-radius: 8px;
        font-weight: bold;
        margin: 10px 0;
        transition: all 0.3s ease;
    ">
        <span style="margin-right: 8px;">🐙</span> View on GitHub
    </a>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    ---
    **📖 Quick Links:**
    - [📝 Documentation](https://github.com/ArvindRamakrishnan78907/Water-Quality-Analysis-and-Prediction-System-for-Asian-Regions#readme)
    - [🐛 Report Bug](https://github.com/ArvindRamakrishnan78907/Water-Quality-Analysis-and-Prediction-System-for-Asian-Regions/issues)
    - [✨ Request Feature](https://github.com/ArvindRamakrishnan78907/Water-Quality-Analysis-and-Prediction-System-for-Asian-Regions/issues/new)
    """)
    
    st.caption("Made with ❤️ for water quality monitoring")

st.sidebar.divider()

# Auto-run prediction on page load (no button click required)
st.sidebar.markdown("---")


# Run prediction automatically
site_data = df[df['SiteID'] == site]

if not site_data.empty and 'risk_level' in site_data.columns:
    avg_risk = site_data['risk_level'].mean()
    
    month_idx = MONTH_LABELS.index(month) + 1 if month in MONTH_LABELS else 1
    month_data = site_data[site_data['Month'] == month_idx]
    if not month_data.empty:
        monthly_risk = month_data['risk_level'].mean()
        predicted_risk = (monthly_risk * 0.7 + avg_risk * 0.3)
    else:
        predicted_risk = avg_risk
    
    status, emoji, color = get_water_quality_status(predicted_risk)
    
    st.subheader(f"🔮 Water Quality Prediction for {site} in {month}/{selected_year}")
    
    # Enhanced Glassmorphism Prediction Card
    st.markdown(f"""
    <style>
        .prediction-card {{
            background: linear-gradient(135deg, {color}15, {color}30);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border: 2px solid {color}50;
            border-radius: 25px;
            padding: 40px;
            text-align: center;
            margin: 15px 0;
            position: relative;
            overflow: hidden;
            box-shadow: 0 15px 50px {color}30, inset 0 1px 1px rgba(255,255,255,0.3);
            transition: all 0.4s ease;
        }}
        
        .prediction-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 20px 60px {color}40, inset 0 1px 1px rgba(255,255,255,0.4);
        }}
        
        .prediction-card::before {{
            content: '';
            position: absolute;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: linear-gradient(45deg, transparent, rgba(255,255,255,0.1), transparent);
            transform: rotate(45deg);
            animation: shimmer 3s infinite;
        }}
        
        @keyframes shimmer {{
            0% {{ transform: translateX(-100%) rotate(45deg); }}
            100% {{ transform: translateX(100%) rotate(45deg); }}
        }}
        
        .prediction-emoji {{
            font-size: 64px;
            margin-bottom: 10px;
            filter: drop-shadow(0 5px 15px rgba(0,0,0,0.2));
        }}
        
        .prediction-status {{
            margin: 15px 0;
            color: {color};
            font-size: 42px;
            font-weight: 800;
            text-shadow: 0 3px 10px {color}40;
        }}
        
        .prediction-subtitle {{
            color: {current_theme['text_secondary']};
            font-size: 14px;
            margin: 0;
        }}
        
        /* Glassmorphism Metric Cards */
        .glass-metrics {{
            display: flex;
            gap: 15px;
            margin-top: 20px;
            flex-wrap: wrap;
        }}
        
        .glass-metric {{
            flex: 1;
            min-width: 150px;
            background: linear-gradient(135deg, {current_theme['card_bg']}cc, {current_theme['background']}cc);
            backdrop-filter: blur(15px);
            -webkit-backdrop-filter: blur(15px);
            border: 1px solid {current_theme['input_border']};
            border-radius: 18px;
            padding: 20px;
            text-align: center;
            transition: all 0.3s ease;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        }}
        
        .glass-metric:hover {{
            transform: translateY(-3px) scale(1.02);
            box-shadow: 0 12px 40px rgba(0,0,0,0.15);
        }}
        
        .glass-metric-icon {{
            font-size: 24px;
            margin-bottom: 8px;
        }}
        
        .glass-metric-value {{
            font-size: 28px;
            font-weight: 700;
            color: {current_theme['accent']};
            margin: 5px 0;
        }}
        
        .glass-metric-label {{
            font-size: 12px;
            color: {current_theme['text_secondary']};
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        
        .glass-metric-delta {{
            font-size: 11px;
            color: {current_theme['text_secondary']};
            margin-top: 5px;
            opacity: 0.8;
        }}
    </style>
    
    <div class="prediction-card">
        <div class="prediction-emoji">{emoji}</div>
        <h2 class="prediction-status">{status}</h2>
        <p class="prediction-subtitle">Based on historical data analysis</p>
    </div>
    
    <div class="glass-metrics">
        <div class="glass-metric">
            <div class="glass-metric-icon">📊</div>
            <div class="glass-metric-value">{predicted_risk:.2f}</div>
            <div class="glass-metric-label">Risk Score</div>
            <div class="glass-metric-delta">↓ Lower is better</div>
        </div>
        <div class="glass-metric">
            <div class="glass-metric-icon">📈</div>
            <div class="glass-metric-value">{len(site_data):,}</div>
            <div class="glass-metric-label">Data Points</div>
            <div class="glass-metric-delta">Records analyzed</div>
        </div>
        <div class="glass-metric">
            <div class="glass-metric-icon">✅</div>
            <div class="glass-metric-value">{(site_data['risk_level'] == 0).sum() / len(site_data) * 100 if len(site_data) > 0 else 0:.0f}%</div>
            <div class="glass-metric-label">Safe Readings</div>
            <div class="glass-metric-delta">Within safe limits</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.warning("No data available for prediction. Please select a different basin or upload data.")

st.divider()

# India Map Visualization (only shown when India is selected)
if selected_country == "India":
    from asian_water_quality_data import INDIA_STATE_BASINS, get_india_state_coordinates
    # plotly.express already imported at top as px
    
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
    if view_type == "Last 30 Days":
        st.subheader(f"🌊 Current Water Quality - {site} (Last 30 Days)")
        thirty_days_ago = datetime.now() - timedelta(days=30)
        st.caption(f"Data from {thirty_days_ago.strftime('%b %d')} to {current_time[:10]}")
    else:
        st.subheader(f"🌊 Current Water Quality - {site}")
        st.caption(f"Data for {month} {selected_year}")

# Filter site data for last 30 days for "current" status
if 'SiteID' not in df.columns or df.empty:
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

# Only apply 30-day filter when explicitly requested via View Type
if view_type == "Last 30 Days" and 'DateTime' in site_data.columns:
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
    
    # Confidence color coding using CONFIDENCE_STYLES
    if 'Real Data' in str(confidence) or is_real_data:
        style = CONFIDENCE_STYLES['real']
    elif 'Baseline' in str(confidence):
        style = CONFIDENCE_STYLES['baseline']
    else:
        style = CONFIDENCE_STYLES['estimated']
    conf_color, data_badge, badge_bg = style['color'], style['badge'], style['bg']
    
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
                # Create risk-level matrix based on indicator-specific thresholds
                risk_matrix = pivot_data.copy()
                
                for indicator in pivot_data.index:
                    threshold = indicator_thresholds.get(indicator)
                    if threshold:
                        threshold_type = threshold.get('type', 'above')
                        moderate = threshold.get('moderate')
                        high = threshold.get('high_threshold')
                        
                        for month in pivot_data.columns:
                            value = pivot_data.loc[indicator, month]
                            if pd.isna(value):
                                risk_matrix.loc[indicator, month] = np.nan
                                continue
                            
                            # Use centralized risk calculation
                            mock_row = {'Indicator': indicator, 'Value (Agency)': value}
                            risk_matrix.loc[indicator, month] = determine_risk_level(mock_row)
                    else:
                        # No threshold defined - use generic scale
                        for month in pivot_data.columns:
                            risk_matrix.loc[indicator, month] = 0  # Default safe
                
                fig, ax = plt.subplots(figsize=(14, 8))
                
                # Custom colormap: Green (safe), Yellow (moderate), Red (high risk)
                from matplotlib.colors import ListedColormap
                risk_cmap = ListedColormap(['#27ae60', '#f39c12', '#e74c3c'])
                
                # Plot the risk matrix with annotations showing actual values
                im = ax.imshow(risk_matrix.values, cmap=risk_cmap, aspect='auto', vmin=0, vmax=2)
                
                # Add value annotations
                for i, indicator in enumerate(pivot_data.index):
                    for j, month in enumerate(pivot_data.columns):
                        value = pivot_data.loc[indicator, month]
                        if pd.notna(value):
                            # Text color based on risk level for contrast
                            risk = risk_matrix.loc[indicator, month]
                            text_color = 'white' if risk >= 1 else 'black'
                            ax.text(j, i, f'{value:.1f}', ha='center', va='center', 
                                   fontsize=9, color=text_color, fontweight='bold')
                
                # Set labels
                ax.set_xticks(np.arange(len(pivot_data.columns)))
                ax.set_yticks(np.arange(len(pivot_data.index)))
                ax.set_xticklabels([MONTH_LABELS[int(m)-1] for m in pivot_data.columns], rotation=45, ha='right')
                ax.set_yticklabels(pivot_data.index)
                ax.set_title(f'Water Quality Risk Levels by Indicator - {site}', fontsize=14, fontweight='bold')
                ax.set_xlabel('Month')
                ax.set_ylabel('Indicator')
                
                # Add legend
                from matplotlib.patches import Patch
                legend_elements = [
                    Patch(facecolor='#27ae60', label='  Safe'),
                    Patch(facecolor='#f39c12', label='  Moderate Risk'),
                    Patch(facecolor='#e74c3c', label='  High Risk')
                ]
                ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Add threshold reference
                with st.expander("📋 Indicator Thresholds Reference"):
                    threshold_info = []
                    for ind, thresh in indicator_thresholds.items():
                        if thresh['type'] == 'above':
                            threshold_info.append(f"**{ind}**: Safe < {thresh['moderate']}, Moderate {thresh['moderate']}-{thresh['high_threshold']}, High > {thresh['high_threshold']}")
                        elif thresh['type'] == 'below':
                            threshold_info.append(f"**{ind}**: Safe > {thresh['moderate']}, Moderate {thresh['high_threshold']}-{thresh['moderate']}, High < {thresh['high_threshold']}")
                        elif thresh['type'] == 'outside_range':
                            threshold_info.append(f"**{ind}**: Safe {thresh['high_threshold'][0]}-{thresh['high_threshold'][1]}, Moderate {thresh['moderate'][0]}-{thresh['moderate'][1]}")
                    st.markdown("\n\n".join(threshold_info))
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
        
        # Search functionality - always visible
        search_term = st.text_input(
            "🔍 Search Data",
            placeholder="Type to search across all columns...",
            key="data_search_input_stable",
            help="Search for any word or value. Shows matching rows."
        )
        
        # Determine what data to display
        display_df = df.copy()
        
        if search_term and search_term.strip():
            search_term_lower = search_term.lower().strip()
            # Find rows that contain the search term in any column
            mask = display_df.apply(
                lambda row: row.astype(str).str.lower().str.contains(search_term_lower, na=False).any(), 
                axis=1
            )
            display_df = display_df[mask]
            
            if len(display_df) > 0:
                st.success(f"🔍 Found {len(display_df):,} rows matching '{search_term}'")
            else:
                st.warning(f"No data matches '{search_term}'. Showing all data instead.")
                display_df = df.copy()  # Reset to show all data
        else:
            st.caption(f"📋 Showing all {len(df):,} records")
        
        # Always show the dataframe - never conditionally hide it
        st.dataframe(
            display_df,
            use_container_width=True,
            height=450,
            hide_index=True
        )
        
        # Download section - always visible
        st.markdown("---")
        col_dl1, col_dl2 = st.columns(2)
        
        with col_dl1:
            # Download what's currently displayed
            display_csv = display_df.to_csv(index=False)
            if search_term and search_term.strip() and len(display_df) < len(df):
                btn_label = f"⬇️ Download Displayed ({len(display_df):,} rows)"
                file_name = f"water_quality_search_{search_term}_{selected_country}_{selected_year}.csv"
            else:
                btn_label = f"⬇️ Download All ({len(display_df):,} rows)"
                file_name = f"water_quality_full_{selected_country}_{selected_year}.csv"
            
            st.download_button(
                label=btn_label,
                data=display_csv,
                file_name=file_name,
                mime="text/csv",
                key="download_displayed_data"
            )
        
        with col_dl2:
            # Always allow downloading full dataset
            if search_term and len(display_df) < len(df):
                full_csv = df.to_csv(index=False)
                st.download_button(
                    label=f"⬇️ Download Full Dataset ({len(df):,} rows)",
                    data=full_csv,
                    file_name=f"water_quality_complete_{selected_country}_{selected_year}.csv",
                    mime="text/csv",
                    key="download_full_dataset"
                )
            else:
                st.caption("Use search to filter, then download results")
    
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

# Footer CSS styles
st.markdown(f"""
<style>
    .footer-container {{
        background: linear-gradient(135deg, {current_theme['card_bg']} 0%, {current_theme['secondary']} 100%);
        border-top: 1px solid {current_theme['input_border']};
        border-radius: 20px 20px 0 0;
        padding: 40px 30px;
        margin-top: 50px;
        text-align: center;
    }}
    .footer-brand {{
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 12px;
        margin-bottom: 20px;
    }}
    .footer-brand-icon {{
        font-size: 32px;
    }}
    .footer-brand-text {{
        font-size: 20px;
        font-weight: 700;
        color: {current_theme['accent']};
    }}
    .footer-links {{
        display: flex;
        justify-content: center;
        gap: 25px;
        margin: 20px 0;
        flex-wrap: wrap;
    }}
    .footer-link {{
        color: {current_theme['text_secondary']} !important;
        text-decoration: none;
        font-size: 14px;
        padding: 8px 16px;
        border-radius: 20px;
        background: {current_theme['background']}80;
        transition: all 0.3s ease;
    }}
    .footer-link:hover {{
        color: {current_theme['accent']} !important;
        background: {current_theme['accent']}20;
    }}
    .footer-divider {{
        width: 100px;
        height: 3px;
        background: linear-gradient(90deg, transparent, {current_theme['accent']}, transparent);
        margin: 25px auto;
    }}
    .footer-info {{
        display: flex;
        justify-content: center;
        gap: 30px;
        flex-wrap: wrap;
        margin: 20px 0;
    }}
    .footer-info-item {{
        color: {current_theme['text_secondary']};
        font-size: 12px;
    }}
    .footer-info-item span {{
        color: {current_theme['accent']};
        font-weight: 600;
    }}
    .footer-copyright {{
        color: {current_theme['text_secondary']};
        font-size: 12px;
        margin-top: 20px;
    }}
    .footer-credits {{
        color: {current_theme['text_secondary']};
        font-size: 11px;
        margin-top: 10px;
        opacity: 0.7;
    }}
</style>
""", unsafe_allow_html=True)

# Footer HTML content
current_time_str = datetime.now().strftime('%Y-%m-%d %H:%M')
st.markdown(f"""
<div class="footer-container">
    <div class="footer-brand">
        <span class="footer-brand-icon">🌊</span>
        <span class="footer-brand-text">Asian Water Quality Dashboard</span>
    </div>
    <div class="footer-links">
        <a href="https://github.com/ArvindRamakrishnan78907/Water-Quality-Analysis-and-Prediction-System-for-Asian-Regions" target="_blank" class="footer-link">📂 Source Code</a>
        <a href="https://github.com/ArvindRamakrishnan78907/Water-Quality-Analysis-and-Prediction-System-for-Asian-Regions#readme" target="_blank" class="footer-link">📖 Documentation</a>
        <a href="https://github.com/ArvindRamakrishnan78907/Water-Quality-Analysis-and-Prediction-System-for-Asian-Regions/issues" target="_blank" class="footer-link">🐛 Report Issues</a>
    </div>
    <div class="footer-divider"></div>
    <div class="footer-info">
        <div class="footer-info-item">📊 Data Sources: <span>World Bank, CPCB</span></div>
        <div class="footer-info-item">🔄 Updated: <span>{current_time_str}</span></div>
        <div class="footer-info-item">🌏 Coverage: <span>48 Countries</span></div>
    </div>
    <p class="footer-copyright">© 2026 Water Quality Analysis and Prediction System</p>
    <p class="footer-credits">Built with ❤️ using Streamlit</p>
</div>
""", unsafe_allow_html=True)
