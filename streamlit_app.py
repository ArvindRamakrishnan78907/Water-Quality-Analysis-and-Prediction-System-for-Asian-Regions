import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from datetime import datetime
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
    # read csv file and format date for processing
    df = pd.read_csv('lawa-lake-monitoring-data-2004-2023_statetrendtli-results_sep2024.csv')
    
    # Convert SampleDateTime to datetime
    df['SampleDateTime'] = pd.to_datetime(df['SampleDateTime'])
    
    # Extract date and month for analysis
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
    # Handle potential None or empty condition
    indicator = row['Indicator']
    condition = indicator_thresholds.get(indicator)
    value = pd.to_numeric(row['Value (Agency)'], errors='coerce')
    # print(f"finding risk value for row, value is {value}")
    if not condition or value is None:
        return 0
    
    condition_type = condition.get("type", "above")
    moderate_threshold = condition.get("moderate")
    
    # Handle both 'high_threshold' and 'threshold' keys
    high_threshold = condition.get("high_threshold") or condition.get("threshold")
    
    # Default risk is safe
    risk_level = 0
    
    # Risk determination logic for different condition types
    if condition_type == "above":
        # Prioritize high threshold if present
        if high_threshold is not None and value > high_threshold:
            risk_level = 2  # High risk
            # st.write(f"High Risk: Value {value} > High Threshold {high_threshold}")
        elif moderate_threshold is not None and value > moderate_threshold:
            risk_level = 1  # Moderate risk
            # st.write(f"Moderate Risk: Value {value} > Moderate Threshold {moderate_threshold}")
    
    elif condition_type == "below":
        # Prioritize high threshold if present
        if high_threshold is not None and value < high_threshold:
            risk_level = 2  # High risk
            # st.write(f"High Risk: Value {value} < High Threshold {high_threshold}")
        elif moderate_threshold is not None and value < moderate_threshold:
            risk_level = 1  # Moderate risk
            # st.write(f"Moderate Risk: Value {value} < Moderate Threshold {moderate_threshold}")
    
    elif condition_type == "outside_range":
        # Ensure we have a valid range for moderate and high thresholds
        if (isinstance(moderate_threshold, list) and len(moderate_threshold) == 2 and 
            isinstance(high_threshold, list) and len(high_threshold) == 2):
            
            mod_lower, mod_upper = moderate_threshold
            high_lower, high_upper = high_threshold
            
            # Check for high risk first
            if value < high_lower or value > high_upper:
                risk_level = 2  # High risk
                # st.write(f"High Risk: Value {value} outside high threshold range [{high_lower}, {high_upper}]")
            
            # Then check for moderate risk
            elif value < mod_lower or value > mod_upper:
                risk_level = 1  # Moderate risk
                # st.write(f"Moderate Risk: Value {value} outside moderate threshold range [{mod_lower}, {mod_upper}]")
    
    # st.write(f"Determined Risk Level: {risk_level}")
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

    # Convert categorical features to appropriate format
    df['SiteID'] = df['SiteID'].astype('category')
    df['Month'] = df['Month'].astype('category')


    # Check the range of years in your dataset
    min_year = df['Year'].min()
    max_year = df['Year'].max()
    unique_years = sorted(df['Year'].unique())

    # st.write(f"Years in dataset: {unique_years}")
    # st.write((f"Range: {min_year} to {max_year} ({len(unique_years)} years)")

    # Calculate a good split point (e.g., use the last 20-25% of years for testing)
    # num_years = len(unique_years)
    # split_index = int(num_years * 0.8)  # Use 80% of years for training
    # split_year = unique_years[split_index]

    # print(f"Suggested split: Train on {unique_years[:split_index]} (years before {split_year})")
    # print(f"                 Test on {unique_years[split_index:]} (years {split_year} and after)")

    # # Create the train/test split
    # train_data = df[df['Year'] < split_year]
    # test_data = df[df['Year'] >= split_year]

    # print(f"Training data: {len(train_data)} samples from years {train_data['Year'].min()}-{train_data['Year'].max()}")
    # print(f"Testing data: {len(test_data)} samples from years {test_data['Year'].min()}-{test_data['Year'].max()}")

    # Alternative approach for imbalanced years: split by percentage of total data
    df_sorted = df.sort_values(['Year', 'Month'])
    train_size = 0.8
    train_data = df_sorted.iloc[:int(len(df_sorted) * train_size)]
    test_data = df_sorted.iloc[int(len(df_sorted) * train_size):]
    # st.write((f"Alternative - Training data: years {train_data['Year'].min()}-{train_data['Year'].max()}")
    # st.write((f"Alternative - Testing data: years {test_data['Year'].min()}-{test_data['Year'].max()}")

    # Create feature and target variables for training
    X_train = train_data.drop('risk_level', axis=1)  # Remove target column
    y_train = train_data['risk_level']               # Just the target column

    # Create feature and target variables for testing
    X_test = test_data.drop('risk_level', axis=1)
    y_test = test_data['risk_level']

    # Specify categorical features for CatBoost
    categorical_features = ['SiteID', 'Month']  # Add any other categorical columns

    # Now train the model
    model = CatBoostClassifier(
        iterations=500,
        learning_rate=0.1,
        cat_features=categorical_features,
        random_seed=42
    )

    # Fit the model
    model.fit(
        X_train, y_train,
        eval_set=(X_test, y_test),
        verbose=100
    )

    # Evaluate performance
    from sklearn.metrics import classification_report, confusion_matrix

    y_pred = model.predict(X_test)
    # st.write(classification_report(y_test, y_pred))
    # st.write(confusion_matrix(y_test, y_pred))

    return model


def aggregate_risk_by_date(df):

    # First let's make sure we have a single date column to work with
    if 'Date' not in df.columns:
        # Create Date from Year, Month, Day if needed
        df['Date'] = pd.to_datetime(df[['Year', 'Month', 'Day']])

    # First ensure 'Date' is in datetime format
    df['Date'] = pd.to_datetime(df['Date'])

    # Extract Year, Month, and Day components
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day

    # Now aggregate to get the maximum risk level for each site and date
    aggregated_df = df.groupby(['SiteID', 'Date']).agg({
        'risk_level': 'max',  # Take highest risk from any indicator
        'Value': ['mean', 'min', 'max'],  # Aggregate value metrics
        # Include other columns you want to keep
        'Year': 'first',
        'Month': 'first',
        'Day': 'first'
    }).reset_index()

    # Flatten multi-level column names
    aggregated_df.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col for col in aggregated_df.columns]
    # st.write('aggregate df columns are ', aggregated_df.columns)
    # st.write(f"Original data: {len(df)} rows")
    # st.write(f"Aggregated data: {len(aggregated_df)} rows")
    # st.write(f"Unique site-date combinations: {df.groupby(['SiteID', 'Date']).ngroups}")

    # Verify our aggregation worked as expected
    # st.write("Risk level distribution before aggregation:")
    # st.write(df['risk_level'].value_counts())
    # st.write("aggregated df columns:", aggregated_df.columns.tolist())
    aggregated_df = aggregated_df.rename(columns={'risk_level_max': 'risk_level'})
    aggregated_df = aggregated_df.rename(columns={'Day_first': 'Day'})
    aggregated_df = aggregated_df.rename(columns={'Month_first': 'Month'})
    aggregated_df = aggregated_df.rename(columns={'Year_first': 'Year'})
    # st.write("Risk level distribution after taking max per site-date:")

    # st.write(aggregated_df['risk_level'].value_counts())
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
    # Filter for the specific site and month (from previous years)
    historical_data = df[(df['SiteID'] == site_id) & 
                         (df['Month'] == target_month) & 
                         (df['Year'] < current_year)]
    
    if len(historical_data) == 0:
        st.write(f"No historical data found for SiteID={site_id}, Month={target_month}")
        return None
    
    # Get the most recent year's data for this site and month
    most_recent_year = historical_data['Year'].max()
    most_recent_data = historical_data[historical_data['Year'] == most_recent_year]
    
    # Create input data for the model using the same structure as training data
    # But with Year updated to current year
    input_data = most_recent_data.copy()
    input_data['Year'] = current_year
    input_data['Month'] = target_month
    input_data['Date'] = pd.to_datetime([f"{current_year}-{target_month}-15"])[0]
    # Select only the columns used by the model
    model_features = [col for col in input_data.columns if col != 'risk_level']
    # st.write('model features:  ', input_data[model_features])
    
    # Make prediction
    predicted_risk = model.predict(input_data[model_features])[0]
    
    return predicted_risk

def aggregate_for_calendar_year(site_id, df):
    # Assuming your dataframe is called 'df' with columns 'siteID', 'date', and 'risk_level'
    site_data = df[df['SiteID'] == site_id].copy()
    # First, make sure your date column is datetime
    site_data['Date'] = pd.to_datetime(df['Date'])

    # Group by siteID, month, and day, then calculate mean risk level
    calendar_df = site_data.groupby(['SiteID', 'Month', 'Day'])['risk_level'].mean().reset_index()

    # Round to integers if your risk levels should be whole numbers
    # If you want to keep decimals, remove this line
    calendar_df['risk_level'] = calendar_df['risk_level'].round().astype(int)

    # Create a date column for visualization (using a reference year, e.g., 2025)
    # st.dataframe(calendar_df)
    # calendar_df['Date'] = pd.to_datetime({
    #     'year': 2024,
    #     'month': calendar_df['Month'],
    #     'day': calendar_df['Day']
    # })
    
    # Use apply to create dates properly
    calendar_df['Date'] = calendar_df.apply(
        lambda row: pd.Timestamp(year=2025, month=row['Month'], day=row['Day']), axis=1
    ) 
    return calendar_df

def heatmap_risk_by_date(site_id, site_data):
    # Create heatmap of risk by day and month
    
    try:
        # Convert day-month to a date string for better sorting
        # site_data['MonthDay'] = site_data['DateObj'].dt.strftime('%m-%d')
        
        # Count occurrences of each risk level by day of year
        daily_data = site_data.copy()
        # st.dataframe(daily_data)
    

        if daily_data['risk_level'].isnull().any():
            st.write("Warning: There are NaN values in 'risk_level' column")
        # Create matrix for heatmap (months x days)
        # Initialize with NaN
        calendar_data = np.full((12, 31), np.nan)
        
        # Fill in with average risk values
        for m in range(1, 13):
            month_data = daily_data[daily_data['Month'] == m]
            for d in range(1, 32):
                day_data = month_data[month_data['Day'] == d]
                if len(day_data) > 0:
                    calendar_data[m-1, d-1] = day_data['risk_level'].mean()
        
        # Create calendar heatmap
        fig = plt.figure(figsize=(14, 8))
        
        # Custom colormap: green for safe, yellow for moderate, red for high
        cmap = plt.cm.colors.ListedColormap(['green', 'yellow', 'red'])
        bounds = [0, 0.67, 1.33, 2]
        norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)
        
        # Create heatmap
        heatmap = plt.pcolormesh(calendar_data.T, cmap=cmap, norm=norm)
        # Set labels
        plt.yticks(np.arange(0.5, 31.5), np.arange(1, 32))
        plt.xticks(np.arange(0.5, 12.5), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                                         'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        
        plt.title(f'Historical Calendar View of Risk Levels for {site_id}', fontsize=18)
        plt.ylabel('Day of Month')
        
        # Add colorbar
        cbar = plt.colorbar(heatmap)
        cbar.set_ticks([0.33, 1, 1.67])
        cbar.set_ticklabels(['Safe', 'Moderate Risk', 'High Risk'])
        
        plt.tight_layout()
        st.pyplot(fig)  # Return the monthly figure for display
    except Exception as e:
        st.write(f"Error creating calendar visualization: {e}")
    
 

# Import the new data module
from asian_water_quality_data import (
    ASIAN_COUNTRIES, 
    get_basins_for_country, 
    fetch_country_metadata,
    fetch_water_quality_data,
    normalize_uploaded_data
)

# ==================== WATER QUALITY STATUS FUNCTIONS ====================

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
    """Calculate overall water quality status for a basin."""
    site_data = data[data['SiteID'] == site_id]
    if site_data.empty or 'risk_level' not in site_data.columns:
        return "Unknown", "⚪", "#95a5a6", {}
    
    avg_risk = site_data['risk_level'].mean()
    status, emoji, color = get_water_quality_status(avg_risk)
    
    # Calculate indicator-specific stats
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
    
    # Main status display
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
    
    # Indicator breakdown
    if stats:
        cols = st.columns(min(4, len(stats)))
        for i, (indicator, info) in enumerate(list(stats.items())[:4]):
            with cols[i % 4]:
                st.metric(
                    label=f"{info['emoji']} {indicator[:12]}...",
                    value=f"{info['value']:.2f}",
                    delta=info['status'],
                    delta_color="off"
                )

# ==================== VISUALIZATION FUNCTIONS ====================

def create_bar_chart(data, site_id, indicator='Chl-a'):
    """Create a bar chart for the selected site and indicator with quality status."""
    site_data = data[data['SiteID'] == site_id]
    if site_data.empty:
        st.warning(f"No data available for {site_id}")
        return
    
    # Display quality summary first
    display_quality_summary(data, site_id)
    
    # Filter by indicator if available
    if 'Indicator' in site_data.columns:
        indicator_data = site_data[site_data['Indicator'] == indicator]
        if indicator_data.empty:
            indicator_data = site_data
    else:
        indicator_data = site_data
    
    # Group by month for bar chart
    if 'Month' in indicator_data.columns and 'Value' in indicator_data.columns:
        monthly_avg = indicator_data.groupby('Month')['Value'].mean().reset_index()
        
        # Get risk levels for coloring
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
        
        # Add legend for colors
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
    """Create a line chart showing trends over time with quality status."""
    site_data = data[data['SiteID'] == site_id]
    if site_data.empty:
        st.warning(f"No data available for {site_id}")
        return
    
    # Display quality summary first
    display_quality_summary(data, site_id)
    
    # Filter by indicator if available
    if 'Indicator' in site_data.columns:
        indicator_data = site_data[site_data['Indicator'] == indicator].copy()
        if indicator_data.empty:
            indicator_data = site_data.copy()
    else:
        indicator_data = site_data.copy()
    
    if 'SampleDateTime' in indicator_data.columns and 'Value' in indicator_data.columns:
        indicator_data = indicator_data.sort_values('SampleDateTime')
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(indicator_data['SampleDateTime'], indicator_data['Value'], 
                color='#3498db', linewidth=2, marker='o', markersize=4, alpha=0.7)
        
        # Add trend line
        if len(indicator_data) > 2:
            z = np.polyfit(range(len(indicator_data)), indicator_data['Value'], 1)
            p = np.poly1d(z)
            ax.plot(indicator_data['SampleDateTime'], p(range(len(indicator_data))), 
                    '--', color='#e74c3c', alpha=0.8, label='Trend')
        
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel(f'{indicator} Value', fontsize=12)
        ax.set_title(f'{indicator} Trend Over Time for {site_id}', fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3)
        ax.legend()
        
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
    
    # Display quality summary first
    display_quality_summary(data, site_id)
    
    # Add Status column based on risk level
    if 'risk_level' in site_data.columns:
        site_data['Status'] = site_data['risk_level'].apply(
            lambda r: '🟢 Excellent' if r <= 0.5 else '🟡 Good' if r <= 1.0 else '🟠 Moderate' if r <= 1.5 else '🔴 Poor'
        )
    
    # Select relevant columns
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


# ==================== MAIN APPLICATION ====================

# Page configuration for auto-refresh
st.set_page_config(
    page_title="Asian Water Quality Dashboard",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🌊 Asian Water Quality Dashboard - LIVE")

# Initialize session state
if 'data_source' not in st.session_state:
    st.session_state.data_source = 'api'
if 'selected_country' not in st.session_state:
    st.session_state.selected_country = 'India'
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = datetime.now()

# ==================== AUTO-REFRESH CONFIGURATION ====================

st.sidebar.header("⚡ Live Updates")

auto_refresh = st.sidebar.toggle("Enable Auto-Refresh", value=True, help="Automatically refresh data")
refresh_interval = st.sidebar.slider(
    "Refresh Interval (seconds)", 
    min_value=10, 
    max_value=60, 
    value=30,
    disabled=not auto_refresh
)

# Manual refresh button
if st.sidebar.button("🔄 Refresh Now"):
    st.cache_data.clear()
    st.session_state.last_refresh = datetime.now()
    st.rerun()

# Auto-refresh logic
if auto_refresh:
    import time
    time_since_refresh = (datetime.now() - st.session_state.last_refresh).total_seconds()
    if time_since_refresh >= refresh_interval:
        st.cache_data.clear()
        st.session_state.last_refresh = datetime.now()
        st.rerun()
    
    # Show countdown
    remaining = int(refresh_interval - time_since_refresh)
    st.sidebar.caption(f"⏱️ Next refresh in: {remaining}s")

st.sidebar.divider()

# ==================== SIDEBAR CONTROLS ====================

st.sidebar.header("🌏 Region Selection")

# Country selector with India as default
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

# Fetch and display country metadata
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
    st.sidebar.caption(f"📊 Data: Simulated (WHO/EPA standards)")

st.sidebar.divider()

# File Upload Section
st.sidebar.header("📁 Data Upload")
uploaded_file = st.sidebar.file_uploader(
    "Upload CSV/Excel (optional)",
    type=['csv', 'xlsx', 'xls'],
    help="Upload your own data to override API data"
)

st.sidebar.divider()

# ==================== DATA LOADING LOGIC ====================

@st.cache_data(ttl=1800)
def load_api_data(country):
    """Load data from API for the selected country."""
    return fetch_water_quality_data(country)

# Determine data source and load data
if uploaded_file is not None:
    # Prioritize uploaded data
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
        df = load_api_data(selected_country)
        st.session_state.data_source = 'api'
else:
    # Use API data
    st.session_state.data_source = 'api'
    df = load_api_data(selected_country)
    st.sidebar.info(f"📡 Live data for {selected_country}")

# Apply risk level calculation
if 'risk_level' not in df.columns:
    df['risk_level'] = df.apply(determine_risk_level, axis=1)

# ==================== BASIN/SITE SELECTION ====================

st.sidebar.header("🏞️ Basin Selection")

# Get sites from current data
if st.session_state.data_source == 'upload':
    available_sites = sorted(df['SiteID'].unique().tolist())
else:
    available_sites = get_basins_for_country(selected_country)

# Filter out numeric-only site IDs
filtered_sites = [s for s in available_sites if not (isinstance(s, str) and s.isdigit())]
if not filtered_sites:
    filtered_sites = available_sites

site = st.sidebar.selectbox('Basin/Site', sorted(filtered_sites))

# Month selection
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
month = st.sidebar.selectbox('Month', months)
current_year = datetime.now().year

st.sidebar.divider()

# ==================== VISUALIZATION CONTROLS ====================

st.sidebar.header("📊 Visualization")

chart_type = st.sidebar.radio(
    "Chart Type",
    ["Heatmap", "Bar Chart", "Line Chart", "Data Table"],
    horizontal=False
)

# Indicator selection for charts
available_indicators = list(indicator_thresholds.keys())
if 'Indicator' in df.columns:
    data_indicators = df['Indicator'].unique().tolist()
    available_indicators = [i for i in available_indicators if i in data_indicators] or data_indicators[:5]

selected_indicator = st.sidebar.selectbox('Indicator', available_indicators)

# ==================== MAIN PANEL ====================

# Risk color mapping
risk_color_map = {
    "safe": "green",
    "moderate": "yellow", 
    "high": "red",
    "unknown": "gray"
}

# Prediction section - Using simplified heuristic approach
if st.sidebar.button("🔮 Run Prediction"):
    with st.spinner("Analyzing water quality..."):
        # Use direct data analysis instead of ML model
        site_data = df[df['SiteID'] == site]
        
        if not site_data.empty and 'risk_level' in site_data.columns:
            # Calculate average risk for this site
            avg_risk = site_data['risk_level'].mean()
            
            # Get month-specific data if available
            month_idx = months.index(month) + 1
            month_data = site_data[site_data['Month'] == month_idx]
            if not month_data.empty:
                monthly_risk = month_data['risk_level'].mean()
                # Weighted average of monthly and overall
                predicted_risk = (monthly_risk * 0.7 + avg_risk * 0.3)
            else:
                predicted_risk = avg_risk
            
            # Get quality status
            status, emoji, color = get_water_quality_status(predicted_risk)
            
            # Display prediction result
            st.subheader(f"Water Quality Prediction for {site} in {month}/{current_year}")
            
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
            
            # Show additional context
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

current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
st.subheader(f"🌊 Current Water Quality - {site}")
st.caption(f"As of {current_time}")

site_data = df[df['SiteID'] == site]
if not site_data.empty and 'risk_level' in site_data.columns:
    current_risk = site_data['risk_level'].mean()
    status, emoji, color = get_water_quality_status(current_risk)
    
    st.markdown(f"""
    <div style='background: linear-gradient(135deg, {color}22, {color}44); 
                border-radius: 15px; 
                padding: 25px; 
                text-align: center;
                margin: 10px 0;'>
        <h1 style='margin:0; font-size: 60px;'>{emoji}</h1>
        <h2 style='margin:10px 0; color: {color}; font-size: 32px; font-weight: bold;'>{status}</h2>
        <p style='margin:0; color: #666;'>Current Water Quality Status</p>
    </div>
    """, unsafe_allow_html=True)
    
    if 'Indicator' in site_data.columns:
        st.markdown("**Latest Readings:**")
        cols = st.columns(4)
        for i, indicator in enumerate(site_data['Indicator'].unique()[:8]):
            ind_data = site_data[site_data['Indicator'] == indicator]
            if not ind_data.empty and 'Value' in ind_data.columns:
                latest_val = ind_data['Value'].iloc[-1]
                ind_risk = ind_data['risk_level'].mean() if 'risk_level' in ind_data.columns else 0
                ind_status, ind_emoji, _ = get_water_quality_status(ind_risk)
                with cols[i % 4]:
                    st.metric(f"{ind_emoji} {indicator[:15]}", f"{latest_val:.2f}", ind_status)

st.divider()

st.subheader(f"📈 {chart_type} View - {site}")

if chart_type == "Heatmap":
    # Display quality summary
    display_quality_summary(df, site)
    
    # Create a simpler, more reliable heatmap
    site_data = df[df['SiteID'] == site]
    if not site_data.empty and 'Indicator' in site_data.columns and 'Month' in site_data.columns:
        # Create pivot table: Indicators vs Months
        if 'Value' in site_data.columns:
            pivot_data = site_data.pivot_table(
                values='Value', 
                index='Indicator', 
                columns='Month', 
                aggfunc='mean'
            )
            
            if not pivot_data.empty:
                fig, ax = plt.subplots(figsize=(14, 8))
                
                # Use seaborn heatmap for better visualization
                import seaborn as sns
                sns.heatmap(
                    pivot_data, 
                    annot=True, 
                    fmt='.1f', 
                    cmap='RdYlGn_r',  # Red=high, Green=low values
                    ax=ax,
                    cbar_kws={'label': 'Average Value'}
                )
                
                # Set month labels
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

# Footer with data source info
st.divider()
col1, col2, col3 = st.columns(3)
with col1:
    st.caption(f"📍 Country: {selected_country}")
with col2:
    st.caption(f"📊 Data Source: {'Uploaded File' if st.session_state.data_source == 'upload' else 'Live API'}")
with col3:
    st.caption(f"📝 Records: {len(df):,}")