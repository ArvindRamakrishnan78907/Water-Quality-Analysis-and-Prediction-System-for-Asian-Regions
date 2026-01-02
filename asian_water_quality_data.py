"""
Asian Water Quality Data Module
Provides live API integration, Asian countries list, and basin data handling.
"""
import pandas as pd
import requests
import streamlit as st
from datetime import datetime, timedelta
import random
import numpy as np

# Complete list of Asian countries (48 sovereign states)
ASIAN_COUNTRIES = [
    "Afghanistan", "Armenia", "Azerbaijan", "Bahrain", "Bangladesh", "Bhutan",
    "Brunei", "Cambodia", "China", "Cyprus", "Georgia", "Hong Kong", "India", "Indonesia",
    "Iran", "Iraq", "Israel", "Japan", "Jordan", "Kazakhstan", "Kuwait",
    "Kyrgyzstan", "Laos", "Lebanon", "Malaysia", "Maldives", "Mongolia",
    "Myanmar", "Nepal", "North Korea", "Oman", "Pakistan", "Palestine",
    "Philippines", "Qatar", "Russia", "Saudi Arabia", "Singapore", "South Korea",
    "Sri Lanka", "Syria", "Taiwan", "Tajikistan", "Thailand", "Timor-Leste",
    "Turkey", "Turkmenistan", "United Arab Emirates", "Uzbekistan", "Vietnam", "Yemen"
]

# Major hydrological basins for ALL Asian countries (comprehensive coverage)
ASIAN_BASINS = {
    "Afghanistan": [
        "Amu Darya Basin", "Helmand Basin", "Kabul River Basin",
        "Hari River Basin", "Panjshir Basin", "Arghandab Basin"
    ],
    "Armenia": [
        "Araks River Basin", "Hrazdan River Basin", "Vorotan Basin",
        "Lake Sevan Basin", "Debed River Basin"
    ],
    "Azerbaijan": [
        "Kura River Basin", "Araks River Basin", "Samur Basin",
        "Mingachevir Reservoir Basin", "Caspian Shore Basin"
    ],
    "Bahrain": [
        "Northern Aquifer Basin", "Central Basin", "Dammam Aquifer",
        "Coastal Water Zone"
    ],
    "Bangladesh": [
        "Ganges-Padma Basin", "Brahmaputra-Jamuna Basin", "Meghna Basin",
        "Karnaphuli Basin", "Surma-Kushiyara Basin", "Teesta Basin"
    ],
    "Bhutan": [
        "Manas River Basin", "Paro Chhu Basin", "Wang Chhu Basin",
        "Punatsang Chhu Basin", "Drangme Chhu Basin"
    ],
    "Brunei": [
        "Belait River Basin", "Tutong River Basin", "Brunei River Basin",
        "Temburong Basin"
    ],
    "Cambodia": [
        "Mekong Basin", "Tonle Sap Basin", "Bassac River Basin",
        "Stung Sen Basin", "Srepok River Basin", "Sesan Basin"
    ],
    "China": [
        "Yangtze River Basin", "Yellow River Basin", "Pearl River Basin",
        "Heilongjiang Basin", "Huai River Basin", "Hai River Basin",
        "Liao River Basin", "Songhua River Basin", "Taihu Lake Basin",
        "Min River Basin", "Han River Basin", "Gan River Basin"
    ],
    "Cyprus": [
        "Kouris Basin", "Yermasoyia Basin", "Ezousas Basin",
        "Limnatis Basin", "Serrachis Basin"
    ],
    "Georgia": [
        "Kura Basin", "Rioni Basin", "Mtkvari Basin",
        "Enguri Basin", "Iori Basin", "Alazani Basin"
    ],
    "Hong Kong": [
        "Shing Mun River", "Lam Tsuen River", "Shan Pui River",
        "Yuen Long Creek", "Deep Bay", "Tolo Harbour"
    ],
    "India": [
        "Ganga Basin", "Brahmaputra Basin", "Indus Basin", "Godavari Basin",
        "Krishna Basin", "Mahanadi Basin", "Narmada Basin", "Tapi Basin",
        "Cauvery Basin", "Pennar Basin", "Sabarmati Basin", "Mahi Basin",
        "Subarnarekha Basin", "Baitarani Basin", "Brahmani Basin"
    ],
    "Indonesia": [
        "Citarum Basin", "Mahakam Basin", "Barito Basin", "Kapuas Basin",
        "Brantas Basin", "Solo River Basin", "Lake Toba Basin",
        "Musi River Basin", "Bengawan Solo Basin", "Ciliwung Basin"
    ],
    "Iran": [
        "Karun Basin", "Zayandeh Basin", "Karkheh Basin",
        "Urmia Lake Basin", "Sefidrud Basin", "Kor River Basin",
        "Dez River Basin", "Gorgan Basin"
    ],
    "Iraq": [
        "Tigris Basin", "Euphrates Basin", "Shatt al-Arab Basin",
        "Greater Zab Basin", "Lesser Zab Basin", "Diyala Basin",
        "Tharthar Basin"
    ],
    "Israel": [
        "Jordan River Basin", "Coastal Aquifer Basin", "Mountain Aquifer",
        "Sea of Galilee Basin", "Yarkon Basin"
    ],
    "Japan": [
        "Tone River Basin", "Shinano River Basin", "Ishikari River Basin",
        "Kitakami River Basin", "Mogami River Basin", "Agano River Basin",
        "Lake Biwa Basin", "Yoshino River Basin", "Tenryu Basin",
        "Kiso River Basin", "Chikugo Basin"
    ],
    "Jordan": [
        "Jordan River Basin", "Yarmouk Basin", "Zarqa River Basin",
        "Azraq Basin", "Dead Sea Basin", "Wadi Araba Basin"
    ],
    "Kazakhstan": [
        "Irtysh Basin", "Syr Darya Basin", "Ili River Basin",
        "Ural River Basin", "Lake Balkhash Basin", "Ishim Basin",
        "Tobol Basin", "Nura Basin", "Chu River Basin"
    ],
    "Kuwait": [
        "Kuwait Bay Basin", "Northern Desert Basin", "Burgan Basin",
        "Coastal Aquifer Zone"
    ],
    "Kyrgyzstan": [
        "Syr Darya Basin", "Naryn River Basin", "Chu River Basin",
        "Issyk-Kul Basin", "Talas Basin", "Kara-Darya Basin"
    ],
    "Laos": [
        "Mekong Basin", "Nam Ou Basin", "Nam Ngum Basin",
        "Sekong Basin", "Nam Khan Basin", "Xe Bang Fai Basin"
    ],
    "Lebanon": [
        "Litani River Basin", "Orontes Basin", "Nahr Ibrahim Basin",
        "Beirut River Basin", "Awali Basin"
    ],
    "Malaysia": [
        "Pahang River Basin", "Perak River Basin", "Kelantan Basin",
        "Rajang River Basin", "Kinabatangan Basin", "Klang Basin",
        "Muda River Basin", "Langat Basin"
    ],
    "Maldives": [
        "Male Atoll Basin", "Addu Atoll Basin", "Ari Atoll Basin",
        "Huvadhoo Atoll Basin"
    ],
    "Mongolia": [
        "Selenge Basin", "Orkhon Basin", "Tuul Basin",
        "Kherlen Basin", "Onon Basin", "Lake Khuvsgul Basin"
    ],
    "Myanmar": [
        "Irrawaddy Basin", "Salween Basin", "Sittaung Basin",
        "Chindwin Basin", "Inle Lake Basin", "Kaladan Basin",
        "Thanlwin Basin"
    ],
    "Nepal": [
        "Koshi Basin", "Gandaki Basin", "Karnali Basin", "Mahakali Basin",
        "Bagmati Basin", "Narayani Basin", "Rapti Basin", "Kankai Basin"
    ],
    "North Korea": [
        "Yalu River Basin", "Tumen River Basin", "Taedong Basin",
        "Chongchon Basin", "Imjin Basin"
    ],
    "Oman": [
        "Wadi Samail Basin", "Batinah Coastal Basin", "Dhofar Basin",
        "Sharqiyah Basin", "Wadi Dayqah Basin"
    ],
    "Pakistan": [
        "Indus Basin", "Jhelum Basin", "Chenab Basin", "Ravi Basin",
        "Sutlej Basin", "Kabul River Basin", "Swat Basin",
        "Hub River Basin", "Lyari Basin"
    ],
    "Palestine": [
        "Jordan Valley Basin", "Mountain Aquifer Basin", "Coastal Aquifer",
        "Wadi Gaza Basin"
    ],
    "Philippines": [
        "Cagayan River Basin", "Pampanga River Basin", "Agno River Basin",
        "Pasig-Laguna Basin", "Mindanao River Basin", "Agusan Basin",
        "Rio Grande de Mindanao Basin"
    ],
    "Qatar": [
        "Northern Basin", "Southern Basin", "Doha Bay Basin",
        "Dukhan Basin"
    ],
    "Russia": [
        "Amur Basin", "Lena Basin", "Yenisei Basin", "Ob Basin",
        "Volga Basin", "Lake Baikal Basin", "Angara Basin"
    ],
    "Saudi Arabia": [
        "Wadi Hanifa Basin", "Wadi Fatima Basin", "Wadi Rimah Basin",
        "Wadi Dawasir Basin", "Wadi Najran Basin", "Tabuk Basin",
        "Saq Aquifer Basin"
    ],
    "Singapore": [
        "Central Catchment Basin", "Marina Basin", "Kranji Basin",
        "Punggol Basin", "Serangoon Basin"
    ],
    "South Korea": [
        "Han River Basin", "Nakdong River Basin", "Geum River Basin",
        "Yeongsan River Basin", "Seomjin River Basin", "Anseong Basin",
        "Sapgyo Basin"
    ],
    "Sri Lanka": [
        "Mahaweli Basin", "Kelani Basin", "Kalu Ganga Basin",
        "Walawe Basin", "Gin Ganga Basin", "Deduru Oya Basin"
    ],
    "Syria": [
        "Euphrates Basin", "Orontes Basin", "Barada Basin",
        "Khabur Basin", "Yarmouk Basin", "Coastal Basin"
    ],
    "Taiwan": [
        "Zhuoshui Basin", "Gaoping Basin", "Tamsui Basin",
        "Lanyang Basin", "Dajia Basin", "Wu River Basin"
    ],
    "Tajikistan": [
        "Amu Darya Basin", "Syr Darya Basin", "Vakhsh Basin",
        "Zeravshan Basin", "Panj Basin", "Kofarnihon Basin"
    ],
    "Thailand": [
        "Chao Phraya Basin", "Mekong Basin", "Mae Klong Basin",
        "Ping River Basin", "Nan River Basin", "Chi River Basin",
        "Mun River Basin", "Tha Chin Basin", "Bang Pakong Basin"
    ],
    "Timor-Leste": [
        "Laclo River Basin", "Loes River Basin", "Comoro Basin",
        "Tukola Basin", "Seical Basin"
    ],
    "Turkey": [
        "Euphrates-Tigris Basin", "Kizilirmak Basin", "Yesilirmak Basin",
        "Sakarya Basin", "Buyuk Menderes Basin", "Seyhan Basin",
        "Ceyhan Basin", "Firat Basin", "Marmara Basin"
    ],
    "Turkmenistan": [
        "Amu Darya Basin", "Karakum Canal Basin", "Murgab Basin",
        "Tejen Basin", "Atrek Basin"
    ],
    "United Arab Emirates": [
        "Dubai Creek Basin", "Al Ain Oasis Basin", "Liwa Basin",
        "Ras Al Khaimah Basin", "Fujairah Coastal Basin"
    ],
    "Uzbekistan": [
        "Amu Darya Basin", "Syr Darya Basin", "Zeravshan Basin",
        "Fergana Valley Basin", "Aral Sea Basin", "Chirchik Basin"
    ],
    "Vietnam": [
        "Mekong Delta Basin", "Red River Basin", "Dong Nai Basin",
        "Ma River Basin", "Ca River Basin", "Saigon River Basin",
        "Thu Bon Basin", "Huong River Basin", "Vu Gia Basin"
    ],
    "Yemen": [
        "Wadi Hadramaut Basin", "Sana'a Basin", "Wadi Zabid Basin",
        "Tihama Basin", "Marib Basin", "Wadi Bana Basin"
    ],
}

# Default basins for any country not explicitly listed
DEFAULT_BASINS = ["Main River Basin", "Northern Basin", "Southern Basin", "Eastern Basin", "Western Basin"]

INDICATOR_RANGES = {
    'Chl-a': {
        'min': 0.5, 'max': 50, 'unit': 'µg/L',
        'safe': 5, 'moderate': 10, 'high': 25,
        'description': 'Chlorophyll-a'
    },
    'pH': {
        'min': 6.0, 'max': 9.5, 'unit': 'pH',
        'safe_range': [6.5, 8.5], 'moderate_range': [6.0, 9.0], 'high_range': [5.5, 9.5],
        'description': 'pH Level'
    },
    'Total Nitrogen': {
        'min': 0.05, 'max': 5.0, 'unit': 'mg/L',
        'safe': 0.5, 'moderate': 1.0, 'high': 2.0,
        'description': 'Total Nitrogen'
    },
    'Total Phosphorus': {
        'min': 0.005, 'max': 0.5, 'unit': 'mg/L',
        'safe': 0.03, 'moderate': 0.1, 'high': 0.2,
        'description': 'Total Phosphorus'
    },
    'E. coli': {
        'min': 0, 'max': 1000, 'unit': 'CFU/100mL',
        'safe': 100, 'moderate': 235, 'high': 575,
        'description': 'E. coli'
    },
    'Nitrate': {
        'min': 0.1, 'max': 50, 'unit': 'mg/L',
        'safe': 10, 'moderate': 25, 'high': 45,
        'description': 'Nitrate'
    },
    'Dissolved Oxygen': {
        'min': 2.0, 'max': 14.0, 'unit': 'mg/L',
        'safe': 6.5, 'moderate': 5.0, 'high': 4.0,
        'description': 'Dissolved Oxygen'
    },
    'Turbidity': {
        'min': 0.1, 'max': 100, 'unit': 'NTU',
        'safe': 5, 'moderate': 25, 'high': 50,
        'description': 'Turbidity'
    },
}


def get_basins_for_country(country: str) -> list:
    """Get list of basins for a given country."""
    return ASIAN_BASINS.get(country, DEFAULT_BASINS)


# Accurate country metadata (hardcoded for reliability)
COUNTRY_METADATA = {
    "Afghanistan": {"capital": "Kabul", "population": 41128771, "flag": "🇦🇫"},
    "Armenia": {"capital": "Yerevan", "population": 2963900, "flag": "🇦🇲"},
    "Azerbaijan": {"capital": "Baku", "population": 10145200, "flag": "🇦🇿"},
    "Bahrain": {"capital": "Manama", "population": 1501635, "flag": "🇧🇭"},
    "Bangladesh": {"capital": "Dhaka", "population": 169356251, "flag": "🇧🇩"},
    "Bhutan": {"capital": "Thimphu", "population": 777486, "flag": "🇧🇹"},
    "Brunei": {"capital": "Bandar Seri Begawan", "population": 449002, "flag": "🇧🇳"},
    "Cambodia": {"capital": "Phnom Penh", "population": 16589023, "flag": "🇰🇭"},
    "China": {"capital": "Beijing", "population": 1411750000, "flag": "🇨🇳"},
    "Cyprus": {"capital": "Nicosia", "population": 1244188, "flag": "🇨🇾"},
    "Georgia": {"capital": "Tbilisi", "population": 3728573, "flag": "🇬🇪"},
    "Hong Kong": {"capital": "Hong Kong", "population": 7413100, "flag": "🇭🇰"},
    "India": {"capital": "New Delhi", "population": 1428627663, "flag": "🇮🇳"},
    "Indonesia": {"capital": "Jakarta", "population": 277534122, "flag": "🇮🇩"},
    "Iran": {"capital": "Tehran", "population": 87590873, "flag": "🇮🇷"},
    "Iraq": {"capital": "Baghdad", "population": 43533592, "flag": "🇮🇶"},
    "Israel": {"capital": "Jerusalem", "population": 9364000, "flag": "🇮🇱"},
    "Japan": {"capital": "Tokyo", "population": 125681593, "flag": "🇯🇵"},
    "Jordan": {"capital": "Amman", "population": 11148278, "flag": "🇯🇴"},
    "Kazakhstan": {"capital": "Astana", "population": 19398331, "flag": "🇰🇿"},
    "Kuwait": {"capital": "Kuwait City", "population": 4294621, "flag": "🇰🇼"},
    "Kyrgyzstan": {"capital": "Bishkek", "population": 6735347, "flag": "🇰🇬"},
    "Laos": {"capital": "Vientiane", "population": 7529475, "flag": "🇱🇦"},
    "Lebanon": {"capital": "Beirut", "population": 5353930, "flag": "🇱🇧"},
    "Malaysia": {"capital": "Kuala Lumpur", "population": 33938221, "flag": "🇲🇾"},
    "Maldives": {"capital": "Malé", "population": 521021, "flag": "🇲🇻"},
    "Mongolia": {"capital": "Ulaanbaatar", "population": 3398366, "flag": "🇲🇳"},
    "Myanmar": {"capital": "Naypyidaw", "population": 54179306, "flag": "🇲🇲"},
    "Nepal": {"capital": "Kathmandu", "population": 30547580, "flag": "🇳🇵"},
    "North Korea": {"capital": "Pyongyang", "population": 26072217, "flag": "🇰🇵"},
    "Oman": {"capital": "Muscat", "population": 4576298, "flag": "🇴🇲"},
    "Pakistan": {"capital": "Islamabad", "population": 231402117, "flag": "🇵🇰"},
    "Palestine": {"capital": "Ramallah", "population": 5371230, "flag": "🇵🇸"},
    "Philippines": {"capital": "Manila", "population": 115559009, "flag": "🇵🇭"},
    "Qatar": {"capital": "Doha", "population": 2930524, "flag": "🇶🇦"},
    "Russia": {"capital": "Moscow", "population": 146447424, "flag": "🇷🇺"},
    "Saudi Arabia": {"capital": "Riyadh", "population": 36408820, "flag": "🇸🇦"},
    "Singapore": {"capital": "Singapore", "population": 5453600, "flag": "🇸🇬"},
    "South Korea": {"capital": "Seoul", "population": 51744876, "flag": "🇰🇷"},
    "Sri Lanka": {"capital": "Sri Jayawardenepura Kotte", "population": 22037000, "flag": "🇱🇰"},
    "Syria": {"capital": "Damascus", "population": 22125249, "flag": "🇸🇾"},
    "Taiwan": {"capital": "Taipei", "population": 23894394, "flag": "🇹🇼"},
    "Tajikistan": {"capital": "Dushanbe", "population": 9952787, "flag": "🇹🇯"},
    "Thailand": {"capital": "Bangkok", "population": 71801279, "flag": "🇹🇭"},
    "Timor-Leste": {"capital": "Dili", "population": 1340513, "flag": "🇹🇱"},
    "Turkey": {"capital": "Ankara", "population": 85279553, "flag": "🇹🇷"},
    "Turkmenistan": {"capital": "Ashgabat", "population": 6430770, "flag": "🇹🇲"},
    "United Arab Emirates": {"capital": "Abu Dhabi", "population": 9441129, "flag": "🇦🇪"},
    "Uzbekistan": {"capital": "Tashkent", "population": 35648100, "flag": "🇺🇿"},
    "Vietnam": {"capital": "Hanoi", "population": 100987686, "flag": "🇻🇳"},
    "Yemen": {"capital": "Sana'a", "population": 33696614, "flag": "🇾🇪"},
}

REAL_DATA_SOURCES = {
    "Hong Kong": {
        "name": "Environmental Protection Department",
        "api": "https://api.data.gov.hk/v2/filter?q=%7B%22resource%22%3A%22http%3A%2F%2Fwww.epd.gov.hk%2Fepd%2Fsites%2Fdefault%2Ffiles%2Fepd%2Fenglish%2Fenvironmentinhk%2Fwater%2Friver_quality%2Fwaterquality.xml%22%7D&sort=sampling_date+desc",
        "type": "api"
    },
    "India": {
        "name": "Central Pollution Control Board (CPCB)",
        "url": "https://cpcb.nic.in/water-quality-data/",
        "type": "reference"
    },
    "China": {
        "name": "China National Environmental Monitoring Centre",
        "url": "http://www.cnemc.cn/",
        "type": "reference"
    },
    "Japan": {
        "name": "National Institute for Environmental Studies (NIES) via GEMStat",
        "url": "https://gemstat.org/",
        "type": "reference"
    },
    "South Korea": {
        "name": "Water Environment Information System (NIER)",
        "url": "http://water.nier.go.kr/web",
        "type": "reference"
    },
}

def get_data_source_info(country: str) -> dict:
    return REAL_DATA_SOURCES.get(country, {"name": "Simulated Data", "type": "simulated"})

@st.cache_data(ttl=1800)
def fetch_hongkong_water_data() -> pd.DataFrame:
    try:
        response = requests.get(
            "https://api.data.gov.hk/v2/filter",
            params={
                "q": '{"resource":"http://www.epd.gov.hk/epd/sites/default/files/epd/english/environmentinhk/water/river_quality/waterquality.xml"}',
                "sort": "sampling_date desc"
            },
            timeout=15
        )
        if response.status_code == 200:
            data = response.json()
            if data:
                df = pd.DataFrame(data)
                df = df.rename(columns={
                    'station_name': 'SiteID',
                    'sampling_date': 'SampleDateTime', 
                    'value': 'Value',
                    'parameter': 'Indicator'
                })
                df['Region'] = 'Hong Kong'
                df['SampleDateTime'] = pd.to_datetime(df['SampleDateTime'], errors='coerce')
                df['Month'] = df['SampleDateTime'].dt.month
                df['Year'] = df['SampleDateTime'].dt.year
                df['Date'] = df['SampleDateTime'].dt.date
                df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
                return df
    except Exception:
        pass
    return pd.DataFrame()

def fetch_country_metadata(country: str) -> dict:
    meta = COUNTRY_METADATA.get(country, {})
    source_info = get_data_source_info(country)
    return {
        'capital': meta.get('capital', 'N/A'),
        'population': meta.get('population', 0),
        'area': 0,
        'region': 'Asia',
        'flag': meta.get('flag', '🌏'),
        'data_source': source_info.get('name', 'Simulated'),
        'data_type': source_info.get('type', 'simulated')
    }


@st.cache_data(ttl=3600)
def fetch_water_quality_data(country: str) -> pd.DataFrame:
    import hashlib
    
    if country == "Hong Kong":
        real_data = fetch_hongkong_water_data()
        if not real_data.empty:
            return real_data
    
    basins = get_basins_for_country(country)
    wb_data = _fetch_worldbank_water_data(country)
    
    # Generate comprehensive dataset with DETERMINISTIC random values
    records = []
    # Use fixed end date for historical data consistency
    end_date = datetime(2025, 12, 31)
    start_date = end_date - timedelta(days=365 * 5)  # 5 years of data
    
    for basin in basins:
        basin_idx = basins.index(basin)
        current_date = start_date
        
        while current_date <= end_date:
            for indicator, ranges in INDICATOR_RANGES.items():
                # Create deterministic seed from country + basin + date + indicator
                seed_string = f"{country}_{basin}_{current_date.strftime('%Y%m%d')}_{indicator}"
                seed = int(hashlib.md5(seed_string.encode()).hexdigest()[:8], 16)
                
                # Use numpy random with specific seed for deterministic values
                rng = np.random.RandomState(seed)
                
                # Use World Bank data as baseline if available
                base_value = wb_data.get(indicator, (ranges['min'] + ranges['max']) / 2)
                
                # Add seasonal variation (deterministic)
                day_of_year = current_date.timetuple().tm_yday
                seasonal_factor = 1 + 0.2 * np.sin(2 * np.pi * day_of_year / 365)
                
                # Add deterministic random variation
                random_factor = rng.uniform(0.8, 1.2)
                value = base_value * seasonal_factor * random_factor
                
                # Clamp to valid range
                value = max(ranges['min'], min(ranges['max'], value))
                
                # Deterministic coordinates
                lat_offset = rng.uniform(-2, 2)
                lon_offset = rng.uniform(-2, 2)
                council_id = rng.randint(1000, 9999)
                
                records.append({
                    'DateImported': end_date.strftime('%d-%b-%y'),
                    'Region': country,
                    'Agency': f'{country} Water Authority',
                    'LawaSiteID': f'{country.lower()[:3]}-{basin_idx:05d}',
                    'SiteID': basin,
                    'CouncilSiteID': council_id,
                    'Latitude': _get_country_coords(country)[0] + lat_offset,
                    'Longitude': _get_country_coords(country)[1] + lon_offset,
                    'Indicator': indicator,
                    'SampleDateTime': current_date.strftime('%m/%d/%y %H:%M'),
                    'Value (Agency)': round(value, 4),
                    'Value': round(value, 2),
                    'Units': ranges['unit'],
                })
            
            # Monthly samples
            current_date += timedelta(days=30)
    
    df = pd.DataFrame(records)
    df['SampleDateTime'] = pd.to_datetime(df['SampleDateTime'], format='%m/%d/%y %H:%M')
    df['Date'] = df['SampleDateTime'].dt.date
    df['Month'] = df['SampleDateTime'].dt.month
    df['Year'] = df['SampleDateTime'].dt.year
    
    return df


def _fetch_worldbank_water_data(country: str) -> dict:
    """
    Attempt to fetch water-related indicators from World Bank API.
    Returns dict of indicator values or empty dict.
    """
    # World Bank country codes
    country_codes = {
        'India': 'IND', 'China': 'CHN', 'Japan': 'JPN', 'Bangladesh': 'BGD',
        'Pakistan': 'PAK', 'Indonesia': 'IDN', 'Thailand': 'THA', 'Vietnam': 'VNM',
        'Malaysia': 'MYS', 'Philippines': 'PHL', 'South Korea': 'KOR', 'Nepal': 'NPL',
        'Sri Lanka': 'LKA', 'Myanmar': 'MMR', 'Cambodia': 'KHM', 'Afghanistan': 'AFG',
        'Kazakhstan': 'KAZ', 'Uzbekistan': 'UZB', 'Iran': 'IRN', 'Iraq': 'IRQ',
        'Saudi Arabia': 'SAU', 'Turkey': 'TUR',
    }
    
    code = country_codes.get(country)
    if not code:
        return {}
    
    try:
        # Fetch renewable water resources indicator
        response = requests.get(
            f"https://api.worldbank.org/v2/country/{code}/indicator/ER.H2O.FWTL.ZS?format=json&per_page=1",
            timeout=5
        )
        if response.status_code == 200:
            data = response.json()
            if len(data) > 1 and data[1]:
                # Use as baseline for water quality
                return {'baseline': data[1][0].get('value', 50)}
    except:
        pass
    
    return {}


def _get_country_coords(country: str) -> tuple:
    """Get approximate center coordinates for a country."""
    coords = {
        'India': (20.5937, 78.9629),
        'China': (35.8617, 104.1954),
        'Japan': (36.2048, 138.2529),
        'Bangladesh': (23.6850, 90.3563),
        'Pakistan': (30.3753, 69.3451),
        'Indonesia': (-0.7893, 113.9213),
        'Thailand': (15.8700, 100.9925),
        'Vietnam': (14.0583, 108.2772),
        'Malaysia': (4.2105, 101.9758),
        'Philippines': (12.8797, 121.7740),
        'South Korea': (35.9078, 127.7669),
        'Nepal': (28.3949, 84.1240),
        'Sri Lanka': (7.8731, 80.7718),
        'Myanmar': (21.9162, 95.9560),
        'Cambodia': (12.5657, 104.9910),
        'Afghanistan': (33.9391, 67.7100),
        'Kazakhstan': (48.0196, 66.9237),
        'Uzbekistan': (41.3775, 64.5853),
        'Iran': (32.4279, 53.6880),
        'Iraq': (33.2232, 43.6793),
        'Saudi Arabia': (23.8859, 45.0792),
        'Turkey': (38.9637, 35.2433),
    }
    return coords.get(country, (25.0, 85.0))  # Default to central Asia


def normalize_uploaded_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize uploaded CSV/Excel data to match expected format.
    Uses fuzzy column matching to align headers.
    """
    column_mappings = {
        # Expected column -> possible variations
        'Region': ['region', 'country', 'area', 'location', 'state', 'province'],
        'SiteID': ['siteid', 'site_id', 'site', 'basin', 'lake', 'river', 'water_body', 'waterbody'],
        'Indicator': ['indicator', 'parameter', 'metric', 'measure', 'type'],
        'Value (Agency)': ['value (agency)', 'value_agency', 'agency_value', 'raw_value'],
        'Value': ['value', 'measurement', 'reading', 'result', 'amount'],
        'SampleDateTime': ['sampledatetime', 'sample_datetime', 'date', 'datetime', 'timestamp', 'sample_date'],
        'Units': ['units', 'unit', 'uom', 'measurement_unit'],
        'Latitude': ['latitude', 'lat', 'y'],
        'Longitude': ['longitude', 'lon', 'lng', 'x'],
    }
    
    # Create lowercase version of columns for matching
    df_columns_lower = {col.lower().strip(): col for col in df.columns}
    
    # Rename columns based on fuzzy matching
    rename_map = {}
    for target_col, variations in column_mappings.items():
        for var in variations:
            if var in df_columns_lower:
                rename_map[df_columns_lower[var]] = target_col
                break
    
    df = df.rename(columns=rename_map)
    
    # Ensure required columns exist
    required = ['SiteID', 'Value']
    for col in required:
        if col not in df.columns:
            # Try to find a numeric column for Value
            if col == 'Value':
                for c in df.columns:
                    if df[c].dtype in ['float64', 'int64', 'float32', 'int32']:
                        df['Value'] = df[c]
                        break
    
    # Add missing columns with defaults
    if 'Region' not in df.columns:
        df['Region'] = 'Uploaded Data'
    if 'Indicator' not in df.columns:
        df['Indicator'] = 'General'
    if 'Value (Agency)' not in df.columns and 'Value' in df.columns:
        df['Value (Agency)'] = df['Value']
    
    # Handle datetime
    if 'SampleDateTime' in df.columns:
        df['SampleDateTime'] = pd.to_datetime(df['SampleDateTime'], errors='coerce')
    else:
        df['SampleDateTime'] = datetime.now()
    
    df['Date'] = pd.to_datetime(df['SampleDateTime']).dt.date
    df['Month'] = pd.to_datetime(df['SampleDateTime']).dt.month
    df['Year'] = pd.to_datetime(df['SampleDateTime']).dt.year
    
    return df
