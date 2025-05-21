import pandas as pd
import numpy as np
import os
import re
from datetime import datetime
import streamlit as st
import tempfile
from typing import Tuple, List

def to_camel_case(snake_str):
    """Convert snake_case to CamelCase"""
    components = snake_str.split('_')
    return ''.join(x.title() for x in components)

def extract_date_info(file_path):
    """Extract month and week information from file path"""
    file_path = file_path.lower()
    
    # Extract month information
    months = {
        'jan': 'January', 'feb': 'February', 'mar': 'March', 'apr': 'April',
        'may': 'May', 'jun': 'June', 'jul': 'July', 'aug': 'August',
        'sep': 'September', 'oct': 'October', 'nov': 'November', 'dec': 'December'
    }
    
    month = None
    for m_abbr, m_full in months.items():
        if m_abbr in file_path:
            month = m_full
            break
    
    # If no month was found, use the current month
    if month is None:
        month = datetime.now().strftime("%B")
    
    # Extract week information if available
    week = "All"  # Default to "All" if no week is specified
    week_match = re.search(r'week\s*(\d+)', file_path, re.IGNORECASE)
    if week_match:
        week = f"Week {week_match.group(1)}"
    
    return month, week

def detect_region_and_country_improved(text: str) -> Tuple[str, str]:
    """
    Improved function to identify region and country based on text content.
    Handles multiple countries and better pattern matching.
    """
    if not isinstance(text, str):
        return "Unknown", "Unknown"
    
    text = text.upper()
    
    # Define regions and their associated countries (full names and codes)
    regions = {
        'APAC': {
            'MALAYSIA': 'Malaysia', 'MY': 'Malaysia',
            'THAILAND': 'Thailand', 'TH': 'Thailand',
            'VIETNAM': 'Vietnam', 'VN': 'Vietnam',
            'SINGAPORE': 'Singapore', 'SG': 'Singapore',
            'INDONESIA': 'Indonesia', 'ID': 'Indonesia',
            'PHILIPPINES': 'Philippines', 'PH': 'Philippines',
            'HONG KONG': 'Hong Kong', 'HONGKONG': 'Hong Kong', 'HK': 'Hong Kong',
            'JAPAN': 'Japan', 'JP': 'Japan',
            'TAIWAN': 'Taiwan', 'TW': 'Taiwan'
        },
        'LATAM': {
            'BRAZIL': 'Brazil', 'BR': 'Brazil',
            'MEXICO': 'Mexico', 'MX': 'Mexico',
            'ARGENTINA': 'Argentina', 'AR': 'Argentina',
            'COLOMBIA': 'Colombia', 'CO': 'Colombia',
            'CHILE': 'Chile', 'CL': 'Chile',
            'PERU': 'Peru', 'PE': 'Peru'
        },
        'Europe': {
            'UNITED KINGDOM': 'United Kingdom', 'UK': 'United Kingdom', 'GB': 'United Kingdom',
            'GERMANY': 'Germany', 'DE': 'Germany',
            'FRANCE': 'France', 'FR': 'France',
            'ITALY': 'Italy', 'IT': 'Italy',
            'SPAIN': 'Spain', 'ES': 'Spain',
            'NETHERLANDS': 'Netherlands', 'NL': 'Netherlands',
            'POLAND': 'Poland', 'PL': 'Poland'
        },
        'Australia': {
            'AUSTRALIA': 'Australia', 'AU': 'Australia', 'AUS': 'Australia',
            'NEW ZEALAND': 'New Zealand', 'NZ': 'New Zealand'
        },
        'Americas': {
            'UNITED STATES': 'United States', 'US': 'United States', 'USA': 'United States',
            'CANADA': 'Canada', 'CA': 'Canada'
        }
    }
    
    def find_countries_in_text(text: str, region_countries: dict) -> List[str]:
        """Find all countries from a specific region mentioned in the text."""
        found_countries = []
        
        for country_code, country_name in region_countries.items():
            # Improved pattern matching for country codes
            patterns = [
                f'\\b{country_code}\\b',  # Word boundary match
                f'_{country_code}_',      # Underscore separated
                f'_{country_code}\\b',    # Underscore prefix with word boundary
                f'\\b{country_code}_',    # Word boundary with underscore suffix
                f'-{country_code}-',      # Dash separated
                f'-{country_code}\\b',    # Dash prefix with word boundary
                f'\\b{country_code}-',    # Word boundary with dash suffix
            ]
            
            # Also check for full country names
            if len(country_code) > 2:  # Only for full names, not codes
                patterns.append(f'\\b{country_code}\\b')
            
            for pattern in patterns:
                if re.search(pattern, text):
                    found_countries.append(country_name)
                    break  # Don't double-count the same country
        
        return list(set(found_countries))  # Remove duplicates
    
    # Check for explicit region mentions first
    for region_name in regions.keys():
        if region_name in text:
            countries_in_region = find_countries_in_text(text, regions[region_name])
            
            if len(countries_in_region) == 1:
                return region_name, countries_in_region[0]
            elif len(countries_in_region) > 1:
                return region_name, 'Multiple'
            else:
                # Region mentioned but no specific countries found
                return region_name, 'Multiple'
    
    # Check for EU specific case
    if re.search(r'\bEU\b|_EU_|_EU\b|\bEU_|-EU-|-EU\b|\bEU-', text):
        return 'Europe', 'Multiple'
    
    # No explicit region mentioned, check for countries
    found_countries_by_region = {}
    
    for region_name, region_countries in regions.items():
        countries_found = find_countries_in_text(text, region_countries)
        if countries_found:
            found_countries_by_region[region_name] = countries_found
    
    # Analyze findings with APAC priority logic
    if not found_countries_by_region:
        return "Unknown", "Unknown"
    
    # BUSINESS RULE: If APAC countries are found with any other countries, 
    # prioritize APAC and ignore others
    if 'APAC' in found_countries_by_region:
        apac_countries = found_countries_by_region['APAC']
        
        if len(apac_countries) == 1:
            # Single APAC country (ignore any other regions found)
            return 'APAC', apac_countries[0]
        else:
            # Multiple APAC countries (ignore any other regions found)
            return 'APAC', 'Multiple'
    
    # No APAC countries found, proceed with normal logic
    if len(found_countries_by_region) == 1:
        # All countries from same non-APAC region
        region = list(found_countries_by_region.keys())[0]
        countries = found_countries_by_region[region]
        
        if len(countries) == 1:
            return region, countries[0]
        else:
            # Multiple countries from same region
            return region, 'Multiple'
    else:
        # Countries from multiple non-APAC regions
        return 'Global', 'Multiple'

def process_region_country_vectorized(df: pd.DataFrame) -> pd.DataFrame:
    """
    Vectorized approach to add region and country information.
    Much faster than iterrows() approach.
    """
    df = df.copy()
    
    # Initialize columns
    df['Region'] = 'Unknown'
    df['Country'] = 'Unknown'
    
    # Create a combined text column for analysis
    text_columns = []
    if 'Campaign' in df.columns:
        text_columns.append('Campaign')
    if 'AdGroup' in df.columns:
        text_columns.append('AdGroup')
    
    if text_columns:
        # Combine relevant text columns
        df['_combined_text'] = df[text_columns].fillna('').apply(
            lambda row: ' '.join(row.astype(str)), axis=1
        )
        
        # Apply the detection function
        region_country = df['_combined_text'].apply(detect_region_and_country_improved)
        df['Region'] = [rc[0] for rc in region_country]
        df['Country'] = [rc[1] for rc in region_country]
        
        # Clean up temporary column
        df.drop('_combined_text', axis=1, inplace=True)
    
    return df

def process_file(file_path, file_name):
    """Enhanced version of process_file with better error handling and performance"""
    try:
        # Extract date information from file path
        month, week = extract_date_info(file_name)
        
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Log original structure
        st.write(f"Processing {file_name}: {len(df)} rows, {len(df.columns)} columns")
        
        # Standardize column names
        df.columns = [col.lower().strip() for col in df.columns]
        df.columns = [to_camel_case(col) if '_' in col else col.capitalize() for col in df.columns]
        
        # Complete column mapping
        column_mapping = {
            'PsChannel': 'Channel', 'Pschannel': 'Channel', 'Channel': 'Channel',
            'PsSubChannel': 'SubChannel', 'Pssubchannel': 'SubChannel',
            'Subchannel': 'SubChannel', 'Subchannels': 'SubChannel', 'Sub_channel': 'SubChannel',
            'Campaign': 'Campaign', 'Campaigns': 'Campaign',
            'AdGroup': 'AdGroup', 'Adgroup': 'AdGroup', 'Ad_group': 'AdGroup', 'Adgroups': 'AdGroup',
            'Impression': 'Impressions', 'Impressions': 'Impressions',
            'Click': 'Clicks', 'Clicks': 'Clicks',
            'Spend': 'Spend', 'Cost': 'Spend',
            'Ql': 'QL', 'Qual': 'QL', 'Qualifieds': 'QL', 'Leads': 'QL',
            'Cpql': 'CpQL', 'Costperql': 'CpQL', 'Cost_per_ql': 'CpQL', 'Costperlead': 'CpQL',
            'Ft': 'FT', 'Firsttime': 'FT', 'First_time': 'FT', 'Firsttrades': 'FT',
            'Cpft': 'CpFT', 'Costperft': 'CpFT', 'Cost_per_ft': 'CpFT',
            'Mnr': 'MNR', 'Marginnetrevenue': 'MNR', 'Margin_net_revenue': 'MNR',
            'Mnr365ql': 'MNR365QL', 'Mnr365': 'MNR365QL', 'Mnr_365_ql': 'MNR365QL',
            'RoiMnr': 'ROIMNR', 'Roi_mnr': 'ROIMNR',
            'RoiMnrql365': 'ROIMNRQL365', 'Roi_mnrql365': 'ROIMNRQL365',
            'React': 'React', 'Reaction': 'React'
        }
        
        df = df.rename(columns={old: new for old, new in column_mapping.items() if old in df.columns})
        
        # Add missing numeric columns
        numeric_cols = ['Impressions', 'Clicks', 'Spend', 'QL', 'FT', 'MNR', 'MNR365QL']
        for col in numeric_cols:
            if col not in df.columns:
                df[col] = 0
            else:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Convert ratio columns to numeric
        ratio_cols = ['CpQL', 'CpFT', 'ROIMNR', 'ROIMNRQL365']
        for col in ratio_cols:
            if col in df.columns:
                # Remove any currency symbols and convert to numeric
                if df[col].dtype == object:  # Only process if it's a string column
                    df[col] = df[col].astype(str).replace('[\$,£,€,¥]', '', regex=True)
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Add date columns
        df['Month'] = month
        df['Week'] = week
        
        # Apply improved region/country detection
        df = process_region_country_vectorized(df)
        
        st.write(f"Completed processing {file_name}")
        return df
        
    except Exception as e:
        st.error(f"Error processing file {file_name}: {str(e)}")
        return None

def combine_dataframes(dataframes):
    """Combine multiple dataframes into one master dataframe"""
    if not dataframes:
        return None
    
    # Ensure all dataframes have the same columns
    all_columns = set()
    for df in dataframes:
        all_columns.update(df.columns)
    
    # Add missing columns to each dataframe
    for i, df in enumerate(dataframes):
        for col in all_columns:
            if col not in df.columns:
                if col in ['Impressions', 'Clicks', 'Spend', 'QL', 'FT', 'MNR', 'MNR365QL', 'CpQL', 'CpFT', 'ROIMNR', 'ROIMNRQL365']:
                    dataframes[i][col] = 0
                else:
                    dataframes[i][col] = None
    
    # Combine all dataframes
    master_df = pd.concat(dataframes, ignore_index=True)
    
    # Reorder columns to have date and region columns first
    date_cols = ['Month', 'Week', 'Region', 'Country']
    other_cols = [col for col in master_df.columns if col not in date_cols]
    master_df = master_df[date_cols + other_cols]
    
    return master_df

def create_excel_output(master_df):
    """Create an Excel output with multiple sheets and comprehensive summary"""
    try:
        # Create a temporary directory for the Excel file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as temp_file:
            temp_path = temp_file.name
        
        # Create a new Excel file with pandas
        with pd.ExcelWriter(temp_path, engine='openpyxl') as writer:
            # Write the master data first
            master_df.to_excel(writer, sheet_name='Master Data', index=False)
            
            # Check which columns actually exist in the DataFrame
            numeric_cols = ['Impressions', 'Clicks', 'Spend', 'QL', 'FT', 'MNR', 'MNR365QL']
            available_numeric_cols = [col for col in numeric_cols if col in master_df.columns]
            
            if not available_numeric_cols:
                st.warning("No numeric columns found for summary")
                return temp_path
            
            # Create comprehensive summary pivot table
            try:
                # Define dimensions
                dimensions = []
                if 'Region' in master_df.columns:
                    dimensions.append('Region')
                if 'Country' in master_df.columns:
                    dimensions.append('Country')
                if 'Month' in master_df.columns:
                    dimensions.append('Month')
                if 'Week' in master_df.columns and (master_df['Week'].nunique() > 1 or master_df['Week'].iloc[0] != "All"):
                    dimensions.append('Week')
                
                if dimensions:
                    # Create the comprehensive summary pivot
                    summary_pivot = pd.pivot_table(
                        master_df, 
                        values=available_numeric_cols,
                        index=dimensions, 
                        aggfunc='sum'
                    )
                    
                    # Add calculated metrics if possible
                    try:
                        if 'Clicks' in available_numeric_cols and 'Impressions' in available_numeric_cols:
                            summary_pivot['CTR'] = summary_pivot['Clicks'] / summary_pivot['Impressions'] * 100
                        
                        if 'Spend' in available_numeric_cols and 'Clicks' in available_numeric_cols and (summary_pivot['Clicks'] > 0).any():
                            summary_pivot['CPC'] = summary_pivot['Spend'] / summary_pivot['Clicks']
                        
                        if 'Spend' in available_numeric_cols and 'QL' in available_numeric_cols and (summary_pivot['QL'] > 0).any():
                            summary_pivot['CpQL'] = summary_pivot['Spend'] / summary_pivot['QL']
                        
                        if 'QL' in available_numeric_cols and 'Clicks' in available_numeric_cols and (summary_pivot['Clicks'] > 0).any():
                            summary_pivot['QL Conv %'] = summary_pivot['QL'] / summary_pivot['Clicks'] * 100
                        
                        if 'FT' in available_numeric_cols and 'QL' in available_numeric_cols and (summary_pivot['QL'] > 0).any():
                            summary_pivot['FT Conv %'] = summary_pivot['FT'] / summary_pivot['QL'] * 100
                        
                        if 'MNR' in available_numeric_cols and 'Spend' in available_numeric_cols and (summary_pivot['Spend'] > 0).any():
                            summary_pivot['ROI'] = summary_pivot['MNR'] / summary_pivot['Spend']
                    except Exception as e:
                        st.warning(f"Could not calculate some metrics: {str(e)}")
                    
                    # Write the summary to Excel
                    summary_pivot.to_excel(writer, sheet_name='Summary')
            except Exception as e:
                st.error(f"Error creating summary: {str(e)}")
            
            # Create additional pivot tables for specific views
            
            # 1. Summary by Time
            try:
                time_cols = []
                if 'Month' in master_df.columns:
                    time_cols.append('Month')
                if 'Week' in master_df.columns and (master_df['Week'].nunique() > 1 or master_df['Week'].iloc[0] != "All"):
                    time_cols.append('Week')
                
                if time_cols:
                    pivot_time = pd.pivot_table(
                        master_df, 
                        values=available_numeric_cols,
                        index=time_cols, 
                        aggfunc='sum'
                    )
                    pivot_time.to_excel(writer, sheet_name='Summary by Time')
            except Exception as e:
                st.error(f"Error creating time summary: {str(e)}")
            
            # 2. Summary by Channel
            try:
                channel_cols = []
                if 'Channel' in master_df.columns:
                    channel_cols.append('Channel')
                if 'SubChannel' in master_df.columns:
                    channel_cols.append('SubChannel')
                
                if channel_cols:
                    pivot_channel = pd.pivot_table(
                        master_df, 
                        values=available_numeric_cols,
                        index=channel_cols, 
                        aggfunc='sum'
                    )
                    pivot_channel.to_excel(writer, sheet_name='Summary by Channel')
            except Exception as e:
                st.error(f"Error creating channel summary: {str(e)}")
            
            # 3. Summary by Region
            try:
                region_cols = []
                if 'Region' in master_df.columns:
                    region_cols.append('Region')
                if 'Country' in master_df.columns:
                    region_cols.append('Country')
                
                if region_cols:
                    pivot_region = pd.pivot_table(
                        master_df, 
                        values=available_numeric_cols,
                        index=region_cols, 
                        aggfunc='sum'
                    )
                    pivot_region.to_excel(writer, sheet_name='Summary by Region')
            except Exception as e:
                st.error(f"Error creating region summary: {str(e)}")
        
        return temp_path
    
    except Exception as e:
        st.error(f"Error creating Excel output: {str(e)}")
        return None

# Streamlit UI
def main():
    st.set_page_config(page_title="Marketing Data Master File Generator", layout="wide")
    
    st.title("Marketing Data Master File Generator")
    st.write("Upload CSV files to generate a master file with added date and region information.")
    
    # Add expander for instructions about file naming
    with st.expander("File Naming Convention"):
        st.markdown("""
        - Include month in the filename (Jan, Feb, Mar, etc.)
        - For weekly data, include 'Week1', 'Week2', etc. in the filename
        - Example: 'Pepperstone_PPC_May_Week1_2025.csv'
        - For monthly data: 'Pepperstone_Social_Feb_2025.csv'
        """)
    
    # File uploader
    uploaded_files = st.file_uploader("Choose CSV files", type="csv", accept_multiple_files=True)
    
    # Processing button
    if uploaded_files:
        st.write(f"Selected {len(uploaded_files)} files")
        
        if st.button("Process Files"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("Starting file processing...")
            dataframes = []
            
            # Process each uploaded file
            for i, uploaded_file in enumerate(uploaded_files):
                status_text.text(f"Processing file {i+1}/{len(uploaded_files)}: {uploaded_file.name}")
                progress_bar.progress((i) / len(uploaded_files))
                
                # Create a temporary file to store the uploaded content
                with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as temp_file:
                    temp_file.write(uploaded_file.getvalue())
                    temp_path = temp_file.name
                
                # Process the file
                df = process_file(temp_path, uploaded_file.name)
                if df is not None:
                    dataframes.append(df)
                
                # Clean up
                os.unlink(temp_path)
                
                # Update progress
                progress_bar.progress((i + 1) / len(uploaded_files))
            
            # Combine dataframes if we have any
            if dataframes:
                status_text.text("Combining data...")
                master_df = combine_dataframes(dataframes)
                
                if master_df is not None:
                    # Show data preview
                    st.subheader("Data Preview")
                    st.dataframe(master_df.head())
                    
                    # Create Excel file
                    status_text.text("Creating Excel output...")
                    
                    # Generate output filename
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_filename = f"Marketing_Master_{timestamp}.xlsx"
                    
                    # Create the Excel file
                    excel_path = create_excel_output(master_df)
                    
                    if excel_path:
                        # Provide download link
                        try:
                            with open(excel_path, "rb") as file:
                                excel_data = file.read()
                            
                            status_text.text("Processing complete! You can download the file below.")
                            progress_bar.progress(1.0)
                            
                            st.download_button(
                                label="📥 Download Excel Report",
                                data=excel_data,
                                file_name=output_filename,
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )
                            
                            # Clean up
                            os.unlink(excel_path)
                            
                        except Exception as e:
                            status_text.text("Error preparing download.")
                            st.error(f"Error preparing download: {str(e)}")
                    else:
                        status_text.text("Failed to create Excel file.")
                        st.error("Failed to create Excel file.")
                else:
                    status_text.text("Failed to combine data.")
                    st.error("Failed to combine dataframes.")
            else:
                status_text.text("No valid data found.")
                st.error("No valid data found in the uploaded files.")
    else:
        st.info("Please upload CSV files to begin processing.")

if __name__ == "__main__":
    main()
