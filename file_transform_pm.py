import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
import os
import re
from datetime import datetime
import streamlit as st

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

def detect_region_and_country(text):
    """Identify region and country based on text content"""
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
            'SPAIN': 'Spain', 'ES': 'Spain'
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
    
    # First check if a region is explicitly mentioned
    for region_name in regions.keys():
        if region_name in text:
            # Region is explicitly mentioned - now check if a specific country is mentioned
            for country_code, country_name in regions[region_name].items():
                if country_code in text.split() or f"_{country_code}_" in text or f"_{country_code}" in text or f"{country_code}_" in text:
                    return region_name, country_name
            
            # Region mentioned but no specific country found
            return region_name, 'Multiple'
    
    # No region explicitly mentioned, so check for country mentions
    for region_name, countries in regions.items():
        for country_code, country_name in countries.items():
            if country_code in text.split() or f"_{country_code}_" in text or f"_{country_code}" in text or f"{country_code}_" in text:
                return region_name, country_name
    
    # Check for region acronyms
    if 'EU' in text.split() or f"_EU_" in text or text.startswith("EU_") or text.endswith("_EU"):
        return 'Europe', 'Multiple'
    
    return "Unknown", "Unknown"

def process_file(file_path):
    """Process a single CSV file"""
    try:
        # Extract date information from file path
        month, week = extract_date_info(file_path)
        
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Print columns for debugging
        print(f"Original columns in {os.path.basename(file_path)}: {df.columns.tolist()}")
        
        # First standardize column names by converting to lowercase and stripping whitespace
        df.columns = [col.lower().strip() for col in df.columns]
        
        # Then convert to CamelCase and apply specific mappings
        df.columns = [to_camel_case(col) if '_' in col else col.capitalize() for col in df.columns]
        
        # Map common column variations to standard names
        column_mapping = {
            'PsChannel': 'Channel',
            'Pschannel': 'Channel',
            'Channel': 'Channel',
            'PsSubChannel': 'SubChannel',
            'Pssubchannel': 'SubChannel',
            'Subchannel': 'SubChannel',
            'Subchannels': 'SubChannel',
            'Sub_channel': 'SubChannel',
            'Campaign': 'Campaign',
            'Campaigns': 'Campaign',
            'AdGroup': 'AdGroup',
            'Adgroup': 'AdGroup',
            'Ad_group': 'AdGroup',
            'Adgroups': 'AdGroup',
            'Impression': 'Impressions',
            'Impressions': 'Impressions',
            'Click': 'Clicks',
            'Clicks': 'Clicks',
            'Spend': 'Spend',
            'Cost': 'Spend',
            'Ql': 'QL',
            'Qual': 'QL',
            'Qualifieds': 'QL',
            'Leads': 'QL',
            'Cpql': 'CpQL',
            'Costperql': 'CpQL',
            'Cost_per_ql': 'CpQL',
            'Costperlead': 'CpQL',
            'Ft': 'FT',
            'Firsttime': 'FT',
            'First_time': 'FT',
            'Firsttrades': 'FT',
            'Cpft': 'CpFT',
            'Costperft': 'CpFT',
            'Cost_per_ft': 'CpFT',
            'Mnr': 'MNR',
            'Marginnetrevenue': 'MNR',
            'Margin_net_revenue': 'MNR',
            'Mnr365ql': 'MNR365QL',
            'Mnr365': 'MNR365QL',
            'Mnr_365_ql': 'MNR365QL',
            'RoiMnr': 'ROIMNR',
            'Roi_mnr': 'ROIMNR',
            'RoiMnrql365': 'ROIMNRQL365',
            'Roi_mnrql365': 'ROIMNRQL365',
            'React': 'React',
            'Reaction': 'React'
        }
        
        # Rename columns according to the mapping
        df = df.rename(columns={old: new for old, new in column_mapping.items() if old in df.columns})
        
        # Print columns after standardization for debugging
        print(f"Standardized columns in {os.path.basename(file_path)}: {df.columns.tolist()}")
        
        # Add clicks column if it doesn't exist
        if 'Clicks' not in df.columns:
            df['Clicks'] = 0
        
        # Add Impressions column if it doesn't exist (needed for pivot tables)
        if 'Impressions' not in df.columns:
            df['Impressions'] = 0
        
        # Add date columns
        df['Month'] = month
        df['Week'] = week
        
        # Convert numeric columns to appropriate types
        numeric_cols = ['Impressions', 'Clicks', 'Spend', 'QL', 'FT', 'MNR', 'MNR365QL']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Convert ratio columns to numeric
        ratio_cols = ['CpQL', 'CpFT', 'ROIMNR', 'ROIMNRQL365']
        for col in ratio_cols:
            if col in df.columns:
                # Remove any currency symbols and convert to numeric
                if df[col].dtype == object:  # Only process if it's a string column
                    df[col] = df[col].astype(str).replace('[\$,£,€,¥]', '', regex=True)
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Add Region and Country columns
        df['Region'] = 'Unknown'
        df['Country'] = 'Unknown'
        
        # Determine region and country from campaign and adgroup
        if 'Campaign' in df.columns:
            for idx, row in df.iterrows():
                campaign_region, campaign_country = detect_region_and_country(str(row['Campaign']))
                if campaign_region != "Unknown":
                    df.at[idx, 'Region'] = campaign_region
                    df.at[idx, 'Country'] = campaign_country
        
        if 'AdGroup' in df.columns:
            for idx, row in df.iterrows():
                # Only update if region is still unknown or if adgroup has more specific info
                if df.at[idx, 'Region'] == 'Unknown':
                    adgroup_region, adgroup_country = detect_region_and_country(str(row['AdGroup']))
                    if adgroup_region != "Unknown":
                        df.at[idx, 'Region'] = adgroup_region
                        df.at[idx, 'Country'] = adgroup_country
        
        # Print sample of region/country assignment
        print(f"Sample region/country assignments in {os.path.basename(file_path)}:")
        print(df[['Region', 'Country']].head())
        
        return df
    
    except Exception as e:
        print(f"Error processing file {os.path.basename(file_path)}: {str(e)}")
        if 'st' in globals():
            st.error(f"Error processing file {os.path.basename(file_path)}: {str(e)}")
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
    
    # Print sample of the master dataframe
    print("Sample of master dataframe:")
    print(master_df[['Month', 'Week', 'Region', 'Country']].head())
    
    return master_df

def create_excel_output(master_df, output_path):
    """Create an Excel output with multiple sheets and comprehensive summary"""
    try:
        # Create a new Excel file with pandas
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Write the master data first
            master_df.to_excel(writer, sheet_name='Master Data', index=False)
            
            # Check which columns actually exist in the DataFrame
            numeric_cols = ['Impressions', 'Clicks', 'Spend', 'QL', 'FT', 'MNR', 'MNR365QL']
            available_numeric_cols = [col for col in numeric_cols if col in master_df.columns]
            
            if not available_numeric_cols:
                print("Warning: No numeric columns found for summary")
                if 'st' in globals():
                    st.warning("No numeric columns found for summary")
                return True
            
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
                        
                        if 'Spend' in available_numeric_cols and 'Clicks' in available_numeric_cols:
                            summary_pivot['CPC'] = summary_pivot['Spend'] / summary_pivot['Clicks']
                        
                        if 'Spend' in available_numeric_cols and 'QL' in available_numeric_cols:
                            summary_pivot['CpQL'] = summary_pivot['Spend'] / summary_pivot['QL']
                        
                        if 'QL' in available_numeric_cols and 'Clicks' in available_numeric_cols:
                            summary_pivot['QL Conv %'] = summary_pivot['QL'] / summary_pivot['Clicks'] * 100
                        
                        if 'FT' in available_numeric_cols and 'QL' in available_numeric_cols:
                            summary_pivot['FT Conv %'] = summary_pivot['FT'] / summary_pivot['QL'] * 100
                        
                        if 'MNR' in available_numeric_cols and 'Spend' in available_numeric_cols:
                            summary_pivot['ROI'] = summary_pivot['MNR'] / summary_pivot['Spend']
                    except Exception as e:
                        print(f"Warning: Could not calculate some metrics: {str(e)}")
                    
                    # Write the summary to Excel
                    summary_pivot.to_excel(writer, sheet_name='Summary')
            except Exception as e:
                print(f"Error creating summary: {str(e)}")
                if 'st' in globals():
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
                print(f"Error creating time summary: {str(e)}")
                if 'st' in globals():
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
                print(f"Error creating channel summary: {str(e)}")
                if 'st' in globals():
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
                print(f"Error creating region summary: {str(e)}")
                if 'st' in globals():
                    st.error(f"Error creating region summary: {str(e)}")
        
        return True
    
    except Exception as e:
        print(f"Error creating Excel output: {str(e)}")
        if 'st' in globals():
            st.error(f"Error creating Excel output: {str(e)}")
        return False

def streamlit_ui():
    st.title("Marketing Data Master File Generator")
    st.write("Upload CSV files to generate a master file with added date and region information.")
    
    # Add instructions about file naming
    st.info("""
    **File Naming Convention:**
    - Include month in the filename (Jan, Feb, Mar, etc.)
    - For weekly data, include 'Week1', 'Week2', etc. in the filename
    - Example: 'Pepperstone_PPC_May_Week1_2025.csv'
    - For monthly data: 'Pepperstone_Social_Feb_2025.csv'
    """)
    
    uploaded_files = st.file_uploader("Choose CSV files", type="csv", accept_multiple_files=True)
    
    if uploaded_files:
        if st.button("Process Files"):
            with st.spinner("Processing files..."):
                dataframes = []
                
                # Process each uploaded file
                for uploaded_file in uploaded_files:
                    # Save the uploaded file temporarily to extract path info
                    temp_path = f"temp_{uploaded_file.name}"
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # Process the file
                    df = process_file(temp_path)
                    if df is not None:
                        dataframes.append(df)
                    
                    # Clean up
                    os.remove(temp_path)
                
                if dataframes:
                    # Combine all dataframes
                    master_df = combine_dataframes(dataframes)
                    
                    if master_df is not None:
                        # Generate output filename
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        output_filename = f"Marketing_Master_{timestamp}.xlsx"
                        
                        # Create Excel output
                        success = create_excel_output(master_df, output_filename)
                        
                        if success:
                            # Provide download link
                            try:
                                with open(output_filename, "rb") as file:
                                    st.download_button(
                                        label="Download Master File",
                                        data=file,
                                        file_name=output_filename,
                                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                    )
                                # Clean up
                                os.remove(output_filename)
                                st.success("Master file generated successfully!")
                            except Exception as e:
                                st.error(f"Error preparing download: {str(e)}")
                        else:
                            st.error("Failed to create Excel file.")
                    else:
                        st.error("Failed to combine dataframes.")
                else:
                    st.error("No valid data found in the uploaded files.")

# Alternative Tkinter UI
def tkinter_ui():
    root = tk.Tk()
    root.title("Marketing Data Master File Generator")
    root.geometry("500x400")
    
    # Variable to store selected file paths
    selected_files = []
    
    def select_files():
        nonlocal selected_files
        file_paths = filedialog.askopenfilenames(filetypes=[("CSV files", "*.csv")])
        if file_paths:
            selected_files = file_paths
            files_label.config(text=f"{len(file_paths)} files selected")
            process_button.config(state=tk.NORMAL)
            
    def process_files_func():
        if not selected_files:
            messagebox.showerror("Error", "No files selected")
            return
            
        dataframes = []
        
        for file_path in selected_files:
            df = process_file(file_path)
            if df is not None:
                dataframes.append(df)
        
        if dataframes:
            # Combine all dataframes
            master_df = combine_dataframes(dataframes)
            
            if master_df is not None:
                # Generate output filename
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = filedialog.asksaveasfilename(
                    defaultextension=".xlsx",
                    filetypes=[("Excel files", "*.xlsx")],
                    initialfile=f"Marketing_Master_{timestamp}.xlsx"
                )
                
                if output_path:
                    # Create Excel output
                    success = create_excel_output(master_df, output_path)
                    if success:
                        messagebox.showinfo("Success", f"Master file saved to {output_path}")
                    else:
                        messagebox.showerror("Error", "Failed to create Excel file")
            else:
                messagebox.showerror("Error", "Failed to combine dataframes.")
        else:
            messagebox.showerror("Error", "No valid data found in the selected files.")
    
    # UI Elements
    header_label = tk.Label(root, text="Marketing Data Master File Generator", font=("Arial", 14))
    header_label.pack(pady=20)
    
    # Instructions label
    instructions = """File Naming Convention:
- Include month in the filename (Jan, Feb, Mar, etc.)
- For weekly data, include 'Week1', 'Week2', etc.
- Example: 'Pepperstone_PPC_May_Week1_2025.csv'
- For monthly: 'Pepperstone_Social_Feb_2025.csv'"""
    
    instructions_label = tk.Label(root, text=instructions, justify="left")
    instructions_label.pack(pady=10)
    
    select_button = tk.Button(root, text="Select CSV Files", command=select_files)
    select_button.pack(pady=10)
    
    files_label = tk.Label(root, text="No files selected")
    files_label.pack(pady=10)
    
    process_button = tk.Button(root, text="Process Files", state=tk.DISABLED, command=process_files_func)
    process_button.pack(pady=10)
    
    root.mainloop()

# Choose which UI to run
if __name__ == "__main__":
    # Choose one:
    streamlit_ui()  # Web-based UI (recommended)
    # tkinter_ui()  # Desktop UI