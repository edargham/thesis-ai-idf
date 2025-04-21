import os
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

if __name__ == "__main__":
    data_path_3h = os.path.join(os.path.dirname(__file__), "data", "trmm-bey-3hrs.csv")

    df = pd.read_csv(data_path_3h)
    # Print available columns to identify precipitation data
    print("Available columns:", df.columns.tolist())
    
    # Assuming there's a precipitation column - adjust column name as needed
    precip_col = 'value'  # Replace with actual column name from your data
    
    # rename precip_col to '3h'
    df.rename(columns={precip_col: '3h'}, inplace=True)
    precip_col = '3h'
    
    # Compute running averages for different time windows to maintain mm/hr
    # Each 3-hour interval corresponds to 1 row
    df['6h'] = df[precip_col].rolling(window=2).sum()
    df['9h'] = df[precip_col].rolling(window=3).sum()
    df['12h'] = df[precip_col].rolling(window=4).sum()
    df['15h'] = df[precip_col].rolling(window=5).sum()
    df['18h'] = df[precip_col].rolling(window=6).sum()
    df['24h'] = df[precip_col].rolling(window=8).sum()
    
    # Display the results
    print(df.head(10))

    # Find columns that might contain date information
    date_cols = [col for col in df.columns if any(term in col.lower() for term in ['date', 'time'])]

    if date_cols:
        # Try to use the first date column found
        date_col = date_cols[0]
        print(f"Using '{date_col}' as date reference")
        df['date'] = pd.to_datetime(df[date_col])
        df['year'] = df['date'].dt.year
    else:
        # If no date column found, try to use the index as date or seek timestamp in column names
        try:
            df.index = pd.to_datetime(df.index)
            df['year'] = df.index.year
            print("Using index as date reference")
        except:
            print("No date column found. Please ensure your data has temporal information.")
            # Assuming data might be organized chronologically
            raise ValueError("Unable to determine year information from data")

    # Step 2: Find yearly maximums for each duration
    durations = ['3h', '6h', '9h', '12h', '15h', '18h', '24h']
    annual_max = pd.DataFrame()

    for duration in durations:
        # Group by year and find maximum for each duration
        annual_max[duration] = df.groupby('year')[duration].max()

    print("\nAnnual Maximum Values:")
    print(annual_max.head(200))

    # Step 3: Fit Gumbel distribution and calculate IDF values using SciPy
    return_periods = [2, 5, 10, 25, 50, 100]
    probabilities = 1 - 1 / np.array(return_periods)
    idf_data = {}

    for duration in durations:
        # Get annual maximum values
        values = annual_max[duration].dropna().values
        
        # Fit Gumbel distribution 
        loc, scale = stats.gumbel_r.fit(values)
        
        print(f"\nGumbel parameters for {duration}:")
        print(f"Location: {loc:.4f}")
        print(f"Scale: {scale:.4f}")
        
        # Calculate intensity for different return periods
        hours = int(duration.replace('h', ''))
        intensities = []
        
        for i, T in enumerate(return_periods):
            # Use SciPy's percent point function to get quantiles
            precip = stats.gumbel_r.ppf(probabilities[i], loc, scale)
            intensity = precip / hours
            intensities.append(intensity)
        
        idf_data[duration] = intensities

    # Create DataFrame for IDF values
    idf_df = pd.DataFrame(idf_data, index=return_periods)
    idf_df.index.name = 'Return Period (years)'
    
    # Print the IDF table
    print("\nIDF Values (mm/hr):")
    print(idf_df)
    
    # Extract duration hours for plotting
    duration_hours = [int(dur.replace('h', '')) for dur in durations]
    
    # Create IDF curve plot
    plt.figure(figsize=(10, 6))
    
    for i, rp in enumerate(return_periods):
        # Get intensities for this return period across all durations
        curve_data = idf_df.loc[rp].values
        plt.plot(duration_hours, curve_data, marker='o', label=f"T = {rp} years")
    
    # plt.xscale('log')  # Log scale for duration axis
    plt.xlabel('Duration (hours)')
    plt.ylabel('Intensity (mm/hr)')
    plt.title('Intensity-Duration-Frequency (IDF) Curve')
    plt.grid(True, which='both', ls='-')
    plt.legend()
    
    # Save the plot and data
    output_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(output_dir, exist_ok=True)
    
    plt.savefig(os.path.join(output_dir, "idf_curve_trmm.png"), dpi=300)
    idf_df.to_csv(os.path.join(output_dir, "idf_data_trmm.csv"))
    
    # Show the plot
    plt.show()
