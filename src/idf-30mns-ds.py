import os
import pandas as pd

if __name__ == '__main__':
    data_path = os.path.join(
        os.path.dirname(__file__),
        'data',
        'gpm-bey-30mns.csv'
    )

    # Read the data.
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)

    # Rename the column 'value' to '30mns'.
    df.rename(columns={'value': '30mns'}, inplace=True)

    # Convert 30-minute intensity (mm/hr) to 30-minute depth (mm)
    df['30mns_depth'] = df['30mns'] * 0.5  # 30 min = 0.5 hours

    # Calculate rolling sums for different durations (in mm)
    # For 1 hour (2 x 30-minute readings)
    df['1hr_depth'] = df['30mns_depth'].rolling(window=2).sum()

    # For 3 hours (6 x 30-minute readings)
    df['3hr_depth'] = df['30mns_depth'].rolling(window=6).sum()

    # For 24 hours (48 x 30-minute readings)
    df['24hr_depth'] = df['30mns_depth'].rolling(window=48).sum()

    # Convert back to intensities (mm/hr)
    df['30mns'] = df['30mns_depth'] #/ 0.5
    df['1hr'] = df['1hr_depth'] #/ 1
    df['3hr'] = df['3hr_depth'] #/ 3
    df['24hr'] = df['24hr_depth'] #/ 24

    # Remove intermediate depth columns
    df.drop(['30mns_depth', '1hr_depth', '3hr_depth', '24hr_depth'], axis=1, inplace=True)

    # Drop NaN values from the rolling calculations
    # df.dropna(inplace=True)

    # Save the new dataframe to a new XSLX file.
    output_path = os.path.join(
        os.path.dirname(__file__),
        'data',
        'gpm-bey-30mns-aggregated.xlsx'
    )
    df.to_excel(output_path, index=True)
    print(f"Aggregated data saved to {output_path}")