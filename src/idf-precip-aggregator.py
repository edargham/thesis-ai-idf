import os
import pandas as pd


if __name__ == '__main__':
    data_path_30mns = os.path.join(
        os.path.dirname(__file__),
        'data',
        'gpm-bey-30mns.csv'
    )
    # data_path_1h = os.path.join(
    #     os.path.dirname(__file__),
    #     'data',
    #     'gpm-gsmap-bey-hourly.csv'
    # )
    data_path_3h = os.path.join(
        os.path.dirname(__file__),
        'data',
        'trmm-bey-3hrs.csv'
    )
    data_path_24h = os.path.join(
        os.path.dirname(__file__),
        'data',
        'gpm-bey-daily.csv'
    )

    # Read the data
    df_30mns = pd.read_csv(data_path_30mns)
    # df_1h = pd.read_csv(data_path_1h)

    # df_hr is the the sum divided by 2 for every hour in the 30mns data
    df_1h = df_30mns.copy()
    df_1h['value'] = df_1h['value'] / 2
    df_1h['date'] = pd.to_datetime(df_1h['date']).dt.ceil('h')
    df_1h.set_index('date', inplace=True)
    df_1h = df_1h.groupby(level=0).sum()

    df_3h = pd.read_csv(data_path_3h)
    df_24h = pd.read_csv(data_path_24h)

    # Convert the time columns to datetime
    df_30mns['date'] = pd.to_datetime(df_30mns['date'])
    df_3h['date'] = pd.to_datetime(df_3h['date'])
    df_24h['date'] = pd.to_datetime(df_24h['date'])

    # Set the time columns as the index
    df_30mns.set_index('date', inplace=True)
    # df_1h.set_index('date', inplace=True)
    df_3h.set_index('date', inplace=True)
    df_24h.set_index('date', inplace=True)

    # Merge the dataframes on the index
    df = pd.concat([
        df_30mns,
        df_1h,
        df_3h,
        df_24h
    ], axis=1)

    df.columns = [
        '30mns',
        '1h',
        '3h',
        '24h'
    ]

    # Drop all the rows with dates after 2019-12-31 23:59:59
    df = df[df.index < '2019-12-31 23:59:59']
    
    df['3h'] = df['3h'] * 3  # 3 hours = 3 hours
    df['30mns'] = df['30mns'] * 0.5  # 30 minutes = 0.5 hours

    # Fill NaN values with 0
    df.fillna(0, inplace=True)

    # Save the new dataframe to a new XSLX file.
    output_path = os.path.join(
        os.path.dirname(__file__),
        'data',
        'bey-aggregated-final.xlsx'
    )
    df.to_excel(output_path, index=True)
    print(f"Aggregated data saved to {output_path}")
   