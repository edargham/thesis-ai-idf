from datetime import datetime
from meteostat import Point, Daily, Hourly, Monthly
import pandas as pd

# Set time period
start = datetime(2000, 6, 1)
end = datetime(2024, 6, 1)

# Create Point for Beirut
beirut = Point(33.83, 35.49)
beirut.radius = 25000

# Get daily data
data = Daily(beirut, start, end, model=False)
data = data.fetch()

# Keep only prcp column
data = data[['prcp']]

# Change the column name to 'value'
data = data.rename(columns={'prcp': 'value'})

# Change the index name to 'date'
data.index.name = 'date'

# Save data to CSV
data.to_csv('data/beirut-meteostat.csv', index=True)
print(data.tail())
print(data['value'].sum())