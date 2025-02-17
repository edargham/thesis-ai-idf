from datetime import datetime
from meteostat import Point, Daily, Hourly
import pandas as pd

# Set time period
start = datetime(2000, 6, 1)
end = datetime(2024, 6, 2)

# Create Point for Beirut
beirut = Point(33.8938, 35.5018)
beirut.radius = 30000

# Get daily data
data = Daily(beirut, start, end)
data = data.fetch()

# Keep only prcp column
data = data[['prcp']]

# Save data to CSV
data.to_csv('data/beirut.csv', index=True)
print(data.tail())
print(data['prcp'].sum())