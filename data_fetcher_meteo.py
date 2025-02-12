from datetime import datetime
from meteostat import Point, Daily
import pandas as pd

# Set time period
start = datetime(2000, 1, 1)
end = datetime(2024, 12, 31)

# Create Point for Beirut
beirut = Point(33.8938, 35.5018, 10)

# Get daily data
data = Daily(beirut, start, end)
data = data.fetch()

# Save data to CSV
data.to_csv('data/beirut.csv', index=False)
print(data.tail())
