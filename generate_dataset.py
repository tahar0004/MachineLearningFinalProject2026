import pandas as pd
import random

data = []

for i in range(1000):

    temperature = random.randint(10, 45)
    humidity = random.randint(20, 100)
    wind_speed = random.randint(0, 40)

    if (humidity > 75 and wind_speed > 10 and temperature < 35) or humidity > 88:
        rain = 1
    else:
        rain = 0

    data.append([temperature, humidity, wind_speed, rain])

df = pd.DataFrame(
    data,
    columns=["Temperature", "Humidity", "Wind_Speed", "Rain"]
)

df.to_csv("weather_dataset.csv", index=False)

print("CSV file created successfully!")
