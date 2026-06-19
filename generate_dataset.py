import pandas as pd
import numpy as np

rows = 600

temperature = np.random.randint(15, 45, rows)
humidity = np.random.randint(20, 100, rows)
wind = np.random.randint(0, 40, rows)

rain = []

for t, h, w in zip(temperature, humidity, wind):

    score = 0

    if h > 65:
        score += 1

    if t < 28:
        score += 1

    if w > 20:
        score += 1

    if np.random.rand() > 0.85:
        score += np.random.randint(-1, 2)

    rain.append(1 if score >= 2 else 0)

data = pd.DataFrame({
    "temperature": temperature,
    "humidity": humidity,
    "wind": wind,
    "rain": rain
})

data.to_csv("dataset.csv", index=False)

print("600 records generated")