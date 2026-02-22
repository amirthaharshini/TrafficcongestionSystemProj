import pandas as pd
import numpy as np
import random
import os

random.seed(42)
rows = []
routes = ['anna-salai', 'omr', 'gst-road', 'ecr', 't-nagar']

for _ in range(50000):
    hour     = random.randint(0, 23)
    dow      = random.randint(0, 6)
    month    = random.randint(1, 12)
    weather  = random.choices([0,1,2,3], weights=[60,25,10,5])[0]
    vehicles = random.randint(100, 4000)
    incident = random.choices([0,1,2,3], weights=[80,10,7,3])[0]
    holiday  = random.choices([0,1], weights=[90,10])[0]

    score = 0
    if 7<=hour<=10 or 17<=hour<=20: score += 40
    elif 11<=hour<=14: score += 20
    score += weather * 12
    score += (vehicles / 4000) * 30
    score += incident * 10
    if holiday: score -= 10

    label = 0 if score < 35 else (1 if score < 65 else 2)

    rows.append([random.choice(routes), hour, dow, month,
                 weather, vehicles, incident, holiday, label])

os.makedirs("../data", exist_ok=True)

df = pd.DataFrame(rows, columns=[
    'route_id','hour','day_of_week','month',
    'weather_encoded','vehicle_count','incident_encoded',
    'is_holiday','congestion_level'
])

df.to_csv("../data/traffic_dataset.csv", index=False)
print("Dataset created! Shape:", df.shape)
print(df['congestion_level'].value_counts())