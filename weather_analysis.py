import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ============================================
# WEATHER CONDITION ANALYSIS USING PYTHON
# ============================================

# Create graphs folder automatically
os.makedirs("graphs", exist_ok=True)

# ============================================
# 1. LOAD DATASET
# ============================================

df = pd.read_csv("weather_data.csv")

print("=" * 60)
print("          WEATHER CONDITION ANALYSIS")
print("=" * 60)

# ============================================
# 2. DISPLAY FIRST 5 ROWS
# ============================================

print("\nFIRST 5 ROWS:")
print(df.head())

# ============================================
# 3. DATASET INFORMATION
# ============================================

print("\nDATASET INFORMATION:")
df.info()

# ============================================
# 4. DATASET SIZE
# ============================================

print("\nDATASET SHAPE:")
print(df.shape)

# ============================================
# 5. COLUMN NAMES
# ============================================

print("\nCOLUMN NAMES:")
print(df.columns.tolist())

# ============================================
# 6. CONVERT DATE
# ============================================

df["Date"] = pd.to_datetime(df["Date"])

# ============================================
# 7. CHECK MISSING VALUES
# ============================================

print("\nMISSING VALUES:")
print(df.isnull().sum())

# ============================================
# 8. CHECK DUPLICATE VALUES
# ============================================

print("\nDUPLICATE ROWS:")
print(df.duplicated().sum())

# ============================================
# 9. STATISTICAL SUMMARY
# ============================================

print("\nSTATISTICAL SUMMARY:")
print(df.describe())

# ============================================
# 10. AVERAGE TEMPERATURE
# ============================================

average_temperature = df["Temperature"].mean()

print("\nAVERAGE TEMPERATURE:")
print(round(average_temperature, 2), "°C")

# ============================================
# 11. HIGHEST TEMPERATURE
# ============================================

highest_temperature = df["Temperature"].max()

print("\nHIGHEST TEMPERATURE:")
print(highest_temperature, "°C")

# ============================================
# 12. LOWEST TEMPERATURE
# ============================================

lowest_temperature = df["Temperature"].min()

print("\nLOWEST TEMPERATURE:")
print(lowest_temperature, "°C")

# ============================================
# 13. AVERAGE HUMIDITY
# ============================================

average_humidity = df["Humidity"].mean()

print("\nAVERAGE HUMIDITY:")
print(round(average_humidity, 2), "%")

# ============================================
# 14. HIGHEST HUMIDITY
# ============================================

highest_humidity = df["Humidity"].max()

print("\nHIGHEST HUMIDITY:")
print(highest_humidity, "%")

# ============================================
# 15. LOWEST HUMIDITY
# ============================================

lowest_humidity = df["Humidity"].min()

print("\nLOWEST HUMIDITY:")
print(lowest_humidity, "%")

# ============================================
# 16. TOTAL RAINFALL
# ============================================

total_rainfall = df["Rainfall"].sum()

print("\nTOTAL RAINFALL:")
print(total_rainfall, "mm")

# ============================================
# 17. AVERAGE RAINFALL
# ============================================

average_rainfall = df["Rainfall"].mean()

print("\nAVERAGE DAILY RAINFALL:")
print(round(average_rainfall, 2), "mm")

# ============================================
# 18. AVERAGE WIND SPEED
# ============================================

average_wind = df["WindSpeed"].mean()

print("\nAVERAGE WIND SPEED:")
print(round(average_wind, 2), "km/h")

# ============================================
# 19. WEATHER CONDITION COUNT
# ============================================

weather_count = df["WeatherCondition"].value_counts()

print("\nWEATHER CONDITION COUNT:")
print(weather_count)

# ============================================
# 20. MOST COMMON WEATHER CONDITION
# ============================================

most_common_weather = df["WeatherCondition"].mode()[0]

print("\nMOST COMMON WEATHER CONDITION:")
print(most_common_weather)

# ============================================
# 21. COUNT WEATHER DAYS
# ============================================

sunny_days = (df["WeatherCondition"] == "Sunny").sum()
cloudy_days = (df["WeatherCondition"] == "Cloudy").sum()
rainy_days = (df["WeatherCondition"] == "Rainy").sum()

print("\nSUNNY DAYS:")
print(sunny_days)

print("\nCLOUDY DAYS:")
print(cloudy_days)

print("\nRAINY DAYS:")
print(rainy_days)

# ============================================
# 22. DAY WITH HIGHEST TEMPERATURE
# ============================================

max_temp_row = df.loc[df["Temperature"].idxmax()]

print("\nDAY WITH HIGHEST TEMPERATURE:")
print(max_temp_row["Date"].date())
print("Temperature:", max_temp_row["Temperature"], "°C")

# ============================================
# 23. DAY WITH LOWEST TEMPERATURE
# ============================================

min_temp_row = df.loc[df["Temperature"].idxmin()]

print("\nDAY WITH LOWEST TEMPERATURE:")
print(min_temp_row["Date"].date())
print("Temperature:", min_temp_row["Temperature"], "°C")

# ============================================
# 24. CORRELATION ANALYSIS
# ============================================

correlation = df[
    ["Temperature", "Humidity", "WindSpeed", "Rainfall"]
].corr()

print("\nCORRELATION MATRIX:")
print(correlation)

# ============================================
#             VISUALIZATION
# ============================================

# ============================================
# GRAPH 1 - TEMPERATURE TREND
# ============================================

plt.figure(figsize=(12, 5))

plt.plot(
    df["Date"],
    df["Temperature"],
    marker="o",
    color="blue",
    linewidth=2,
    markerfacecolor="red",
    markeredgecolor="black"
)

# Date labels on every dot
for i in range(len(df)):
    plt.annotate(
        df["Date"].iloc[i].strftime("%d-%m-%Y"),
        (df["Date"].iloc[i], df["Temperature"].iloc[i]),
        textcoords="offset points",
        xytext=(0, 10),
        ha="center",
        fontsize=8,
        color="black"
    )

plt.title("Temperature Trend")
plt.xlabel("Date")
plt.ylabel("Temperature (°C)")
plt.xticks(rotation=45)
plt.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig("graphs/temperature_trend.png")
plt.show()


# ============================================
# GRAPH 2 - WEATHER CONDITION
# ============================================

plt.figure(figsize=(7, 5))

weather_count.plot(
    kind="bar",
    color=["orange", "skyblue", "green"]
)

plt.title("Weather Condition Frequency")
plt.xlabel("Weather Condition")
plt.ylabel("Number of Days")

plt.tight_layout()
plt.savefig("graphs/weather_condition.png")
plt.show()


# ============================================
# GRAPH 3 - RAINFALL
# ============================================

plt.figure(figsize=(10, 5))

plt.bar(
    df["Date"],
    df["Rainfall"],
    color="skyblue",
    edgecolor="blue"
)

plt.title("Daily Rainfall")
plt.xlabel("Date")
plt.ylabel("Rainfall (mm)")

plt.xticks(
    df["Date"],
    df["Date"].dt.strftime("%d-%m-%Y"),
    rotation=45
)

plt.tight_layout()
plt.savefig("graphs/rainfall.png")
plt.show()


# ============================================
# GRAPH 4 - HUMIDITY
# ============================================
plt.figure(figsize=(12, 5))

plt.plot(
    df["Date"],
    df["Humidity"],
    marker="o",
    color="green",
    linewidth=2,
    markerfacecolor="yellow",
    markeredgecolor="black"
)

# Date labels on every dot
for i in range(len(df)):
    plt.annotate(
        df["Date"].iloc[i].strftime("%d-%m-%Y"),
        (df["Date"].iloc[i], df["Humidity"].iloc[i]),
        textcoords="offset points",
        xytext=(0, 10),
        ha="center",
        fontsize=8,
        color="black"
    )

plt.title("Humidity Trend")
plt.xlabel("Date")
plt.ylabel("Humidity (%)")

plt.xticks(
    df["Date"],
    df["Date"].dt.strftime("%d-%m-%Y"),
    rotation=45
)

plt.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig("graphs/humidity.png")
plt.show()


# ============================================
# GRAPH 5 - WIND SPEED
# ============================================
plt.figure(figsize=(12, 5))

plt.plot(
    df["Date"],
    df["WindSpeed"],
    marker="o",
    color="purple",
    linewidth=2,
    markerfacecolor="pink",
    markeredgecolor="black"
)

# Date labels on every dot
for i in range(len(df)):
    plt.annotate(
        df["Date"].iloc[i].strftime("%d-%m-%Y"),
        (df["Date"].iloc[i], df["WindSpeed"].iloc[i]),
        textcoords="offset points",
        xytext=(0, 10),
        ha="center",
        fontsize=8,
        color="black"
    )

plt.title("Wind Speed Trend")
plt.xlabel("Date")
plt.ylabel("Wind Speed (km/h)")

plt.xticks(
    df["Date"],
    df["Date"].dt.strftime("%d-%m-%Y"),
    rotation=45
)

plt.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig("graphs/wind_speed.png")
plt.show()

# ============================================
# GRAPH 6 - TEMPERATURE DISTRIBUTION
# ============================================

plt.figure(figsize=(8, 5))

plt.hist(
    df["Temperature"],
    bins=8,
    color="orange",
    edgecolor="black"
)

plt.title("Temperature Distribution")
plt.xlabel("Temperature (°C)")
plt.ylabel("Frequency")

plt.tight_layout()
plt.savefig("graphs/temperature_distribution.png")
plt.show()


# ============================================
# GRAPH 7 - TEMPERATURE VS RAINFALL
# ============================================

plt.figure(figsize=(8, 5))

plt.scatter(
    df["Temperature"],
    df["Rainfall"],
    color="red",
    marker="o",
    s=80,
    edgecolor="black"
)

plt.title("Temperature vs Rainfall")
plt.xlabel("Temperature (°C)")
plt.ylabel("Rainfall (mm)")

plt.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig("graphs/temp_vs_rainfall.png")
plt.show()


# ============================================
# FINAL MESSAGE
# ============================================

print("\n" + "=" * 60)
print("       WEATHER ANALYSIS COMPLETED")
print("=" * 60)