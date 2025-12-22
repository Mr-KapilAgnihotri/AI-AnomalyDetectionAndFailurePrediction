from data_generator import generate_time_series

df = generate_time_series(
    n_minutes=2880,
    anomaly_windows=[(600, 650), (1200, 1250)],
    failure_window=(1800, 2100)
)

print(df.head())
