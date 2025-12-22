import numpy as np
import pandas as pd

def generate_time_series(
    n_minutes: int = 1440,
    anomaly_windows: list = None,
    failure_window: tuple = None,
    random_seed=42
) -> pd.DataFrame:
    """
    generate synthetic multivariate time series data
    """
    np.random.seed(random_seed)
    
    #Time index
    time_index=pd.date_range(
        start="2025-01-01",
        periods=n_minutes,
        freq="T"
    )
    
    #Base normal behavior 
    cpu=np.clip(np.random.normal(0.4,0.05,n_minutes),0,1)
    memory= np.clip(cpu + np.random.normal(0.1,0.03,n_minutes),0,1)
    latency = np.random.normal(120, 15, n_minutes)
    temperature = np.random.normal(60, 4, n_minutes)
    error_rate = np.abs(np.random.normal(0.01, 0.005, n_minutes))
    
    #Inject point and contextual anomalies
    if anomaly_windows:
        for start, end in anomaly_windows:
            cpu[start:end] += np.random.uniform(0.3, 0.5)
            latency[start:end] += np.random.uniform(200, 400)
            temperature[start:end] += np.random.uniform(10, 15)
            error_rate[start:end] += np.random.uniform(0.05, 0.1)
    
    #Inject gradual failure degradation
    if failure_window:
        f_start, f_end = failure_window
        degradation = np.linspace(0, 1, f_end - f_start)

        cpu[f_start:f_end] += degradation * 0.5
        memory[f_start:f_end] += degradation * 0.4
        latency[f_start:f_end] += degradation * 600
        temperature[f_start:f_end] += degradation * 25
        error_rate[f_start:f_end] += degradation * 0.15

    #Clip final values
    cpu = np.clip(cpu, 0, 1)
    memory = np.clip(memory, 0, 1)
    error_rate = np.clip(error_rate, 0, 1)
    
    
    # Create DF
    df = pd.DataFrame({
        "timestamp": time_index,
        "cpu": cpu,
        "memory": memory,
        "latency": latency,
        "temperature": temperature,
        "error_rate": error_rate
    })
    
    return df