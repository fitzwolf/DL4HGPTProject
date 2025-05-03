import os
import numpy as np
from tqdm import tqdm

def simulate_hirid_patient_realistic_with_labels(seed=None, out_label=False):
    if seed is not None:
        np.random.seed(seed)

    duration_days = np.random.randint(1, 15)
    timesteps = duration_days * 24 * 12  # 5-min intervals

    # Vitals initialization
    heart_rate = np.random.normal(80, 10, timesteps)
    map_pressure = np.random.normal(75, 8, timesteps)
    respiratory_rate = np.random.normal(18, 3, timesteps)
    spo2 = np.random.normal(97, 2, timesteps)
    temperature = np.random.normal(37.0, 0.5, timesteps)
    lactate = np.random.normal(1.2, 0.4, timesteps)
    creatinine = np.random.normal(1.0, 0.3, timesteps)
    glucose = np.random.normal(110, 20, timesteps)
    vasopressor_rate = np.zeros(timesteps)
    mech_ventilation = np.zeros(timesteps)
    label = 0  # Default healthy

    if np.random.rand() < 0.4:
        shock_start = np.random.randint(timesteps // 4, timesteps // 2)
        shock_duration = np.random.randint(100, 500)
        shock_duration = min(shock_duration, timesteps - shock_start)
        heart_rate[shock_start:shock_start+shock_duration] += np.random.normal(10, 3, shock_duration)
        map_pressure[shock_start:shock_start+shock_duration] -= np.random.normal(10, 3, shock_duration)
        lactate[shock_start:shock_start+shock_duration] += np.random.normal(2, 0.5, shock_duration)
        vasopressor_rate[shock_start:shock_start+shock_duration] = np.random.uniform(0.05, 0.2)
        label = 1  # Shock

    if np.random.rand() < 0.4 and timesteps > 1000:
        rf_start = np.random.randint(timesteps // 2, timesteps - 500)
        rf_duration = np.random.randint(100, 400)
        rf_duration = min(rf_duration, timesteps - rf_start)
        spo2[rf_start:rf_start+rf_duration] -= np.random.normal(10, 3, rf_duration)
        respiratory_rate[rf_start:rf_start+rf_duration] += np.random.normal(5, 2, rf_duration)
        mech_ventilation[rf_start:rf_start+rf_duration] = 1
        label = 2  # Respiratory Failure

    drift = np.cumsum(np.random.normal(0, 0.01, timesteps))
    heart_rate += drift
    map_pressure += np.random.normal(0, 0.5, timesteps)
    respiratory_rate += np.random.normal(0, 0.5, timesteps)

    # Clamp values
    heart_rate = np.clip(heart_rate, 30, 200).astype(int)
    map_pressure = np.clip(map_pressure, 30, 130).astype(int)
    respiratory_rate = np.clip(respiratory_rate, 5, 50).astype(int)
    spo2 = np.clip(spo2, 70, 100).astype(int)
    temperature = np.clip(temperature, 35.0, 41.0)
    lactate = np.clip(lactate, 0.5, 15.0)
    creatinine = np.clip(creatinine, 0.2, 5.0)
    glucose = np.clip(glucose, 40, 400).astype(int)
    vasopressor_rate = np.round(vasopressor_rate, 3)
    mech_ventilation = mech_ventilation.astype(int)

    patient = np.stack([
        heart_rate,
        map_pressure,
        respiratory_rate,
        spo2,
        temperature,
        lactate,
        creatinine,
        glucose,
        vasopressor_rate,
        mech_ventilation
    ], axis=0)

    # Missing values
    missing_rate = np.random.uniform(0.05, 0.15)
    missing_mask = np.random.rand(*patient.shape) < missing_rate
    patient[missing_mask] = 0

    if np.random.rand() < 0.1:
        for _ in range(np.random.randint(1, 3)):
            ch = np.random.randint(0, patient.shape[0])
            start = np.random.randint(0, max(1, patient.shape[1] - 200))
            end = min(start + np.random.randint(50, 200), patient.shape[1])
            patient[ch, start:end] = 0

    return (patient, label) if out_label else patient

def simulate_longevity_score(patient_signal):
    mean_vitals = np.mean(np.abs(patient_signal[0:4, :]))
    return mean_vitals + np.random.normal(0, 0.1)

def simulate_hirid_dataset(save_dir, num_runs=1, patients_per_run=1000):
    os.makedirs(save_dir, exist_ok=True)
    for run_idx in range(num_runs):
        print(f'Generating run {run_idx+1}/{num_runs}...')
        patients, labels, longevity_scores = [], [], []

        for _ in tqdm(range(patients_per_run)):
            patient, label = simulate_hirid_patient_realistic_with_labels(out_label=True)
            patients.append(patient)
            labels.append(label)
            longevity_scores.append(simulate_longevity_score(patient))

        max_length = max(p.shape[1] for p in patients)
        padded = [np.pad(p, ((0, 0), (0, max_length - p.shape[1])), 'constant') for p in patients]
        X_run = np.stack(padded)
        y_run = np.array(labels)
        lon_labels = (np.array(longevity_scores) > np.median(longevity_scores)).astype(int)

        run_folder = os.path.join(save_dir, f'run_{run_idx}')
        os.makedirs(run_folder, exist_ok=True)
        np.save(os.path.join(run_folder, 'X.npy'), X_run)
        np.save(os.path.join(run_folder, 'y.npy'), y_run)
        np.save(os.path.join(run_folder, 'longevity_labels.npy'), lon_labels)
        print(f'Saved run {run_idx}: X shape {X_run.shape}, y shape {y_run.shape}, longevity shape {lon_labels.shape}')
