"""
Phase 1: Synthetic Crash Detection Data Generator
Generates crash and normal driving scenarios using physics models.
"""

import numpy as np
import pandas as pd
import os


class SyntheticCrashGenerator:
    def __init__(self, sampling_rate=100, output_dir='data/raw'):
        self.sampling_rate = sampling_rate
        self.dt = 1.0 / sampling_rate
        self.output_dir = output_dir
        os.makedirs(f"{output_dir}/crashes", exist_ok=True)
        os.makedirs(f"{output_dir}/normal_driving", exist_ok=True)
        print(f"Generator ready | {sampling_rate} Hz | Output: {output_dir}/")

    def _base_signals(self, duration):
        n = int(duration * self.sampling_rate)
        t = np.linspace(0, duration, n)
        return t, n

    def _base_gps(self, n):
        return (
            30.0 + np.linspace(0, 0.001, n),
            31.0 + np.linspace(0, 0.001, n),
            50 + np.random.normal(0, 0.5, n)
        )

    def _make_record(self, t, ax, ay, az, gx, gy, gz, label):
        n = len(t)
        lat, lon, alt = self._base_gps(n)
        return {
            'timestamp': t, 'accel_x': ax, 'accel_y': ay, 'accel_z': az,
            'gyro_x': gx, 'gyro_y': gy, 'gyro_z': gz,
            'gps_lat': lat, 'gps_lon': lon, 'gps_alt': alt, 'label': label
        }

    # -- Crash Scenarios --

    def generate_frontal_crash(self, initial_speed=60, duration=5.0):
        """Frontal collision: sudden deceleration, peak 20-50g."""
        t, n = self._base_signals(duration)
        idx = int(2.0 * self.sampling_rate)
        w = int(0.1 * self.sampling_rate)

        ax = np.random.normal(0, 0.1, n)
        peak = -np.random.uniform(20, 50) * 9.81
        ax[idx:idx+w] = peak * np.exp(-np.linspace(0, 3, w))

        ay = np.random.normal(0, 2, n)
        az = np.random.normal(9.81, 1, n)
        az[idx:idx+w] += np.random.uniform(5, 15) * 9.81

        gx = np.random.normal(0, 0.05, n)
        gx[idx:idx+w*2] += np.random.uniform(2, 5) * np.exp(-np.linspace(0, 2, w*2))
        gy = np.random.normal(0, 0.02, n)
        gz = np.random.normal(0, 0.02, n)

        return self._make_record(t, ax, ay, az, gx, gy, gz, label=1)

    def generate_side_crash(self, initial_speed=50, duration=5.0):
        """Side collision (T-bone): strong lateral impact, 15-40g."""
        t, n = self._base_signals(duration)
        idx = int(2.0 * self.sampling_rate)
        w = int(0.15 * self.sampling_rate)

        ax = np.random.normal(0, 0.5, n)
        ay = np.random.normal(0, 0.5, n)
        ay[idx:idx+w] = np.random.uniform(15, 40) * 9.81 * np.exp(-np.linspace(0, 3, w))
        ax[idx:] += -np.random.uniform(5, 15) * 9.81 * np.exp(-np.linspace(0, 2, n - idx))

        az = np.random.normal(9.81, 1, n)
        az[idx:idx+w] += np.random.uniform(3, 8) * 9.81

        gx = np.random.normal(0, 0.05, n)
        gy = np.random.normal(0, 0.05, n)
        gy[idx:idx+w*2] += np.random.uniform(3, 7) * np.exp(-np.linspace(0, 2, w*2))
        gz = np.random.normal(0, 0.05, n)

        return self._make_record(t, ax, ay, az, gx, gy, gz, label=1)

    def generate_rollover(self, initial_speed=70, duration=6.0):
        """Vehicle rollover: oscillating forces, continuous rotation."""
        t, n = self._base_signals(duration)
        si = int(2.0 * self.sampling_rate)
        rs = int(2.0 * self.sampling_rate)

        ax = np.random.normal(0, 1, n)
        ay = np.random.normal(0, 1, n)
        az = np.random.normal(9.81, 1, n)

        for i in range(si, min(si + rs, n)):
            phase = (i - si) / rs * 2 * np.pi
            ax[i] += 10 * np.sin(phase * 2)
            ay[i] += 15 * np.sin(phase * 2 + np.pi / 2)
            az[i] = 9.81 * np.cos(phase) + np.random.uniform(-5, 5) * 9.81

        gx = np.random.normal(0, 0.1, n)
        gy = np.random.normal(0, 0.1, n)
        gy[si:si+rs] = 4 * np.pi * (1 + 0.2 * np.sin(np.linspace(0, 4 * np.pi, rs)))
        gz = np.random.normal(0, 0.1, n)

        return self._make_record(t, ax, ay, az, gx, gy, gz, label=1)

    # -- Normal / False-Positive Scenarios --

    def generate_normal_highway(self, duration=10.0):
        """Normal highway driving: smooth, low g-forces."""
        t, n = self._base_signals(duration)
        ax = np.gradient(5 * np.sin(0.5 * t), self.dt) + np.random.normal(0, 0.3, n)
        ay = np.random.normal(0, 0.2, n)
        az = np.random.normal(9.81, 0.5, n)
        gx = np.random.normal(0, 0.01, n)
        gy = np.random.normal(0, 0.01, n)
        gz = np.random.normal(0, 0.02, n)
        return self._make_record(t, ax, ay, az, gx, gy, gz, label=0)

    def generate_hard_brake(self, duration=5.0):
        """Hard braking: strong but controlled decel, 5-8g max."""
        t, n = self._base_signals(duration)
        idx = int(2.0 * self.sampling_rate)
        bd = int(0.5 * self.sampling_rate)

        ax = np.random.normal(0, 0.2, n)
        ax[idx:idx+bd] = -np.random.uniform(5, 8) * 9.81 * (1 - np.linspace(0, 1, bd))

        ay = np.random.normal(0, 0.3, n)
        az = np.random.normal(9.81, 0.8, n)

        gx = np.random.normal(0, 0.03, n)
        gx[idx:idx+bd] += 0.5 * np.exp(-np.linspace(0, 2, bd))
        gy = np.random.normal(0, 0.02, n)
        gz = np.random.normal(0, 0.02, n)

        return self._make_record(t, ax, ay, az, gx, gy, gz, label=0)

    def generate_pothole(self, duration=5.0):
        """Pothole impact: sharp vertical spike only, 3-6g."""
        t, n = self._base_signals(duration)
        idx = int(2.0 * self.sampling_rate)
        w = int(0.05 * self.sampling_rate)

        ax = np.random.normal(0, 0.2, n)
        ay = np.random.normal(0, 0.2, n)
        az = np.random.normal(9.81, 0.5, n)
        az[idx:idx+w] += np.random.uniform(3, 6) * 9.81

        gx = np.random.normal(0, 0.02, n)
        gy = np.random.normal(0, 0.02, n)
        gz = np.random.normal(0, 0.02, n)

        return self._make_record(t, ax, ay, az, gx, gy, gz, label=0)

    # -- Save and Run --

    def save_to_csv(self, data, category, scenario_name, index):
        folder = 'crashes' if category == 'crash' else 'normal_driving'
        filename = f"{scenario_name}_{index:03d}.csv"
        filepath = os.path.join(self.output_dir, folder, filename)
        pd.DataFrame(data).to_csv(filepath, index=False)
        return filepath

    def generate_dataset(self, samples_per_scenario=50):
        scenarios = {
            'crash': [
                ('frontal', self.generate_frontal_crash),
                ('side', self.generate_side_crash),
                ('rollover', self.generate_rollover),
            ],
            'normal': [
                ('highway', self.generate_normal_highway),
                ('hard_brake', self.generate_hard_brake),
                ('pothole', self.generate_pothole),
            ]
        }

        total = 0
        for cat_key, cat_name in [('crash', 'CRASH'), ('normal', 'NORMAL')]:
            print(f"\nGenerating {cat_name} scenarios...")
            for name, func in scenarios[cat_key]:
                for i in range(samples_per_scenario):
                    data = func()
                    self.save_to_csv(data, cat_key, name, i)
                    total += 1
                print(f"  {name}: {samples_per_scenario} files")

        crash_count = samples_per_scenario * len(scenarios['crash'])
        normal_count = samples_per_scenario * len(scenarios['normal'])
        print(f"\nDone! {total} files | {crash_count} crash + {normal_count} normal")
        print(f"Output: {self.output_dir}/")


def main():
    gen = SyntheticCrashGenerator(sampling_rate=100, output_dir='data/raw')
    gen.generate_dataset(samples_per_scenario=50)


if __name__ == '__main__':
    main()