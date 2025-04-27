use rustfft::{num_complex::Complex, FftPlanner};
use serde::Deserialize;

#[derive(Deserialize, Clone)]
pub struct Signal {
    pub label: String,
    pub _times: Vec<f64>,
    pub values: Vec<f64>,
    min: f64,
    max: f64,
}

#[derive(Debug)]
pub enum FeatureType {
    Mean,
    FftStd,
}

pub struct SignalFeatures {
    pub type_: FeatureType,
    pub values: Vec<f64>,
}

impl std::fmt::Debug for Signal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Signal {{ label: {}, min: {}, max: {} }}. Values length: {}",
            self.label,
            self.min,
            self.max,
            self.values.len()
        )
    }
}

impl std::fmt::Debug for SignalFeatures {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "SignalFeatures {{ type: {:?}, values: {:?} }}",
            self.type_, self.values
        )
    }
}

impl Signal {
    pub fn new(label: String, values: Vec<f64>) -> Signal {
        let min = values.iter().fold(f64::MAX, |a, b| a.min(*b));
        let max = values.iter().fold(f64::MIN, |a, b| a.max(*b));
        Signal {
            label,
            _times: Vec::new(),
            values,
            min,
            max,
        }
    }

    /// Crea señales a partir de archivos que coincidan con un patrón.
    /// Devuelve un vector con las señales leídas.
    pub fn _from_file_pattern(file_pattern: &str) -> Vec<Signal> {
        let mut signals = Vec::new();
        let paths = glob::glob(file_pattern).expect("Error reading file pattern");
        for path in paths {
            let path = path.unwrap();
            let file_path = path.to_str().unwrap();
            let signal = Signal::_from_file(file_path);
            signals.push(signal);
        }
        signals
    }

    /// Crea una señal a partir de un archivo.
    pub fn _from_file(file_path: &str) -> Signal {
        let mut times = Vec::new();
        let mut values = Vec::new();
        let mut min = f64::MAX;
        let mut max = f64::MIN;

        let file = std::fs::read_to_string(file_path).expect("Error reading file");
        for line in file.lines() {
            let parts: Vec<&str> = line.trim().split(" ").filter(|s| !s.is_empty()).collect();

            let time = parts[0].parse::<f64>().unwrap();
            let value = parts[1].parse::<f64>().unwrap();

            times.push(time);
            values.push(value);
            if value < min {
                min = value;
            }
            if value > max {
                max = value;
            }
        }

        Signal {
            label: file_path.to_string(),
            _times: times,
            values,
            min,
            max,
        }
    }

    pub fn _normalize(&mut self) {
        self.values
            .iter_mut()
            .for_each(|v| *v = (*v - self.min) / (self.max - self.min));
        self.min = self.values.iter().fold(f64::MAX, |a, b| a.min(*b));
        self.max = self.values.iter().fold(f64::MIN, |a, b| a.max(*b));
    }

    /// Recibe un vector de señales y las normaliza, según la fórmula:
    /// \[ x_{norm} = \frac{x - x_{min}}{x_{max} - x_{min}} \]
    /// donde \( x_{min} \) y \( x_{max} \) son los valores mínimo y máximo de todas las señales.
    /// Devuelve un vector con las señales normalizadas.
    pub fn normalize_vec(signals: Vec<Signal>) -> Vec<Signal> {
        let mut signals_norm = Vec::new();
        let global_min = signals
            .iter()
            .map(|s| s.min)
            .fold(f64::MAX, |a, b| a.min(b));
        let global_max = signals
            .iter()
            .map(|s| s.max)
            .fold(f64::MIN, |a, b| a.max(b));

        for signal in signals {
            let mut values_norm = Vec::new();
            for value in signal.values {
                let value_norm = (value - global_min) / (global_max - global_min);
                values_norm.push(value_norm);
            }
            let local_min = values_norm.iter().fold(f64::MAX, |a, b| a.min(*b));
            let local_max = values_norm.iter().fold(f64::MIN, |a, b| a.max(*b));
            signals_norm.push(Signal {
                label: signal.label,
                _times: signal._times,
                values: values_norm,
                min: local_min,
                max: local_max,
            });
        }

        signals_norm
    }

    /// Calcula las características de la señal en ventanas de tamaño `window_size`.
    /// Devuelve dos vectores con los valores medios y desviaciones estándar de la FFT.
    /// **Se asume que la señal ya está normalizada.**
    pub fn get_features(&self, window_size: usize) -> (SignalFeatures, SignalFeatures) {
        let num_windows = self.values.len() / window_size;

        // Vectores para almacenar resultados
        let mut mean_values = Vec::with_capacity(num_windows);
        let mut fft_std_values = Vec::with_capacity(num_windows);

        // Configurar FFT
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(window_size);

        for k in 0..num_windows {
            let start_idx = k * window_size;
            let end_idx = (k + 1) * window_size;
            let window = &self.values[start_idx..end_idx];

            // 1. Calcular valor medio de la ventana
            let mean_value = window.iter().sum::<f64>() / window_size as f64;
            mean_values.push(mean_value);

            // 2. Calcular FFT
            // Convertir a números complejos para la FFT
            let mut fft_input: Vec<Complex<f64>> =
                window.iter().map(|&x| Complex::new(x, 0.0)).collect();

            fft.process(&mut fft_input);

            // Obtener magnitudes de frecuencias positivas (sin DC)
            // Solo necesitamos hasta window_size/2 por simetría
            let magnitudes: Vec<f64> = fft_input[1..=window_size / 2]
                .iter()
                .map(|c| c.norm())
                .collect();

            // Calcular desviación estándar de las magnitudes
            let fft_mean = magnitudes.iter().sum::<f64>() / magnitudes.len() as f64;
            let fft_variance = magnitudes
                .iter()
                .map(|&m| (m - fft_mean) * (m - fft_mean))
                .sum::<f64>()
                / magnitudes.len() as f64;

            let fft_std = fft_variance.sqrt();
            fft_std_values.push(fft_std);
        }

        (
            SignalFeatures {
                type_: FeatureType::Mean,
                values: mean_values,
            },
            SignalFeatures {
                type_: FeatureType::FftStd,
                values: fft_std_values,
            },
        )
    }

    pub fn _min(&self) -> f64 {
        self.min
    }

    pub fn _max(&self) -> f64 {
        self.max
    }
}
