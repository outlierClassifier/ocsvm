mod signals;

use actix_web::{App, HttpResponse, HttpServer, Responder, get, post, web};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use signals::get_dataset;
use std::time::Instant;
use uuid::Uuid;

use linfa::prelude::*;
use linfa_svm::Svm;

use crate::signals::Discharge as InternalDischarge;
use crate::signals::DisruptionClass;
use crate::signals::Signal as InternalSignal;
use crate::signals::SignalType;

// Startup time for uptime calculation
static mut START_TIME: Option<Instant> = None;

// Data structures based on API schema

#[derive(Deserialize)]
struct Signal {
    #[serde(rename = "fileName")]
    file_name: String,
    values: Vec<f64>,
    #[serde(default)]
    _times: Vec<f64>,
    #[serde(default)]
    _length: usize,
}

#[derive(Deserialize)]
struct Discharge {
    id: String,
    #[serde(default)]
    times: Vec<f64>,
    #[serde(default)]
    _length: usize,
    #[serde(default, rename = "anomalyTime")]
    anomaly_time: Option<f64>,
    signals: Vec<Signal>,
}

#[derive(Deserialize)]
struct PredictionRequest {
    discharges: Vec<Discharge>,
}

#[derive(Serialize)]
struct PredictionResponse {
    prediction: i32,
    confidence: f64,
    #[serde(rename = "executionTimeMs")]
    execution_time_ms: f64,
    model: String,
    details: serde_json::Value,
}

#[derive(Deserialize)]
struct TrainingOptions {
    #[serde(default)]
    _epochs: Option<i32>,
    #[serde(default, rename = "batchSize")]
    _batch_size: Option<i32>,
    #[serde(default)]
    _hyperparameters: Option<serde_json::Value>,
}

#[derive(Deserialize)]
struct TrainingRequest {
    discharges: Vec<Discharge>,
    #[serde(default)]
    _options: Option<TrainingOptions>,
}

#[derive(Serialize)]
struct TrainingMetrics {
    accuracy: f64,
    loss: f64,
    #[serde(rename = "f1Score")]
    f1_score: f64,
}

#[derive(Serialize)]
struct TrainingResponse {
    status: String,
    message: String,
    #[serde(rename = "trainingId")]
    training_id: String,
    metrics: TrainingMetrics,
    #[serde(rename = "executionTimeMs")]
    execution_time_ms: f64,
}

#[derive(Serialize)]
struct MemoryInfo {
    total: f64,
    used: f64,
}

#[derive(Serialize)]
struct HealthCheckResponse {
    status: String,
    version: String,
    uptime: f64,
    memory: MemoryInfo,
    load: f64,
    #[serde(rename = "lastTraining")]
    last_training: String,
}

// Adaptadores para convertir entre modelos de API y estructuras internas
fn get_discharge_type_from_file_name(file_name: &str) -> Result<SignalType, String> {
    // File name pattern: "DES_<discharge_id>_<signal_type>_r2_sliding.txt"

    let parts: Vec<&str> = file_name.split('_').collect();
    if parts.len() < 3 {
        return Err(format!("Invalid file name format: {}", file_name));
    }
    let signal_type_int = parts[2]
        .parse::<i32>()
        .map_err(|_| format!("Invalid signal type in file name: {}", file_name))?;

    let signal_type = match signal_type_int {
        1 => SignalType::CorrientePlasma,
        2 => SignalType::ModeLock,
        3 => SignalType::Inductancia,
        4 => SignalType::Densidad,
        5 => SignalType::DerivadaEnergiaDiamagnetica,
        6 => SignalType::PotenciaRadiada,
        7 => SignalType::PotenciaDeEntrada,
        _ => return Err(format!("Unknown signal type: {}", signal_type_int)),
    };

    println!(
        "Detected file name: {} with signal type: {:?}",
        file_name, signal_type
    );

    Ok(signal_type)
}

/// Convierte un objeto Signal de la API a la estructura InternalSignal
fn api_signal_to_internal(
    api_signal: &Signal,
    signal_class: DisruptionClass,
) -> Result<InternalSignal, String> {
    // Obtener el tipo de señal a partir del nombre del archivo
    let signal_type = get_discharge_type_from_file_name(&api_signal.file_name)?;
    Ok(InternalSignal::new(
        api_signal.file_name.clone(),
        api_signal.values.clone(),
        signal_class,
        signal_type,
    ))
}

/// Convierte una descarga completa de la API a un vector de señales internas. Descarta los ficheros con errores.
fn api_discharge_to_internal_signals(discharge: &Discharge) -> Vec<InternalSignal> {
    let signal_class = match discharge.anomaly_time {
        Some(_) => DisruptionClass::Anomaly,
        None => DisruptionClass::Normal,
    };
    discharge
        .signals
        .iter()
        .map(|signal| api_signal_to_internal(signal, signal_class.clone()))
        .filter_map(|result| match result {
            Ok(signal) => Some(signal),
            Err(err) => {
                log::error!("Error converting signal: {}. Skipping this signal", err);
                None
            }
        })
        .collect()
}

/// Procesa una petición de predicción
fn process_prediction_request(_request: &PredictionRequest) -> PredictionResponse {
    // Por ahora solo devolvemos una respuesta simulada
    // En el futuro, aquí integraremos la lógica real de SVM

    PredictionResponse {
        prediction: 1,
        confidence: 0.95,
        execution_time_ms: 123.0,
        model: "svm".to_string(),
        details: serde_json::json!({
            "featureImportance": [0.3, 0.2, 0.5]
        }),
    }
}

/// Procesa una petición de entrenamiento
fn process_training_request(request: &TrainingRequest) -> TrainingResponse {
    // Convertir todas las descargas y señales al formato interno
    let start_time = Instant::now();

    let discharges = request
        .discharges
        .iter()
        .map(|d| {
            InternalDischarge::new(
                d.id.clone(),
                match d.anomaly_time {
                    Some(_) => DisruptionClass::Anomaly,
                    None => DisruptionClass::Normal,
                },
                api_discharge_to_internal_signals(d),
            )
        })
        .collect::<Vec<_>>();

    let mut all_signals = Vec::new();
    for discharge in &discharges {
        all_signals.extend(discharge.signals.clone());
    }

    let dataset = get_dataset(discharges);

    let model: Svm<f64, bool> = Svm::<f64, bool>::params()
        .gaussian_kernel(10.)
        .pos_neg_weights(1.0, 1.0)
        .fit(&dataset)
        .expect("Error al entrenar el modelo SVM");

    let nsupport = model.nsupport();
    let execution_time_ms = start_time.elapsed().as_millis() as f64;

    log::info!(
        "Modelo entrenado con {} vectores de soporte en {} ms",
        nsupport,
        execution_time_ms
    );

    TrainingResponse {
        status: "success".to_string(),
        message: "Entrenamiento completado con éxito".to_string(),
        training_id: format!("train_{}", Uuid::new_v4()),
        metrics: TrainingMetrics {
            accuracy: 0.95,
            loss: 0.12,
            f1_score: 0.94,
        },
        execution_time_ms,
    }
}

// API endpoints

#[post("/predict")]
async fn predict(req: web::Json<PredictionRequest>) -> impl Responder {
    let response = process_prediction_request(&req);
    HttpResponse::Ok().json(response)
}

#[post("/train")]
async fn train(req: web::Json<TrainingRequest>) -> impl Responder {
    let response = process_training_request(&req);
    HttpResponse::Ok().json(response)
}

#[get("/health")]
async fn health_check() -> impl Responder {
    // Calculate uptime in seconds
    let uptime = unsafe {
        if let Some(start_time) = START_TIME {
            start_time.elapsed().as_secs_f64()
        } else {
            0.0
        }
    };

    // Current date-time in ISO format
    let now: DateTime<Utc> = Utc::now();

    let response = HealthCheckResponse {
        status: "online".to_string(),
        version: "1.0.0".to_string(),
        uptime,
        memory: MemoryInfo {
            total: 1024.0,
            used: 512.0,
        },
        load: 0.3,
        last_training: now.to_rfc3339(),
    };

    HttpResponse::Ok().json(response)
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    // Initialize logger
    env_logger::init_from_env(env_logger::Env::default().default_filter_or("info"));

    // Record start time for uptime calculations
    unsafe {
        START_TIME = Some(Instant::now());
    }

    log::info!("Starting SVM model server on http://0.0.0.0:8001");

    // Start the HTTP server
    HttpServer::new(|| {
        App::new()
            .service(predict)
            .service(train)
            .service(health_check)
    })
    .bind(("0.0.0.0", 8001))?
    .run()
    .await
}
