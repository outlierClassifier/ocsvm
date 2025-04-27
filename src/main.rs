mod sets;
mod signals;

use actix_web::{get, post, web, App, HttpResponse, HttpServer, Responder};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::time::Instant;
use uuid::Uuid;

use crate::signals::Signal as InternalSignal;
use crate::sets::{Set, SetCollection};

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

/// Convierte un objeto Signal de la API a la estructura InternalSignal
fn api_signal_to_internal(api_signal: &Signal, discharge_id: &str) -> InternalSignal {        
    InternalSignal::new(api_signal.file_name.clone(), api_signal.values.clone())
}

/// Convierte una descarga completa de la API a un vector de señales internas
fn api_discharge_to_internal_signals(discharge: &Discharge) -> Vec<InternalSignal> {
    discharge
        .signals
        .iter()
        .map(|signal| api_signal_to_internal(signal, &discharge.id))
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
    
    // Transformar todas las descargas en señales internas
    let all_signals: Vec<InternalSignal> = request
        .discharges
        .iter()
        .flat_map(api_discharge_to_internal_signals)
        .collect();
    
    log::info!("Procesando {} señales para entrenamiento", all_signals.len());
    
    // Normalizar las señales
    let normalized_signals = InternalSignal::normalize_vec(all_signals);
    
    // Extraer características de cada señal
    const WINDOW_SIZE: usize = 16;
    
    let mut feature_sets = SetCollection::new();
    
    for signal in &normalized_signals {
        // Extraer características de la señal
        let (mean_features, fft_features) = signal.get_features(WINDOW_SIZE);
        
        // Crear conjuntos a partir de las características
        let mean_set = Set::from(&mean_features);
        let fft_set = Set::from(&fft_features);
        
        // Añadir los conjuntos al colector
        feature_sets.add_set(mean_set);
        feature_sets.add_set(fft_set);
    }
    
    // TODO: Entrenar el modelo SVM con los conjuntos de características
    
    let execution_time_ms = start_time.elapsed().as_millis() as f64;
    
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
