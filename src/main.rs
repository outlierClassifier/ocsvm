use actix_web::{get, post, web, App, HttpResponse, HttpServer, Responder};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};
use uuid::Uuid;

static mut START_TIME: Option<Instant> = None;

// Data structures based on API schema

#[derive(Deserialize)]
struct Signal {
    #[serde(rename = "fileName")]
    file_name: String,
    values: Vec<f64>,
    #[serde(default)]
    times: Vec<f64>,
    #[serde(default)]
    length: usize,
}

#[derive(Deserialize)]
struct Discharge {
    id: String,
    #[serde(default)]
    times: Vec<f64>,
    #[serde(default)]
    length: usize,
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
    epochs: Option<i32>,
    #[serde(default, rename = "batchSize")]
    batch_size: Option<i32>,
    #[serde(default)]
    hyperparameters: Option<serde_json::Value>,
}

#[derive(Deserialize)]
struct TrainingRequest {
    discharges: Vec<Discharge>,
    #[serde(default)]
    options: Option<TrainingOptions>,
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

// API endpoints

#[post("/predict")]
async fn predict(_req: web::Json<PredictionRequest>) -> impl Responder {
    // Simple mock response as requested (no actual SVM logic)
    let response = PredictionResponse {
        prediction: 1,
        confidence: 0.95,
        execution_time_ms: 123.0,
        model: "svm".to_string(),
        details: serde_json::json!({
            "featureImportance": [0.3, 0.2, 0.5]
        }),
    };
    
    HttpResponse::Ok().json(response)
}

#[post("/train")]
async fn train(_req: web::Json<TrainingRequest>) -> impl Responder {
    // Simple mock response as requested (no actual training logic)
    let response = TrainingResponse {
        status: "success".to_string(),
        message: "Entrenamiento completado con éxito".to_string(),
        training_id: format!("train_{}", Uuid::new_v4()),
        metrics: TrainingMetrics {
            accuracy: 0.95,
            loss: 0.12,
            f1_score: 0.94,
        },
        execution_time_ms: 15000.0,
    };
    
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
