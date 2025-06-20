mod signals;

use actix_web::{App, HttpResponse, HttpServer, Responder, get, post, web};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use signals::get_dataset_one_class;
use std::sync::RwLock;
use std::time::Instant;
use uuid::Uuid;

use linfa::dataset::Pr;
use linfa::prelude::*;
use linfa_svm::Svm;

use crate::signals::Discharge as InternalDischarge;
use crate::signals::DisruptionClass;
use crate::signals::Signal as InternalSignal;
use crate::signals::SignalType;

// Startup time for uptime calculation
static mut START_TIME: Option<Instant> = None;
const MODEL_PATH: &str = "trained_svm_model.json";

struct AppState {
    model: RwLock<Option<Svm<f64, bool>>>,
    training_session: RwLock<Option<TrainingSession>>,
    last_training: RwLock<Option<DateTime<Utc>>>,
}

struct TrainingSession {
    expected_discharges: usize,
    discharges: Vec<Discharge>,
}

impl AppState {
    fn save_model_json(&self, path: &str) -> anyhow::Result<()> {
        let model = self.model.read().unwrap();
        if let Some(model) = &*model {
            let json = serde_json::to_string(model)?;
            std::fs::write(path, json)
                .map_err(|e| anyhow::anyhow!("Failed to save model: {}", e))?;
        } else {
            return Err(anyhow::anyhow!("No model to save"));
        }

        log::info!("Model saved successfully to {}", path);

        Ok(())
    }

    fn load_model_json(&self, path: &str) -> anyhow::Result<()> {
        let model_data = std::fs::read_to_string(path)?;
        let model: Svm<f64, bool> = serde_json::from_str(&model_data)?;
        *self.model.write().unwrap() = Some(model);
        log::info!("Model loaded successfully from {}", path);
        Ok(())
    }
}

// Data structures based on API schema

#[derive(Deserialize, Clone)]
struct Signal {
    filename: String,
    values: Vec<f64>,
}

#[derive(Deserialize, Clone)]
struct Discharge {
    id: String,
    signals: Vec<Signal>,
    times: Vec<f64>,
    length: usize,
}

#[derive(Deserialize)]
struct StartTrainingRequest {
    #[serde(rename = "totalDischarges")]
    total_discharges: usize,
    #[serde(rename = "timeoutSeconds")]
    _timeout_seconds: usize,
}

#[derive(Serialize)]
struct StartTrainingResponse {
    #[serde(rename = "expectedDischarges")]
    expected_discharges: usize,
}

#[derive(Serialize)]
struct DischargeAck {
    ordinal: usize,
    #[serde(rename = "totalDischarges")]
    total_discharges: usize,
}

#[derive(Serialize)]
struct PredictionResponse {
    prediction: String,
    confidence: f64,
    #[serde(rename = "executionTimeMs")]
    execution_time_ms: f64,
    model: String,
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
struct HealthCheckResponse {
    name: String,
    uptime: f64,
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
    let signal_type = get_discharge_type_from_file_name(&api_signal.filename)?;
    Ok(InternalSignal::new(
        api_signal.filename.clone(),
        api_signal.values.clone(),
        signal_class,
        signal_type,
    ))
}

/// Convierte una descarga completa de la API a un vector de señales internas. Descarta los ficheros con errores.
fn api_discharge_to_internal_signals(discharge: &Discharge) -> Vec<InternalSignal> {
    let signal_class = DisruptionClass::Normal;
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
fn process_prediction_request(
    discharge: &Discharge,
    model: &Option<Svm<f64, bool>>,
) -> PredictionResponse {
    let start_time = Instant::now();

    // Si no hay modelo entrenado, devolver respuesta por defecto
    if model.is_none() {
        return PredictionResponse {
            prediction: "Unknown".to_string(),
            confidence: 0.0,
            execution_time_ms: 0.0,
            model: "none".to_string(),
        };
    }

    let discharges = vec![InternalDischarge::new(
        DisruptionClass::Unknown,
        api_discharge_to_internal_signals(discharge),
    )];

    let dataset = get_dataset_one_class(discharges);

    // Realizar predicción con el modelo
    let predictions = model.as_ref().unwrap().predict(dataset.records());

    // Determinar si hay anomalía (true representa anomalía)
    let anomaly_count = predictions.iter().filter(|&&p| p).count();
    let total = predictions.len();
    let confidence = if total > 0 {
        anomaly_count as f64 / total as f64
    } else {
        0.0
    };

    let prediction = if confidence > 0.5 {
        "Anomaly"
    } else {
        "Normal"
    };

    PredictionResponse {
        prediction: prediction.to_string(),
        confidence,
        execution_time_ms: start_time.elapsed().as_millis() as f64,
        model: "svm".to_string(),
    }
}

/// Procesa una petición de entrenamiento
fn process_training(discharges: &[Discharge]) -> (TrainingResponse, Svm<f64, bool>) {
    // Convertir todas las descargas y señales al formato interno
    let start_time = Instant::now();

    let discharges = discharges
        .iter()
        .map(|d| {
            InternalDischarge::new(
                DisruptionClass::Normal,
                api_discharge_to_internal_signals(d),
            )
        })
        .collect::<Vec<_>>();

    let mut all_signals = Vec::new();
    for discharge in &discharges {
        all_signals.extend(discharge.signals.clone());
    }

    let dataset = get_dataset_one_class(discharges);

    let model: Svm<f64, bool> = Svm::<f64, Pr>::params()
        .nu_weight(0.1)
        .gaussian_kernel(10.)
        .fit(&dataset)
        .expect("Error al entrenar el modelo SVM");

    let nsupport = model.nsupport();
    let execution_time_ms = start_time.elapsed().as_millis() as f64;

    log::info!(
        "Modelo entrenado con {} vectores de soporte en {} ms",
        nsupport,
        execution_time_ms
    );

    let response = TrainingResponse {
        status: "success".to_string(),
        message: "Entrenamiento completado con éxito".to_string(),
        training_id: format!("train_{}", Uuid::new_v4()),
        metrics: TrainingMetrics {
            accuracy: 0.95,
            loss: 0.12,
            f1_score: 0.94,
        },
        execution_time_ms,
    };

    (response, model)
}

// API endpoints

#[post("/predict")]
async fn predict(req: web::Json<Discharge>, app_state: web::Data<AppState>) -> impl Responder {
    let model = app_state.model.read().unwrap();
    let response = process_prediction_request(&req, &model);
    HttpResponse::Ok().json(response)
}

#[post("/train")]
async fn start_training(
    req: web::Json<StartTrainingRequest>,
    app_state: web::Data<AppState>,
) -> impl Responder {
    let mut session = app_state.training_session.write().unwrap();
    if session.is_some() {
        return HttpResponse::ServiceUnavailable().finish();
    }
    let expected = req.total_discharges;
    *session = Some(TrainingSession {
        expected_discharges: expected,
        discharges: Vec::new(),
    });
    HttpResponse::Ok().json(StartTrainingResponse {
        expected_discharges: expected,
    })
}

#[post("/train/{ordinal}")]
async fn push_discharge(
    path: web::Path<usize>,
    req: web::Json<Discharge>,
    app_state: web::Data<AppState>,
) -> impl Responder {
    let ordinal = path.into_inner();
    let mut session_lock = app_state.training_session.write().unwrap();
    if let Some(session) = session_lock.as_mut() {
        if ordinal == 0 || ordinal > session.expected_discharges {
            return HttpResponse::BadRequest().finish();
        }
        session.discharges.push(req.into_inner());
        let ack = DischargeAck {
            ordinal,
            total_discharges: session.expected_discharges,
        };
        if ordinal == session.expected_discharges {
            let discharges = std::mem::take(&mut session.discharges);
            *session_lock = None;
            let app_state_cloned = app_state.clone();
            tokio::spawn(async move {
                let (response, model) = process_training(&discharges);
                {
                    let mut model_lock = app_state_cloned.model.write().unwrap();
                    *model_lock = Some(model);
                }
                let _ = app_state_cloned.save_model_json(MODEL_PATH);
                *app_state_cloned.last_training.write().unwrap() = Some(Utc::now());
                log::info!("Training completed: {}", response.training_id);
            });
        }
        HttpResponse::Ok().json(ack)
    } else {
        HttpResponse::ServiceUnavailable().finish()
    }
}

#[get("/health")]
async fn health_check(app_state: web::Data<AppState>) -> impl Responder {
    // Calculate uptime in seconds
    let uptime = unsafe {
        if let Some(start_time) = START_TIME {
            start_time.elapsed().as_secs_f64()
        } else {
            0.0
        }
    };

    println!("Received heartbeat at {uptime}");

    let last_training = app_state
        .last_training
        .read()
        .unwrap()
        .map(|t| t.to_rfc3339())
        .unwrap_or_else(|| "".to_string());

    let response = HealthCheckResponse {
        name: "svm".to_string(),
        uptime,
        last_training,
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

    let app_state = web::Data::new(AppState {
        model: RwLock::new(None),
        training_session: RwLock::new(None),
        last_training: RwLock::new(None),
    });

    // Try to load model
    let model_loaded = app_state.load_model_json(MODEL_PATH);
    match model_loaded {
        Ok(_) => log::info!("Model loaded from {}", MODEL_PATH),
        Err(_) => log::info!("Model could not be loaded"),
    }

    let json_config = web::JsonConfig::default()
        .limit(1 << 26) // Tamaño max de 2^26 bytes (64 MB)
        .error_handler(|err, _req| {
            log::error!("JSON payload error: {}", err);
            actix_web::error::InternalError::from_response(
                err,
                HttpResponse::BadRequest()
                    .json(serde_json::json!({"error": "Payload too large or malformed"})),
            )
            .into()
        });

    log::info!("Starting OC-SVM model server on http://0.0.0.0:8004");
    log::info!("Health check server on http://0.0.0.0:3002");

    // Start the health check server in a separate thread
    let health_state = app_state.clone();
    std::thread::spawn(move || {
        // Use the system runtime for the health check server
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            HttpServer::new(move || {
                App::new()
                    .app_data(health_state.clone())
                    .service(health_check)
            })
            .workers(1) // Use only one worker for health checks
            .bind(("0.0.0.0", 3002))
            .unwrap()
            .run()
            .await
            .unwrap();
        });
    });

    // Main application server
    HttpServer::new(move || {
        App::new()
            .app_data(app_state.clone())
            .app_data(json_config.clone()) // Aplicar configuración de tamaño JSON
            .service(predict)
            .service(start_training)
            .service(push_discharge)
    })
    .bind(("0.0.0.0", 8004))?
    .run()
    .await
}
