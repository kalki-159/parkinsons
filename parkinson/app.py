import os
import time
import tempfile
import traceback
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# Import our custom modules
from extract_features import HandFeatureExtractor
from model_inference import predict_parkinson, get_model

ROOT = Path(__file__).resolve().parent

app = FastAPI(title="CogniVue - Parkinson's Detection API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _pick_existing(*names: str) -> Path | None:
    for name in names:
        p = ROOT / name
        if p.exists():
            return p
    return None


def _get_index_html() -> Path | None:
    return _pick_existing("index-full.html", "index.html", "parkinson.html")


@app.get("/")
def serve_index():
    index_html = _get_index_html()
    if not index_html:
        raise HTTPException(status_code=404, detail="No index.html found")
    return FileResponse(str(index_html))


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "service": "CogniVue Parkinson's Detection",
        "version": "2.0",
        "index_file": (_get_index_html().name if _get_index_html() else None),
        "cwd": str(ROOT),
        "has_model_h5": (ROOT / "model.h5").exists(),
        "has_scaler": (ROOT / "scaler.pkl").exists(),
        "has_label_encoder": (ROOT / "label_encoder.pkl").exists(),
        "has_hand_model": (ROOT / "hand_landmarker.task").exists(),
    }


@app.post("/predict-video")
async def predict_video(video: UploadFile = File(...), source: str = "upload") -> JSONResponse:
    """
    Process uploaded video and return Parkinson's risk assessment.
    
    Pipeline:
    1. Save uploaded video to temp file
    2. Extract hand features using MediaPipe Hands
    3. Make prediction using ML model
    4. Return combined results
    """
    t0 = time.perf_counter()

    if not video or not video.filename:
        raise HTTPException(status_code=400, detail="Missing video file")

    # Validate file type
    allowed_types = {"video/mp4", "video/webm", "video/quicktime", "video/x-msvideo", "video/mov", "video/avi"}
    content_type = video.content_type or ""
    
    video_extensions = {".mp4", ".webm", ".mov", ".avi", ".mkv"}
    file_ext = Path(video.filename).suffix.lower()
    
    if content_type not in allowed_types and file_ext not in video_extensions:
        return JSONResponse({
            "success": False,
            "error": "Unsupported video format",
            "message": f"Format '{content_type or file_ext}' is not supported.",
            "suggestion": "Please upload MP4, WebM, MOV, or AVI format video.",
        }, status_code=400)

    try:
        content = await video.read()
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": "Could not read video file",
            "message": f"Error reading upload: {str(e)}",
            "suggestion": "Please try uploading the video again.",
        }, status_code=400)

    if not content:
        return JSONResponse({
            "success": False,
            "error": "Empty video file",
            "message": "The uploaded file appears to be empty.",
            "suggestion": "Please select a valid video file.",
        }, status_code=400)

    # Save to temp file for processing
    temp_video_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
            tmp.write(content)
            temp_video_path = tmp.name

        # Step 1: Extract hand features using MediaPipe
        try:
            extractor = HandFeatureExtractor(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            
            feature_result = extractor.extract_features_from_video(temp_video_path)
        except FileNotFoundError as e:
            return JSONResponse({
                "success": False,
                "error": "MediaPipe model not found",
                "message": str(e),
                "suggestion": "Please ensure hand_landmarker.task file exists in the project directory.",
            }, status_code=500)
        except Exception as e:
            error_trace = traceback.format_exc()
            print(f"Hand detection error: {error_trace}")
            return JSONResponse({
                "success": False,
                "error": "Hand detection failed",
                "message": f"Error during hand detection: {str(e)}",
                "suggestion": "Try with a different video or ensure the video shows hands clearly.",
            }, status_code=500)
        
        # Check if hands were detected
        if not feature_result.get('hand_detected', False):
            return JSONResponse({
                "success": False,
                "error": "No hands detected",
                "message": "We couldn't detect any hands in your video.",
                "suggestion": "Please record/upload a video that clearly shows your hands in the frame. Hold your hands steady for 5-15 seconds in front of the camera.",
                "frames_with_hands": feature_result.get('frames_with_hands', 0),
                "total_frames": feature_result.get('total_frames', 0),
                "video_duration_seconds": round(feature_result.get('video_duration_seconds', 0), 1),
            }, status_code=422)

        # Check video quality (lowered threshold for videos with intermittent hand visibility)
        detection_ratio = feature_result.get('hand_detection_ratio', 0)
        if detection_ratio < 0.05:
            return JSONResponse({
                "success": False,
                "error": "Low hand visibility",
                "message": "Hands were only visible in a very small portion of the video.",
                "suggestion": "Ensure good lighting and keep hands clearly visible throughout the recording. Try to minimize camera movement.",
                "detection_ratio": round(detection_ratio * 100, 1),
            }, status_code=422)

        # Step 2: Make prediction using ML model or rule-based fallback
        try:
            prediction_result = predict_parkinson(feature_result)
        except Exception as e:
            # If model prediction fails, use rule-based calculation
            print(f"Model prediction error: {e}")
            from model_inference import calculate_rule_based_risk
            prediction_result = calculate_rule_based_risk(feature_result)

        # Calculate inference time
        inference_ms = round((time.perf_counter() - t0) * 1000, 2)

        # Count hands detected
        hand_count = sum([
            feature_result.get('left_hand_detected', False),
            feature_result.get('right_hand_detected', False)
        ])

        # Combine results
        return JSONResponse({
            "success": True,
            "label": prediction_result.get('prediction', 'Unknown'),
            "risk_score": prediction_result.get('risk_score', 0),
            "confidence": prediction_result.get('confidence', 0),
            "risk_category": prediction_result.get('risk_category', 'Unknown'),
            "tremor_frequency_hz": feature_result.get('tremor_frequency_hz', 0),
            "amplitude_mm": feature_result.get('amplitude_mm', 0),
            "rhythm_consistency": feature_result.get('rhythm_consistency', 0),
            "movement_smoothness": feature_result.get('movement_smoothness', 0),
            "dominant_hand": feature_result.get('dominant_hand', 'unknown'),
            "hand_detected": True,
            "hand_count": hand_count,
            "video_duration_seconds": round(feature_result.get('video_duration_seconds', 0), 1),
            "meta": {
                "filename": video.filename,
                "content_type": video.content_type,
                "bytes": len(content),
                "source": source,
                "inference_ms": inference_ms,
                "frames_processed": feature_result.get('frames_processed', 0),
                "frames_with_hands": feature_result.get('frames_with_hands', 0),
                "hand_detection_ratio": round(detection_ratio * 100, 1),
            }
        })

    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"Unexpected error: {error_trace}")
        return JSONResponse({
            "success": False,
            "error": "Processing error",
            "message": f"An unexpected error occurred: {str(e)}",
            "suggestion": "Please try again with a different video. If the issue persists, contact support.",
        }, status_code=500)

    finally:
        # Clean up temp file
        if temp_video_path and os.path.exists(temp_video_path):
            try:
                os.unlink(temp_video_path)
            except Exception:
                pass


@app.get("/model-info")
def model_info() -> Dict[str, Any]:
    """Get information about the loaded ML model"""
    try:
        model = get_model()
        model.load()
        info = model.get_model_info()
        info['status'] = 'loaded'
        return info
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e)
        }


@app.get("/test-hand-detection")
async def test_hand_detection():
    """Test endpoint to verify MediaPipe is working"""
    try:
        extractor = HandFeatureExtractor()
        return {
            "status": "ok",
            "message": "HandLandmarker initialized successfully",
            "model_path": extractor.model_path
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


# Mount static files last (after API routes)
app.mount("/", StaticFiles(directory=str(ROOT), html=True), name="static")


if __name__ == "__main__":
    import wsgiref.simple_server as _wsgi

    from a2wsgi import ASGIMiddleware

    host, port = "127.0.0.1", 8000
    print(f"============================================")
    print(f"  CogniVue - Parkinson's Detection System")
    print(f"============================================")
    print(f"  Server: http://{host}:{port}")
    print(f"  API Docs: http://{host}:{port}/docs")
    print(f"  Test Hand Detection: http://{host}:{port}/test-hand-detection")
    print(f"============================================")
    print()
    
    wsgi_app = ASGIMiddleware(app)
    with _wsgi.make_server(host, port, wsgi_app) as server:
        server.serve_forever()
