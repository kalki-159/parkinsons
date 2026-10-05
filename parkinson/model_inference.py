import os
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
import joblib


class ParkinsonModel:
    def __init__(self, model_path: str, scaler_path: str, encoder_path: str):
        self.model_path = Path(model_path)
        self.scaler_path = Path(scaler_path)
        self.encoder_path = Path(encoder_path)
        
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self._loaded = False
    
    def load(self):
        """Load model, scaler, and label encoder"""
        if self._loaded:
            return
        
        try:
            # Load TensorFlow/Keras model
            import tensorflow as tf
            self.model = tf.keras.models.load_model(str(self.model_path), compile=False)
            
            # Load scaler if exists
            if self.scaler_path.exists():
                self.scaler = joblib.load(str(self.scaler_path))
            
            # Load label encoder
            if self.encoder_path.exists():
                self.label_encoder = joblib.load(str(self.encoder_path))
            
            self._loaded = True
        except Exception as e:
            print(f"Warning: Could not load ML model: {e}")
            print("Will use rule-based risk calculation instead.")
            self._loaded = True  # Mark as loaded to prevent retry
    
    def predict(self, feature_vector: np.ndarray) -> Dict[str, Any]:
        """Make prediction using feature vector"""
        if not self._loaded:
            self.load()
        
        if self.model is None:
            return None  # Signal to use rule-based fallback
        
        try:
            # Apply scaler if available
            if self.scaler is not None:
                features_scaled = self.scaler.transform(feature_vector)
            else:
                features_scaled = feature_vector
            
            # Get model prediction
            prediction = self.model.predict(features_scaled, verbose=0)
            
            if len(prediction.shape) > 1:
                prediction = prediction[0]
            
            pred_class = int(np.argmax(prediction))
            confidence = float(prediction[pred_class])
            
            if self.label_encoder is not None:
                try:
                    label = str(self.label_encoder.inverse_transform([pred_class])[0])
                except Exception:
                    label = f"Class_{pred_class}"
            else:
                label = f"Class_{pred_class}"
            
            risk_score = confidence * 100
            
            return {
                'prediction': label,
                'class': pred_class,
                'confidence': round(confidence * 100, 2),
                'risk_score': round(risk_score, 2),
                'all_probabilities': prediction.tolist(),
                'method': 'ml_model'
            }
        except Exception as e:
            print(f"Model prediction error: {e}")
            return None  # Signal to use rule-based fallback
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        return {
            'model_path': str(self.model_path),
            'model_exists': self.model_path.exists(),
            'scaler_path': str(self.scaler_path),
            'scaler_exists': self.scaler_path.exists(),
            'encoder_path': str(self.encoder_path),
            'encoder_exists': self.encoder_path.exists(),
            'loaded': self._loaded,
            'model_loaded': self.model is not None
        }


# Global model instance
_model_instance: Optional[ParkinsonModel] = None


def get_model(
    model_path: str = None,
    scaler_path: str = None,
    encoder_path: str = None
) -> ParkinsonModel:
    """Get or create the global model instance"""
    global _model_instance
    
    if _model_instance is not None:
        return _model_instance
    
    if model_path is None:
        model_path = Path(__file__).parent / "model.h5"
    if scaler_path is None:
        scaler_path = Path(__file__).parent / "scaler.pkl"
    if encoder_path is None:
        encoder_path = Path(__file__).parent / "label_encoder.pkl"
    
    _model_instance = ParkinsonModel(
        model_path=str(model_path),
        scaler_path=str(scaler_path),
        encoder_path=str(encoder_path)
    )
    
    return _model_instance


def calculate_rule_based_risk(features: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate Parkinson's risk using rule-based analysis of tremor features.
    
    Risk is calculated based on:
    1. Tremor Frequency (4-8 Hz is typical Parkinson's range)
    2. Amplitude (higher = more severe)
    3. Rhythm Consistency (more rhythmic = more likely Parkinson's)
    4. Movement Smoothness (less smooth = more tremor)
    
    Args:
        features: Dictionary with tremor feature values
        
    Returns:
        Dictionary with risk assessment
    """
    # Extract features
    tremor_freq = features.get('tremor_frequency_hz', 0)
    amplitude = features.get('amplitude_mm', 0)
    rhythm = features.get('rhythm_consistency', 50)
    smoothness = features.get('movement_smoothness', 50)
    velocity = features.get('average_velocity', 0)
    freq_conf = features.get('frequency_confidence', 0)
    
    # === TREMOR FREQUENCY SCORE (0-40 points) ===
    # Parkinson's tremor is typically 4-8 Hz
    if tremor_freq == 0 or freq_conf < 20:
        freq_score = 20  # Uncertain - moderate score
    elif 4 <= tremor_freq <= 8:
        # Peak in Parkinson's range
        if 5 <= tremor_freq <= 6:
            freq_score = 40  # Classic Parkinson's frequency
        else:
            freq_score = 30 + (8 - abs(tremor_freq - 6)) * 5
    elif 3 <= tremor_freq < 4 or 8 < tremor_freq <= 9:
        freq_score = 20  # Near range
    else:
        freq_score = max(0, 15 - abs(tremor_freq - 6) * 2)
    
    freq_score = min(40, max(0, freq_score))
    
    # === AMPLITUDE SCORE (0-30 points) ===
    # Higher amplitude = more severe tremor
    if amplitude == 0:
        amp_score = 0
    elif amplitude < 1:
        amp_score = 5  # Minimal tremor
    elif amplitude < 3:
        amp_score = 10  # Mild tremor
    elif amplitude < 5:
        amp_score = 18  # Moderate tremor
    elif amplitude < 8:
        amp_score = 25  # Significant tremor
    else:
        amp_score = 30  # Severe tremor
    
    # === RHYTHM CONSISTENCY SCORE (0-20 points) ===
    # Parkinson's tremors are often rhythmic (regular oscillations)
    if rhythm == 0:
        rhythm_score = 0
    elif rhythm < 40:
        rhythm_score = rhythm * 0.2  # Irregular - low score
    elif rhythm < 60:
        rhythm_score = 8 + (rhythm - 40) * 0.3
    else:
        rhythm_score = 14 + (rhythm - 60) * 0.15
        rhythm_score = min(20, rhythm_score)
    
    # === SMOOTHNESS SCORE (0-10 points, inverse) ===
    # Less smooth = more jerkiness = more tremor
    if smoothness == 0:
        smooth_score = 5
    elif smoothness < 50:
        smooth_score = 10  # Very jerky
    elif smoothness < 70:
        smooth_score = 7
    elif smoothness < 85:
        smooth_score = 4
    else:
        smooth_score = 1  # Very smooth
    
    # === TOTAL RISK SCORE ===
    total_risk = freq_score + amp_score + rhythm_score + smooth_score
    total_risk = min(100, total_risk)
    
    # === DETERMINE RISK CATEGORY ===
    if total_risk < 25:
        risk_category = "Normal"
        label = "Normal"
    elif total_risk < 45:
        risk_category = "Early Stage"
        label = "Early Stage"
    elif total_risk < 65:
        risk_category = "Moderate"
        label = "Moderate"
    else:
        risk_category = "Severe"
        label = "Severe"
    
    # === CALCULATE CONFIDENCE ===
    # Higher frequency confidence = more reliable assessment
    confidence = min(95, max(50, freq_conf + 30))
    
    return {
        'prediction': label,
        'risk_score': round(total_risk, 2),
        'confidence': round(confidence, 2),
        'risk_category': risk_category,
        'method': 'rule_based',
        'all_probabilities': [],
        'class': 0,
        # Breakdown for transparency
        'score_breakdown': {
            'frequency_score': round(freq_score, 1),
            'amplitude_score': round(amp_score, 1),
            'rhythm_score': round(rhythm_score, 1),
            'smoothness_score': round(smooth_score, 1),
            'tremor_frequency_hz': tremor_freq,
            'amplitude_mm': amplitude,
            'rhythm_consistency': rhythm,
            'movement_smoothness': smoothness
        }
    }


def predict_parkinson(features: Dict[str, Any]) -> Dict[str, Any]:
    """
    Make prediction from tremor features.
    Tries ML model first, falls back to rule-based if model fails.
    """
    # Try ML model first
    try:
        model = get_model()
        model.load()
        
        if model.model is not None:
            # Build feature vector
            feature_order = [
                'tremor_frequency_hz',
                'amplitude_mm',
                'rhythm_consistency',
                'movement_smoothness',
                'average_velocity',
                'frequency_confidence'
            ]
            
            vector = [features.get(key, 0.0) for key in feature_order]
            feature_vector = np.array(vector).reshape(1, -1)
            
            ml_result = model.predict(feature_vector)
            if ml_result:
                ml_result['features_used'] = feature_order
                ml_result['feature_values'] = vector
                ml_result['risk_category'] = _map_to_risk_category(ml_result['risk_score'])
                return ml_result
    except Exception as e:
        print(f"ML model prediction failed: {e}")
    
    # Fall back to rule-based calculation
    print("Using rule-based risk calculation (ML model unavailable)")
    result = calculate_rule_based_risk(features)
    return result


def _map_to_risk_category(risk_score: float) -> str:
    """Map risk score to risk category"""
    if risk_score < 25:
        return "Normal"
    elif risk_score < 45:
        return "Early Stage"
    elif risk_score < 65:
        return "Moderate"
    else:
        return "Severe"


if __name__ == "__main__":
    # Test with sample features
    print("=== Testing Rule-Based Risk Calculation ===\n")
    
    # Test case 1: Classic Parkinson's tremor
    parkinson_features = {
        'tremor_frequency_hz': 5.5,  # Classic Parkinson's range
        'amplitude_mm': 4.5,
        'rhythm_consistency': 75,  # Rhythmic
        'movement_smoothness': 55,  # Some jerkiness
        'average_velocity': 0.1,
        'frequency_confidence': 80
    }
    
    print("Test Case 1: Classic Parkinson's tremor")
    result1 = calculate_rule_based_risk(parkinson_features)
    print(f"Risk Score: {result1['risk_score']}%")
    print(f"Category: {result1['risk_category']}")
    print(f"Confidence: {result1['confidence']}%")
    print(f"Method: {result1['method']}")
    print(f"Breakdown: {result1['score_breakdown']}")
    print()
    
    # Test case 2: Normal
    normal_features = {
        'tremor_frequency_hz': 2.5,  # Low frequency
        'amplitude_mm': 1.2,
        'rhythm_consistency': 30,  # Irregular
        'movement_smoothness': 90,  # Smooth
        'average_velocity': 0.05,
        'frequency_confidence': 70
    }
    
    print("Test Case 2: Normal (minimal tremor)")
    result2 = calculate_rule_based_risk(normal_features)
    print(f"Risk Score: {result2['risk_score']}%")
    print(f"Category: {result2['risk_category']}")
    print(f"Confidence: {result2['confidence']}%")
    print(f"Method: {result2['method']}")
    print(f"Breakdown: {result2['score_breakdown']}")
