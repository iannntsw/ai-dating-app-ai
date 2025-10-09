# InsightFace-based AI Verification Service
# This uses the state-of-the-art InsightFace library for face analysis

import warnings
import numpy as np
import requests
import base64
from typing import List, Dict, Any, Optional
from PIL import Image
import io
import cv2

# Suppress known deprecation warnings from InsightFace dependencies
warnings.filterwarnings('ignore', category=DeprecationWarning, module='insightface')
warnings.filterwarnings('ignore', category=FutureWarning, module='insightface')

try:
    import insightface
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    print("Warning: InsightFace not available. Please install with: pip install insightface")

class InsightFaceVerificationAI:
    """Advanced AI verification system using InsightFace"""
    
    def __init__(self):
        self.face_app = None
        self.known_faces_db = {}  # Face database for consistency checking
        
        if INSIGHTFACE_AVAILABLE:
            try:
                # Initialize InsightFace with default models
                self.face_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
                self.face_app.prepare(ctx_id=0, det_size=(640, 640))
                print("InsightFace initialized successfully with buffalo_l model")
            except Exception as e:
                print(f"Error initializing InsightFace: {e}")
                self.face_app = None
        else:
            print("InsightFace not available - falling back to simplified analysis")
    
    def _convert_numpy_types(self, obj):
        """Convert numpy types to Python native types for JSON serialization"""
        if isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        else:
            return obj
    
    def _download_image(self, url: str) -> Optional[np.ndarray]:
        """Download image from URL and convert to numpy array"""
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Convert to PIL Image then numpy array
            image = Image.open(io.BytesIO(response.content))
            image_array = np.array(image)
            
            return image_array
        except Exception as e:
            print(f"Error downloading image {url}: {e}")
            return None
    
    def _process_base64_image(self, base64_data: str) -> Optional[np.ndarray]:
        """Process base64 encoded image and return as numpy array"""
        try:
            if ',' in base64_data:
                base64_data = base64_data.split(',')[1]
            image_data = base64.b64decode(base64_data)
            image = Image.open(io.BytesIO(image_data))
            return np.array(image)
        except Exception as e:
            print(f"Error processing base64 image: {e}")
            return None
    
    def _analyze_faces_insightface(self, image: np.ndarray) -> Dict[str, Any]:
        """Advanced face analysis using InsightFace"""
        try:
            if not self.face_app:
                return self._get_fallback_face_analysis()
            
            # Convert image to BGR format (OpenCV format)
            if len(image.shape) == 3:
                image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            else:
                image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            
            # Detect faces using InsightFace
            faces = self.face_app.get(image_bgr)
            
            analysis = {
                'face_detected': len(faces) > 0,
                'face_count': len(faces),
                'face_locations': [],
                'face_encodings': [],
                'face_sizes': [],
                'face_quality_scores': [],
                'face_angles': [],
                'lighting_quality': self._assess_lighting_advanced(image),
                'blur_detection': self._detect_blur_advanced(image),
                'image_dimensions': tuple(int(x) for x in image.shape[:2]),
                'face_confidence_scores': []
            }
            
            # Analyze each detected face
            for face in faces:
                # Face bounding box
                bbox = face.bbox.astype(int)
                x1, y1, x2, y2 = bbox
                analysis['face_locations'].append([int(x1), int(y1), int(x2-x1), int(y2-y1)])
                
                # Face embedding (for consistency checking)
                embedding = face.embedding
                analysis['face_encodings'].append(embedding.tolist())
                
                # Face size ratio
                face_area = (x2 - x1) * (y2 - y1)
                image_area = image.shape[0] * image.shape[1]
                face_size_ratio = float(face_area / image_area)
                analysis['face_sizes'].append(face_size_ratio)
                
                # Face quality assessment
                face_quality = self._assess_face_quality_advanced(image, bbox, face)
                analysis['face_quality_scores'].append(face_quality)
                
                # Face angle (pose estimation)
                if hasattr(face, 'pose'):
                    analysis['face_angles'].append({
                        'yaw': float(face.pose[0]),
                        'pitch': float(face.pose[1]),
                        'roll': float(face.pose[2])
                    })
                else:
                    analysis['face_angles'].append({'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0})
                
                # Detection confidence
                if hasattr(face, 'det_score'):
                    analysis['face_confidence_scores'].append(float(face.det_score))
                else:
                    analysis['face_confidence_scores'].append(1.0)
            
            return analysis
            
        except Exception as e:
            print(f"Error in InsightFace face analysis: {e}")
            return self._get_fallback_face_analysis()
    
    def _assess_face_quality_advanced(self, image: np.ndarray, bbox: np.ndarray, face) -> float:
        """Advanced face quality assessment using InsightFace features"""
        try:
            x1, y1, x2, y2 = bbox.astype(int)
            face_image = image[y1:y2, x1:x2]
            
            if face_image.size == 0:
                return 0.0
            
            # Convert to grayscale for analysis
            if len(face_image.shape) == 3:
                gray_face = cv2.cvtColor(face_image, cv2.COLOR_RGB2GRAY)
            else:
                gray_face = face_image
            
            # Sharpness assessment using Laplacian variance
            laplacian_var = float(cv2.Laplacian(gray_face, cv2.CV_64F).var())
            
            # Brightness and contrast
            brightness = float(np.mean(gray_face))
            contrast = float(np.std(gray_face))
            
            # Face size factor (larger faces are generally better for verification)
            face_area = (x2 - x1) * (y2 - y1)
            image_area = image.shape[0] * image.shape[1]
            size_factor = min(1.0, face_area / (image_area * 0.1))  # Optimal if face is 10% of image
            
            # Pose quality (if available)
            pose_factor = 1.0
            if hasattr(face, 'pose'):
                yaw, pitch, roll = face.pose
                # Penalize extreme angles
                pose_factor = max(0.0, 1.0 - (abs(yaw) + abs(pitch) + abs(roll)) / 90.0)
            
            # Combine metrics
            quality_score = (
                min(1.0, laplacian_var / 1000) * 0.3 +  # Sharpness
                min(1.0, brightness / 255) * 0.2 +      # Brightness
                min(1.0, contrast / 100) * 0.2 +         # Contrast
                size_factor * 0.2 +                      # Size
                pose_factor * 0.1                        # Pose
            )
            
            return float(round(quality_score, 3))
            
        except Exception as e:
            print(f"Error assessing face quality: {e}")
            return 0.0
    
    def _assess_lighting_advanced(self, image: np.ndarray) -> str:
        """Advanced lighting quality assessment"""
        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image
            
            # Calculate brightness and contrast
            brightness = float(np.mean(gray))
            contrast = float(np.std(gray))
            
            # Check for overexposure/underexposure
            overexposed_pixels = float(np.sum(gray > 240)) / gray.size
            underexposed_pixels = float(np.sum(gray < 15)) / gray.size
            
            # Lighting quality assessment
            if overexposed_pixels > 0.1 or underexposed_pixels > 0.1:
                return 'poor'
            elif brightness > 200 or brightness < 30:
                return 'poor'
            elif brightness > 180 or brightness < 50:
                return 'fair'
            elif brightness > 150 or brightness < 80:
                return 'good'
            else:
                return 'excellent'
                
        except Exception as e:
            print(f"Error assessing lighting: {e}")
            return 'poor'
    
    def _detect_blur_advanced(self, image: np.ndarray) -> bool:
        """Advanced blur detection using multiple methods"""
        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image
            
            # Method 1: Laplacian variance
            laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            
            # Method 2: Sobel edge detection
            sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
            edge_strength = float(np.mean(sobel_magnitude))
            
            # Combined blur detection
            is_blurry = laplacian_var < 100 or edge_strength < 50
            
            return bool(is_blurry)
            
        except Exception as e:
            print(f"Error detecting blur: {e}")
            return True  # Assume blurry if error
    
    def _calculate_face_consistency_advanced(self, face_analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Advanced face consistency analysis using embeddings"""
        try:
            if len(face_analyses) < 2:
                return {'consistency_score': 100.0, 'face_matches': [], 'details': 'Single photo - no consistency check needed'}
            
            embeddings = []
            for analysis in face_analyses:
                if analysis.get('face_encodings'):
                    embeddings.extend(analysis['face_encodings'])
            
            if len(embeddings) < 2:
                return {'consistency_score': 0.0, 'face_matches': [], 'details': 'No face embeddings available'}
            
            # Calculate cosine similarity between all face pairs
            similarities = []
            for i in range(len(embeddings)):
                for j in range(i + 1, len(embeddings)):
                    # Cosine similarity
                    dot_product = np.dot(embeddings[i], embeddings[j])
                    norm_i = float(np.linalg.norm(embeddings[i]))
                    norm_j = float(np.linalg.norm(embeddings[j]))
                    
                    if norm_i > 0 and norm_j > 0:
                        similarity = dot_product / (norm_i * norm_j)
                        similarities.append(similarity)
            
            if not similarities:
                return {'consistency_score': 0.0, 'face_matches': [], 'details': 'Could not calculate similarities'}
            
            # Average similarity as consistency score
            avg_similarity = float(np.mean(similarities))  # Ensure scalar value
            consistency_score = float(avg_similarity * 100)  # Convert to 0-100 scale
            
            # Determine if faces are consistent (threshold: 0.6 similarity)
            is_consistent = avg_similarity > 0.6
            
            return {
                'consistency_score': consistency_score,
                'face_matches': similarities,
                'is_consistent': bool(is_consistent),
                'details': f'Average similarity: {avg_similarity:.3f}'
            }
            
        except Exception as e:
            print(f"Error calculating face consistency: {e}")
            return {'consistency_score': 0.0, 'face_matches': [], 'details': f'Error: {str(e)}'}
    
    def _detect_deepfake_advanced(self, image: np.ndarray) -> Dict[str, Any]:
        """Advanced deepfake detection using multiple techniques"""
        try:
            # Convert to grayscale for analysis
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image
            
            # Method 1: Frequency domain analysis
            f_transform = np.fft.fft2(gray)
            f_shift = np.fft.fftshift(f_transform)
            magnitude_spectrum = np.log(np.abs(f_shift) + 1)
            
            # Check for unusual frequency patterns
            high_freq_energy = float(np.sum(magnitude_spectrum[gray.shape[0]//4:3*gray.shape[0]//4, 
                                                       gray.shape[1]//4:3*gray.shape[1]//4]))
            total_energy = float(np.sum(magnitude_spectrum))
            freq_ratio = high_freq_energy / total_energy if total_energy > 0 else 0
            
            # Method 2: Edge consistency analysis
            edges = cv2.Canny(gray, 50, 150)
            edge_density = float(np.sum(edges > 0)) / edges.size
            
            # Method 3: Color consistency (if color image)
            color_inconsistency = 0.0
            if len(image.shape) == 3:
                # Check for unusual color patterns
                color_std = np.std(image, axis=2)
                color_inconsistency = float(np.mean(color_std)) / 255.0
            
            # Combine indicators
            manipulation_score = 0.0
            
            # Frequency analysis indicator
            if freq_ratio < 0.1 or freq_ratio > 0.8:
                manipulation_score += 0.3
            
            # Edge analysis indicator
            if edge_density < 0.05 or edge_density > 0.5:
                manipulation_score += 0.3
            
            # Color analysis indicator
            if color_inconsistency > 0.3:
                manipulation_score += 0.2
            
            # Additional checks
            if manipulation_score > 0.5:
                manipulation_detected = True
                deepfake_probability = min(0.9, manipulation_score)
            else:
                manipulation_detected = False
                deepfake_probability = manipulation_score
            
            return {
                'deepfake_probability': float(deepfake_probability),
                'manipulation_detected': bool(manipulation_detected),
                'frequency_anomaly': bool(freq_ratio < 0.1 or freq_ratio > 0.8),
                'edge_anomaly': bool(edge_density < 0.05 or edge_density > 0.5),
                'color_anomaly': bool(color_inconsistency > 0.3),
                'metadata_analysis': {
                    'suspicious_editing': bool(manipulation_detected),
                    'compression_artifacts': bool(manipulation_score > 0.3),
                    'exif_data_present': False  # Would need additional EXIF analysis
                }
            }
            
        except Exception as e:
            print(f"Error in deepfake detection: {e}")
            return {
                'deepfake_probability': 0.5,
                'manipulation_detected': True,
                'metadata_analysis': {
                    'suspicious_editing': True,
                    'compression_artifacts': True,
                    'exif_data_present': False
                }
            }
    
    def _assess_image_quality_advanced(self, image: np.ndarray) -> Dict[str, Any]:
        """Advanced image quality assessment"""
        try:
            # Convert to grayscale for analysis
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image
            
            # Technical quality metrics
            laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            brightness = float(np.mean(gray))
            contrast = float(np.std(gray))
            
            # Noise estimation
            noise = cv2.fastNlMeansDenoising(gray)
            noise_level = float(np.mean(np.abs(gray.astype(float) - noise.astype(float))))
            
            # Technical score calculation
            technical_score = 0.0
            
            # Sharpness (40% weight)
            if laplacian_var > 1000:
                technical_score += 40
            elif laplacian_var > 500:
                technical_score += 30
            elif laplacian_var > 200:
                technical_score += 20
            elif laplacian_var > 100:
                technical_score += 10
            
            # Brightness (30% weight)
            if 50 <= brightness <= 200:
                technical_score += 30
            elif 30 <= brightness <= 220:
                technical_score += 20
            elif 20 <= brightness <= 230:
                technical_score += 10
            
            # Contrast (20% weight)
            if contrast > 50:
                technical_score += 20
            elif contrast > 30:
                technical_score += 15
            elif contrast > 20:
                technical_score += 10
            
            # Noise (10% weight)
            if noise_level < 5:
                technical_score += 10
            elif noise_level < 10:
                technical_score += 7
            elif noise_level < 15:
                technical_score += 5
            
            technical_score = min(100, technical_score)
            
            # Aesthetic quality (simplified)
            aesthetic_score = 70  # Base score
            
            # Penalize extreme brightness
            if brightness < 40 or brightness > 210:
                aesthetic_score -= 20
            
            # Penalize low contrast
            if contrast < 20:
                aesthetic_score -= 15
            
            # Penalize blur
            if laplacian_var < 100:
                aesthetic_score -= 15
            
            aesthetic_score = max(0, min(100, aesthetic_score))
            
            return {
                'technical_score': float(technical_score),
                'aesthetic_score': float(aesthetic_score),
                'aggregate_score': float((technical_score + aesthetic_score) / 2),
                'sharpness_score': float(laplacian_var),
                'brightness_score': float(brightness),
                'contrast_score': float(contrast),
                'noise_level': float(noise_level)
            }
            
        except Exception as e:
            print(f"Error assessing image quality: {e}")
            return {
                'technical_score': 0.0,
                'aesthetic_score': 0.0,
                'aggregate_score': 0.0,
                'sharpness_score': 0.0,
                'brightness_score': 0.0,
                'contrast_score': 0.0,
                'noise_level': 100.0
            }
    
    def _get_fallback_face_analysis(self) -> Dict[str, Any]:
        """Fallback face analysis when InsightFace is not available"""
        return {
            'face_detected': False,
            'face_count': 0,
            'face_locations': [],
            'face_encodings': [],
            'face_sizes': [],
            'face_quality_scores': [],
            'face_angles': [],
            'lighting_quality': 'poor',
            'blur_detection': True,
            'image_dimensions': (0, 0),
            'face_confidence_scores': [],
            'error': 'InsightFace not available'
        }
    
    def analyze_verification_photos(self, photo_urls: List[str], user_id: str, verification_level: str = "basic") -> Dict[str, Any]:
        """
        Performs comprehensive AI analysis on verification photos using InsightFace
        """
        try:
            results = {
                'face_analysis': [],
                'consistency_scores': [],
                'quality_scores': [],
                'deepfake_analysis': [],
                'metadata_analysis': [],
                'overall_assessment': {}
            }
            
            # Download and analyze each photo
            for i, photo_url in enumerate(photo_urls):
                # Download image
                image_data = self._download_image(photo_url)
                if image_data is None:
                    results['face_analysis'].append(self._get_fallback_face_analysis())
                    results['quality_scores'].append({'aggregate_score': 0.0})
                    results['deepfake_analysis'].append({
                        'deepfake_probability': 0.9,
                        'manipulation_detected': True,
                        'metadata_analysis': {'suspicious_editing': True, 'compression_artifacts': True, 'exif_data_present': False}
                    })
                    continue
                
                # Perform InsightFace analysis
                face_analysis = self._analyze_faces_insightface(image_data)
                results['face_analysis'].append(face_analysis)
                
                # Perform quality analysis
                quality_analysis = self._assess_image_quality_advanced(image_data)
                results['quality_scores'].append(quality_analysis)
                
                # Perform deepfake detection
                deepfake_analysis = self._detect_deepfake_advanced(image_data)
                results['deepfake_analysis'].append(deepfake_analysis)
                
                # Extract metadata analysis
                results['metadata_analysis'].append(deepfake_analysis['metadata_analysis'])
            
            # Calculate consistency scores if multiple photos
            if len(results['face_analysis']) > 1:
                consistency_result = self._calculate_face_consistency_advanced(results['face_analysis'])
                results['consistency_scores'] = [consistency_result['consistency_score']]
            else:
                results['consistency_scores'] = [100.0]  # Single photo - assume consistent
            
            # Generate overall assessment
            results['overall_assessment'] = self._generate_overall_assessment_advanced(results)
            
            # Convert numpy types to Python native types for JSON serialization
            return self._convert_numpy_types(results)
            
        except Exception as e:
            print(f"Error in analyze_verification_photos: {e}")
            return self._convert_numpy_types(self._get_fallback_analysis())
    
    def _generate_overall_assessment_advanced(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overall assessment using advanced metrics"""
        try:
            face_analyses = results.get('face_analysis', [])
            quality_scores = results.get('quality_scores', [])
            deepfake_analyses = results.get('deepfake_analysis', [])
            consistency_scores = results.get('consistency_scores', [])
            
            # Calculate average scores
            avg_face_consistency = float(np.mean(consistency_scores)) if consistency_scores else 0
            avg_photo_quality = float(np.mean([q.get('aggregate_score', 0) for q in quality_scores])) if quality_scores else 0
            avg_deepfake_risk = float(np.mean([d.get('deepfake_probability', 0) for d in deepfake_analyses])) * 100 if deepfake_analyses else 100
            
            # Face detection score
            total_faces = sum(fa.get('face_count', 0) for fa in face_analyses)
            face_detection_score = min(100, (total_faces / len(face_analyses)) * 100) if face_analyses else 0
            
            # Behavioral score (placeholder - would come from backend)
            behavioral_score = 70.0
            
            # Overall score calculation
            overall_score = (
                avg_face_consistency * 0.3 +
                avg_photo_quality * 0.25 +
                (100 - avg_deepfake_risk) * 0.25 +
                behavioral_score * 0.2
            )
            overall_score = max(0, min(100, overall_score))
            
            # Determine verification status
            verification_status = "pending"
            requires_manual_review = False
            recommendations = []
            
            if overall_score >= 70 and avg_deepfake_risk < 35 and avg_face_consistency >= 70:
                verification_status = "verified"
                recommendations.append("Photos meet verification requirements.")
            elif overall_score < 50 or avg_deepfake_risk >= 50:
                verification_status = "rejected"
                recommendations.append("Photos do not meet verification requirements. High risk of manipulation or poor quality.")
                requires_manual_review = True
            else:
                verification_status = "flagged"
                recommendations.append("Verification requires manual review due to moderate risk or inconsistent data.")
                requires_manual_review = True
            
            # Additional checks
            if not any(fa.get('face_detected', False) for fa in face_analyses):
                verification_status = "rejected"
                requires_manual_review = True
                recommendations.append("No faces detected in verification photos.")
            elif any(fa.get('face_count', 0) > 1 for fa in face_analyses):
                verification_status = "flagged"
                requires_manual_review = True
                recommendations.append("Multiple faces detected in one or more photos.")
            
            return {
                'overall_score': round(overall_score, 2),
                'face_consistency_score': round(avg_face_consistency, 2),
                'photo_quality_score': round(avg_photo_quality, 2),
                'deepfake_risk_score': round(avg_deepfake_risk, 2),
                'face_detection_score': round(face_detection_score, 2),
                'verification_status': verification_status,
                'confidence_level': 'high' if overall_score >= 80 else 'low',
                'requires_manual_review': requires_manual_review,
                'recommendations': recommendations
            }
            
        except Exception as e:
            print(f"Error generating overall assessment: {e}")
            return {
                'overall_score': 0.0,
                'face_consistency_score': 0.0,
                'photo_quality_score': 0.0,
                'deepfake_risk_score': 100.0,
                'face_detection_score': 0.0,
                'verification_status': 'rejected',
                'confidence_level': 'low',
                'requires_manual_review': True,
                'recommendations': ['Error in analysis']
            }
    
    def _detect_deepfake(self, image: np.ndarray) -> Dict[str, Any]:
        """Detect deepfake/manipulation in image - wrapper for advanced method"""
        return self._detect_deepfake_advanced(image)
    
    def _calculate_face_consistency(self, face_analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate face consistency - wrapper for advanced method"""
        return self._calculate_face_consistency_advanced(face_analyses)
    
    def _get_fallback_analysis(self) -> Dict[str, Any]:
        """Fallback analysis when InsightFace is not available"""
        return {
            'face_analysis': [],
            'consistency_scores': [],
            'quality_scores': [],
            'deepfake_analysis': [],
            'metadata_analysis': [],
            'overall_assessment': {
                'overall_score': 0.0,
                'face_consistency_score': 0.0,
                'photo_quality_score': 0.0,
                'deepfake_risk_score': 100.0,
                'face_detection_score': 0.0,
                'verification_status': 'rejected',
                'confidence_level': 'low',
                'requires_manual_review': True,
                'recommendations': ['InsightFace not available']
            }
        }

# Create global instance
insightface_verification_ai = InsightFaceVerificationAI()
