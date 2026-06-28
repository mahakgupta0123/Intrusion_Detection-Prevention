"""Real-time IPS monitoring"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime

logger = logging.getLogger(__name__)


class RealTimeIPS:
    """Real-time intrusion prevention system"""
    
    def __init__(self, prediction_engine, alert_threshold=0.7):
        self.engine = prediction_engine
        self.alert_threshold = alert_threshold
        self.alert_history = []
    
    def analyze_packet(self, packet_features):
        """Analyze single packet"""
        if isinstance(packet_features, dict):
            packet_features = pd.DataFrame([packet_features])
        elif isinstance(packet_features, (list, np.ndarray)):
            packet_features = pd.DataFrame([packet_features])
        
        result = self.engine.predict_with_confidence(packet_features)
        
        is_attack = result['predictions'][0] == 1
        confidence = result['probabilities'][0]
        
        if is_attack and confidence > self.alert_threshold:
            alert = {
                'timestamp': datetime.now(),
                'confidence': confidence,
                'action': 'block'
            }
            self.alert_history.append(alert)
            logger.warning(f"🚨 Attack detected! Confidence: {confidence:.2f}")
            return True, alert
        
        return False, {'action': 'allow', 'confidence': confidence}
    
    def analyze_traffic(self, traffic_df):
        """Analyze traffic batch"""
        results = self.engine.predict_with_confidence(traffic_df)
        
        detections = results['predictions'].sum()
        logger.info(f"📊 Detected {detections} potential attacks in {len(traffic_df)} packets")
        
        return results
    
    def get_alert_summary(self):
        """Get alert summary"""
        return {
            'total_alerts': len(self.alert_history),
            'recent_alerts': self.alert_history[-10:] if self.alert_history else []
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Real-time IPS module loaded")
