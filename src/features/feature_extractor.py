"""
Feature Extraction for Network Intrusion Detection

Extract features from network packets and flows.
"""

import time
import ipaddress
from collections import defaultdict
import pandas as pd


# UNSW-NB15 dataset features
UNSW_NB15_FEATURES = [
    'dur', 'proto', 'service', 'state', 'spkts', 'dpkts', 'sbytes', 'dbytes',
    'sload', 'dload', 'sttl', 'dttl', 'sloss', 'dloss', 'Sintpkt', 'Dintpkt',
    'tcprtt', 'synack', 'ackdat', 'is_sm_ips_ports', 'ct_srv_src',
    'ct_dst_ltm', 'ct_src_ltm', 'ct_srv_dst', 'ct_dst_src_ltm',
    'ct_srcdst_ltm', 'ct_flw_http_mthd', 'ct_ftp_cmd', 'is_ftp_login',
    'ct_dst_sport_ltm', 'ct_src_dport_ltm', 'ct_state_ttl', 'sjit', 'djit'
]


class FeatureExtractor:
    """Extract features from network traffic."""
    
    def __init__(self):
        """Initialize feature extractor."""
        self.active_flows = {}
        self.last_ten_minutes_activity = defaultdict(lambda: defaultdict(int))
        self.state_ttl_counts = defaultdict(lambda: defaultdict(int))
    
    def get_feature_template(self):
        """Get empty feature template with all UNSW-NB15 features."""
        template = {col: 0 for col in UNSW_NB15_FEATURES}
        template['proto'] = 'tcp'
        template['state'] = 'INT'
        template['last_src_pkt_time'] = 0
        template['last_dst_pkt_time'] = 0
        template['prev_sintpkt'] = 0
        template['prev_dintpkt'] = 0
        return template
    
    def get_proto_name(self, proto_num):
        """Convert protocol number to name."""
        protos = {6: 'tcp', 17: 'udp', 1: 'icmp'}
        return protos.get(proto_num, 'other')
    
    def get_service_name(self, dport, proto):
        """Determine service from destination port and protocol."""
        if proto == 'tcp':
            services = {
                80: 'http', 443: 'ssl', 21: 'ftp', 22: 'ssh',
                23: 'smtp', 25: 'smtp', 53: 'dns'
            }
            return services.get(dport, '-')
        if proto == 'udp' and dport == 53:
            return 'dns'
        return '-'
    
    def update_last_ten_minutes_counts(self, key, type_key):
        """Update and get 10-minute window packet count."""
        current_minute_bucket = int(time.time() / 60)
        self.last_ten_minutes_activity[(key, type_key)][current_minute_bucket] += 1
        
        # Remove old buckets (> 10 minutes)
        buckets_to_remove = [b for b in self.last_ten_minutes_activity[(key, type_key)].keys()
                            if current_minute_bucket - b > 10]
        for b in buckets_to_remove:
            del self.last_ten_minutes_activity[(key, type_key)][b]
        
        # Return total count in last 10 minutes
        return sum(self.last_ten_minutes_activity[(key, type_key)].values())
    
    def is_private_ip(self, ip_str):
        """Check if IP address is private."""
        try:
            ip = ipaddress.ip_address(ip_str)
            return ip.is_private
        except ValueError:
            return False
    
    def is_multicast_ip(self, ip_str):
        """Check if IP address is multicast."""
        try:
            ip = ipaddress.ip_address(ip_str)
            return ip.is_multicast
        except ValueError:
            return False
    
    def extract_from_packet(self, packet_data):
        """
        Extract features from packet data.
        
        Args:
            packet_data: Dictionary with packet information
                Required keys: src_ip, dst_ip, src_port, dst_port,
                              proto, ttl, payload_size, flags, etc.
        
        Returns:
            Dictionary with extracted features
        """
        features = self.get_feature_template()
        
        # Basic protocol and service features
        features['proto'] = self.get_proto_name(packet_data.get('proto', 6))
        features['service'] = self.get_service_name(
            packet_data.get('dst_port', 0),
            features['proto']
        )
        
        # Connection state
        flags = packet_data.get('flags', '')
        if 'S' in flags and 'A' not in flags:
            features['state'] = 'SYN'
        elif 'A' in flags and 'S' not in flags:
            features['state'] = 'ACK'
        elif 'F' in flags:
            features['state'] = 'FIN'
        else:
            features['state'] = 'INT'
        
        # Packet counts and sizes
        features['spkts'] = packet_data.get('src_packets', 1)
        features['dpkts'] = packet_data.get('dst_packets', 1)
        features['sbytes'] = packet_data.get('src_bytes', packet_data.get('payload_size', 0))
        features['dbytes'] = packet_data.get('dst_bytes', packet_data.get('payload_size', 0))
        
        # TTL information
        features['sttl'] = packet_data.get('src_ttl', 64)
        features['dttl'] = packet_data.get('dst_ttl', 64)
        
        # Flow characteristics
        src_ip = packet_data.get('src_ip', '0.0.0.0')
        dst_ip = packet_data.get('dst_ip', '0.0.0.0')
        
        features['is_sm_ips_ports'] = 1 if (src_ip == dst_ip) else 0
        features['dur'] = packet_data.get('duration', 0)
        
        return features
    
    def extract_batch(self, packets):
        """
        Extract features from batch of packets.
        
        Args:
            packets: List of packet data dictionaries
            
        Returns:
            DataFrame with extracted features
        """
        features_list = [self.extract_from_packet(pkt) for pkt in packets]
        return pd.DataFrame(features_list)


def get_feature_template():
    """Get empty feature template with all UNSW-NB15 features."""
    template = {col: 0 for col in UNSW_NB15_FEATURES}
    template['proto'] = 'tcp'
    template['state'] = 'INT'
    return template


def get_proto_name(proto_num):
    """Convert protocol number to name."""
    protos = {6: 'tcp', 17: 'udp', 1: 'icmp'}
    return protos.get(proto_num, 'other')


def get_service_name(dport, proto):
    """Determine service from destination port and protocol."""
    if proto == 'tcp':
        services = {
            80: 'http', 443: 'ssl', 21: 'ftp', 22: 'ssh',
            23: 'smtp', 25: 'smtp', 53: 'dns'
        }
        return services.get(dport, '-')
    if proto == 'udp' and dport == 53:
        return 'dns'
    return '-'
