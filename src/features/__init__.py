"""
Feature Extraction Module

This module provides feature extraction utilities for real-time
network traffic analysis.

Classes:
    FeatureExtractor - Extract features from network packets

Functions:
    get_feature_template() - Get empty feature template
    get_proto_name() - Convert protocol number to name
    get_service_name() - Determine service from port and protocol
    update_flow_statistics() - Update flow-based statistics

Usage:
    from src.features import FeatureExtractor
    
    extractor = FeatureExtractor()
    features = extractor.get_feature_template()
    features = extractor.extract_from_packet(packet_data)
"""

from .feature_extractor import (
    FeatureExtractor,
    get_feature_template,
    get_proto_name,
    get_service_name
)

__all__ = [
    'FeatureExtractor',
    'get_feature_template',
    'get_proto_name',
    'get_service_name'
]
