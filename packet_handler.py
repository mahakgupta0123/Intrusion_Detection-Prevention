from scapy.all import sniff
from ips_config import NETWORK_INTERFACE, ENABLE_PREVENTION_ACTIONS, PREVENTION_THRESHOLD
from prediction_model import predict_packet, block_source_ip
from feature_extractor import extract_features


def packet_callback(packet):
    try:
        features = extract_features(packet)  # You write this
        prediction, confidence = predict_packet(features)

        if prediction == "malicious" and confidence > PREVENTION_THRESHOLD:
            src_ip = packet[0][1].src
            print(f"🚨 Malicious packet detected from {src_ip} (Confidence: {confidence})")

            if ENABLE_PREVENTION_ACTIONS:
                block_source_ip(src_ip)
    except Exception as e:
        print(f"❌ Error processing packet: {e}")

sniff(iface=NETWORK_INTERFACE, prn=packet_callback, store=0)
