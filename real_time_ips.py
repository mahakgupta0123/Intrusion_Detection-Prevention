# real_time_ips.py

import os
import sys
import time
import joblib
import pandas as pd
import logging
import subprocess
from scapy.all import sniff
from threading import Thread, Event
import queue
from collections import deque, defaultdict
import psutil

# Custom modules
import ips_config1
from feature_extractor import process_packet, cleanup_old_flows, active_flows, get_processed_features_for_prediction, UNSW_NB15_FEATURES
from metrics_logger import update_metric

# --- Logging Setup ---
os.makedirs(ips_config1.LOG_DIR, exist_ok=True)

logging.basicConfig(level=getattr(logging, ips_config1.LOG_LEVEL.upper()),
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(ips_config1.DETECTION_LOG_FILE),
                        logging.StreamHandler(sys.stdout)
                    ])

prevention_logger = logging.getLogger('prevention_logger')
prevention_logger.setLevel(logging.INFO)
prevention_logger.addHandler(logging.FileHandler(ips_config1.PREVENTION_LOG_FILE))
prevention_logger.addHandler(logging.StreamHandler(sys.stderr))

# --- Globals ---
model = None
scaler = None
label_encoders = None
feature_columns_for_model = []

packet_queue = queue.Queue()
stop_event = Event()
recent_packets = deque(maxlen=20)
blocked_entities = {}

# --- Optional Debug: Show Interfaces ---
def print_available_interfaces():
    interfaces = psutil.net_if_addrs()
    print("🔧 Available network interfaces:")
    for iface in interfaces:
        print(f" - {iface}")

# --- IPS Actions ---
def execute_prevention_action(ip_address, action_type="DROP", block_duration=ips_config1.BLOCK_DURATION_SECONDS):
    if not ips_config1.ENABLE_PREVENTION_ACTIONS:
        prevention_logger.info(f"[SIMULATED BLOCK] {action_type} for {ip_address} for {block_duration}s.")
        return

    try:
        if action_type == "DROP":
            subprocess.run(["sudo", "iptables", "-A", "INPUT", "-s", ip_address, "-j", "DROP"], check=True)
            subprocess.run(["sudo", "iptables", "-A", "FORWARD", "-s", ip_address, "-j", "DROP"], check=True)
            prevention_logger.warning(f"[IPTABLES] Blocked {ip_address}.")
        elif action_type == "REJECT":
            subprocess.run(["sudo", "iptables", "-A", "INPUT", "-s", ip_address, "-j", "REJECT"], check=True)
            prevention_logger.warning(f"[IPTABLES] Rejected {ip_address}.")

        if block_duration > 0:
            unblock_time = time.time() + block_duration
            blocked_entities[ip_address] = unblock_time
            prevention_logger.info(f"Scheduled unblock at {time.ctime(unblock_time)}")

    except Exception as e:
        prevention_logger.error(f"Prevention error: {e}")

def unblock_entities():
    current_time = time.time()
    to_unblock = []
    for ip, unblock_time in list(blocked_entities.items()):
        if current_time >= unblock_time:
            try:
                subprocess.run(["sudo", "iptables", "-D", "INPUT", "-s", ip, "-j", "DROP"], check=True)
                subprocess.run(["sudo", "iptables", "-D", "FORWARD", "-s", ip, "-j", "DROP"], check=True)
                prevention_logger.info(f"Unblocked {ip}.")
                to_unblock.append(ip)
            except Exception as e:
                prevention_logger.error(f"Unblock error for {ip}: {e}")

    for ip in to_unblock:
        del blocked_entities[ip]

# --- Threads ---
def packet_sniffer(interface, packet_q, stop_event):
    logging.info(f"🌐 Starting packet sniffer on interface: {interface}")
    try:
        sniff(iface=interface, prn=lambda p: packet_q.put((p, time.time())), store=0, stop_filter=lambda x: stop_event.is_set())
    except Exception as e:
        logging.critical(f"Packet sniffing failed: {e}")
        stop_event.set()

def packet_processor(packet_q, stop_event, model, scaler, label_encoders, feature_columns):
    logging.info("⚙️ Starting packet processor.")
    packet_count_in_flow = defaultdict(int)

    while not stop_event.is_set():
        try:
            packet, timestamp = packet_q.get(timeout=1)
            recent_packets.append(packet)

            flow_key, raw_features = process_packet(packet, timestamp,
                                                    ips_config1.FLOW_IDLE_TIMEOUT,
                                                    ips_config1.FLOW_ACTIVE_TIMEOUT)

            if flow_key and raw_features:
                packet_count_in_flow[flow_key] += 1
                update_metric("total_flows")

                if packet_count_in_flow[flow_key] % ips_config1.PREDICTION_PACKET_INTERVAL == 0:
                    processed_df = get_processed_features_for_prediction(raw_features, scaler, label_encoders, feature_columns)
                    prediction_proba = model.predict_proba(processed_df)[0][1]
                    is_attack = prediction_proba >= ips_config1.PREVENTION_THRESHOLD

                    if is_attack:
                        src_ip = flow_key[0]
                        dst_ip = flow_key[1]
                        logging.warning(f"🚨 ATTACK DETECTED (Prob: {prediction_proba:.4f}) {src_ip} -> {dst_ip}")
                        prevention_logger.info(f"Prediction triggered prevention for {src_ip}")
                        update_metric("attacks_detected")

                        top_feats = processed_df.iloc[0].nlargest(5)
                        explanation = ', '.join([f'{col}={val}' for col, val in top_feats.items()])
                        prevention_logger.info(f"XAI Explanation: {explanation}")

                        if src_ip not in blocked_entities or time.time() > blocked_entities[src_ip]:
                            execute_prevention_action(src_ip)
                            update_metric("blocks_performed")

                    else:
                        logging.info(f"✅ Normal traffic (Prob: {prediction_proba:.4f}) for flow: {flow_key}")

            cleanup_old_flows(timestamp, ips_config1.FLOW_IDLE_TIMEOUT, ips_config1.FLOW_ACTIVE_TIMEOUT)
            unblock_entities()

        except queue.Empty:
            continue
        except Exception as e:
            logging.error(f"Error in packet processor: {e}")

    logging.info("Packet processor stopped.")

# --- Main Entry ---
def start_ips():
    global model, scaler, label_encoders, feature_columns_for_model

    logging.info("📦 Loading models...")
    try:
        model = joblib.load(ips_config1.XGB_MODEL_PATH)
        scaler = joblib.load(ips_config1.SCALER_PATH)
        label_encoders = joblib.load(ips_config1.LABEL_ENCODERS_PATH)
        dummy_X_train = pd.read_csv(os.path.join(ips_config1.PROCESSED_DATA_DIR, 'X_train_unsw_augmented.csv'), nrows=1)
        feature_columns_for_model = dummy_X_train.columns.tolist()
        del dummy_X_train
    except Exception as e:
        logging.critical(f"Model load error: {e}")
        sys.exit(1)

    print_available_interfaces()
    print(f"🌐 Using interface: {ips_config1.NETWORK_INTERFACE}")

    sniffer_thread = Thread(target=packet_sniffer, args=(ips_config1.NETWORK_INTERFACE, packet_queue, stop_event))
    processor_thread = Thread(target=packet_processor, args=(packet_queue, stop_event, model, scaler, label_encoders, feature_columns_for_model))

    sniffer_thread.start()
    processor_thread.start()

    logging.info("🛡️ IPS is running. Press Ctrl+C to stop.")
    try:
        while not stop_event.is_set():
            time.sleep(1)
            unblock_entities()
    except KeyboardInterrupt:
        logging.info("🛑 Interrupt detected. Shutting down IPS...")
        stop_event.set()
    finally:
        sniffer_thread.join()
        processor_thread.join()
        logging.info("✅ IPS shutdown complete.")

if __name__ == "__main__":
    start_ips()
