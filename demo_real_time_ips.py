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
from scapy.all import sniff, IP, TCP, UDP, ICMP, Raw, Ether, PcapReader

# Custom modules
import demo_config as ips_config1
from feature_extractor import process_packet, cleanup_old_flows, active_flows, get_processed_features_for_prediction, UNSW_NB15_FEATURES
from demo_metrics_logger import update_metric

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
def packet_sniffer(source, packet_q, stop_event):
    if os.path.exists(source) and source.endswith('.pcap'):
        logging.info(f"Starting packet sniffer from PCAP file: {source}")
        try:
            with PcapReader(source) as pcap_reader:
                for packet in pcap_reader:
                    if stop_event.is_set():
                        break
                    packet_q.put((packet, time.time())) # Use current time for timestamp
                    time.sleep(0.0001) # Small delay to simulate real-time processing, adjust if too fast/slow
            logging.info("Finished reading all packets from PCAP file.")
            stop_event.set() # Signal processor to stop after file is read
        except Exception as e:
            logging.critical(f"Error reading PCAP file {source}: {e}")
            stop_event.set()
    else:
        logging.info(f"Starting packet sniffer on interface: {source}")
        try:
            sniff(iface=source, prn=lambda p: packet_q.put((p, time.time())), store=0, stop_filter=lambda x: stop_event.is_set())
        except Exception as e:
            logging.critical(f"Packet sniffing failed on interface {source}: {e}. Ensure you have root privileges and the interface exists.")
            stop_event.set()

# real_time_ips.py (inside the file, replace existing packet_processor)

def packet_processor(packet_q, stop_event, model, scaler, label_encoders, feature_columns):
    logging.info("Starting packet processor and prediction engine.")
    packet_count_in_flow = defaultdict(int)

    while not stop_event.is_set() or not packet_q.empty(): # Keep running if not stopped OR if queue still has packets
        try:
            packet, timestamp = packet_q.get(timeout=0.1) # Shorter timeout for quicker shutdown
            recent_packets.append(packet)

            flow_key, raw_features = process_packet(packet, timestamp,
                                                    ips_config1.FLOW_IDLE_TIMEOUT,
                                                    ips_config1.FLOW_ACTIVE_TIMEOUT)

            if flow_key and raw_features:
                packet_count_in_flow[flow_key] += 1
                update_metric("total_flows")

                if packet_count_in_flow[flow_key] % ips_config1.PREDICTION_PACKET_INTERVAL == 0:
                    logging.debug(f"Processing flow {flow_key} at packet count {packet_count_in_flow[flow_key]}")
                    try:
                        processed_df = get_processed_features_for_prediction(raw_features, scaler, label_encoders, feature_columns)
                        prediction_proba = model.predict_proba(processed_df)[0][1] # Probability of attack (class 1)
                        is_attack = (prediction_proba >= ips_config1.PREVENTION_THRESHOLD)

                        if is_attack:
                            src_ip = flow_key[0]
                            dst_ip = flow_key[1]
                            logging.warning(f"🚨 ATTACK DETECTED (Prob: {prediction_proba:.4f}) for flow: {src_ip}:{flow_key[2]} -> {dst_ip}:{flow_key[3]} ({flow_key[4]})")
                            prevention_logger.info(f"ATTACK PREDICTED. Initiating prevention for {src_ip}.")
                            update_metric("attacks_detected")

                            # XAI Explanation - make sure it works even if DataFrame is small
                            if not processed_df.empty and len(processed_df.columns) > 0:
                                top_features = processed_df.iloc[0].nlargest(5)
                                explanation_parts = []
                                for col, val in top_features.items():
                                    if isinstance(val, (int, float)):
                                        explanation_parts.append(f"{col}={val:.2f}")
                                    else:
                                        explanation_parts.append(f"{col}={val}")
                                explanation = f"Top features contributing to prediction: {', '.join(explanation_parts)}"
                                prevention_logger.info(f"XAI Explanation: {explanation}")
                            else:
                                prevention_logger.info("XAI Explanation: No features to display or DataFrame is empty.")

                            if src_ip not in blocked_entities or time.time() > blocked_entities[src_ip]:
                                execute_prevention_action(src_ip, block_duration=ips_config1.BLOCK_DURATION_SECONDS)
                                update_metric("blocks_performed")
                            else:
                                prevention_logger.info(f"{src_ip} is already blocked.")
                        else:
                            logging.info(f"Normal traffic (Prob: {prediction_proba:.4f}) for flow: {flow_key}")

                    except Exception as e:
                        logging.error(f"Error during ML prediction for flow {flow_key}: {e}")

            cleanup_old_flows(timestamp, ips_config1.FLOW_IDLE_TIMEOUT, ips_config1.FLOW_ACTIVE_TIMEOUT)
            unblock_entities() # This will just log simulated unblocks if prevention is disabled

        except queue.Empty:
            if stop_event.is_set(): # If queue is empty and stop event is set, truly done
                break
            continue # Otherwise, keep waiting for packets

        except Exception as e:
            logging.error(f"Error in packet processor: {e}")

    logging.info("Packet processor stopping.")

# --- Main Entry ---
# real_time_ips.py (inside the file, replace existing start_ips)

def start_ips(pcap_file_path=None): # Added pcap_file_path parameter
    global model, scaler, label_encoders, feature_columns_for_model

    logging.info("Loading pre-trained models and scalers...")
    try:
        model = joblib.load(ips_config1.XGB_MODEL_PATH)
        scaler = joblib.load(ips_config1.SCALER_PATH)
        label_encoders = joblib.load(ips_config1.LABEL_ENCODERS_PATH)

        # Determine path to X_train_unsw_augmented.csv
        # It's assumed to be in the same directory as your .pkl models
        processed_data_dir = os.path.dirname(os.path.abspath(ips_config1.XGB_MODEL_PATH))
        dummy_X_train_path = os.path.join(r"data/processed/", 'X_train_unsw_augmented.csv')

        if not os.path.exists(dummy_X_train_path):
             logging.critical(f"Dummy X_train file not found: {dummy_X_train_path}. Cannot determine feature columns. Please ensure it's in the same directory as your models (or adjust MODEL_DIR in demo_config.py).")
             sys.exit(1)

        dummy_X_train = pd.read_csv(dummy_X_train_path, nrows=1)
        feature_columns_for_model = dummy_X_train.columns.tolist()
        # Remove target/label columns if they exist in the dummy file
        if 'Label' in feature_columns_for_model:
            feature_columns_for_model.remove('Label')
        if 'attack_cat' in feature_columns_for_model:
            feature_columns_for_model.remove('attack_cat')

        del dummy_X_train
        logging.info(f"Models loaded successfully. Identified {len(feature_columns_for_model)} features.")
    except FileNotFoundError as e:
        logging.critical(f"Required trained_models/scaler file not found: {e}. Please ensure you've run preprocessing_data.py, train_gan_and_augment.py, and train_xgboost_model.py. Looked in: {ips_config1.XGB_MODEL_PATH}")
        sys.exit(1)
    except Exception as e:
        logging.critical(f"Error loading models or feature columns: {e}. Exiting.")
        sys.exit(1)

    logging.info("Starting IPS threads...")

    # Determine sniffer source based on whether pcap_file_path is provided
    sniffer_source = pcap_file_path if pcap_file_path else ips_config1.NETWORK_INTERFACE

    sniffer_thread = Thread(target=packet_sniffer, args=(sniffer_source, packet_queue, stop_event))
    processor_thread = Thread(target=packet_processor, args=(packet_queue, stop_event, model, scaler, label_encoders, feature_columns_for_model))

    sniffer_thread.start()
    processor_thread.start()

    logging.info("IPS is running. Press Ctrl+C to stop.")

    try:
        while not stop_event.is_set():
            time.sleep(1)
            # unblock_entities() # Not strictly necessary if prevention is disabled
    except KeyboardInterrupt:
        logging.info("Ctrl+C detected. Shutting down IPS...")
        stop_event.set()
    finally:
        sniffer_thread.join()
        processor_thread.join()
        logging.info("IPS shutdown complete.")

# real_time_ips.py (at the very end of the file)

if __name__ == "__main__":
    # --- For the demo, specify the PCAP file path here ---
    # Make sure 'demo_attacks.pcap' is in the same directory as this script,
    # or provide the full path to it.
    demo_pcap_file = "demo_attacks.pcap"
    start_ips(pcap_file_path=demo_pcap_file)

    # --- For live sniffing later, you would use: ---
    # start_ips()