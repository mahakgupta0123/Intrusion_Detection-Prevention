# real_time_ips.py

import os
import sys
import time
import joblib
import pandas as pd
import logging
import subprocess
from scapy.all import sniff, IP, TCP, UDP, ICMP, Raw, Ether
from threading import Thread, Event
import queue # For inter-thread communication
from collections import deque # For recent packet history for XAI
from collections import defaultdict
from metrics_logger import update_metric  # Removed save_metrics since it's now inside update_metric

# Custom modules
import ips_config
from feature_extractor import process_packet, cleanup_old_flows, active_flows, get_processed_features_for_prediction, UNSW_NB15_FEATURES

# --- Logging Setup ---
# Create logs directory if it doesn't exist
os.makedirs(ips_config.LOG_DIR, exist_ok=True)

logging.basicConfig(level=getattr(logging, ips_config.LOG_LEVEL.upper()),
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(ips_config.DETECTION_LOG_FILE),
                        logging.StreamHandler(sys.stdout)
                    ])

prevention_logger = logging.getLogger('prevention_logger')
prevention_logger.setLevel(logging.INFO)
prevention_logger.addHandler(logging.FileHandler(ips_config.PREVENTION_LOG_FILE))
prevention_logger.addHandler(logging.StreamHandler(sys.stderr)) # Also output to console for prevention actions

# --- Global Variables ---
model = None
scaler = None
label_encoders = None
feature_columns_for_model = [] # Will be populated from training data columns

packet_queue = queue.Queue() # Queue for packets between sniffer and processor threads
stop_event = Event() # Event to signal threads to stop

# To store recent packets for XAI context (optional but useful)
recent_packets = deque(maxlen=20) # Store last 20 packets globally for context

# --- IPS Action Manager ---
# Keep track of currently blocked IPs/flows and their unblock times
blocked_entities = {} # {(ip, port, proto, direction): unblock_time} or just {ip: unblock_time}

def execute_prevention_action(ip_address, action_type="DROP", block_duration=ips_config.BLOCK_DURATION_SECONDS):
    if not ips_config.ENABLE_PREVENTION_ACTIONS:
        prevention_logger.info(f"[SIMULATED BLOCK] {action_type} action for {ip_address} for {block_duration}s.")
        return

    try:
        if action_type == "DROP":
            subprocess.run(["sudo", "iptables", "-A", "INPUT", "-s", ip_address, "-j", "DROP"], check=True)
            subprocess.run(["sudo", "iptables", "-A", "FORWARD", "-s", ip_address, "-j", "DROP"], check=True)
            prevention_logger.warning(f"[IPTABLES] Successfully blocked incoming/forward traffic from {ip_address}.")
            if block_duration > 0:
                unblock_time = time.time() + block_duration
                blocked_entities[ip_address] = unblock_time
                prevention_logger.info(f"Scheduled unblock for {ip_address} at {time.ctime(unblock_time)}")

        elif action_type == "REJECT":
            subprocess.run(["sudo", "iptables", "-A", "INPUT", "-s", ip_address, "-j", "REJECT"], check=True)
            prevention_logger.warning(f"[IPTABLES] Successfully rejected incoming traffic from {ip_address}.")
            if block_duration > 0:
                unblock_time = time.time() + block_duration
                blocked_entities[ip_address] = unblock_time
                prevention_logger.info(f"Scheduled unblock for {ip_address} at {time.ctime(unblock_time)}")
        else:
            prevention_logger.error(f"Unknown action type: {action_type}")

    except subprocess.CalledProcessError as e:
        prevention_logger.error(f"Failed to execute iptables command for {ip_address}: {e}")
    except FileNotFoundError:
        prevention_logger.error("iptables command not found. Make sure iptables is installed and in your PATH.")
    except Exception as e:
        prevention_logger.error(f"An unexpected error occurred during prevention action: {e}")

def unblock_entities():
    current_time = time.time()
    to_unblock = []
    for ip_address, unblock_time in list(blocked_entities.items()):
        if current_time >= unblock_time:
            try:
                subprocess.run(["sudo", "iptables", "-D", "INPUT", "-s", ip_address, "-j", "DROP"], check=True)
                subprocess.run(["sudo", "iptables", "-D", "FORWARD", "-s", ip_address, "-j", "DROP"], check=True)
                prevention_logger.info(f"[IPTABLES] Successfully unblocked {ip_address}.")
                to_unblock.append(ip_address)
            except subprocess.CalledProcessError as e:
                prevention_logger.error(f"Failed to unblock {ip_address}: {e}")
            except Exception as e:
                prevention_logger.error(f"Error during unblock for {ip_address}: {e}")

    for ip in to_unblock:
        del blocked_entities[ip]

def packet_sniffer(interface, packet_q, stop_event):
    logging.info(f"Starting packet sniffer on interface: {interface}")
    try:
        sniff(iface=interface, prn=lambda p: packet_q.put((p, time.time())), store=0, stop_filter=lambda x: stop_event.is_set())
    except Exception as e:
        logging.critical(f"Packet sniffing failed: {e}. Ensure you have root privileges and the interface exists.")
        stop_event.set()

def packet_processor(packet_q, stop_event, model, scaler, label_encoders, feature_columns):
    logging.info("Starting packet processor and prediction engine.")
    packet_count_in_flow = defaultdict(int)

    while not stop_event.is_set():
        try:
            packet, timestamp = packet_q.get(timeout=1)
            recent_packets.append(packet)

            flow_key, raw_features = process_packet(packet, timestamp,
                                                    ips_config.FLOW_IDLE_TIMEOUT,
                                                    ips_config.FLOW_ACTIVE_TIMEOUT)

            if flow_key and raw_features:
                packet_count_in_flow[flow_key] += 1
                update_metric("total_flows")

                if packet_count_in_flow[flow_key] % ips_config.PREDICTION_PACKET_INTERVAL == 0:
                    logging.debug(f"Processing flow {flow_key} at packet count {packet_count_in_flow[flow_key]}")
                    try:
                        processed_df = get_processed_features_for_prediction(raw_features, scaler, label_encoders, feature_columns)
                        prediction_proba = model.predict_proba(processed_df)[0][1]
                        is_attack = (prediction_proba >= ips_config.PREVENTION_THRESHOLD)

                        if is_attack:
                            src_ip = flow_key[0]
                            dst_ip = flow_key[1]
                            logging.warning(f"🚨 ATTACK DETECTED (Prob: {prediction_proba:.4f}) for flow: {src_ip}:{flow_key[2]} -> {dst_ip}:{flow_key[3]} ({flow_key[4]})")
                            prevention_logger.info(f"ATTACK PREDICTED. Initiating prevention for {src_ip}.")
                            update_metric("attacks_detected")

                            explanation = f"Top features contributing to prediction: {', '.join([f'{col}={val}' for col, val in processed_df.iloc[0].nlargest(5).items()])}"
                            prevention_logger.info(f"XAI Explanation: {explanation}")

                            if src_ip not in blocked_entities or time.time() > blocked_entities[src_ip]:
                                execute_prevention_action(src_ip, block_duration=ips_config.BLOCK_DURATION_SECONDS)
                                update_metric("blocks_performed")
                            else:
                                prevention_logger.info(f"{src_ip} is already blocked.")

                        else:
                            logging.info(f"Normal traffic (Prob: {prediction_proba:.4f}) for flow: {flow_key}")

                    except Exception as e:
                        logging.error(f"Error during ML prediction for flow {flow_key}: {e}")

            cleanup_old_flows(timestamp, ips_config.FLOW_IDLE_TIMEOUT, ips_config.FLOW_ACTIVE_TIMEOUT)
            unblock_entities()

        except queue.Empty:
            continue
        except Exception as e:
            logging.error(f"Error in packet processor: {e}")

    logging.info("Packet processor stopping.")

def start_ips():
    global model, scaler, label_encoders, feature_columns_for_model

    logging.info("Loading pre-trained models and scalers...")
    try:
        model = joblib.load(ips_config.XGB_MODEL_PATH)
        scaler = joblib.load(ips_config.SCALER_PATH)
        label_encoders = joblib.load(ips_config.LABEL_ENCODERS_PATH)
        dummy_X_train = pd.read_csv(os.path.join(ips_config.PROCESSED_DATA_DIR, 'X_train_unsw_augmented.csv'), nrows=1)
        feature_columns_for_model = dummy_X_train.columns.tolist()
        del dummy_X_train
        logging.info("Models loaded successfully.")
    except FileNotFoundError as e:
        logging.critical(f"Required model/scaler file not found: {e}. Please ensure you've run preprocessing_data.py, train_gan_and_augment.py, and train_xgboost_model.py.")
        sys.exit(1)
    except Exception as e:
        logging.critical(f"Error loading models: {e}. Exiting.")
        sys.exit(1)

    logging.info("Starting IPS threads...")
    sniffer_thread = Thread(target=packet_sniffer, args=(ips_config.NETWORK_INTERFACE, packet_queue, stop_event))
    processor_thread = Thread(target=packet_processor, args=(packet_queue, stop_event, model, scaler, label_encoders, feature_columns_for_model))

    sniffer_thread.start()
    processor_thread.start()

    logging.info("IPS is running. Press Ctrl+C to stop.")

    try:
        while not stop_event.is_set():
            time.sleep(1)
            unblock_entities()
    except KeyboardInterrupt:
        logging.info("Ctrl+C detected. Shutting down IPS...")
        stop_event.set()
    finally:
        sniffer_thread.join()
        processor_thread.join()
        logging.info("IPS shutdown complete.")

if __name__ == "__main__":
    start_ips()
