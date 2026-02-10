# feature_extractor.py

import time
import ipaddress
from collections import defaultdict
import pandas as pd

UNSW_NB15_FEATURES = [
    'dur', 'proto', 'service', 'state', 'spkts', 'dpkts', 'sbytes', 'dbytes',
    'sload', 'dload', 'sttl', 'dttl', 'sloss', 'dloss', 'Sintpkt', 'Dintpkt',
    'tcprtt', 'synack', 'ackdat', 'is_sm_ips_ports', 'ct_srv_src',
    'ct_dst_ltm', 'ct_src_ltm', 'ct_srv_dst', 'ct_dst_src_ltm',
    'ct_srcdst_ltm', 'ct_flw_http_mthd', 'ct_ftp_cmd', 'is_ftp_login',
    'ct_dst_sport_ltm', 'ct_src_dport_ltm', 'ct_state_ttl', 'sjit', 'djit'
]

active_flows = {}
last_ten_minutes_activity = defaultdict(lambda: defaultdict(int))
state_ttl_counts = defaultdict(lambda: defaultdict(int))


def get_feature_template():
    template = {col: 0 for col in UNSW_NB15_FEATURES}
    template['proto'] = 'tcp'
    template['state'] = 'INT'
    template['last_src_pkt_time'] = 0
    template['last_dst_pkt_time'] = 0
    template['prev_sintpkt'] = 0
    template['prev_dintpkt'] = 0
    return template

def get_proto_name(proto_num):
    protos = {6: 'tcp', 17: 'udp', 1: 'icmp'}
    return protos.get(proto_num, 'other')

def get_service_name(dport, proto):
    if proto == 'tcp':
        return {
            80: 'http', 443: 'ssl', 21: 'ftp', 22: 'ssh',
            23: 'smtp', 25: 'smtp', 53: 'dns'
        }.get(dport, '-')
    if proto == 'udp' and dport == 53:
        return 'dns'
    return '-'

def update_last_ten_minutes_counts(key, type_key):
    current_minute_bucket = int(time.time() / 60)
    last_ten_minutes_activity[(key, type_key)][current_minute_bucket] += 1
    return sum(count for bucket, count in last_ten_minutes_activity[(key, type_key)].items())

def process_packet(packet, timestamp, flow_idle_timeout, flow_active_timeout):
    if not packet.haslayer('IP'):
        return None, None

    ip_layer = packet.getlayer('IP')
    src_ip = ip_layer.src
    dst_ip = ip_layer.dst
    sttl = ip_layer.ttl
    proto, src_port, dst_port, state = 'tcp', 0, 0, 'INT'

    if packet.haslayer('TCP'):
        tcp = packet.getlayer('TCP')
        src_port, dst_port = tcp.sport, tcp.dport
        proto = 'tcp'
        state = 'EST' if (tcp.flags & 0x02 and tcp.flags & 0x10) else 'REQ'
    elif packet.haslayer('UDP'):
        udp = packet.getlayer('UDP')
        src_port, dst_port = udp.sport, udp.dport
        proto = 'udp'
        state = 'CON'
    elif packet.haslayer('ICMP'):
        proto = 'icmp'
        state = 'CON'

    flow_key = (src_ip, dst_ip, src_port, dst_port, proto)
    reverse_flow_key = (dst_ip, src_ip, dst_port, src_port, proto)
    is_reverse = False

    if flow_key not in active_flows:
        if reverse_flow_key in active_flows:
            flow_key = reverse_flow_key
            is_reverse = True
        else:
            active_flows[flow_key] = get_feature_template()
            active_flows[flow_key]['start_time'] = timestamp
            active_flows[flow_key]['last_packet_time'] = timestamp
    flow = active_flows[flow_key]
    flow['last_packet_time'] = timestamp
    flow['dur'] = timestamp - flow['start_time']
    pkt_len = len(packet)

    if not is_reverse:
        flow['spkts'] += 1
        flow['sbytes'] += pkt_len
        flow['sload'] = (flow['sbytes'] / flow['dur']) * 8 if flow['dur'] > 0 else 0
        if flow['last_src_pkt_time']:
            sint = timestamp - flow['last_src_pkt_time']
            flow['Sintpkt'] = sint
            flow['sjit'] = abs(sint - flow['prev_sintpkt'])
            flow['prev_sintpkt'] = sint
        flow['last_src_pkt_time'] = timestamp
    else:
        flow['dpkts'] += 1
        flow['dbytes'] += pkt_len
        flow['dload'] = (flow['dbytes'] / flow['dur']) * 8 if flow['dur'] > 0 else 0
        if flow['last_dst_pkt_time']:
            dint = timestamp - flow['last_dst_pkt_time']
            flow['Dintpkt'] = dint
            flow['djit'] = abs(dint - flow['prev_dintpkt'])
            flow['prev_dintpkt'] = dint
        flow['last_dst_pkt_time'] = timestamp

    flow['sttl'] = sttl
    flow['proto'] = proto
    flow['service'] = get_service_name(dst_port, proto)
    flow['state'] = state
    flow['is_sm_ips_ports'] = int(src_ip == dst_ip and src_port == dst_port)

    flow['ct_srv_src'] = update_last_ten_minutes_counts((flow['service'], src_ip), 'srv_src')
    flow['ct_dst_ltm'] = update_last_ten_minutes_counts(dst_ip, 'dst_ip')
    flow['ct_src_ltm'] = update_last_ten_minutes_counts(src_ip, 'src_ip')
    flow['ct_srv_dst'] = update_last_ten_minutes_counts((flow['service'], dst_ip), 'srv_dst')
    flow['ct_dst_src_ltm'] = update_last_ten_minutes_counts((dst_ip, src_ip), 'dst_src')
    flow['ct_srcdst_ltm'] = update_last_ten_minutes_counts((src_ip, dst_ip), 'src_dst')
    flow['ct_dst_sport_ltm'] = update_last_ten_minutes_counts((dst_ip, src_port), 'dst_sport_ltm')
    flow['ct_src_dport_ltm'] = update_last_ten_minutes_counts((src_ip, dst_port), 'src_dport_ltm')

    state_ttl_counts[src_ip][(flow['state'], flow['sttl'])] += 1
    flow['ct_state_ttl'] = len(state_ttl_counts[src_ip])

    return flow_key, flow

def cleanup_old_flows(current_time, flow_idle_timeout, flow_active_timeout):
    expired = [fk for fk, f in active_flows.items()
               if (current_time - f['last_packet_time'] > flow_idle_timeout) or
                  (current_time - f['start_time'] > flow_active_timeout)]
    for fk in expired:
        del active_flows[fk]
    return expired

def get_processed_features_for_prediction(raw_features, scaler, label_encoders, feature_columns):
    df = pd.DataFrame([raw_features])
    df.drop(columns=['start_time', 'last_packet_time', 'last_src_pkt_time', 'last_dst_pkt_time',
                     'prev_sintpkt', 'prev_dintpkt'], errors='ignore', inplace=True)

    # Encode categorical features
    for col, le in label_encoders.items():
        if col in df.columns:
            df[col] = df[col].map(lambda s: s if s in le.classes_ else le.classes_[0])
            df[col] = le.transform(df[col])
        else:
            df[col] = 0  # Missing encoded column

    # Fill missing numeric features and scale
    for col in scaler.feature_names_in_:
        if col not in df.columns:
            df[col] = 0
        df[col] = df[col].fillna(0)

    # Apply scaler
    df[scaler.feature_names_in_] = scaler.transform(df[scaler.feature_names_in_])

    # Ensure all model-required features are present
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0

    # Reorder to match model input
    df = df[feature_columns]

    return df
# feature_extractor.py

def extract_features(packet):
    """
    Converts a packet into numerical features for prediction.
    Replace with the same feature extraction logic you used while training.
    """
    try:
        ip_layer = packet.getlayer("IP")
        if not ip_layer:
            return [0, 0, 0, 0, 0]  # dummy fallback for non-IP packets

        src_len = len(ip_layer.src)
        dst_len = len(ip_layer.dst)
        ttl = ip_layer.ttl
        proto = ip_layer.proto
        total_len = ip_layer.len

        return [src_len, dst_len, ttl, proto, total_len]
    except Exception as e:
        print(f"❌ Feature extraction error: {e}")
        return [0, 0, 0, 0, 0]

