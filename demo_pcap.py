# create_demo_pcap.py
from scapy.all import IP, TCP, UDP, ICMP, Ether, Raw, wrpcap
import time

packets = []

# --- Normal Traffic Simulation ---
print("Generating normal traffic...")
for i in range(5): # Generate a few normal HTTP and DNS flows
    # Basic HTTP request (SYN, SYN-ACK, ACK, PSH/ACK for GET, FIN/ACK, ACK)
    # Source IP and Port for this flow
    src_ip_normal = f"192.168.1.10{i}"
    dst_ip_normal = "192.168.1.1"
    src_port_normal = 10000 + i
    dst_port_normal = 80

    # SYN
    p_syn = Ether()/IP(src=src_ip_normal, dst=dst_ip_normal)/TCP(sport=src_port_normal, dport=dst_port_normal, flags="S", seq=1000)
    packets.append(p_syn)
    # SYN-ACK
    p_synack = Ether()/IP(src=dst_ip_normal, dst=src_ip_normal)/TCP(sport=dst_port_normal, dport=src_port_normal, flags="SA", seq=2000, ack=p_syn.seq + 1)
    packets.append(p_synack)
    # ACK (completes handshake)
    p_ack = Ether()/IP(src=src_ip_normal, dst=dst_ip_normal)/TCP(sport=src_port_normal, dport=dst_port_normal, flags="A", seq=p_syn.seq + 1, ack=p_synack.seq + 1)
    packets.append(p_ack)
    # PSH/ACK (HTTP GET)
    p_get = Ether()/IP(src=src_ip_normal, dst=dst_ip_normal)/TCP(sport=src_port_normal, dport=dst_port_normal, flags="PA", seq=p_ack.seq, ack=p_ack.ack)/Raw(load="GET /index.html HTTP/1.1\r\nHost: example.com\r\n\r\n")
    packets.append(p_get)
    # FIN/ACK (server closes)
    p_finack = Ether()/IP(src=dst_ip_normal, dst=src_ip_normal)/TCP(sport=dst_port_normal, dport=src_port_normal, flags="FA", seq=p_synack.seq + len(p_get[Raw].load) + 1, ack=p_get.seq + len(p_get[Raw].load))
    packets.append(p_finack)
    # ACK (client acknowledges FIN)
    packets.append(Ether()/IP(src=src_ip_normal, dst=dst_ip_normal)/TCP(sport=src_port_normal, dport=dst_port_normal, flags="A", seq=p_get.seq + len(p_get[Raw].load), ack=p_finack.seq + 1))


    # UDP DNS query
    packets.append(Ether()/IP(src=f"192.168.1.20{i}", dst="8.8.8.8")/UDP(sport=50000+i, dport=53)/Raw(load=b"\x00\x01\x01\x00\x00\x01\x00\x00\x00\x00\x00\x00\x03www\x06google\x03com\x00\x00\x01\x00\x01"))


# --- Attack 1: SYN Flood Simulation ---
# Many SYN packets from a single source to one destination/port
print("Generating SYN Flood attack...")
attacker_syn_ip = "10.0.0.10"
target_syn_ip = "192.168.1.50"
target_syn_port = 80
for i in range(150): # Send 150 SYN packets
    pkt = Ether()/IP(src=attacker_syn_ip, dst=target_syn_ip)/TCP(sport=1024 + i, dport=target_syn_port, flags="S", seq=i*100)
    packets.append(pkt)
    # Optionally, add some return RSTs from target if port is closed/busy
    if i % 10 == 0:
        packets.append(Ether()/IP(src=target_syn_ip, dst=attacker_syn_ip)/TCP(sport=target_syn_port, dport=1024 + i, flags="R", seq=i*200))


# --- Attack 2: Port Scan Simulation (many ports from one source to one destination) ---
print("Generating Port Scan attack...")
scanner_ip = "10.0.0.20"
scanned_ip = "192.168.1.50" # Same target as SYN flood, to aggregate features
for i in range(1, 80): # Scan ports 1-79
    # SYN packet to initiate scan
    pkt_scan = Ether()/IP(src=scanner_ip, dst=scanned_ip)/TCP(dport=i, flags="S", sport=50000+i, seq=i*50)
    packets.append(pkt_scan)
    # Simulate RST for closed ports or SYN/ACK then RST for open ports after initial SYN
    if i % 5 == 0: # Simulate an open port and then tear down
        packets.append(Ether()/IP(src=scanned_ip, dst=scanner_ip)/TCP(sport=i, dport=50000+i, flags="SA", ack=pkt_scan.seq+1, seq=i*60))
        packets.append(Ether()/IP(src=scanner_ip, dst=scanned_ip)/TCP(sport=50000+i, dport=i, flags="R", ack=pkt_scan[TCP].ack+1, seq=pkt_scan.seq+1)) # Client sends RST
    else: # Simulate closed port with RST
        packets.append(Ether()/IP(src=scanned_ip, dst=scanner_ip)/TCP(sport=i, dport=50000+i, flags="R", ack=pkt_scan.seq+1, seq=i*70))


# --- Attack 3: ICMP Flood (Ping Flood) ---
print("Generating ICMP Flood attack...")
icmp_attacker = "10.0.0.30"
icmp_target = "192.168.1.60"
for i in range(100): # Send 100 ICMP echo requests
    pkt = Ether()/IP(src=icmp_attacker, dst=icmp_target)/ICMP(type=8, code=0, id=1, seq=i)
    packets.append(pkt)

# Save all packets to a PCAP file
pcap_filename = "demo_attacks.pcap" # This is the file name used in real_time_ips.py
wrpcap(pcap_filename, packets)
print(f"Generated {len(packets)} packets and saved to {pcap_filename}")