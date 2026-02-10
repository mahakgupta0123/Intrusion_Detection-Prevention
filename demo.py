from scapy.all import rdpcap, IP, TCP, UDP, ICMP

pcap_file = "demo_attacks.pcap"
packets = rdpcap(pcap_file)

print(f"Total packets in {pcap_file}: {len(packets)}")

# Print a summary of the first few packets
for i, pkt in enumerate(packets[:20]): # Adjust 20 to see more or fewer
    print(f"Packet {i}: {pkt.summary()}")

# You can filter and count specific types of packets/flows as well
syn_packets = [p for p in packets if p.haslayer(TCP) and p[TCP].flags == 'S']
print(f"Total SYN packets: {len(syn_packets)}")

icmp_echo_requests = [p for p in packets if p.haslayer(ICMP) and p[ICMP].type == 8]
print(f"Total ICMP Echo Requests: {len(icmp_echo_requests)}")



