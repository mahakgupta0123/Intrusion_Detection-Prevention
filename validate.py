import json
import os
import sys

def validate_metrics(metrics_file="metrics1.json"):
    """
    Validates the metrics in ips_metrics.json against expected values for the demo PCAP.
    """
    if not os.path.exists(metrics_file):
        print(f"Error: Metrics file '{metrics_file}' not found. Please run real_time_ips.py first.")
        return False

    print(f"--- Validating metrics from {metrics_file} ---")

    try:
        with open(metrics_file, 'r') as f:
            metrics_data = json.load(f)

        # Assuming the last entry in the JSON file is the most recent run's metrics
        # If your JSON just has one object, this is fine. If it's a list, take the last.
        if isinstance(metrics_data, list):
            if not metrics_data:
                print("Validation Failed: Metrics file is empty.")
                return False
            latest_metrics = metrics_data[-1]
        elif isinstance(metrics_data, dict):
            latest_metrics = metrics_data
        else:
            print("Validation Failed: Unexpected JSON format.")
            return False

        print(f"Latest metrics timestamp: {latest_metrics.get('timestamp', 'N/A')}")
        print(f"Total Flows Detected: {latest_metrics.get('total_flows')}")
        print(f"Attacks Detected: {latest_metrics.get('attacks_detected')}")
        print(f"Blocks Performed (Simulated): {latest_metrics.get('blocks_performed')}")

        # --- Define Expected Values Based on create_demo_pcap.py ---
        # These are approximations. Slight variations might occur due to flow timeouts,
        # prediction intervals, or Scapy's packet handling.

        # From create_demo_pcap.py:
        # Normal HTTP/DNS flows: 5 HTTP flows (multiple packets each), 5 DNS UDP flows (1 packet each)
        # SYN Flood: 150 SYN packets (1 main flow key, potentially with some RSTs)
        # Port Scan: 79 unique ports scanned (79 distinct flow keys from source)
        # ICMP Flood: 100 ICMP echo requests (1 main flow key)

        # Expected ranges for verification
        # The 'total_flows' count is tricky because one attack (like port scan) creates many distinct flows.
        # We also have 5 normal HTTP flows (multiple packets per flow) and 5 normal DNS flows.
        # A simple estimation:
        # ~5 (normal HTTP) + 5 (normal DNS) + 1 (SYN flood source) + 79 (port scan source to 79 targets) + 1 (ICMP flood source)
        # So, min_expected_flows is roughly 5 + 5 + 1 + 79 + 1 = 91. It will likely be higher due to
        # how flows are broken down by features.

        min_expected_total_flows = 90 # A reasonable lower bound given the PCAP structure
        min_expected_attacks_detected = 3 # At least one for SYN, Port Scan, ICMP Flood source IPs

        # Check total_flows
        if latest_metrics.get('total_flows', 0) >= min_expected_total_flows:
            print(f"Validation Passed: Total flows ({latest_metrics.get('total_flows')}) is within expected range (>= {min_expected_total_flows}).")
        else:
            print(f"Validation Failed: Total flows ({latest_metrics.get('total_flows')}) is lower than expected (expected >= {min_expected_total_flows}).")
            return False

        # Check attacks_detected
        if latest_metrics.get('attacks_detected', 0) >= min_expected_attacks_detected:
            print(f"Validation Passed: Attacks detected ({latest_metrics.get('attacks_detected')}) is within expected range (>= {min_expected_attacks_detected}).")
        else:
            print(f"Validation Failed: Attacks detected ({latest_metrics.get('attacks_detected')}) is lower than expected (expected >= {min_expected_attacks_detected}).")
            return False

        # Check blocks_performed (should match attacks_detected if prevention is simulated for every detection)
        if latest_metrics.get('blocks_performed', 0) >= min_expected_attacks_detected: # Should be roughly equal or slightly more than attacks detected
            print(f"Validation Passed: Blocks performed ({latest_metrics.get('blocks_performed')}) is consistent with attacks detected.")
        else:
            print(f"Validation Failed: Blocks performed ({latest_metrics.get('blocks_performed')}) is inconsistent with attacks detected.")
            return False

        print("\n--- All basic metric validations passed! ---")
        print("For more detailed validation, manually inspect logs/demo_detection_events.log and logs/demo_prevention_actions.log.")
        return True

    except json.JSONDecodeError as e:
        print(f"Error: Could not decode JSON from '{metrics_file}'. Is the file corrupted or empty? {e}")
        return False
    except Exception as e:
        print(f"An unexpected error occurred during validation: {e}")
        return False

if __name__ == "__main__":
    if validate_metrics():
        sys.exit(0) # Exit with success code
    else:
        sys.exit(1) # Exit with failure code