"""
Generate synthetic botnet and normal network traffic data
based on CTU-13 style characteristics.
"""

import random
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

DATA_DIR = Path(__file__).parent / "data"
RANDOM_SEED = 42
random.seed(RANDOM_SEED); np.random.seed(RANDOM_SEED)

NUM_SAMPLES = 20000
BOTNET_RATIO = 0.20

PROTOCOLS = ["tcp", "udp", "icmp"]
SERVICES = ["dns", "http", "https", "smtp", "ssh", "ftp", "irc", "unknown"]
BOTNET_C2_PORTS = [443, 8080, 6667, 7777, 8443, 12345, 31337, 4444, 1337]
NORMAL_PORTS = [80, 443, 22, 25, 53, 110, 143, 993, 8080]
TCP_FLAGS = ["SYN", "SYN-ACK", "ACK", "FIN", "RST", "PSH", "URG"]


PROTOCOL_ENCODER = {"tcp": 0, "udp": 1, "icmp": 2}


def generate_normal_flow():
    """Generate a normal network flow."""
    feats = {}
    feats["duration_s"] = random.uniform(0.01, 300)
    feats["protocol"] = float(PROTOCOL_ENCODER[random.choice(PROTOCOLS)])
    feats["src_port"] = random.randint(1024, 65535)
    feats["dst_port"] = random.choice(NORMAL_PORTS)
    feats["src_bytes"] = float(np.random.lognormal(5, 2))
    feats["dst_bytes"] = float(np.random.lognormal(4, 2))
    feats["packets_sent"] = max(1, int(np.random.lognormal(2, 1.5)))
    feats["packets_received"] = max(1, int(np.random.lognormal(2, 1.5)))
    feats["bytes_per_packet"] = float(np.random.lognormal(6, 1))
    feats["packet_rate"] = feats["packets_sent"] / max(feats["duration_s"], 0.1)
    feats["byte_rate"] = (feats["src_bytes"] + feats["dst_bytes"]) / max(feats["duration_s"], 0.1)
    feats["flow_duration_ms"] = feats["duration_s"] * 1000
    feats["tcp_flag_count"] = random.randint(1, 5)
    feats["syn_count"] = random.randint(1, 3)
    feats["rst_count"] = random.randint(0, 1)
    feats["fin_count"] = random.randint(0, 1)
    feats["ack_count"] = random.randint(1, 5)
    feats["inter_arrival_mean_ms"] = float(np.random.lognormal(1, 1))
    feats["inter_arrival_std_ms"] = float(np.random.lognormal(0.5, 1))
    feats["inter_arrival_min_ms"] = random.uniform(0.1, 10)
    feats["inter_arrival_max_ms"] = random.uniform(100, 5000)
    feats["small_packets_ratio"] = random.uniform(0, 0.2)
    feats["large_packets_ratio"] = random.uniform(0.3, 0.8)
    feats["payload_bytes_mean"] = float(np.random.lognormal(5, 2))
    feats["payload_bytes_std"] = float(np.random.lognormal(4, 2))
    feats["num_distinct_src_ports"] = 1
    feats["num_distinct_dst_ports"] = random.randint(1, 3)
    feats["dns_query_count"] = random.randint(0, 5)
    feats["http_request_count"] = random.randint(0, 20)
    feats["irc_message_count"] = 0
    feats["c2_communication_score"] = 0
    feats["connection_entropy"] = random.uniform(0.3, 0.7)
    feats["is_botnet"] = 0
    return feats


def generate_botnet_flow():
    """Generate a botnet network flow with C2 characteristics."""
    feats = {}
    is_c2 = random.random() < 0.4  # 40% are C2 connections
    feats["duration_s"] = random.uniform(1, 3600) if is_c2 else random.uniform(0.01, 600)
    feats["protocol"] = float(PROTOCOL_ENCODER.get("tcp" if random.random() < 0.9 else random.choice(PROTOCOLS), 0))
    feats["src_port"] = random.randint(1024, 65535)
    feats["dst_port"] = random.choice(BOTNET_C2_PORTS)
    feats["src_bytes"] = float(np.random.lognormal(4, 2)) if is_c2 else float(np.random.lognormal(5, 2))
    feats["dst_bytes"] = float(np.random.lognormal(6, 2)) if is_c2 else float(np.random.lognormal(4, 2))
    feats["packets_sent"] = max(1, int(np.random.lognormal(1, 1.5))) if is_c2 else max(1, int(np.random.lognormal(3, 1.5)))
    feats["packets_received"] = max(1, int(np.random.lognormal(1, 1.5))) if is_c2 else max(1, int(np.random.lognormal(3, 1.5)))
    feats["bytes_per_packet"] = float(np.random.lognormal(4, 1)) if is_c2 else float(np.random.lognormal(6, 1))
    feats["packet_rate"] = feats["packets_sent"] / max(feats["duration_s"], 0.1)
    feats["byte_rate"] = (feats["src_bytes"] + feats["dst_bytes"]) / max(feats["duration_s"], 0.1)
    feats["flow_duration_ms"] = feats["duration_s"] * 1000
    feats["tcp_flag_count"] = random.randint(2, 8)
    # Botnets often have unusual TCP flag patterns
    feats["syn_count"] = random.randint(0, 5)
    feats["rst_count"] = random.randint(0, 5)
    feats["fin_count"] = random.randint(0, 5)
    feats["ack_count"] = random.randint(0, 10)
    # Periodic communication → regular inter-arrival times
    feats["inter_arrival_mean_ms"] = float(np.random.lognormal(2, 0.5)) if is_c2 else float(np.random.lognormal(0.5, 2))
    feats["inter_arrival_std_ms"] = float(np.random.lognormal(0.5, 0.5)) if is_c2 else float(np.random.lognormal(1, 2))
    feats["inter_arrival_min_ms"] = random.uniform(1, 50)
    feats["inter_arrival_max_ms"] = random.uniform(100, 10000)
    feats["small_packets_ratio"] = random.uniform(0.3, 0.9) if is_c2 else random.uniform(0, 0.4)
    feats["large_packets_ratio"] = random.uniform(0, 0.3)
    feats["payload_bytes_mean"] = float(np.random.lognormal(3, 1)) if is_c2 else float(np.random.lognormal(5, 2))
    feats["payload_bytes_std"] = float(np.random.lognormal(2, 1)) if is_c2 else float(np.random.lognormal(4, 2))
    feats["num_distinct_src_ports"] = random.randint(1, 5)
    feats["num_distinct_dst_ports"] = random.randint(1, 10)
    feats["dns_query_count"] = random.randint(5, 50)
    feats["http_request_count"] = random.randint(0, 5)
    feats["irc_message_count"] = random.randint(0, 100) if is_c2 else random.randint(0, 10)
    feats["c2_communication_score"] = random.uniform(0.6, 1.0)
    feats["connection_entropy"] = random.uniform(0.1, 0.4)
    feats["is_botnet"] = 1
    return feats


FEATURE_NAMES = [
    "duration_s", "protocol", "src_port", "dst_port", "src_bytes", "dst_bytes",
    "packets_sent", "packets_received", "bytes_per_packet", "packet_rate",
    "byte_rate", "flow_duration_ms", "tcp_flag_count", "syn_count", "rst_count",
    "fin_count", "ack_count", "inter_arrival_mean_ms", "inter_arrival_std_ms",
    "inter_arrival_min_ms", "inter_arrival_max_ms", "small_packets_ratio",
    "large_packets_ratio", "payload_bytes_mean", "payload_bytes_std",
    "num_distinct_src_ports", "num_distinct_dst_ports", "dns_query_count",
    "http_request_count", "irc_message_count", "c2_communication_score",
    "connection_entropy",
]


def main():
    DATA_DIR.mkdir(exist_ok=True)
    print("=" * 50); print("  Botnet Traffic — Data Generator"); print("=" * 50)
    n_bot = int(NUM_SAMPLES * BOTNET_RATIO); n_norm = NUM_SAMPLES - n_bot
    print(f"\n  Generating {n_norm} normal flows...")
    normal = [generate_normal_flow() for _ in tqdm(range(n_norm))]
    print(f"  Generating {n_bot} botnet flows...")
    botnet = [generate_botnet_flow() for _ in tqdm(range(n_bot))]
    all_flows = normal + botnet; labels = [0]*n_norm + [1]*n_bot
    combined = list(zip(all_flows, labels)); random.shuffle(combined)
    all_flows, labels = zip(*combined)
    df = pd.DataFrame(all_flows); df["is_botnet"] = labels
    out_path = DATA_DIR / "botnet_traffic.csv"
    df.to_csv(out_path, index=False)
    X = df[FEATURE_NAMES].values.astype(np.float64); y = np.array(labels, dtype=np.int64)
    np.savez_compressed(DATA_DIR / "botnet_traffic.npz", X=X, y=y)
    print(f"\n  Total: {len(df)} samples, {len(FEATURE_NAMES)} features")
    print(f"  Botnet ratio: {sum(labels)/len(labels):.1%}")

if __name__ == "__main__":
    main()
