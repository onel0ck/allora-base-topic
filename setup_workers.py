import json
import yaml
import random
import os
from typing import Dict, List, Tuple

RPC_URL = ""

TOPIC_CONFIGS = {
    1: {"window": 12, "interval": (15, 25), "workers_percent": 15},
    3: {"window": 12, "interval": (15, 25), "workers_percent": 17},
    5: {"window": 12, "interval": (15, 25), "workers_percent": 17},
    7: {"window": 24, "interval": (50, 70), "workers_percent": 17},
    8: {"window": 24, "interval": (50, 70), "workers_percent": 17},
    9: {"window": 24, "interval": (50, 70), "workers_percent":  17},
    2: {"window": 60, "interval": (90, 120), "workers_percent": 0},
    4: {"window": 60, "interval": (90, 120), "workers_percent": 0},
    6: {"window": 60, "interval": (90, 120), "workers_percent": 0}
}


def generate_offsets(num_workers: int, window: int) -> List[int]:
    if num_workers <= window:
        return list(range(num_workers))

    offsets = []
    workers_per_slot = num_workers / window
    for i in range(num_workers):
        offset = int(i / workers_per_slot) % window
        offsets.append(offset)
    random.shuffle(offsets)
    return offsets

def create_base_wallet_config(seed_phrase: str, idx: int) -> dict:
    return {
        "addressKeyName": f"WALLET_{random.randint(1000, 9999)}",
        "addressRestoreMnemonic": seed_phrase,
        "alloraHomeDir": f"./root/.allorad_{idx + 1}",
        "gas": "auto",
        "gasAdjustment": 1.5,
        "gasPrices": "20",
        "gasPriceUpdateInterval": 60,
        "maxFees": 10000000,
        "nodeRpc": RPC_URL,
        "maxRetries": 1,
        "retryDelay": 3,
        "accountSequenceRetryDelay": 5,
        "submitTx": True,
        "blockDurationEstimated": 7,
        "windowCorrectionFactor": 0.8
    }
def distribute_topics(total_workers: int) -> List[int]:
    topic_distribution = []
    
    valid_topics = {k: v for k, v in TOPIC_CONFIGS.items() if v["workers_percent"] > 0}
    
    for topic_id, config in valid_topics.items():
        num_workers = int((config["workers_percent"] / 100) * total_workers)
        topic_distribution.extend([topic_id] * num_workers)
    
    valid_topic_ids = list(valid_topics.keys())
    while len(topic_distribution) < total_workers:
        topic_distribution.append(random.choice(valid_topic_ids))

    random.shuffle(topic_distribution)
    return topic_distribution

def main():
    with open('seed_phrases.txt', 'r') as f:
        seed_phrases = [line.strip() for line in f if line.strip()]

    total_workers = len(seed_phrases)
    config = {"workers": []}
    docker_compose = {'version': '3', 'services': {}}

    worker_template = {
        'image': 'alloranetwork/allora-offchain-node:v0.7.0',
        'depends_on': {'inference': {'condition': 'service_healthy'}},
        'deploy': {
            'resources': {
                'limits': {}
            }
        },
        'restart': 'on-failure:5'
    }

    topic_assignment = distribute_topics(total_workers)

    for worker_idx in range(total_workers):
        assigned_topic = topic_assignment[worker_idx]
        topic_config = TOPIC_CONFIGS[assigned_topic]
        min_interval, max_interval = topic_config["interval"]
        loop_seconds = random.randint(min_interval, max_interval)

        worker_task = {
            "topicId": assigned_topic,
            "inferenceEntrypointName": "apiAdapter",
            "loopSeconds": loop_seconds,
            "parameters": {
                "InferenceEndpoint": f"http://inference:8000/inference/{assigned_topic}?worker_id={worker_idx + 1}"
            }
        }

        node_config = {
            "wallet": create_base_wallet_config(seed_phrases[worker_idx], worker_idx),
            "worker": [worker_task]
        }

        config['workers'].append(node_config)
        docker_compose['services'][f'worker{worker_idx + 1}'] = {
            **worker_template,
            'container_name': f'allora-worker{worker_idx + 1}',
            'environment': [
                f'ALLORA_OFFCHAIN_NODE_CONFIG_JSON=${{WORKER{worker_idx + 1}_CONFIG}}',
                f'WORKER_ID={worker_idx + 1}'
            ]
        }

    docker_compose['services']['inference'] = {
        'build': '.',
        'command': 'python -u /app/app.py',
        'container_name': 'allora-inference',
        'env_file': ['.env'],
        'environment': [f'RPC_URL={RPC_URL}'],
        'healthcheck': {
            'test': ['CMD', 'curl', '-f', 'http://localhost:8000/health'],
            'interval': '30s',
            'timeout': '20s',
            'retries': 10,
            'start_period': '300s'
        },
        'ports': ['8000:8000'],
        'volumes': [
            './inference-data:/app/data',
            './logs:/app/logs'
        ],
        'restart': 'always',
        'deploy': {
            'resources': {'limits': {}}
        }
    }

    with open('multi_worker_config.json', 'w') as f:
        json.dump(config, f, indent=2)

    with open('docker-compose.yml', 'w') as f:
        yaml.dump(docker_compose, f)

    tokens = "ETH,BTC,SOL,BNB,ARB"
    env_content = [f"TOKENS={tokens}", "MODEL=SVR", f"RPC_URL={RPC_URL}"]
    for i, worker in enumerate(config['workers'], 1):
        env_content.append(f"WORKER{i}_CONFIG='{json.dumps(worker)}'")

    with open('.env', 'w') as f:
        f.write('\n'.join(env_content))

    print(f"\nTotal nodes: {total_workers}")
    print(f"Active topics: {list(set(topic_assignment))}")
    print("\nNodes configured in worker mode only")

if __name__ == "__main__":
    main()
