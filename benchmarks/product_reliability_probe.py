"""Reproducible write-transaction and recall-SQL probe; no neural quality claims.

Run against two source trees with the same Python environment, for example:
  python benchmarks/product_reliability_probe.py --source-tree /tmp/memoria-baseline
  python benchmarks/product_reliability_probe.py --source-tree .

All data lives in temporary databases. Provider output and reranking are fixed;
this measures database overhead, not inference latency or benchmark accuracy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import tempfile
import time

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-tree', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--facts', type=int, default=1000)
    parser.add_argument('--runs', type=int, default=3)
    args = parser.parse_args()
    if args.facts < 1 or args.runs < 1:
        parser.error('--facts and --runs must be positive')
    sys.path.insert(0, str(args.source_tree.resolve()))
    from memoria.core import Memoria

    class FixedReranker:
        def predict(self, pairs):
            return np.ones(len(pairs), dtype=float)

    extraction = json.dumps({
        'entities': [{'name': f'Entity{index}'} for index in range(args.facts)],
        'facts': [{'subject': f'Entity{index}', 'predicate': 'records',
                   'object': f'reliability marker {index}'} for index in range(args.facts)],
    })
    measurements = []
    for _ in range(args.runs):
        with tempfile.TemporaryDirectory(prefix='memoria-probe-') as directory:
            with Memoria(Path(directory) / 'probe.db', enable_embeddings=False,
                         llm_call=lambda _: extraction) as memory:
                statements = []
                memory.db.set_trace_callback(statements.append)
                started = time.perf_counter()
                memory.remember('fixed synthetic provenance')
                elapsed = time.perf_counter() - started
                commits = sum(sql.strip().upper() == 'COMMIT' for sql in statements)
                memory.retriever.llm_call = None
                memory.retriever.enable_reranking = True
                memory.retriever._reranker = FixedReranker()
                statements.clear()
                recalled = memory.recall('reliability marker', top_k=10)
                reads = sum(sql.lstrip().upper().startswith('SELECT') for sql in statements)
                measurements.append({
                    'write_seconds': round(elapsed, 6), 'write_commits': commits,
                    'recall_selects': reads, 'results': len(recalled.results),
                })
    print(json.dumps({
        'facts': args.facts, 'runs': args.runs,
        'write_seconds_median': statistics.median(row['write_seconds'] for row in measurements),
        'measurements': measurements,
    }, indent=2))


if __name__ == '__main__':
    main()
