# Part D: GPU Baseline Comparison (NVIDIA V100)

## System Configuration Comparison

### Eyeriss 16×16 (ASIC)
- **Array:** 16×16 spatial array (256 PEs)
- **Frequency:** 1 GHz
- **Compute Peak:** 256 GMACs/s (FP16)
- **Power Budget:** 33.3 W peak
- **Energy per MAC:** 0.130 pJ
- **Memory:** 64 KB GLB + 896 KB distributed SRAM + GDDR6

### NVIDIA V100 (GPU)
- **Cores:** 5,120 CUDA cores
- **Frequency:** 1.3 GHz
- **Compute Peak (FP16):** 13.3 TFLOPs/s = 13,300 GMACs/s
- **Power Budget:** 250 W TDP
- **Energy per MAC:** 18.8 pJ (estimated)
- **Memory:** 32 GB HBM2 (900 GB/s bandwidth)

---

## Performance Comparison (BERT Attention 768×768)

### Throughput
| System | Peak Throughput | Achieved on BERT | Efficiency |
|--------|-----------------|------------------|------------|
| **Eyeriss 16×16** | 256 GMACs/s | 256 GMACs/s | 100% |
| **V100** | 13,300 GMACs/s | 8,000–10,000 GMACs/s | 60–75% |
| **Ratio** | 52× (V100 vs Eyeriss) | 31–39× (V100 vs Eyeriss) | — |

**Note:** V100 is faster but has lower utilization on small/medium workloads (BERT attention is compute-light for GPUs).

### Latency
| System | Cycles (1GHz) | Wall-clock Time |
|--------|---------------|-----------------|
| **Eyeriss 16×16** | 294,912 | 295 µs |
| **V100 @ 1.3 GHz** | ~50,000–80,000 equivalent | 40–60 µs |
| **Ratio** | 4–6× faster (V100) | — |

---

## Energy Comparison (BERT Attention 768×768)

### Per-Workload Energy

**Eyeriss 16×16:**
- Total: 9.83 mJ
- Peak power: 33.3 W
- Duration: 295 µs
- Avg power: 33.3 W

**V100:**
- Peak power: 250 W
- Latency: 50 µs
- Energy estimate: 250 W × 50 µs = 12.5 mJ (conservative)
- More realistic (partial utilization): 150 W × 50 µs = 7.5 mJ

### Energy per MAC (Efficiency)

| System | Total Energy | MACs | Energy/MAC | Efficiency |
|--------|-------------|------|-----------|------------|
| **Eyeriss 16×16** | 9.83 mJ | 904M | 0.130 pJ | Baseline |
| **V100 (75% util)** | 7.5 mJ | 904M | 0.0083 pJ | **62× better** |
| **V100 (50% util)** | 12.5 mJ | 904M | 0.0138 pJ | **9× better** |

**V100 has better energy efficiency per MAC** due to:
1. Advanced 7nm process (vs Eyeriss 45nm equivalent)
2. Highly optimized data paths
3. Efficient memory subsystem (HBM2)

---

## Power Consumption Comparison

### Peak Power
| System | Peak Power | Operating Mode |
|--------|-----------|-----------------|
| **Eyeriss 16×16** | 33.3 W | Sustained (all PEs active) |
| **V100** | 250 W | Full utilization |
| **V100 (50% util)** | 125 W | Typical BERT inference |
| **Ratio** | 3.75–7.5× (V100 > Eyeriss) | — |

### Power Budget Compliance
| System | Power Limit | Peak Power | Status |
|--------|-----------|-----------|--------|
| **Eyeriss 16×16** | 120 W (edge) | 33.3 W | ✅ Safe (27.6%) |
| **V100** | 250 W (datacenter) | 250 W | ✅ At limit |
| **V100 in edge** | 120 W | 250 W | ❌ Infeasible |

---

## Use Case Analysis

### Edge Deployment (120 W Budget)
| Scenario | Eyeriss 16×16 | V100 | Winner |
|----------|---------------|------|--------|
| **Power compliance** | ✅ 33.3 W | ❌ 250 W | Eyeriss |
| **Throughput** | 256 GMACs/s | 13,300 GMACs/s | V100 (but violates budget) |
| **Energy/MAC** | 0.130 pJ | 0.0083 pJ | V100 (if power-limited: tie) |
| **Suitable?** | YES | NO | **Eyeriss** |

### Datacenter Deployment (No Power Limit)
| Scenario | Eyeriss 16×16 | V100 | Winner |
|----------|---------------|------|--------|
| **Throughput** | 256 GMACs/s | 8,000–13,300 GMACs/s | V100 (31–52×) |
| **Energy/MAC** | 0.130 pJ | 0.0083 pJ | V100 (62×) |
| **Cost per inference** | ~$2 (ASIC amortized) | ~$15,000 (GPU hardware) | Eyeriss (at scale) |
| **Latency** | 295 µs | 50 µs | V100 (6×) |
| **Suitable?** | Acceptable | YES | **V100** |

---

## Specialized Workload: INT8 Precision

**With INT8 quantization:**

| System | Compute Peak | Energy/MAC (INT8) | Status |
|--------|------------|-------------------|--------|
| **Eyeriss INT8** | 512 GMACs/s | 0.065 pJ | Best for edge |
| **V100 INT8** | 26,600 GMACs/s | 0.004 pJ | Best for datacenter |

INT8 narrows the energy gap but doesn't change the use-case winner.

---

## Cost-Performance Analysis

### Hardware Cost
| System | Unit Cost | Amortized Cost/Inference (1M inferences) |
|--------|----------|----------------------------------------|
| **Eyeriss ASIC** | $50–200 (estimated) | $0.00005–0.0002 |
| **V100 GPU** | $10,000–15,000 | $0.01–0.015 |
| **Ratio** | 50–300× cheaper (ASIC) | — |

### Total Cost of Ownership (1M inferences)

**Eyeriss (edge device):**
- Hardware: $100 (×100 devices) = $10,000
- Power: 33.3 W × 295 µs × 1M = 9.8 kWh @ $0.10/kWh = $1
- **Total: ~$10,000**

**V100 (datacenter):**
- Hardware: $12,000
- Power: 150 W × 50 µs × 1M = 8.3 kWh @ $0.10/kWh = $0.83
- Amortization (3-year lifetime, 1B inferences): $12,000/1000 = $12
- **Total: ~$13**

---

## Summary Table

| Metric | Eyeriss 16×16 | V100 | Winner |
|--------|---|---|---|
| **Throughput** | 256 GMACs/s | 10,000 GMACs/s | V100 (39×) |
| **Energy/MAC** | 0.130 pJ | 0.0083 pJ | V100 (16×)* |
| **Peak Power** | 33.3 W | 250 W | Eyeriss |
| **Edge deployment** | ✅ Suitable | ❌ Over budget | Eyeriss |
| **Datacenter** | ⚠️ Acceptable | ✅ Ideal | V100 |
| **Cost/inference** | $0.00005 | $0.01 | Eyeriss (200×) |
| **Latency** | 295 µs | 50 µs | V100 (6×) |

*V100 is more energy-efficient per MAC due to advanced process; however, at edge power budgets, Eyeriss dominates.

---

## Recommendation

- **Edge IoT devices (120 W limit):** Deploy Eyeriss INT8
- **Cloud inference (no limit):** Use V100 or newer GPUs (A100)
- **Hybrid approach:** Edge (Eyeriss) + occasional cloud offload (V100) for large batches
