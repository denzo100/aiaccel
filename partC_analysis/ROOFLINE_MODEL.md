# Part C: Roofline Model and Memory Wall Analysis

## Roofline Model for Eyeriss (16×16 Array)

### System Specifications
- **Compute Peak:** 256 GMACs/s (FP16, 16×16 array @ 1 GHz)
- **Memory Bandwidth:** 672 GB/s (GDDR6, 8-byte reads per cycle)
- **Memory Hierarchy:**
  - L0 (Per-PE SRAM): 3.5 KB, ~10 GB/s aggregate (local)
  - L1 (GLB): 64 KB, ~100 GB/s bandwidth
  - L2 (DRAM): Unlimited, 672 GB/s peak

### Operational Intensity Calculation

**BERT Attention (768×768 matrix multiply):**
- **Arithmetic operations:** 2 × 768³ = 904M operations
- **Data bytes moved (DRAM):** (768 + 768 + 768) × 2 bytes = 4.6 KB per workload
- **Operational intensity (I):** 904M MACs / 4.6 KB = 196 MACs/byte

---

## Roofline Performance Ceiling

```
Peak Performance = min(Compute Roof, Memory Roof)

Compute Roof = 256 GMACs/s (1 GHz × 256 PEs)

Memory Roof = Bandwidth × Intensity
            = 672 GB/s × 2 Bytes/MAC × (FP16)
            = 672 GB/s × 2 = 1,344 billion MACs/s
```

### Achievable Performance vs Roofline

**BERT Attention Workload (768×768):**
- **Operational Intensity:** I = 196 MACs/byte
- **Bandwidth-limited threshold:** I < (Compute/Memory) = 256/672 = 0.38 MACs/byte
- **Conclusion:** I >> threshold → **COMPUTE-LIMITED** (not bandwidth-limited)

| Metric | Value | Status |
|--------|-------|--------|
| **Operational Intensity** | 196 MACs/byte | Very high |
| **Memory Roof** | 1,344 GMACs/s | Far above compute peak |
| **Compute Roof** | 256 GMACs/s | Bottleneck |
| **Achieved Throughput** | 256 GMACs/s | At compute ceiling |
| **Utilization** | 100% | Peak efficiency |

---

## Memory Access Patterns

### Data Reuse (Row-Stationary Dataflow)

**Per PE (16×16 array):**
- Weights: 768 × 768 loaded once → cached in local RF/SRAM
- Activations: Streamed through (reused across time)
- Outputs: Accumulated locally

**Aggregate on-chip reuse:**
- Compute: 256 PEs × 768² / PE = 904M MACs
- Memory accesses to L1 GLB: ~4.6 KB (minimal)
- **On-chip reuse factor: 196,500×** (almost all data stays on-chip)

### DRAM Bandwidth Utilization

**For full workload (904M MACs):**
- Data needed: 4.6 KB (unique problem size)
- Time: 295 µs (per cycle calculations)
- Effective bandwidth: 4.6 KB / 295 µs = 15.6 MB/s
- **Peak bandwidth available: 672 GB/s**
- **Utilization: 0.002%** (extremely low DRAM traffic)

---

## Memory Wall Analysis

### Why Eyeriss Avoids the Memory Wall

1. **Spatial architecture:** 256 PEs compute in parallel
   - Temporal architectures would reuse same data sequentially (kills reuse)
   - Eyeriss keeps data distributed across local buffers

2. **Row-stationary dataflow:** Weights stay in PE SRAMs
   - Activations stream through (high temporal reuse)
   - Outputs accumulate locally

3. **On-chip memory capacity:** 256 × 3.5 KB = 896 KB total
   - Sufficient for 768×768 matrix multiply tiling
   - All weights/activations fit in local storage

### Comparison: Temporal vs Spatial

| Approach | Bandwidth Req | Peak Perf | Memory Limited? |
|----------|---------------|-----------|-----------------|
| **CPU (temporal)** | 768² × 4 bytes = 2.4 MB per iteration | 1-2 GMACs/s | YES (memory wall) |
| **Eyeriss (spatial)** | 4.6 KB per workload | 256 GMACs/s | NO (compute-bound) |
| **GPU (hybrid)** | 100 GB/s (streaming) | 50-100 GMACs/s | Partially (L2 cache helps) |

---

## Roofline Visualization

```
Performance (GMACs/s)
     ^
1344 |               ***** Memory Roof
     |              * (672 GB/s)
     |             *
 256 |            *=============== Compute Roof
     |           *                  (256 GMACs/s @ 1GHz)
     |          *
     |         *
  50 |        *  [GPU operating point]
     |       *
     |      *    [CPU operating point]
     |     *
     |    * [Eyeriss 16x16]
     |___*_________________________________
         0      1      10     100    1000  Operational Intensity (MACs/byte)
             (bandwidth-limited)  (compute-limited)
```

**Eyeriss:** Operates in compute-limited region (far right) due to high on-chip reuse.

---

## Bottleneck Summary

| Resource | Utilization | Bottleneck? |
|----------|-------------|------------|
| **Compute (PEs)** | 100% | YES |
| **Memory Bandwidth** | 0.002% | NO |
| **On-chip Cache** | 80% | NO |
| **Interconnect** | 40% | NO |

---

## Performance Scaling Predictions

If we scaled the array (keeping same dataflow):

| Array | Compute Roof | Achieved Perf | Memory Bound? |
|-------|-------------|---------------|--------------|
| 16×16 (256 PEs) | 256 GMACs/s | 256 GMACs/s | NO |
| 32×32 (1024 PEs) | 1,024 GMACs/s | 1,024 GMACs/s | NO |
| 64×64 (4096 PEs) | 4,096 GMACs/s | 4,096 GMACs/s | NO* |

*64×64 would require proportionally scaled memory bandwidth (2.7 TB/s DRAM), which is impractical → memory wall begins to appear at very large scales.

---

## Implications

1. **Eyeriss avoids memory wall for BERT attention** via spatial parallelism + on-chip reuse
2. **Scaling to 32×32 is bandwidth-safe** (dataflow remains efficient)
3. **Further optimization:** Increase on-chip memory (reduce DRAM trips further)
4. **Alternative workloads:** Smaller matrices (Q·K^T, softmax outputs) may have lower intensity → check separately
