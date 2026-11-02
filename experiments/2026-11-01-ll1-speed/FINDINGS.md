# LongLive 1: speed and smoothness after the LongLive 2.0 lessons (2026-11-01)

Setup: the desk webcam stock clip (looped), 640×352, prompt "man in a cabin", noise 0.9, run through Scope and the bridge, 60 s per run, with `--latency-probe` for frame-gap stats.

## Fixes carried over from the LongLive 2.0 port (committed)
1. **Output as CPU uint8, one copy per chunk** (`process.to_cpu_uint8`, applied in LongLive, MemFlow and Reward Forcing). Before, Scope's output track copied each GPU frame separately, and each copy waited behind the next chunk's queued DiT work.
2. **Input frames uploaded on a side CUDA stream** (`frame_processor._frame_array_to_gpu`). Before, each camera frame's upload waited for all in-flight GPU work and blocked the WebRTC event loop.

Results, 1 step with depth:

| | fps | stutters > 66 ms | p95 frame gap |
|---|---|---|---|
| before | 17.1 | 157 / 803 | 115 ms |
| **after** | **17.4** | **27 / 831** | **64 ms** |

**Fixed along the way:** `attention_steer.set_weights` had been deleted by an earlier edit (commit 7305bdb), which broke LongLive, MemFlow and Reward Forcing. It is restored.

## Where 3 steps lose speed
| | 1 step | 3 steps, clean every 2nd |
|---|---|---|
| depth off | 21.3 | 13.4 |
| depth on | 17.4 | 10.1 (11.4 with VACE hint reuse) |

- **The cost is linear in DiT passes per chunk:** 3.5 passes against 2, a ratio of 1.75, which matches 21.3 → 13.4.
- **Depth/VACE costs about 20–25%.** Reusing the step-1 VACE hints for later steps (new switch, off by default) recovers only about 13%, and it is an approximation that still needs a visual check.
- **Conclusion:** overhead tricks can't give 3–5-step quality at 20 fps on this laptop. It needs cheaper passes or a model that is better at fewer steps: Causal Forcing++ 2-step ("better than 4-step") or the OPSD-V LoRA. Those are the research pass's top picks.
