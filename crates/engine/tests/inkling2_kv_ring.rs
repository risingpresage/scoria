//! The windowed KV ring against the unbounded history, on the real checkpoint.
//!
//! A sliding-window visit only attends to the last `window` positions, so evicting older
//! ones cannot change the answer -- but only if the ring maps absolute positions, biases and
//! masks correctly. The oracle is the *unbounded* store with the same window: same visible
//! range, same absolute distances, same arithmetic, so the logits must match **bit for
//! bit**.
//!
//! The window is overridden to 64 so that 150 tokens evict ~86 times: an off-by-one in the
//! ring, an absolute-versus-relative distance slip, or a stale slot surviving eviction all
//! show up as a different logit. At the checkpoint's real window of 1024 the same bug would
//! need 1024+ tokens to appear, which is minutes of compute per run.

use scoria_engine::inkling2_model::Model;

const GGUF: &str = "/home/noorm/models/inkling/Inkling-2-Mini-UD-Q3_K_XL.gguf";
const WINDOW: usize = 64;
const TOKENS: usize = 150;

fn run(ring: bool) -> Vec<f32> {
    let mut model = Model::open(GGUF).expect("checkpoint");
    let _ = model.enable_gpu();
    model.configure_kv(Some(WINDOW), ring);
    // A varied token stream, deterministic: token ids from the checkpoint's own vocabulary.
    let ids: Vec<u32> = (0..TOKENS)
        .map(|i| 300 + ((i * 37) % 4000) as u32)
        .collect();
    let mut logits = Vec::new();
    for id in ids {
        logits = model.forward(id);
    }
    let (windowed, full) = model.kv_bytes();
    if ring {
        // The ring must be bounded: 40 windowed visits (20 sliding layers across 3 loop
        // passes, minus the full-history ones) x 2 tensors x 64 slots x 1024 values, and no
        // growth with the sequence length.
        assert!(
            windowed <= 40 * 2 * WINDOW * 1024 * 4,
            "windowed KV is not bounded by the window: {windowed} bytes"
        );
    } else {
        assert!(
            windowed > 40 * 2 * WINDOW * 1024 * 4,
            "the unbounded store should exceed the ring bound: {windowed} bytes"
        );
    }
    assert!(full > 0, "full-history visits must retain history");
    logits
}

#[test]
fn windowed_ring_reproduces_the_unbounded_history_exactly() {
    // This runs the full 48-visit stack twice over the real checkpoint, which is ~20x
    // slower in an unoptimised build -- long enough to stall the gate. It is not skipped:
    // `scripts/check.sh` runs this target again under `--release`, where it takes ~3 min.
    // The ring's own invariants (sliding, growth, slot mapping) are unit tests in
    // `inkling2_model` and run in both builds.
    if cfg!(debug_assertions) {
        eprintln!("deferring to the release run in scripts/check.sh");
        return;
    }
    let ring = run(true);
    let full = run(false);
    assert_eq!(ring.len(), full.len());
    let differing = ring.iter().zip(&full).filter(|(a, b)| a != b).count();
    let worst = ring
        .iter()
        .zip(&full)
        .fold(0f32, |m, (a, b)| m.max((a - b).abs()));
    assert!(
        differing == 0,
        "{differing} logits differ between the ring and the unbounded history (max |diff| {worst})"
    );
    assert!(
        ring.iter().all(|v| v.is_finite()),
        "logits must stay finite past the window"
    );
}
