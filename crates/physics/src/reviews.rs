//! Agent review pipeline: error detection modeling.
//!
//! Models multi-agent review pipelines vs human-only review,
//! based on OpenAI's finding that agent-to-agent reviews can
//! replace most human reviews with 80%+ positive comment rates.

/// Simulate a multi-stage review pipeline.
///
/// Returns vectors of (errors_caught, false_positives, time_hours) for each stage.
pub struct ReviewPipelineResult {
    pub stages: Vec<String>,
    pub errors_caught: Vec<usize>,
    pub false_positives: Vec<usize>,
    pub time_hours: Vec<f64>,
}

pub fn simulate_review_pipeline(
    total_errors: usize,
    human_detection: f64,
    agent_self_review: f64,
    agent_cross_review: f64,
    human_fpr: f64,
    agent_fpr: f64,
) -> ReviewPipelineResult {
    let mut stages = Vec::new();
    let mut errors = Vec::new();
    let mut fps = Vec::new();
    let mut times = Vec::new();

    // No review
    stages.push("No Review".into());
    errors.push(0);
    fps.push(0);
    times.push(0.0);

    // Human only
    stages.push("Human Only".into());
    errors.push((total_errors as f64 * human_detection) as usize);
    fps.push((total_errors as f64 * human_fpr) as usize);
    times.push(4.0);

    // Agent self-review
    stages.push("Agent Self-Review".into());
    errors.push((total_errors as f64 * agent_self_review) as usize);
    fps.push((total_errors as f64 * agent_fpr) as usize);
    times.push(0.1);

    // Agent cross-review
    stages.push("Agent Cross-Review".into());
    errors.push((total_errors as f64 * agent_cross_review) as usize);
    fps.push((total_errors as f64 * agent_fpr) as usize);
    times.push(0.15);

    // Self + Cross
    let after_self = total_errors as f64 * (1.0 - agent_self_review);
    let caught2 = after_self * agent_cross_review;
    let total_caught = (total_errors as f64 * agent_self_review + caught2) as usize;
    stages.push("Self + Cross Review".into());
    errors.push(total_caught.min(total_errors));
    fps.push((total_errors as f64 * agent_fpr * 1.5) as usize);
    times.push(0.25);

    // Human + Agent
    let ho = total_errors as f64 * human_detection;
    let ao = total_errors as f64 * agent_cross_review;
    let overlap = ho * agent_cross_review * 0.5;
    let combined = (ho + ao - overlap).min(total_errors as f64) as usize;
    stages.push("Human + Agent".into());
    errors.push(combined);
    fps.push((total_errors as f64 * (human_fpr + agent_fpr) * 0.5) as usize);
    times.push(2.5);

    // Self → Cross → Human
    let after_self2 = total_errors as f64 * (1.0 - agent_self_review);
    let after_cross = after_self2 * (1.0 - agent_cross_review);
    let after_human = after_cross * (1.0 - human_detection);
    let full = total_errors - (after_human as usize);
    stages.push("Self → Cross → Human".into());
    errors.push(full.min(total_errors));
    fps.push((total_errors as f64 * 0.15) as usize);
    times.push(2.8);

    ReviewPipelineResult {
        stages,
        errors_caught: errors,
        false_positives: fps,
        time_hours: times,
    }
}

/// Efficiency: errors caught per hour of review time.
pub fn review_efficiency(errors_caught: usize, time_hours: f64) -> f64 {
    if time_hours > 0.0 {
        errors_caught as f64 / time_hours
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_review_catches_nothing() {
        let result = simulate_review_pipeline(1000, 0.70, 0.55, 0.65, 0.05, 0.08);
        assert_eq!(result.errors_caught[0], 0);
    }

    #[test]
    fn full_pipeline_high_detection() {
        let result = simulate_review_pipeline(1000, 0.70, 0.55, 0.65, 0.05, 0.08);
        let full_idx = result
            .stages
            .iter()
            .position(|s| s == "Self → Cross → Human")
            .unwrap();
        // Full pipeline should have > 90% detection
        let detection_pct = result.errors_caught[full_idx] as f64 / 1000.0 * 100.0;
        assert!(
            detection_pct > 90.0,
            "full pipeline should be > 90% detection, got {detection_pct}%"
        );
    }

    #[test]
    fn agent_review_is_faster_than_human() {
        let result = simulate_review_pipeline(1000, 0.70, 0.55, 0.65, 0.05, 0.08);
        let human_idx = 1;
        let agent_idx = 2;

        // Agent review time should be much smaller
        assert!(result.time_hours[agent_idx] < result.time_hours[human_idx]);
    }

    #[test]
    fn detection_never_exceeds_total() {
        let result = simulate_review_pipeline(1000, 0.70, 0.55, 0.65, 0.05, 0.08);
        for &caught in &result.errors_caught {
            assert!(caught <= 1000, "caught {caught} exceeds total 1000");
        }
    }

    #[test]
    fn all_stages_present() {
        let result = simulate_review_pipeline(1000, 0.70, 0.55, 0.65, 0.05, 0.08);
        assert_eq!(result.stages.len(), 7);
    }

    #[test]
    fn review_efficiency_normal_case() {
        let eff = review_efficiency(700, 4.0);
        assert!((eff - 175.0).abs() < 0.01);
    }

    #[test]
    fn review_efficiency_zero_time() {
        let eff = review_efficiency(100, 0.0);
        assert_eq!(eff, 0.0);
    }
}
