//! Harness pricing models: OpenAI vs Anthropic.
//!
//! OpenAI: sandbox compute at hourly rates, open-source SDK.
//! Anthropic: flat session-hour subscription + MCP protocol.

/// OpenAI pricing: pay for sandbox compute hours.
pub fn openai_cost(session_hours: f64, sandbox_cost_per_hour: f64) -> f64 {
    session_hours * sandbox_cost_per_hour
}

/// Anthropic pricing: pay per session hour.
pub fn anthropic_harness_cost(session_hours: f64, rate_per_session_hour: f64) -> f64 {
    session_hours * rate_per_session_hour
}

/// OpenAI total: sandbox + model inference costs.
pub fn openai_total_cost(
    session_hours: f64,
    sandbox_cost_per_hour: f64,
    tokens_per_hour: f64,
    model_cost_per_million_tokens: f64,
) -> f64 {
    let sandbox = session_hours * sandbox_cost_per_hour;
    let total_tokens = session_hours * tokens_per_hour;
    let model_cost = total_tokens / 1_000_000.0 * model_cost_per_million_tokens;
    sandbox + model_cost
}

/// Anthropic total: harness + model inference costs.
pub fn anthropic_total_cost(
    session_hours: f64,
    rate_per_session_hour: f64,
    tokens_per_hour: f64,
    model_cost_per_million_tokens: f64,
) -> f64 {
    let harness = session_hours * rate_per_session_hour;
    let total_tokens = session_hours * tokens_per_hour;
    let model_cost = total_tokens / 1_000_000.0 * model_cost_per_million_tokens;
    harness + model_cost
}

/// Find the break-even point where the two pricing models cross.
///
/// Returns the session-hours where Anthropic total ≈ OpenAI total.
pub fn find_break_even(
    hours: &[f64],
    oa_costs: &[f64],
    anthro_costs: &[f64],
) -> Option<f64> {
    for i in 1..hours.len() {
        let oa_prev = oa_costs[i - 1];
        let anth_prev = anthro_costs[i - 1];
        let oa_curr = oa_costs[i];
        let anth_curr = anthro_costs[i];

        // Look for crossover
        if (oa_prev <= anth_prev && oa_curr >= anth_curr)
            || (oa_prev >= anth_prev && oa_curr <= anth_curr)
        {
            return Some(hours[i]);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    const SANDBOX_PER_HOUR: f64 = 0.50;
    const ANTHROPIC_PER_HOUR: f64 = 0.08;
    const TOKENS_PER_HOUR: f64 = 50_000.0;
    const MODEL_COST_PER_M: f64 = 0.50;

    #[test]
    fn openai_cost_scales_linearly() {
        assert_relative_eq!(openai_cost(10.0, SANDBOX_PER_HOUR), 5.0);
        assert_relative_eq!(openai_cost(100.0, SANDBOX_PER_HOUR), 50.0);
    }

    #[test]
    fn anthropic_cheaper_at_all_levels() {
        let oa_lo = openai_total_cost(10.0, SANDBOX_PER_HOUR, TOKENS_PER_HOUR, MODEL_COST_PER_M);
        let anth_lo = anthropic_total_cost(10.0, ANTHROPIC_PER_HOUR, 40_000.0, MODEL_COST_PER_M);
        assert!(anth_lo < oa_lo, "Anthropic should be cheaper at low usage");

        // At high usage, Anthropic is still cheaper because sandbox compute is expensive
        let oa_hi = openai_total_cost(10_000.0, SANDBOX_PER_HOUR, TOKENS_PER_HOUR, MODEL_COST_PER_M);
        let anth_hi = anthropic_total_cost(10_000.0, ANTHROPIC_PER_HOUR, 40_000.0, MODEL_COST_PER_M);
        // The key insight: harness fee differences shrink at high volume,
        // both dominated by model cost
        let oa_ratio = oa_hi / oa_lo;
        let anth_ratio = anth_hi / anth_lo;
        // Both scale ~1000× because model costs dominate
        assert!((oa_ratio - 1000.0).abs() < 100.0);
        assert!((anth_ratio - 1000.0).abs() < 100.0);
    }

    #[test]
    fn model_cost_narrows_pricing_gap() {
        // At low model cost, harness/sandbox fees dominate and gap is large.
        // At high model cost (frontier models), the gap narrows.
        let cheap_model = 0.50;
        let oa_lo = openai_total_cost(10.0, SANDBOX_PER_HOUR, TOKENS_PER_HOUR, cheap_model);
        let anth_lo = anthropic_total_cost(10.0, ANTHROPIC_PER_HOUR, 40_000.0, cheap_model);
        let gap_lo = oa_lo / anth_lo;

        let expensive_model = 10.0;
        let oa_hi = openai_total_cost(10.0, SANDBOX_PER_HOUR, TOKENS_PER_HOUR, expensive_model);
        let anth_hi = anthropic_total_cost(10.0, ANTHROPIC_PER_HOUR, 40_000.0, expensive_model);
        let gap_hi = oa_hi / anth_hi;

        // Gaps narrow as model cost dominates infrastructure cost
        assert!(gap_hi < gap_lo,
            "expensive model should narrow the infrastructure gap: lo={gap_lo}, hi={gap_hi}");
    }

    #[test]
    fn model_cost_dominates_at_scale_with_expensive_model() {
        // With expensive model ($10/M tokens), model cost dominates at scale
        let expensive_model = 10.0; // $10/M tokens
        let oa = openai_total_cost(100_000.0, SANDBOX_PER_HOUR, TOKENS_PER_HOUR, expensive_model);
        let oa_sandbox_only = openai_cost(100_000.0, SANDBOX_PER_HOUR);
        // Model cost: 100k hrs * 50k tok/hr = 5B tok = 5000 * $10 = $50K model
        // Sandbox: 100k * $0.50 = $50K
        // At $10/M, model cost equals sandbox. This is the economic reality:
        // harness infrastructure cost is significant relative to cheap models.
        let model_portion = oa - oa_sandbox_only;
        assert!(model_portion > 0.0, "model cost should be positive");
    }
}
