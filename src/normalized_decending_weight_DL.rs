use crate::weighted_DL::generic_weighted_damerau_levenshtein;

/// Generate a descending geometric weight sequence of length `n` with ratio `k`,
/// normalized so the weights sum exactly to `n`.
///
/// Notes:
/// - If `k == 1.0`, all weights are `1`.
/// - If `k > 1.0`, we invert it (use `1/k`) to keep the sequence descending.
/// - Panics if `k <= 0.0`.
fn normalized_geometric_descending_weights(n: usize, k: f64) -> Vec<f64> {
    if n == 0 {
        return Vec::new();
    }

    assert!(k > 0.0, "Geometric ratio k must be positive");

    if (k - 1.0).abs() < f64::EPSILON {
        // Equal weights that sum to n
        return vec![1.0; n];
    }

    let ratio = if k > 1.0 { 1.0 / k } else { k };

    // Build raw geometric sequence and accumulate sum in one pass
    let mut weights: Vec<f64> = Vec::with_capacity(n);
    let mut current = 1.0_f64;
    let mut sum = 0.0_f64;
    for _ in 0..n {
        weights.push(current);
        sum += current;
        current *= ratio;
    }

    // Scale so the sum equals n (within floating-point precision)
    let scale = (n as f64) / sum;
    for w in &mut weights {
        *w *= scale;
    }
    weights
}

/// Wrapper over generic weighted Damerau-Levenshtein that uses normalized
/// descending geometric weights for both strings, parameterized by `k`.
///
/// To keep relative costs comparable across different string lengths, we
/// normalize using a shared scale based on `max(len(a), len(b))` and then
/// slice the weights for each string. This avoids making early-character
/// weights larger solely because one string is longer.
pub fn normalized_descending_weighted_damerau_levenshtein(a: &str, b: &str, k: f64) -> f64 {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();

    let max_len = a_chars.len().max(b_chars.len());
    let shared_weights = normalized_geometric_descending_weights(max_len, k);
    let weight_a = shared_weights[0..a_chars.len()].to_vec();
    let weight_b = shared_weights[0..b_chars.len()].to_vec();

    generic_weighted_damerau_levenshtein(&a_chars, &b_chars, &weight_a, &weight_b)
}

#[cfg(test)]
mod tests {
    use super::{
        normalized_descending_weighted_damerau_levenshtein, normalized_geometric_descending_weights,
    };

    #[test]
    fn weights_sum_to_n_and_descend() {
        let n = 10;
        let k = 0.8;
        let w = normalized_geometric_descending_weights(n, k);
        // println!("{:?}", w);
        assert_eq!(w.len(), n);
        assert!((w.iter().sum::<f64>() - (n as f64)).abs() < 1e-9);
        // Check generally descending (allow ties due to integer rounding)
        for i in 1..n {
            assert!(w[i - 1] >= w[i]);
        }
    }

    #[test]
    fn k_equal_one_gives_all_ones() {
        let n = 7;
        let w = normalized_geometric_descending_weights(n, 1.0);
        assert_eq!(w, vec![1.0; n]);
    }

    #[test]
    fn wrapper_basic_equivalence_for_equal_strings() {
        let a = "weighted";
        let b = "weighted";
        let d = normalized_descending_weighted_damerau_levenshtein(a, b, 0.7);
        assert!((d - 0.0).abs() < 1e-9);
    }

    #[test]
    fn wrapper_bigger_difference_for_different_strings() {
        let a = "MEFA groups";
        let b = "MEFA group";
        let c = "BEFA groups";
        let d1 = normalized_descending_weighted_damerau_levenshtein(a, b, 0.7);
        let d2 = normalized_descending_weighted_damerau_levenshtein(a, c, 0.7);
        let d3 = normalized_descending_weighted_damerau_levenshtein(b, c, 0.7);
        println!("d1: {}, d2: {}, d3: {}", d1, d2, d3);
        assert!(d1 < d2);
        assert!(d2 < d3);
    }

    #[test]
    fn deletion_and_substitution_cost_parity() {
        // With shared-normalized weights, deleting leading 'a' from "abc" to get "bc"
        // should cost the same as substituting it to 'z' when comparing to "zbc".
        let k = 0.8;
        let d_del = normalized_descending_weighted_damerau_levenshtein("abc", "bc", k);
        let d_sub = normalized_descending_weighted_damerau_levenshtein("abc", "zbc", k);
        assert!((d_del - d_sub).abs() < 1e-9);
    }
}
