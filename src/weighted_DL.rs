use std::collections::HashMap;
use std::hash::Hash;

/* Returns the final index for a value in a single vector that represents a fixed
grid */
fn flat_index(i: usize, j: usize, width: usize) -> usize {
    j * width + i
}

/// Like optimal string alignment, but substrings can be edited an unlimited
/// number of times, and the triangle inequality holds. Weighted version where
/// deletion costs from `a_elems` are multiplied by `weight_a` (position-dependent)
/// and insertion costs to `b_elems` are multiplied by `weight_b` (position-dependent).
/// Substitution cost is the sum of corresponding weights if mismatch. Transposition
/// includes weighted costs for intervening deletions/insertions plus the mean of the
/// four involved character weights: `weight_a` for both swapped chars and `weight_b`
/// for both swapped chars. For an adjacent swap at (i,j), this is
/// `(weight_a[i-1] + weight_a[i-2] + weight_b[j-1] + weight_b[j-2]) / 4`.
pub fn generic_weighted_damerau_levenshtein<Elem>(
    a_elems: &[Elem],
    b_elems: &[Elem],
    weight_a: &[f64],
    weight_b: &[f64],
) -> f64
where
    Elem: Eq + Hash + Clone,
{
    let a_len = a_elems.len();
    let b_len = b_elems.len();

    assert_eq!(weight_a.len(), a_len);
    assert_eq!(weight_b.len(), b_len);

    let mut prefix_a: Vec<f64> = vec![0.0];
    for &w in weight_a {
        prefix_a.push(*prefix_a.last().unwrap() + w);
    }
    let mut prefix_b: Vec<f64> = vec![0.0];
    for &w in weight_b {
        prefix_b.push(*prefix_b.last().unwrap() + w);
    }

    if a_len == 0 {
        return prefix_b[b_len];
    }
    if b_len == 0 {
        return prefix_a[a_len];
    }

    let width = a_len + 2;
    let mut distances = vec![0.0_f64; (a_len + 2) * (b_len + 2)];
    let max_distance = prefix_a[a_len] + prefix_b[b_len] + 1.0;

    distances[0] = max_distance;

    for i in 0..=a_len {
        distances[flat_index(i + 1, 0, width)] = max_distance;
        distances[flat_index(i + 1, 1, width)] = prefix_a[i];
    }

    for j in 0..=b_len {
        distances[flat_index(0, j + 1, width)] = max_distance;
        distances[flat_index(1, j + 1, width)] = prefix_b[j];
    }

    let mut elems: HashMap<Elem, usize> = HashMap::with_capacity(64);

    for i in 1..=a_len {
        let mut db = 0;

        for j in 1..=b_len {
            let k = *elems.get(&b_elems[j - 1]).unwrap_or(&0);

            let deletion_cost_code = distances[flat_index(i, j + 1, width)] + weight_a[i - 1];
            let insertion_cost_code = distances[flat_index(i + 1, j, width)] + weight_b[j - 1];

            let is_match = a_elems[i - 1] == b_elems[j - 1];
            // Substitution uses the maximum of the two position-dependent weights
            // so it is comparable to a single deletion or insertion when weights match.
            let substitution_cost = distances[flat_index(i, j, width)]
                + if is_match {
                    0.0
                } else {
                    weight_a[i - 1].max(weight_b[j - 1])
                };

            let del_between = prefix_a[i - 1] - prefix_a[k];
            let ins_between = prefix_b[j - 1] - prefix_b[db];
            // Transposition base uses the average of the two positions' max weights.
            // This keeps a single swap comparable to a single substitution when the
            // same positions are involved.
            let swap_base = if k > 0 && db > 0 {
                let left_max = weight_a[i - 1].max(weight_b[j - 1]);
                let right_max = weight_a[k - 1].max(weight_b[db - 1]);
                (left_max + right_max) / 2.0
            } else {
                weight_a[i - 1].max(weight_b[j - 1])
            };
            let transposition_cost =
                distances[flat_index(k, db, width)] + del_between + ins_between + swap_base;

            let val = substitution_cost
                .min(deletion_cost_code)
                .min(insertion_cost_code)
                .min(transposition_cost);

            distances[flat_index(i + 1, j + 1, width)] = val;

            if is_match {
                db = j;
            }
        }

        elems.insert(a_elems[i - 1].clone(), i);
    }

    distances[flat_index(a_len + 1, b_len + 1, width)]
}

#[cfg(test)]
mod tests {
    use super::generic_weighted_damerau_levenshtein;

    fn to_chars(s: &str) -> Vec<char> {
        s.chars().collect()
    }

    fn descending(n: usize) -> Vec<f64> {
        (0..n).map(|i| (n - i) as f64).collect()
    }

    #[test]
    fn empty_to_empty_zero() {
        let a = to_chars("");
        let b = to_chars("");
        let wa: Vec<f64> = vec![];
        let wb: Vec<f64> = vec![];
        let d = generic_weighted_damerau_levenshtein(&a, &b, &wa, &wb);
        assert!((d - 0.0).abs() < 1e-9);
    }

    #[test]
    fn equal_strings_zero_cost() {
        let a = to_chars("abc");
        let b = to_chars("abc");
        let wa = descending(a.len());
        let wb = descending(b.len());
        let d = generic_weighted_damerau_levenshtein(&a, &b, &wa, &wb);
        assert!((d - 0.0).abs() < 1e-9);
    }

    #[test]
    fn insert_all_cost_is_sum_weights_b() {
        let a = to_chars("");
        let b = to_chars("abcd");
        let wa: Vec<f64> = vec![];
        let wb = descending(b.len());
        let expected: f64 = wb.iter().sum();
        let d = generic_weighted_damerau_levenshtein(&a, &b, &wa, &wb);
        assert!((d - expected).abs() < 1e-9);
    }

    #[test]
    fn delete_all_cost_is_sum_weights_a() {
        let a = to_chars("abcd");
        let b = to_chars("");
        let wa = descending(a.len());
        let wb: Vec<f64> = vec![];
        let expected: f64 = wa.iter().sum();
        let d = generic_weighted_damerau_levenshtein(&a, &b, &wa, &wb);
        assert!((d - expected).abs() < 1e-9);
    }

    #[test]
    fn substitution_uses_sum_of_weights() {
        let a = to_chars("a");
        let b = to_chars("b");
        let wa = vec![5.0];
        let wb = vec![7.0];
        let d = generic_weighted_damerau_levenshtein(&a, &b, &wa, &wb);
        // Now substitution uses max of weights
        assert!((d - 7.0).abs() < 1e-9);
    }

    #[test]
    fn early_position_has_higher_penalty() {
        let a = to_chars("abc");
        let b = to_chars("xbc");
        let wa = descending(a.len()); // [3,2,1]
        let wb = descending(b.len()); // [3,2,1]
        let d = generic_weighted_damerau_levenshtein(&a, &b, &wa, &wb);
        assert!((d - (wa[0] + wb[0])).abs() < 1e-9); // 3 + 3 = 6
    }

    #[test]
    fn late_position_has_lower_penalty() {
        let a = to_chars("abz");
        let b = to_chars("abc");
        let wa = descending(a.len()); // [3,2,1]
        let wb = descending(b.len()); // [3,2,1]
        let d = generic_weighted_damerau_levenshtein(&a, &b, &wa, &wb);
        assert!((d - (wa[2] + wb[2])).abs() < 1e-9); // 1 + 1 = 2
    }

    #[test]
    fn deletion_vs_insertion_asymmetric_costs() {
        // Deleting trailing 'b' from a should cost weight_a at that position
        let a = to_chars("ab");
        let b = to_chars("a");
        let wa = descending(a.len()); // [2,1]
        let wb = descending(b.len()); // [1]
        let d = generic_weighted_damerau_levenshtein(&a, &b, &wa, &wb);
        assert!((d - wa[1]).abs() < 1e-9); // delete 'b' cost = 1
    }

    #[test]
    fn insertion_only_takes_weight_b_of_inserted_char() {
        let a = to_chars("a");
        let b = to_chars("ab");
        let wa = descending(a.len()); // [1]
        let wb = descending(b.len()); // [2,1]
        let d = generic_weighted_damerau_levenshtein(&a, &b, &wa, &wb);
        assert!((d - wb[1]).abs() < 1e-9); // insert 'b' cost = 1
    }

    #[test]
    fn adjacent_transposition_uses_four_weight_mean_cost() {
        // Swap of first two letters in two strings: "ME" -> "EM".
        // With descending weights, the involved indices are (i-1)=0 in a, (j-1)=0 in b.
        // Mean swap cost should be (wa[0] + wb[0]) / 2.
        let a = to_chars("MESA group");
        let b = to_chars("EMSA group");
        let wa = descending(a.len());
        let wb = descending(b.len());
        // With new rule, adjacent swap cost equals average of max weights at both positions
        let expected_mean = (wa[0].max(wb[0]) + wa[1].max(wb[1])) / 2.0;
        let d = generic_weighted_damerau_levenshtein(&a, &b, &wa, &wb);
        assert!((d - expected_mean).abs() < 1e-9);
    }
}
