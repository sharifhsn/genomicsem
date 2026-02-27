pub fn is_strand_ambiguous(a1: &str, a2: &str) -> bool {
    matches!((a1, a2), ("A", "T") | ("T", "A") | ("C", "G") | ("G", "C"))
}
