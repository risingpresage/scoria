use std::collections::HashMap;

use crate::{GgufError, GgufFile};

pub struct BpeTokenizer {
    pub tokens: Vec<String>,
    pub token_types: Vec<i32>,
    token_to_id: HashMap<String, u32>,
    merge_ranks: HashMap<(String, String), usize>,
    special_tokens: Vec<(String, u32)>,
    byte_to_char: [char; 256],
    char_to_byte: HashMap<char, u8>,
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<u32>,
    pub add_bos_token: bool,
}

fn build_gpt2_byte_tables() -> ([char; 256], HashMap<char, u8>) {
    let mut is_direct = [false; 256];
    for b in 33u8..=126 {
        is_direct[b as usize] = true;
    }
    for b in 161u8..=172 {
        is_direct[b as usize] = true;
    }
    for b in 174u8..=255 {
        is_direct[b as usize] = true;
    }

    let mut byte_to_char = ['\0'; 256];
    let mut char_to_byte = HashMap::with_capacity(256);
    let mut extra = 0u32;
    for b in 0..256usize {
        let ch = if is_direct[b] {
            char::from_u32(b as u32).unwrap()
        } else {
            let c = char::from_u32(256 + extra).unwrap();
            extra += 1;
            c
        };
        byte_to_char[b] = ch;
        char_to_byte.insert(ch, b as u8);
    }
    (byte_to_char, char_to_byte)
}

impl BpeTokenizer {
    pub fn from_gguf(file: &GgufFile) -> Result<Self, GgufError> {
        let raw_tokens = file
            .meta("tokenizer.ggml.tokens")
            .and_then(|v| v.as_array())
            .ok_or_else(|| GgufError::MissingMetadata("tokenizer.ggml.tokens".to_string()))?;

        let mut tokens = Vec::with_capacity(raw_tokens.len());
        let mut token_to_id = HashMap::with_capacity(raw_tokens.len());
        for (idx, val) in raw_tokens.iter().enumerate() {
            let s = val
                .as_str()
                .ok_or_else(|| GgufError::Malformed("non-string token in vocab".to_string()))?
                .to_string();
            token_to_id.entry(s.clone()).or_insert(idx as u32);
            tokens.push(s);
        }

        let token_types = if let Some(arr) = file
            .meta("tokenizer.ggml.token_type")
            .and_then(|v| v.as_array())
        {
            arr.iter()
                .map(|v| v.as_i32().unwrap_or(1))
                .collect::<Vec<_>>()
        } else {
            vec![1; tokens.len()]
        };

        let mut special_tokens = Vec::new();
        for (idx, (tok, &tt)) in tokens.iter().zip(token_types.iter()).enumerate() {
            if (tt == 3 || tt == 4) && !tok.is_empty() {
                special_tokens.push((tok.clone(), idx as u32));
            }
        }
        special_tokens.sort_by(|a, b| b.0.len().cmp(&a.0.len()).then_with(|| a.1.cmp(&b.1)));

        let mut merge_ranks = HashMap::new();
        if let Some(raw_merges) = file
            .meta("tokenizer.ggml.merges")
            .and_then(|v| v.as_array())
        {
            merge_ranks.reserve(raw_merges.len());
            for (rank, val) in raw_merges.iter().enumerate() {
                let s = val
                    .as_str()
                    .ok_or_else(|| GgufError::Malformed("non-string merge entry".to_string()))?;
                if let Some((a, b)) = s.split_once(' ') {
                    merge_ranks
                        .entry((a.to_string(), b.to_string()))
                        .or_insert(rank);
                }
            }
        }

        let (byte_to_char, char_to_byte) = build_gpt2_byte_tables();
        let bos_token_id = file
            .meta("tokenizer.ggml.bos_token_id")
            .and_then(|v| v.as_u32());
        let eos_token_id = file
            .meta("tokenizer.ggml.eos_token_id")
            .and_then(|v| v.as_u32());
        let add_bos_token = file
            .meta("tokenizer.ggml.add_bos_token")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        Ok(Self {
            tokens,
            token_types,
            token_to_id,
            merge_ranks,
            special_tokens,
            byte_to_char,
            char_to_byte,
            bos_token_id,
            eos_token_id,
            add_bos_token,
        })
    }

    pub fn encode(&self, text: &str, add_bos: bool) -> Vec<u32> {
        let mut out = Vec::new();
        if add_bos
            && self.add_bos_token
            && let Some(bos) = self.bos_token_id
        {
            out.push(bos);
        }

        let mut pos = 0usize;
        while pos < text.len() {
            let mut next_hit: Option<(usize, usize, u32)> = None;
            let tail = &text[pos..];
            for (sp, id) in &self.special_tokens {
                if let Some(rel) = tail.find(sp.as_str()) {
                    let abs = pos + rel;
                    match next_hit {
                        None => next_hit = Some((abs, sp.len(), *id)),
                        Some((best_abs, best_len, _)) => {
                            if abs < best_abs || (abs == best_abs && sp.len() > best_len) {
                                next_hit = Some((abs, sp.len(), *id));
                            }
                        }
                    }
                }
            }

            match next_hit {
                Some((hit_start, hit_len, hit_id)) => {
                    if hit_start > pos {
                        self.encode_normal(&text[pos..hit_start], &mut out);
                    }
                    out.push(hit_id);
                    pos = hit_start + hit_len;
                }
                None => {
                    self.encode_normal(&text[pos..], &mut out);
                    break;
                }
            }
        }
        out
    }

    fn encode_normal(&self, text: &str, out: &mut Vec<u32>) {
        for piece in qwen2_pretokenize(text) {
            let mut word: Vec<String> = piece
                .as_bytes()
                .iter()
                .map(|&b| self.byte_to_char[b as usize].to_string())
                .collect();

            if word.len() > 1 {
                loop {
                    let mut best_rank = usize::MAX;
                    let mut best_idx = None;
                    for i in 0..word.len() - 1 {
                        if let Some(&rank) = self
                            .merge_ranks
                            .get(&(word[i].clone(), word[i + 1].clone()))
                            && rank < best_rank
                        {
                            best_rank = rank;
                            best_idx = Some(i);
                        }
                    }
                    let Some(idx) = best_idx else {
                        break;
                    };
                    let merged = format!("{}{}", word[idx], word[idx + 1]);
                    word[idx] = merged;
                    word.remove(idx + 1);
                    if word.len() < 2 {
                        break;
                    }
                }
            }

            for tok in word {
                if let Some(&id) = self.token_to_id.get(&tok) {
                    out.push(id);
                }
            }
        }
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        let mut out = String::new();
        let mut byte_buf = Vec::new();
        for &id in ids {
            let idx = id as usize;
            if idx >= self.tokens.len() {
                continue;
            }
            let tt = self.token_types.get(idx).copied().unwrap_or(1);
            if tt == 3 || tt == 4 {
                if !byte_buf.is_empty() {
                    out.push_str(&String::from_utf8_lossy(&byte_buf));
                    byte_buf.clear();
                }
                out.push_str(&self.tokens[idx]);
            } else {
                for ch in self.tokens[idx].chars() {
                    if let Some(&b) = self.char_to_byte.get(&ch) {
                        byte_buf.push(b);
                    } else {
                        if !byte_buf.is_empty() {
                            out.push_str(&String::from_utf8_lossy(&byte_buf));
                            byte_buf.clear();
                        }
                        out.push(ch);
                    }
                }
            }
        }
        if !byte_buf.is_empty() {
            out.push_str(&String::from_utf8_lossy(&byte_buf));
        }
        out
    }
}

/// Deterministic implementation of the Qwen2 / DeepSeek-R1-Qwen pre-tokenizer regex
/// from llama.cpp (`llama-vocab.cpp`):
/// `(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`
pub fn qwen2_pretokenize(text: &str) -> Vec<&str> {
    let mut pieces = Vec::new();
    let mut i = 0usize;

    while i < text.len() {
        let rest = &text[i..];

        // 1. English contractions: '(s|t|re|ve|m|ll|d) case-insensitive
        if let Some(after) = rest.strip_prefix('\'') {
            let mut matched = 0usize;
            for suf in ["re", "ve", "ll"] {
                if after.len() >= 2 && after[..2].eq_ignore_ascii_case(suf) {
                    matched = 3;
                    break;
                }
            }
            if matched == 0 && !after.is_empty() {
                let b = after.as_bytes()[0].to_ascii_lowercase();
                if matches!(b, b's' | b't' | b'm' | b'd') {
                    matched = 2;
                }
            }
            if matched > 0 {
                pieces.push(&rest[..matched]);
                i += matched;
                continue;
            }
        }

        let mut chars = rest.char_indices();
        let (_, c0) = chars.next().unwrap();
        let c0_len = c0.len_utf8();

        // 2. [^\r\n\p{L}\p{N}]?\p{L}+
        if c0.is_alphabetic() {
            let mut end = c0_len;
            for (idx, ch) in chars {
                if ch.is_alphabetic() {
                    end = idx + ch.len_utf8();
                } else {
                    break;
                }
            }
            pieces.push(&rest[..end]);
            i += end;
            continue;
        }
        if c0 != '\r'
            && c0 != '\n'
            && !c0.is_numeric()
            && let Some((_, c1)) = chars.next()
            && c1.is_alphabetic()
        {
            let mut end = c0_len + c1.len_utf8();
            for (idx, ch) in chars {
                if ch.is_alphabetic() {
                    end = idx + ch.len_utf8();
                } else {
                    break;
                }
            }
            pieces.push(&rest[..end]);
            i += end;
            continue;
        }

        // 3. \p{N} (single numeric character)
        if c0.is_numeric() {
            pieces.push(&rest[..c0_len]);
            i += c0_len;
            continue;
        }

        // 4.  ?[^\s\p{L}\p{N}]+[\r\n]*
        let is_sym = |c: char| !c.is_whitespace() && !c.is_alphabetic() && !c.is_numeric();
        if is_sym(c0) {
            let mut end = c0_len;
            let mut it = rest[c0_len..].char_indices().peekable();
            while let Some(&(rel, ch)) = it.peek() {
                if is_sym(ch) {
                    end = c0_len + rel + ch.len_utf8();
                    it.next();
                } else {
                    break;
                }
            }
            while let Some(&(rel, ch)) = it.peek() {
                if ch == '\r' || ch == '\n' {
                    end = c0_len + rel + ch.len_utf8();
                    it.next();
                } else {
                    break;
                }
            }
            pieces.push(&rest[..end]);
            i += end;
            continue;
        }
        if c0 == ' '
            && let Some(c1) = rest[c0_len..].chars().next()
            && is_sym(c1)
        {
            let mut end = c0_len + c1.len_utf8();
            let mut it = rest[end..].char_indices().peekable();
            let base = end;
            while let Some(&(rel, ch)) = it.peek() {
                if is_sym(ch) {
                    end = base + rel + ch.len_utf8();
                    it.next();
                } else {
                    break;
                }
            }
            while let Some(&(rel, ch)) = it.peek() {
                if ch == '\r' || ch == '\n' {
                    end = base + rel + ch.len_utf8();
                    it.next();
                } else {
                    break;
                }
            }
            pieces.push(&rest[..end]);
            i += end;
            continue;
        }

        // 5/6/7. Whitespace runs: \s*[\r\n]+ | \s+(?!\S) | \s+
        if c0.is_whitespace() {
            let mut ws_end = c0_len;
            let mut last_nl_end = if c0 == '\r' || c0 == '\n' {
                Some(c0_len)
            } else {
                None
            };
            let mut ws_count = 1usize;
            let mut prev_ws_end = 0usize;

            for (idx, ch) in rest[c0_len..].char_indices() {
                if ch.is_whitespace() {
                    prev_ws_end = c0_len + idx;
                    ws_end = prev_ws_end + ch.len_utf8();
                    ws_count += 1;
                    if ch == '\r' || ch == '\n' {
                        last_nl_end = Some(ws_end);
                    }
                } else {
                    break;
                }
            }

            if let Some(nl_end) = last_nl_end {
                pieces.push(&rest[..nl_end]);
                i += nl_end;
                continue;
            }

            if ws_end < rest.len() && ws_count >= 2 {
                pieces.push(&rest[..prev_ws_end]);
                i += prev_ws_end;
                continue;
            }

            pieces.push(&rest[..ws_end]);
            i += ws_end;
            continue;
        }

        // Unreachable fallback
        pieces.push(&rest[..c0_len]);
        i += c0_len;
    }

    pieces
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qwen2_pretokenize_patterns() {
        assert_eq!(
            qwen2_pretokenize("Hello, world! Don't stop 2026."),
            vec![
                "Hello", ",", " world", "!", " Don", "'t", " stop", " ", "2", "0", "2", "6", "."
            ]
        );
        assert_eq!(
            qwen2_pretokenize("a   b\n\nc"),
            vec!["a", "  ", " b", "\n\n", "c"]
        );
        assert_eq!(
            qwen2_pretokenize("foo\n  bar"),
            vec!["foo", "\n", " ", " bar"]
        );
    }
}
