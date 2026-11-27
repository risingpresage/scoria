use crate::{GgmlType, GgufError, iq_tables};

#[inline]
pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let exp = (bits >> 10) & 0x1F;
    let mant = (bits & 0x03FF) as u32;

    if exp == 0 {
        if mant == 0 {
            return f32::from_bits(sign);
        }
        // Subnormal half -> normalized float32
        let mut m = mant;
        let mut e: u32 = 127 - 14;
        while (m & 0x0400) == 0 {
            m <<= 1;
            e -= 1;
        }
        m &= 0x03FF;
        return f32::from_bits(sign | (e << 23) | (m << 13));
    }
    if exp == 31 {
        if mant == 0 {
            return f32::from_bits(sign | 0x7F80_0000);
        }
        return f32::from_bits(sign | 0x7FC0_0000 | (mant << 13));
    }
    let f_exp = ((exp as u32) + (127 - 15)) << 23;
    f32::from_bits(sign | f_exp | (mant << 13))
}

#[inline]
fn read_u16_le(bytes: &[u8]) -> u16 {
    u16::from_le_bytes([bytes[0], bytes[1]])
}

#[inline]
fn read_u32_le(bytes: &[u8]) -> u32 {
    u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
}

pub fn dequant_row_f32(src: &[u8], dst: &mut [f32]) {
    assert_eq!(src.len(), dst.len() * 4);
    for (i, out) in dst.iter_mut().enumerate() {
        *out = f32::from_le_bytes(src[i * 4..i * 4 + 4].try_into().unwrap());
    }
}

pub fn dequant_row_f16(src: &[u8], dst: &mut [f32]) {
    assert_eq!(src.len(), dst.len() * 2);
    for (i, out) in dst.iter_mut().enumerate() {
        *out = f16_to_f32(read_u16_le(&src[i * 2..]));
    }
}

pub fn dequant_row_q5_0(src: &[u8], dst: &mut [f32]) {
    const QK: usize = 32;
    const BLK_BYTES: usize = 22;
    assert_eq!(dst.len() % QK, 0);
    let nb = dst.len() / QK;
    assert_eq!(src.len(), nb * BLK_BYTES);

    for i in 0..nb {
        let blk = &src[i * BLK_BYTES..(i + 1) * BLK_BYTES];
        let d = f16_to_f32(read_u16_le(&blk[0..2]));
        let qh = read_u32_le(&blk[2..6]);
        let qs = &blk[6..22];
        let y = &mut dst[i * QK..(i + 1) * QK];
        for j in 0..16 {
            let xh_0 = ((qh >> j) << 4) & 0x10;
            let xh_1 = (qh >> (j + 12)) & 0x10;
            let x0 = (((qs[j] & 0x0F) as u32 | xh_0) as i32) - 16;
            let x1 = (((qs[j] >> 4) as u32 | xh_1) as i32) - 16;
            y[j] = (x0 as f32) * d;
            y[j + 16] = (x1 as f32) * d;
        }
    }
}

pub fn dequant_row_q8_0(src: &[u8], dst: &mut [f32]) {
    const QK: usize = 32;
    const BLK_BYTES: usize = 34;
    assert_eq!(dst.len() % QK, 0);
    let nb = dst.len() / QK;
    assert_eq!(src.len(), nb * BLK_BYTES);

    for i in 0..nb {
        let blk = &src[i * BLK_BYTES..(i + 1) * BLK_BYTES];
        let d = f16_to_f32(read_u16_le(&blk[0..2]));
        let qs = &blk[2..34];
        let y = &mut dst[i * QK..(i + 1) * QK];
        for j in 0..QK {
            y[j] = (qs[j] as i8 as f32) * d;
        }
    }
}

#[inline]
fn get_scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        let d = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
        let m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
        (d, m)
    }
}

pub fn dequant_row_q4_k(src: &[u8], dst: &mut [f32]) {
    const QK_K: usize = 256;
    const BLK_BYTES: usize = 144;
    assert_eq!(dst.len() % QK_K, 0);
    let nb = dst.len() / QK_K;
    assert_eq!(src.len(), nb * BLK_BYTES);

    for i in 0..nb {
        let blk = &src[i * BLK_BYTES..(i + 1) * BLK_BYTES];
        let d = f16_to_f32(read_u16_le(&blk[0..2]));
        let dmin = f16_to_f32(read_u16_le(&blk[2..4]));
        let scales = &blk[4..16];
        let qs = &blk[16..144];
        let y = &mut dst[i * QK_K..(i + 1) * QK_K];

        let mut is = 0usize;
        for c in 0..4 {
            let (sc1, m1) = get_scale_min_k4(is, scales);
            let d1 = d * (sc1 as f32);
            let min1 = dmin * (m1 as f32);
            let (sc2, m2) = get_scale_min_k4(is + 1, scales);
            let d2 = d * (sc2 as f32);
            let min2 = dmin * (m2 as f32);
            let q = &qs[c * 32..(c + 1) * 32];
            let base = c * 64;
            for l in 0..32 {
                y[base + l] = d1 * ((q[l] & 0x0F) as f32) - min1;
                y[base + 32 + l] = d2 * ((q[l] >> 4) as f32) - min2;
            }
            is += 2;
        }
    }
}

pub fn dequant_row_q6_k(src: &[u8], dst: &mut [f32]) {
    const QK_K: usize = 256;
    const BLK_BYTES: usize = 210;
    assert_eq!(dst.len() % QK_K, 0);
    let nb = dst.len() / QK_K;
    assert_eq!(src.len(), nb * BLK_BYTES);

    for i in 0..nb {
        let blk = &src[i * BLK_BYTES..(i + 1) * BLK_BYTES];
        let ql_all = &blk[0..128];
        let qh_all = &blk[128..192];
        let sc_all = &blk[192..208];
        let d = f16_to_f32(read_u16_le(&blk[208..210]));
        let y_all = &mut dst[i * QK_K..(i + 1) * QK_K];

        for h in 0..2 {
            let ql = &ql_all[h * 64..(h + 1) * 64];
            let qh = &qh_all[h * 32..(h + 1) * 32];
            let sc = &sc_all[h * 8..(h + 1) * 8];
            let y = &mut y_all[h * 128..(h + 1) * 128];

            for l in 0..32 {
                let is = l / 16;
                let q1 = (((ql[l] & 0x0F) | ((qh[l] & 0x03) << 4)) as i8) - 32;
                let q2 = (((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 0x03) << 4)) as i8) - 32;
                let q3 = (((ql[l] >> 4) | (((qh[l] >> 4) & 0x03) << 4)) as i8) - 32;
                let q4 = (((ql[l + 32] >> 4) | (((qh[l] >> 6) & 0x03) << 4)) as i8) - 32;

                y[l] = d * ((sc[is] as i8) as f32) * (q1 as f32);
                y[l + 32] = d * ((sc[is + 2] as i8) as f32) * (q2 as f32);
                y[l + 64] = d * ((sc[is + 4] as i8) as f32) * (q3 as f32);
                y[l + 96] = d * ((sc[is + 6] as i8) as f32) * (q4 as f32);
            }
        }
    }
}

/// E2M1 values (doubled), shared by MXFP4 and NVFP4. The factor of two is absorbed by
/// `ue4m3_to_f32`, which halves the decoded scale.
const KVALUES_FP4: [i8; 16] = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12];

/// IQ4_NL codebook: 16 non-uniform signed levels.
const KVALUES_IQ4NL: [i8; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
];

/// Decodes an unsigned E4M3 scale (sign bit reused as the top exponent bit).
fn ue4m3_to_f32(x: u8) -> f32 {
    if x == 0 || x == 0x7F {
        return 0.0;
    }
    let exp = ((x >> 3) & 0xF) as i32;
    let man = (x & 0x7) as f32;
    let raw = if exp == 0 {
        man * 2f32.powi(-9)
    } else {
        (1.0 + man / 8.0) * 2f32.powi(exp - 7)
    };
    raw * 0.5
}

pub fn dequant_row_bf16(src: &[u8], dst: &mut [f32]) {
    assert_eq!(src.len(), dst.len() * 2);
    for (i, out) in dst.iter_mut().enumerate() {
        let bits = u16::from_le_bytes([src[i * 2], src[i * 2 + 1]]);
        *out = f32::from_bits((bits as u32) << 16);
    }
}

/// Q5_K: 176 bytes per 256 elements: d, dmin, 12 packed 6-bit scale/min pairs, a 32-byte
/// high-bit plane, and 128 bytes of low nibbles.
pub fn dequant_row_q5_k(src: &[u8], dst: &mut [f32]) {
    const QK_K: usize = 256;
    const BLK_BYTES: usize = 176;
    assert_eq!(dst.len() % QK_K, 0);
    let nb = dst.len() / QK_K;
    assert_eq!(src.len(), nb * BLK_BYTES);

    for i in 0..nb {
        let blk = &src[i * BLK_BYTES..(i + 1) * BLK_BYTES];
        let d = f16_to_f32(read_u16_le(&blk[0..2]));
        let dmin = f16_to_f32(read_u16_le(&blk[2..4]));
        let scales = &blk[4..16];
        let qh = &blk[16..48];
        let qs = &blk[48..176];
        let y = &mut dst[i * QK_K..(i + 1) * QK_K];

        let mut is = 0usize;
        let mut u1 = 1u8;
        let mut u2 = 2u8;
        for j in 0..4 {
            let (sc1, m1) = get_scale_min_k4(is, scales);
            let d1 = d * sc1 as f32;
            let min1 = dmin * m1 as f32;
            let (sc2, m2) = get_scale_min_k4(is + 1, scales);
            let d2 = d * sc2 as f32;
            let min2 = dmin * m2 as f32;
            let q = &qs[j * 32..(j + 1) * 32];
            let base = j * 64;
            for l in 0..32 {
                let hi1 = if qh[l] & u1 != 0 { 16.0 } else { 0.0 };
                let hi2 = if qh[l] & u2 != 0 { 16.0 } else { 0.0 };
                y[base + l] = d1 * ((q[l] & 0x0F) as f32 + hi1) - min1;
                y[base + 32 + l] = d2 * ((q[l] >> 4) as f32 + hi2) - min2;
            }
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
    }
}

/// Q3_K: 110 bytes per 256 elements: 32-byte high mask, 64 bytes of 2-bit quants, 12
/// bytes of packed 6-bit scales, then the super-block scale.
pub fn dequant_row_q3_k(src: &[u8], dst: &mut [f32]) {
    const QK_K: usize = 256;
    const BLK_BYTES: usize = 110;
    const KMASK1: u32 = 0x0303_0303;
    const KMASK2: u32 = 0x0f0f_0f0f;
    assert_eq!(dst.len() % QK_K, 0);
    let nb = dst.len() / QK_K;
    assert_eq!(src.len(), nb * BLK_BYTES);

    for i in 0..nb {
        let blk = &src[i * BLK_BYTES..(i + 1) * BLK_BYTES];
        let hmask = &blk[0..32];
        let qsrc = &blk[32..96];
        let d_all = f16_to_f32(read_u16_le(&blk[108..110]));
        let y = &mut dst[i * QK_K..(i + 1) * QK_K];

        // Unpack 16 six-bit scales out of 12 bytes, exactly as ggml does.
        let mut aux = [0u32; 4];
        for (k, a) in aux.iter_mut().take(3).enumerate() {
            *a = u32::from_le_bytes([
                blk[96 + k * 4],
                blk[97 + k * 4],
                blk[98 + k * 4],
                blk[99 + k * 4],
            ]);
        }
        let tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & KMASK2) | (((tmp >> 4) & KMASK1) << 4);
        aux[3] = ((aux[1] >> 4) & KMASK2) | (((tmp >> 6) & KMASK1) << 4);
        aux[0] = (aux[0] & KMASK2) | ((tmp & KMASK1) << 4);
        aux[1] = (aux[1] & KMASK2) | (((tmp >> 2) & KMASK1) << 4);
        let scales: [i8; 16] = {
            let mut sc = [0i8; 16];
            for (k, byte) in aux.iter().flat_map(|a| a.to_le_bytes()).enumerate() {
                sc[k] = byte as i8;
            }
            sc
        };

        let mut is = 0usize;
        for n in 0..2 {
            let mut shift = 0u32;
            let mut m: u8 = 1;
            for _ in 0..4 {
                let dl = d_all * (scales[is] as f32 - 32.0);
                is += 1;
                for l in 0..16 {
                    let qv = ((qsrc[n * 32 + l] >> shift) & 3) as i8;
                    let sub = if hmask[l] & m != 0 { 0 } else { 4 };
                    y[n * 128 + (shift as usize / 2) * 32 + l] = dl * (qv - sub) as f32;
                }
                let dl = d_all * (scales[is] as f32 - 32.0);
                is += 1;
                for l in 0..16 {
                    let qv = ((qsrc[n * 32 + 16 + l] >> shift) & 3) as i8;
                    let sub = if hmask[16 + l] & m != 0 { 0 } else { 4 };
                    y[n * 128 + (shift as usize / 2) * 32 + 16 + l] = dl * (qv - sub) as f32;
                }
                shift += 2;
                m <<= 1;
            }
        }
    }
}

/// IQ4_NL: 18 bytes per 32 elements: an f16 scale and 16 bytes of nibble indices into a
/// 16-level non-uniform codebook.
pub fn dequant_row_iq4_nl(src: &[u8], dst: &mut [f32]) {
    const QK4_NL: usize = 32;
    const BLK_BYTES: usize = 18;
    assert_eq!(dst.len() % QK4_NL, 0);
    let nb = dst.len() / QK4_NL;
    assert_eq!(src.len(), nb * BLK_BYTES);

    for i in 0..nb {
        let blk = &src[i * BLK_BYTES..(i + 1) * BLK_BYTES];
        let d = f16_to_f32(read_u16_le(&blk[0..2]));
        let qs = &blk[2..18];
        let y = &mut dst[i * QK4_NL..(i + 1) * QK4_NL];
        for j in 0..16 {
            y[j] = d * KVALUES_IQ4NL[(qs[j] & 0x0F) as usize] as f32;
            y[j + 16] = d * KVALUES_IQ4NL[(qs[j] >> 4) as usize] as f32;
        }
    }
}

/// NVFP4 (GGML type 40): 36 bytes per 64 elements. Four unsigned-E4M3 scales, one per
/// 16-element sub-block, followed by 32 bytes of packed E2M1 nibbles.
pub fn dequant_row_nvfp4(src: &[u8], dst: &mut [f32]) {
    const QK: usize = 64;
    const SUB: usize = 16;
    const BLK_BYTES: usize = 36;
    assert_eq!(dst.len() % QK, 0);
    let nb = dst.len() / QK;
    assert_eq!(src.len(), nb * BLK_BYTES);

    for i in 0..nb {
        let blk = &src[i * BLK_BYTES..(i + 1) * BLK_BYTES];
        let y = &mut dst[i * QK..(i + 1) * QK];
        for s in 0..4 {
            let d = ue4m3_to_f32(blk[s]);
            let qs = &blk[4 + s * 8..4 + (s + 1) * 8];
            for j in 0..8 {
                y[s * SUB + j] = KVALUES_FP4[(qs[j] & 0x0F) as usize] as f32 * d;
                y[s * SUB + 8 + j] = KVALUES_FP4[(qs[j] >> 4) as usize] as f32 * d;
            }
        }
    }
}

fn grid8(grid: u64) -> [u8; 8] {
    grid.to_le_bytes()
}

fn grid4(grid: u32) -> [u8; 4] {
    grid.to_le_bytes()
}

/// Applies ggml's sign mask: `kmask_iq2xs[j]` selects a bit of the sign byte.
#[inline]
fn signed(v: f32, signs: u8, j: usize) -> f32 {
    if signs & iq_tables::KMASK_IQ2XS[j] != 0 {
        -v
    } else {
        v
    }
}

/// IQ2_XS: 74 bytes per 256 elements. `qs` holds 32 sixteen-bit codes (9-bit grid index,
/// 7-bit sign index); eight packed nibble-pair scales give two sub-block scales each.
pub fn dequant_row_iq2_xs(src: &[u8], dst: &mut [f32]) {
    const QK_K: usize = 256;
    const BLK_BYTES: usize = 74;
    assert_eq!(dst.len() % QK_K, 0);
    let nb = dst.len() / QK_K;
    assert_eq!(src.len(), nb * BLK_BYTES);

    for i in 0..nb {
        let blk = &src[i * BLK_BYTES..(i + 1) * BLK_BYTES];
        let d = f16_to_f32(read_u16_le(&blk[0..2]));
        let scales = &blk[66..74];
        let y = &mut dst[i * QK_K..(i + 1) * QK_K];
        for ib32 in 0..8 {
            let db = [
                d * (0.5 + f32::from(scales[ib32] & 0x0F)) * 0.25,
                d * (0.5 + f32::from(scales[ib32] >> 4)) * 0.25,
            ];
            for l in 0..4 {
                let code = read_u16_le(&blk[2 + 2 * (4 * ib32 + l)..]);
                let grid = grid8(iq_tables::IQ2XS_GRID[(code & 511) as usize]);
                let signs = iq_tables::KSIGNS_IQ2XS[(code >> 9) as usize];
                let scale = db[l / 2];
                for j in 0..8 {
                    y[ib32 * 32 + l * 8 + j] = scale * f32::from(grid[j]) * signed(1.0, signs, j);
                }
            }
        }
    }
}

/// IQ2_S: 82 bytes per 256 elements. Like IQ2_XS but the grid index gains two high bits
/// from a separate `qh` plane, and the sign bytes sit in the second half of `qs`.
pub fn dequant_row_iq2_s(src: &[u8], dst: &mut [f32]) {
    const QK_K: usize = 256;
    const BLK_BYTES: usize = 82;
    assert_eq!(dst.len() % QK_K, 0);
    let nb = dst.len() / QK_K;
    assert_eq!(src.len(), nb * BLK_BYTES);

    for i in 0..nb {
        let blk = &src[i * BLK_BYTES..(i + 1) * BLK_BYTES];
        let d = f16_to_f32(read_u16_le(&blk[0..2]));
        let qs = &blk[2..66];
        let qh = &blk[66..74];
        let scales = &blk[74..82];
        let y = &mut dst[i * QK_K..(i + 1) * QK_K];
        for ib32 in 0..8 {
            let db = [
                d * (0.5 + f32::from(scales[ib32] & 0x0F)) * 0.25,
                d * (0.5 + f32::from(scales[ib32] >> 4)) * 0.25,
            ];
            for l in 0..4 {
                let hi = u16::from((qh[ib32] >> (2 * l)) & 0x03) << 8;
                let index = (u16::from(qs[4 * ib32 + l]) | hi) as usize;
                let grid = grid8(iq_tables::IQ2S_GRID[index]);
                let signs = qs[32 + 4 * ib32 + l];
                let scale = db[l / 2];
                for j in 0..8 {
                    y[ib32 * 32 + l * 8 + j] = scale * f32::from(grid[j]) * signed(1.0, signs, j);
                }
            }
        }
    }
}

/// IQ3_XXS: 98 bytes per 256 elements. Two four-element grids per eight outputs, with the
/// scale and four sign indices packed into one u32 per 32-element sub-block.
pub fn dequant_row_iq3_xxs(src: &[u8], dst: &mut [f32]) {
    const QK_K: usize = 256;
    const BLK_BYTES: usize = 98;
    assert_eq!(dst.len() % QK_K, 0);
    let nb = dst.len() / QK_K;
    assert_eq!(src.len(), nb * BLK_BYTES);

    for i in 0..nb {
        let blk = &src[i * BLK_BYTES..(i + 1) * BLK_BYTES];
        let d = f16_to_f32(read_u16_le(&blk[0..2]));
        let qs = &blk[2..98];
        let y = &mut dst[i * QK_K..(i + 1) * QK_K];
        for ib32 in 0..8 {
            let aux = u32::from_le_bytes([
                qs[64 + 4 * ib32],
                qs[65 + 4 * ib32],
                qs[66 + 4 * ib32],
                qs[67 + 4 * ib32],
            ]);
            let db = d * (0.5 + f32::from((aux >> 28) as u8)) * 0.5;
            for l in 0..4 {
                let signs = iq_tables::KSIGNS_IQ2XS[((aux >> (7 * l)) & 127) as usize];
                let g1 = grid4(iq_tables::IQ3XXS_GRID[qs[8 * ib32 + 2 * l] as usize]);
                let g2 = grid4(iq_tables::IQ3XXS_GRID[qs[8 * ib32 + 2 * l + 1] as usize]);
                for j in 0..4 {
                    y[ib32 * 32 + l * 8 + j] = db * f32::from(g1[j]) * signed(1.0, signs, j);
                    y[ib32 * 32 + l * 8 + 4 + j] =
                        db * f32::from(g2[j]) * signed(1.0, signs, j + 4);
                }
            }
        }
    }
}

/// IQ3_S: 110 bytes per 256 elements. Four scales of 4 bits, a 9th grid-index bit per
/// output taken from `qh`, and 32 bytes of sign masks.
pub fn dequant_row_iq3_s(src: &[u8], dst: &mut [f32]) {
    const QK_K: usize = 256;
    const BLK_BYTES: usize = 110;
    assert_eq!(dst.len() % QK_K, 0);
    let nb = dst.len() / QK_K;
    assert_eq!(src.len(), nb * BLK_BYTES);

    for i in 0..nb {
        let blk = &src[i * BLK_BYTES..(i + 1) * BLK_BYTES];
        let d = f16_to_f32(read_u16_le(&blk[0..2]));
        let qs = &blk[2..66];
        let qh = &blk[66..74];
        let signs = &blk[74..106];
        let scales = &blk[106..110];
        let y = &mut dst[i * QK_K..(i + 1) * QK_K];

        for pair in 0..4 {
            let db = [
                d * (1.0 + 2.0 * f32::from(scales[pair] & 0x0F)),
                d * (1.0 + 2.0 * f32::from(scales[pair] >> 4)),
            ];
            for half in 0..2 {
                let qs_base = 16 * pair + 8 * half;
                let sign_base = 8 * pair + 4 * half;
                let hi = qh[2 * pair + half];
                for l in 0..4 {
                    let bit1 = (hi >> (2 * l)) & 1;
                    let bit2 = (hi >> (2 * l + 1)) & 1;
                    let i1 = usize::from(qs[qs_base + 2 * l]) | usize::from(bit1) << 8;
                    let i2 = usize::from(qs[qs_base + 2 * l + 1]) | usize::from(bit2) << 8;
                    let g1 = grid4(iq_tables::IQ3S_GRID[i1]);
                    let g2 = grid4(iq_tables::IQ3S_GRID[i2]);
                    let sign = signs[sign_base + l];
                    let base = pair * 64 + half * 32 + l * 8;
                    for j in 0..4 {
                        y[base + j] = db[half] * f32::from(g1[j]) * signed(1.0, sign, j);
                        y[base + 4 + j] = db[half] * f32::from(g2[j]) * signed(1.0, sign, j + 4);
                    }
                }
            }
        }
    }
}

pub fn dequant_row(ggml_type: GgmlType, src: &[u8], dst: &mut [f32]) -> Result<(), GgufError> {
    match ggml_type {
        GgmlType::F32 => dequant_row_f32(src, dst),
        GgmlType::F16 => dequant_row_f16(src, dst),
        GgmlType::Q5_0 => dequant_row_q5_0(src, dst),
        GgmlType::Q8_0 => dequant_row_q8_0(src, dst),
        GgmlType::Q3K => dequant_row_q3_k(src, dst),
        GgmlType::Q4K => dequant_row_q4_k(src, dst),
        GgmlType::Q5K => dequant_row_q5_k(src, dst),
        GgmlType::Q6K => dequant_row_q6_k(src, dst),
        GgmlType::BF16 => dequant_row_bf16(src, dst),
        GgmlType::IQ2XS => dequant_row_iq2_xs(src, dst),
        GgmlType::IQ2S => dequant_row_iq2_s(src, dst),
        GgmlType::IQ3XXS => dequant_row_iq3_xxs(src, dst),
        GgmlType::IQ3S => dequant_row_iq3_s(src, dst),
        GgmlType::IQ4NL => dequant_row_iq4_nl(src, dst),
        GgmlType::NVFP4 => dequant_row_nvfp4(src, dst),
        other => return Err(GgufError::UnsupportedQuant(other.as_u32())),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---------------------------------------------------------------- IQ family
    //
    // These formats are grid lookups plus a sign mask, so the hand-computable
    // invariants are: a zero block decodes to zero, the scale and sign plumbing
    // behaves, and the tables are bit-identical to ggml's.

    const IQ_CASES: [(GgmlType, usize, usize); 4] = [
        (GgmlType::IQ2XS, 256, 74),
        (GgmlType::IQ2S, 256, 82),
        (GgmlType::IQ3XXS, 256, 98),
        (GgmlType::IQ3S, 256, 110),
    ];

    #[test]
    fn iq_tables_are_unchanged_from_ggml() {
        // FNV-1a over every value, regenerated from ggml-common.h.
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        let mut mix = |bytes: [u8; 8]| {
            for b in bytes {
                h = (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3);
            }
        };
        for v in crate::iq_tables::KMASK_IQ2XS {
            mix(u64::from(v).to_le_bytes());
        }
        for v in crate::iq_tables::KSIGNS_IQ2XS {
            mix(u64::from(v).to_le_bytes());
        }
        for v in crate::iq_tables::IQ2XS_GRID {
            mix(v.to_le_bytes());
        }
        for v in crate::iq_tables::IQ2S_GRID {
            mix(v.to_le_bytes());
        }
        for v in crate::iq_tables::IQ3XXS_GRID {
            mix(u64::from(v).to_le_bytes());
        }
        for v in crate::iq_tables::IQ3S_GRID {
            mix(u64::from(v).to_le_bytes());
        }
        assert_eq!(h, crate::iq_tables::GGML_TABLE_DIGEST);
    }

    #[test]
    fn iq_zero_block_decodes_to_zero() {
        for (ty, elems, bytes) in IQ_CASES {
            let src = vec![0u8; bytes];
            let mut dst = vec![0f32; elems];
            dequant_row(ty, &src, &mut dst).unwrap();
            assert!(dst.iter().all(|v| *v == 0.0), "{ty:?} zero block");
        }
    }

    #[test]
    fn iq_max_index_block_stays_finite() {
        // Every grid/sign index at its maximum must still land inside the tables.
        for (ty, elems, bytes) in IQ_CASES {
            let mut src = vec![0xFFu8; bytes];
            // A non-zero, finite scale in the first two bytes.
            src[0..2].copy_from_slice(&f16_block(1.0));
            let mut dst = vec![0f32; elems];
            dequant_row(ty, &src, &mut dst).unwrap();
            assert!(
                dst.iter().all(|v| v.is_finite()),
                "{ty:?} produced non-finite"
            );
            assert!(dst.iter().any(|v| *v != 0.0), "{ty:?} produced all zeros");
        }
    }

    #[test]
    fn iq2_xs_scale_and_sign_plumbing() {
        // d = 1, grid index 0, sign index 0, scale nibble 0 -> db = 0.5*0.25 = 0.125.
        let mut src = [0u8; 74];
        src[0..2].copy_from_slice(&f16_block(1.0));
        let mut dst = [0f32; 256];
        dequant_row_iq2_xs(&src, &mut dst);
        let grid = crate::iq_tables::IQ2XS_GRID[0].to_le_bytes();
        for j in 0..8 {
            assert_eq!(dst[j], 0.125 * f32::from(grid[j]));
        }
        // Scale nibble 1 in the low half doubles the base to 0.5 + 1.
        src[66] = 0x01;
        dequant_row_iq2_xs(&src, &mut dst);
        assert_eq!(dst[0], (0.5 + 1.0) * 0.25 * f32::from(grid[0]));
        // Sign index 1 must flip the sign bits selected by KMASK for the first code.
        src[2] = 0x00;
        src[3] = 0x02; // code >> 9 == 1
        let signs = crate::iq_tables::KSIGNS_IQ2XS[1];
        dequant_row_iq2_xs(&src, &mut dst);
        for j in 0..8 {
            let want = (0.5 + 1.0) * 0.25 * f32::from(grid[j]);
            let want = if signs & crate::iq_tables::KMASK_IQ2XS[j] != 0 {
                -want
            } else {
                want
            };
            assert_eq!(dst[j], want, "sign bit {j}");
        }
    }

    #[test]
    fn iq3_s_ninth_index_bit_and_signs() {
        // d = 1, scale nibble 0 -> db = 1; grid index 0 for both halves.
        let mut src = [0u8; 110];
        src[0..2].copy_from_slice(&f16_block(1.0));
        let mut dst = [0f32; 256];
        dequant_row_iq3_s(&src, &mut dst);
        let g = crate::iq_tables::IQ3S_GRID[0].to_le_bytes();
        for j in 0..4 {
            assert_eq!(dst[j], f32::from(g[j]));
            assert_eq!(dst[j + 4], f32::from(g[j]));
        }
        // Setting bit 0 of qh[0] moves the first grid index to 256.
        src[66] = 0x01;
        let g256 = crate::iq_tables::IQ3S_GRID[256].to_le_bytes();
        dequant_row_iq3_s(&src, &mut dst);
        assert_eq!(dst[0], f32::from(g256[0]));
        // Scale nibble 1 -> db = 1 + 2 = 3.
        src[106] = 0x01;
        dequant_row_iq3_s(&src, &mut dst);
        assert_eq!(dst[0], 3.0 * f32::from(g256[0]));
    }

    #[test]
    fn iq_types_dispatch_with_their_block_sizes() {
        for (ty, elems, bytes) in IQ_CASES {
            assert_eq!(ty.tensor_byte_len(elems).unwrap(), bytes, "{ty:?}");
            let mut dst = vec![0f32; elems];
            dequant_row(ty, &vec![0u8; bytes], &mut dst).unwrap();
            // Wrong length is rejected by the size formula, not by a panic inside.
            assert!(ty.tensor_byte_len(elems + 1).is_err(), "{ty:?}");
        }
    }

    fn f16_block(d: f32) -> [u8; 2] {
        // Only exact powers of two are needed by these tests.
        let bits: u16 = match d {
            0.0 => 0x0000,
            0.5 => 0x3800,
            1.0 => 0x3C00,
            2.0 => 0x4000,
            _ => panic!("test only uses exact halves/powers of two"),
        };
        bits.to_le_bytes()
    }

    #[test]
    fn bf16_is_a_shifted_f32() {
        let words: [u16; 5] = [0x3F80, 0xC000, 0x7F80, 0x0001, 0x8000];
        let src: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let mut dst = [0f32; 5];
        dequant_row_bf16(&src, &mut dst);
        assert_eq!(dst[0], 1.0);
        assert_eq!(dst[1], -2.0);
        assert_eq!(dst[2], f32::INFINITY);
        assert_eq!(dst[3], f32::from_bits(1 << 16)); // smallest normal f32
        assert_eq!(dst[4], 0.0);
        assert!(dst[4].is_sign_negative(), "bf16 -0.0 must keep its sign");
    }

    #[test]
    fn iq4_nl_uses_the_codebook_and_both_nibbles() {
        let mut src = [0u8; 18];
        src[0..2].copy_from_slice(&f16_block(1.0));
        src[2] = 0x98; // low 8 -> +1, high 9 -> +13
        src[3] = 0x70; // low 0 -> -127, high 7 -> -10
        let mut dst = [0f32; 32];
        dequant_row_iq4_nl(&src, &mut dst);
        assert_eq!(dst[0], 1.0);
        assert_eq!(dst[16], 13.0);
        assert_eq!(dst[1], -127.0);
        assert_eq!(dst[17], -10.0);
        // Every value must come from the codebook, scaled by d.
        for (i, v) in dst.iter().enumerate() {
            let code = if i < 16 {
                src[2 + i] & 0x0F
            } else {
                src[2 + i - 16] >> 4
            };
            assert_eq!(*v, KVALUES_IQ4NL[code as usize] as f32);
        }
    }

    #[test]
    fn nvfp4_scales_per_sixteen_element_sub_block() {
        let mut src = [0u8; 36];
        // ue4m3 0x40 decodes to 1.0 (exponent 8, mantissa 0, halved).
        src[0] = 0x40;
        src[4] = 0x70; // sub-block 0, byte 0: low 0 -> 0, high 7 -> +12
        let mut dst = [0f32; 64];
        dequant_row_nvfp4(&src, &mut dst);
        assert_eq!(dst[0], 0.0);
        assert_eq!(dst[8], 12.0);
        // Sub-block 1 has scale 0, so its 16 values are all zero even with nibbles set.
        src[1] = 0x00;
        src[12] = 0xFF;
        dequant_row_nvfp4(&src, &mut dst);
        assert!(dst[16..32].iter().all(|v| *v == 0.0));

        // A scale of 0x40 in every sub-block with all-zero nibbles is all zeros; the
        // zero scale byte 0x00 must also decode to zero rather than a subnormal.
        let zero = [0u8; 36];
        dequant_row_nvfp4(&zero, &mut dst);
        assert!(dst.iter().all(|v| *v == 0.0));
    }

    #[test]
    fn q5_k_high_bit_plane_and_scale_packing() {
        let mut src = [0u8; 176];
        src[0..2].copy_from_slice(&f16_block(1.0)); // d
        src[2..4].copy_from_slice(&f16_block(0.0)); // dmin
        // scales[0..4] low 6 bits = 1 (sc=1 for the first two groups), mins 0.
        for b in src.iter_mut().skip(4).take(4) {
            *b = 1;
        }
        let mut dst = [0f32; 256];
        dequant_row_q5_k(&src, &mut dst);
        assert!(
            dst.iter().all(|v| *v == 0.0),
            "all-zero quants must decode to zero"
        );

        // High-bit plane: u1 = 1 for the first 32 elements of the first 64-group,
        // u2 = 2 for the second 32, then the bits shift left by two per group.
        src[16] = 1; // qh[0] bit 0 -> element 0 gets +16
        dequant_row_q5_k(&src, &mut dst);
        assert_eq!(dst[0], 16.0);
        assert_eq!(dst[1], 0.0);

        src[16] = 4; // bit 2 -> element 64 (group 1, u1 = 4)
        dequant_row_q5_k(&src, &mut dst);
        assert_eq!(dst[64], 16.0);
        assert_eq!(dst[0], 0.0);
    }

    #[test]
    fn q3_k_unpacks_six_bit_scales_and_the_high_mask() {
        let mut src = [0u8; 110];
        src[0..32].fill(0xFF); // hmask: all high bits set
        src[108..110].copy_from_slice(&f16_block(1.0));
        // Pack sixteen 6-bit scales of value 33 so that (scale - 32) == 1. The packing
        // is the documented inverse of the unpack in `dequant_row_q3_k`: scales[0..4]
        // carry the low nibbles, scales[4..8] the second nibble set, and the third word
        // contributes the top two bits for all four sets.
        src[96..100].fill(0x11);
        src[100..104].fill(0x11);
        src[104..108].fill(0xAA);

        let mut dst = [0f32; 256];
        dequant_row_q3_k(&src, &mut dst);
        assert!(dst.iter().all(|v| *v == 0.0));

        // q = 1 in the lowest 2 bits of the first byte, high bit set -> value 1*1.
        src[32] = 1;
        dequant_row_q3_k(&src, &mut dst);
        assert_eq!(dst[0], 1.0);
        assert_eq!(dst[1], 0.0);

        // Clear the element's high bit: the mask subtracts 4 instead of 0.
        src[0] = 0xFE;
        dequant_row_q3_k(&src, &mut dst);
        assert_eq!(dst[0], -3.0);
    }

    #[test]
    fn new_types_dispatch_and_reject_wrong_lengths() {
        for (ty, elems, bytes) in [
            (GgmlType::BF16, 3usize, 6usize),
            (GgmlType::Q3K, 256, 110),
            (GgmlType::Q5K, 256, 176),
            (GgmlType::IQ4NL, 32, 18),
            (GgmlType::NVFP4, 64, 36),
        ] {
            assert_eq!(ty.tensor_byte_len(elems).unwrap(), bytes);
            let src = vec![0u8; bytes];
            let mut dst = vec![0f32; elems];
            dequant_row(ty, &src, &mut dst).unwrap();
        }
        // Not-yet-implemented types must report the type id, not panic.
        let err = dequant_row(GgmlType::IQ2XXS, &[0u8; 66], &mut [0f32; 256]).unwrap_err();
        assert!(format!("{err}").contains("unsupported"));
    }

    #[test]
    fn f16_known_values() {
        assert_eq!(f16_to_f32(0x0000), 0.0);
        assert_eq!(f16_to_f32(0x3C00), 1.0);
        assert_eq!(f16_to_f32(0xBC00), -1.0);
        assert_eq!(f16_to_f32(0x3800), 0.5);
        assert_eq!(f16_to_f32(0x4000), 2.0);
        assert!((f16_to_f32(0x0001) - 5.960_464_5e-8).abs() < 1e-12);
    }

    #[test]
    fn dequant_q8_0_q5_0_q4_k_q6_k_smoke() {
        // Q8_0: d=0.5 (0x3800), qs=[0..32]
        let mut q8 = [0u8; 34];
        q8[0..2].copy_from_slice(&0x3800u16.to_le_bytes());
        for i in 0..32 {
            q8[2 + i] = (i as i8 - 16) as u8;
        }
        let mut out32 = [0f32; 32];
        dequant_row_q8_0(&q8, &mut out32);
        assert_eq!(out32[0], -8.0);
        assert_eq!(out32[16], 0.0);
        assert_eq!(out32[31], 7.5);

        // Q5_0: d=1.0 (0x3C00), qh=0x00010001 (bit 0 and bit 16 set), qs[0]=0x21
        let mut q5 = [0u8; 22];
        q5[0..2].copy_from_slice(&0x3C00u16.to_le_bytes());
        q5[2..6].copy_from_slice(&0x0001_0001u32.to_le_bytes());
        q5[6] = 0x21;
        dequant_row_q5_0(&q5, &mut out32);
        // elem 0: low=1 | 0x10 = 17, 17-16 = 1
        assert_eq!(out32[0], 1.0);
        // elem 16: high=2 | 0x10 = 18, 18-16 = 2
        assert_eq!(out32[16], 2.0);

        // Q4_K: d=1.0, dmin=0.5, scale[0]=2, scale[4]=1 => d1=2, min1=0.5, q[0]=0x35
        let mut q4k = [0u8; 144];
        q4k[0..2].copy_from_slice(&0x3C00u16.to_le_bytes());
        q4k[2..4].copy_from_slice(&0x3800u16.to_le_bytes());
        q4k[4] = 2;
        q4k[8] = 1;
        q4k[16] = 0x35;
        let mut out256 = [0f32; 256];
        dequant_row_q4_k(&q4k, &mut out256);
        assert_eq!(out256[0], 2.0 * 5.0 - 0.5);

        // Q6_K: d=0.5, sc[0]=3, ql[0]=0x01, qh[0]=0x02 => q1 = (1 | (2<<4)) - 32 = 33 - 32 = 1
        let mut q6k = [0u8; 210];
        q6k[0] = 0x01;
        q6k[128] = 0x02;
        q6k[192] = 3;
        q6k[208..210].copy_from_slice(&0x3800u16.to_le_bytes());
        dequant_row_q6_k(&q6k, &mut out256);
        assert_eq!(out256[0], 0.5 * 3.0 * 1.0);
    }
}
