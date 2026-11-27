use crate::{GgmlType, GgufError};

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

pub fn dequant_row(ggml_type: GgmlType, src: &[u8], dst: &mut [f32]) -> Result<(), GgufError> {
    match ggml_type {
        GgmlType::F32 => dequant_row_f32(src, dst),
        GgmlType::F16 => dequant_row_f16(src, dst),
        GgmlType::Q5_0 => dequant_row_q5_0(src, dst),
        GgmlType::Q8_0 => dequant_row_q8_0(src, dst),
        GgmlType::Q4K => dequant_row_q4_k(src, dst),
        GgmlType::Q6K => dequant_row_q6_k(src, dst),
        other => return Err(GgufError::UnsupportedQuant(other.as_u32())),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

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
