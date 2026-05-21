use rustler::{Encoder, Env, Error, NifResult, ResourceArc, Term};

pub mod ffi;

// NOTE: CubeCL GPU integration placeholder.
// When the cubecl crate is available, uncomment the feature flag in Cargo.toml
// and add cubecl::wgpu::WgpuRuntime backend support here.
// For now, all operations run on CPU with optimized integer-aware paths.

// ── DType ─────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    F32,
    F64,
    S32,
    S64,
    U32,
    U8,
}

impl DType {
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "f32" => Some(DType::F32),
            "f64" => Some(DType::F64),
            "s32" => Some(DType::S32),
            "s64" => Some(DType::S64),
            "u32" => Some(DType::U32),
            "u8" => Some(DType::U8),
            _ => None,
        }
    }
    pub fn size_in_bytes(self) -> usize {
        match self {
            DType::F32 => 4,
            DType::F64 => 8,
            DType::S32 => 4,
            DType::S64 => 8,
            DType::U32 => 4,
            DType::U8 => 1,
        }
    }
}

// ── BufResource ──────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct BufResource {
    pub data: Vec<u8>,
    pub shape: Vec<usize>,
    pub dtype: DType,
}

impl BufResource {
    pub fn num_elements(&self) -> usize {
        self.shape.iter().product()
    }
}

// ── Atoms ────────────────────────────────────────────────────

mod atoms {
    rustler::atoms! { ok, error }
}

// ── Helpers ──────────────────────────────────────────────────

fn decode_dtype(s: &str) -> NifResult<DType> {
    DType::from_str(s.trim())
        .ok_or_else(|| Error::RaiseTerm(Box::new(format!("unknown dtype: {}", s))))
}

fn strides_for(shape: &[usize]) -> Vec<usize> {
    let mut s = vec![1usize; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        s[i] = s[i + 1] * shape[i + 1];
    }
    s
}

fn to_f64(data: &[u8], dtype: DType) -> Vec<f64> {
    match dtype {
        DType::F32 => bytemuck::cast_slice::<u8, f32>(data)
            .iter()
            .map(|v| *v as f64)
            .collect(),
        DType::F64 => bytemuck::cast_slice::<u8, f64>(data).to_vec(),
        DType::S32 => bytemuck::cast_slice::<u8, i32>(data)
            .iter()
            .map(|v| *v as f64)
            .collect(),
        DType::S64 => bytemuck::cast_slice::<u8, i64>(data)
            .iter()
            .map(|v| *v as f64)
            .collect(),
        DType::U32 => bytemuck::cast_slice::<u8, u32>(data)
            .iter()
            .map(|v| *v as f64)
            .collect(),
        DType::U8 => data.iter().map(|v| *v as f64).collect(),
    }
}

fn from_f64(vals: Vec<f64>, dtype: DType) -> Vec<u8> {
    match dtype {
        DType::F32 => {
            let v: Vec<f32> = vals.iter().map(|x| *x as f32).collect();
            bytemuck::cast_slice(&v).to_vec()
        }
        DType::F64 => bytemuck::cast_slice(&vals).to_vec(),
        DType::S32 => {
            let v: Vec<i32> = vals.iter().map(|x| *x as i32).collect();
            bytemuck::cast_slice(&v).to_vec()
        }
        DType::S64 => {
            let v: Vec<i64> = vals.iter().map(|x| *x as i64).collect();
            bytemuck::cast_slice(&v).to_vec()
        }
        DType::U32 => {
            let v: Vec<u32> = vals.iter().map(|x| *x as u32).collect();
            bytemuck::cast_slice(&v).to_vec()
        }
        DType::U8 => vals.iter().map(|x| *x as u8).collect(),
    }
}

// Integer-aware binary ops (no f64 roundtrip for integer types)
fn binary_op_int(
    data_a: &[u8],
    dtype_a: DType,
    data_b: &[u8],
    dtype_b: DType,
    op_i64: impl Fn(i64, i64) -> i64,
    op_f64: impl Fn(f64, f64) -> f64,
) -> Vec<u8> {
    match (dtype_a, dtype_b) {
        (DType::S32, DType::S32) => {
            let va: &[i32] = bytemuck::cast_slice(data_a);
            let vb: &[i32] = bytemuck::cast_slice(data_b);
            let v: Vec<i32> = va
                .iter()
                .zip(vb.iter())
                .map(|(a, b)| op_i64(*a as i64, *b as i64) as i32)
                .collect();
            bytemuck::cast_slice(&v).to_vec()
        }
        (DType::S64, DType::S64) => {
            let va: &[i64] = bytemuck::cast_slice(data_a);
            let vb: &[i64] = bytemuck::cast_slice(data_b);
            let v: Vec<i64> = va
                .iter()
                .zip(vb.iter())
                .map(|(a, b)| op_i64(*a, *b))
                .collect();
            bytemuck::cast_slice(&v).to_vec()
        }
        (DType::U32, DType::U32) => {
            let va: &[u32] = bytemuck::cast_slice(data_a);
            let vb: &[u32] = bytemuck::cast_slice(data_b);
            let v: Vec<u32> = va
                .iter()
                .zip(vb.iter())
                .map(|(a, b)| op_i64(*a as i64, *b as i64) as u32)
                .collect();
            bytemuck::cast_slice(&v).to_vec()
        }
        (DType::U8, DType::U8) => data_a
            .iter()
            .zip(data_b.iter())
            .map(|(a, b)| op_i64(*a as i64, *b as i64) as u8)
            .collect(),
        _ => {
            let va = to_f64(data_a, dtype_a);
            let vb = to_f64(data_b, dtype_b);
            from_f64(
                va.iter()
                    .zip(vb.iter())
                    .map(|(a, b)| op_f64(*a, *b))
                    .collect(),
                dtype_a,
            )
        }
    }
}

fn unary_op_int(
    data: &[u8],
    dtype: DType,
    op_i64: impl Fn(i64) -> i64,
    op_f64: impl Fn(f64) -> f64,
) -> Vec<u8> {
    match dtype {
        DType::S32 => {
            let va: &[i32] = bytemuck::cast_slice(data);
            let v: Vec<i32> = va.iter().map(|a| op_i64(*a as i64) as i32).collect();
            bytemuck::cast_slice(&v).to_vec()
        }
        DType::S64 => {
            let va: &[i64] = bytemuck::cast_slice(data);
            let v: Vec<i64> = va.iter().map(|a| op_i64(*a)).collect();
            bytemuck::cast_slice(&v).to_vec()
        }
        DType::U32 => {
            let va: &[u32] = bytemuck::cast_slice(data);
            let v: Vec<u32> = va.iter().map(|a| op_i64(*a as i64) as u32).collect();
            bytemuck::cast_slice(&v).to_vec()
        }
        DType::U8 => data.iter().map(|a| op_i64(*a as i64) as u8).collect(),
        _ => {
            let v = to_f64(data, dtype);
            from_f64(v.iter().map(|x| op_f64(*x)).collect(), dtype)
        }
    }
}

fn unary_op(data: &[u8], dtype: DType, op: impl Fn(f64) -> f64 + Copy) -> Vec<u8> {
    unary_op_int(data, dtype, |x| op(x as f64) as i64, op)
}

fn binary_op(
    data_a: &[u8],
    dtype_a: DType,
    data_b: &[u8],
    dtype_b: DType,
    op: impl Fn(f64, f64) -> f64 + Copy,
) -> Vec<u8> {
    binary_op_int(
        data_a,
        dtype_a,
        data_b,
        dtype_b,
        |a, b| op(a as f64, b as f64) as i64,
        op,
    )
}

fn unary_op_bool(data: &[u8], dtype: DType, op: impl Fn(f64) -> bool) -> Vec<u8> {
    let v = to_f64(data, dtype);
    v.iter().map(|x| if op(*x) { 1u8 } else { 0u8 }).collect()
}

fn binary_op_bool(
    data_a: &[u8],
    dtype_a: DType,
    data_b: &[u8],
    dtype_b: DType,
    op: impl Fn(f64, f64) -> bool,
) -> Vec<u8> {
    let va = to_f64(data_a, dtype_a);
    let vb = to_f64(data_b, dtype_b);
    va.iter()
        .zip(vb.iter())
        .map(|(a, b)| if op(*a, *b) { 1u8 } else { 0u8 })
        .collect()
}

// ── NIF init ─────────────────────────────────────────────────

#[allow(non_local_definitions)]
fn on_load(env: Env, _info: Term) -> bool {
    let _ = rustler::resource!(BufResource, env);
    true
}

rustler::init!("Elixir.ExCubecl.NIF", load = on_load);

// ═══════════════════════════════════════════════════════════════
// Buffer lifecycle
// ═══════════════════════════════════════════════════════════════

#[rustler::nif(schedule = "DirtyCpu")]
fn new_tensor<'a>(
    env: Env<'a>,
    data: rustler::Binary<'a>,
    shape: Vec<usize>,
    dtype_str: String,
) -> NifResult<Term<'a>> {
    let dtype = decode_dtype(&dtype_str)?;
    let expected = shape.iter().product::<usize>() * dtype.size_in_bytes();
    if data.len() != expected {
        return Err(Error::RaiseTerm(Box::new(format!(
            "size mismatch: got {} expected {}",
            data.len(),
            expected
        ))));
    }
    Ok((
        atoms::ok(),
        ResourceArc::new(BufResource {
            data: data.as_slice().to_vec(),
            shape,
            dtype,
        }),
    )
        .encode(env))
}

#[rustler::nif]
fn read_tensor<'a>(env: Env<'a>, buf: ResourceArc<BufResource>) -> NifResult<Term<'a>> {
    let mut nb = rustler::types::binary::NewBinary::new(env, buf.data.len());
    nb.as_mut_slice().copy_from_slice(&buf.data);
    Ok((atoms::ok(), rustler::Binary::from(nb)).encode(env))
}

#[rustler::nif]
fn deallocate_tensor(env: Env, _buf: ResourceArc<BufResource>) -> NifResult<Term> {
    Ok(atoms::ok().encode(env))
}

#[rustler::nif]
fn tensor_shape(env: Env, buf: ResourceArc<BufResource>) -> NifResult<Term> {
    Ok((atoms::ok(), buf.shape.clone()).encode(env))
}

#[rustler::nif]
fn tensor_dtype(env: Env, buf: ResourceArc<BufResource>) -> NifResult<Term> {
    Ok((atoms::ok(), buf.dtype as usize).encode(env))
}

// ═══════════════════════════════════════════════════════════════
// Binary ops
// ═══════════════════════════════════════════════════════════════

#[rustler::nif(schedule = "DirtyCpu")]
fn add<'a>(
    env: Env<'a>,
    a: ResourceArc<BufResource>,
    b: ResourceArc<BufResource>,
) -> NifResult<Term<'a>> {
    Ok((
        atoms::ok(),
        ResourceArc::new(BufResource {
            data: binary_op(&a.data, a.dtype, &b.data, b.dtype, |x, y| x + y),
            shape: a.shape.clone(),
            dtype: a.dtype,
        }),
    )
        .encode(env))
}

#[rustler::nif(schedule = "DirtyCpu")]
fn subtract<'a>(
    env: Env<'a>,
    a: ResourceArc<BufResource>,
    b: ResourceArc<BufResource>,
) -> NifResult<Term<'a>> {
    Ok((
        atoms::ok(),
        ResourceArc::new(BufResource {
            data: binary_op(&a.data, a.dtype, &b.data, b.dtype, |x, y| x - y),
            shape: a.shape.clone(),
            dtype: a.dtype,
        }),
    )
        .encode(env))
}

#[rustler::nif(schedule = "DirtyCpu")]
fn multiply<'a>(
    env: Env<'a>,
    a: ResourceArc<BufResource>,
    b: ResourceArc<BufResource>,
) -> NifResult<Term<'a>> {
    Ok((
        atoms::ok(),
        ResourceArc::new(BufResource {
            data: binary_op(&a.data, a.dtype, &b.data, b.dtype, |x, y| x * y),
            shape: a.shape.clone(),
            dtype: a.dtype,
        }),
    )
        .encode(env))
}

#[rustler::nif(schedule = "DirtyCpu")]
fn divide<'a>(
    env: Env<'a>,
    a: ResourceArc<BufResource>,
    b: ResourceArc<BufResource>,
) -> NifResult<Term<'a>> {
    Ok((
        atoms::ok(),
        ResourceArc::new(BufResource {
            data: binary_op(&a.data, a.dtype, &b.data, b.dtype, |x, y| {
                if y == 0.0 {
                    f64::NAN
                } else {
                    x / y
                }
            }),
            shape: a.shape.clone(),
            dtype: a.dtype,
        }),
    )
        .encode(env))
}

#[rustler::nif(schedule = "DirtyCpu")]
fn pow<'a>(
    env: Env<'a>,
    a: ResourceArc<BufResource>,
    b: ResourceArc<BufResource>,
) -> NifResult<Term<'a>> {
    Ok((
        atoms::ok(),
        ResourceArc::new(BufResource {
            data: binary_op(&a.data, a.dtype, &b.data, b.dtype, |x, y| x.powf(y)),
            shape: a.shape.clone(),
            dtype: a.dtype,
        }),
    )
        .encode(env))
}

#[rustler::nif(schedule = "DirtyCpu")]
fn remainder<'a>(
    env: Env<'a>,
    a: ResourceArc<BufResource>,
    b: ResourceArc<BufResource>,
) -> NifResult<Term<'a>> {
    Ok((
        atoms::ok(),
        ResourceArc::new(BufResource {
            data: binary_op(&a.data, a.dtype, &b.data, b.dtype, |x, y| x % y),
            shape: a.shape.clone(),
            dtype: a.dtype,
        }),
    )
        .encode(env))
}

#[rustler::nif(schedule = "DirtyCpu")]
fn atan2<'a>(
    env: Env<'a>,
    a: ResourceArc<BufResource>,
    b: ResourceArc<BufResource>,
) -> NifResult<Term<'a>> {
    Ok((
        atoms::ok(),
        ResourceArc::new(BufResource {
            data: binary_op(&a.data, a.dtype, &b.data, b.dtype, |x, y| x.atan2(y)),
            shape: a.shape.clone(),
            dtype: a.dtype,
        }),
    )
        .encode(env))
}

#[rustler::nif(schedule = "DirtyCpu")]
fn min_tensor<'a>(
    env: Env<'a>,
    a: ResourceArc<BufResource>,
    b: ResourceArc<BufResource>,
) -> NifResult<Term<'a>> {
    Ok((
        atoms::ok(),
        ResourceArc::new(BufResource {
            data: binary_op(&a.data, a.dtype, &b.data, b.dtype, |x, y| x.min(y)),
            shape: a.shape.clone(),
            dtype: a.dtype,
        }),
    )
        .encode(env))
}

#[rustler::nif(schedule = "DirtyCpu")]
fn max_tensor<'a>(
    env: Env<'a>,
    a: ResourceArc<BufResource>,
    b: ResourceArc<BufResource>,
) -> NifResult<Term<'a>> {
    Ok((
        atoms::ok(),
        ResourceArc::new(BufResource {
            data: binary_op(&a.data, a.dtype, &b.data, b.dtype, |x, y| x.max(y)),
            shape: a.shape.clone(),
            dtype: a.dtype,
        }),
    )
        .encode(env))
}

#[rustler::nif(schedule = "DirtyCpu")]
fn quotient<'a>(
    env: Env<'a>,
    a: ResourceArc<BufResource>,
    b: ResourceArc<BufResource>,
) -> NifResult<Term<'a>> {
    Ok((
        atoms::ok(),
        ResourceArc::new(BufResource {
            data: binary_op(&a.data, a.dtype, &b.data, b.dtype, |x, y| {
                if y == 0.0 {
                    0.0
                } else {
                    (x / y).trunc()
                }
            }),
            shape: a.shape.clone(),
            dtype: a.dtype,
        }),
    )
        .encode(env))
}

// ── Bitwise ops (integer-efficient) ──────────────────────────

#[rustler::nif(schedule = "DirtyCpu")]
fn bitwise_and<'a>(
    env: Env<'a>,
    a: ResourceArc<BufResource>,
    b: ResourceArc<BufResource>,
) -> NifResult<Term<'a>> {
    Ok((
        atoms::ok(),
        ResourceArc::new(BufResource {
            data: binary_op_int(
                &a.data,
                a.dtype,
                &b.data,
                b.dtype,
                |x, y| x & y,
                |x, y| (x as i64 & y as i64) as f64,
            ),
            shape: a.shape.clone(),
            dtype: a.dtype,
        }),
    )
        .encode(env))
}

#[rustler::nif(schedule = "DirtyCpu")]
fn bitwise_or<'a>(
    env: Env<'a>,
    a: ResourceArc<BufResource>,
    b: ResourceArc<BufResource>,
) -> NifResult<Term<'a>> {
    Ok((
        atoms::ok(),
        ResourceArc::new(BufResource {
            data: binary_op_int(
                &a.data,
                a.dtype,
                &b.data,
                b.dtype,
                |x, y| x | y,
                |x, y| (x as i64 | y as i64) as f64,
            ),
            shape: a.shape.clone(),
            dtype: a.dtype,
        }),
    )
        .encode(env))
}

#[rustler::nif(schedule = "DirtyCpu")]
fn bitwise_xor<'a>(
    env: Env<'a>,
    a: ResourceArc<BufResource>,
    b: ResourceArc<BufResource>,
) -> NifResult<Term<'a>> {
    Ok((
        atoms::ok(),
        ResourceArc::new(BufResource {
            data: binary_op_int(
                &a.data,
                a.dtype,
                &b.data,
                b.dtype,
                |x, y| x ^ y,
                |x, y| (x as i64 ^ y as i64) as f64,
            ),
            shape: a.shape.clone(),
            dtype: a.dtype,
        }),
    )
        .encode(env))
}

#[rustler::nif(schedule = "DirtyCpu")]
fn left_shift<'a>(
    env: Env<'a>,
    a: ResourceArc<BufResource>,
    b: ResourceArc<BufResource>,
) -> NifResult<Term<'a>> {
    Ok((
        atoms::ok(),
        ResourceArc::new(BufResource {
            data: binary_op_int(
                &a.data,
                a.dtype,
                &b.data,
                b.dtype,
                |x, y| x << y,
                |x, y| ((x as i64) << (y as i64)) as f64,
            ),
            shape: a.shape.clone(),
            dtype: a.dtype,
        }),
    )
        .encode(env))
}

#[rustler::nif(schedule = "DirtyCpu")]
fn right_shift<'a>(
    env: Env<'a>,
    a: ResourceArc<BufResource>,
    b: ResourceArc<BufResource>,
) -> NifResult<Term<'a>> {
    Ok((
        atoms::ok(),
        ResourceArc::new(BufResource {
            data: binary_op_int(
                &a.data,
                a.dtype,
                &b.data,
                b.dtype,
                |x, y| x >> y,
                |x, y| ((x as i64) >> (y as i64)) as f64,
            ),
            shape: a.shape.clone(),
            dtype: a.dtype,
        }),
    )
        .encode(env))
}

// ═══════════════════════════════════════════════════════════════
// Comparison ops
// ═══════════════════════════════════════════════════════════════

#[rustler::nif(schedule = "DirtyCpu")]
fn equal<'a>(
    env: Env<'a>,
    a: ResourceArc<BufResource>,
    b: ResourceArc<BufResource>,
) -> NifResult<Term<'a>> {
    Ok((
        atoms::ok(),
        ResourceArc::new(BufResource {
            data: binary_op_bool(&a.data, a.dtype, &b.data, b.dtype, |x, y| {
                (x - y).abs() < f64::EPSILON
            }),
            shape: a.shape.clone(),
            dtype: DType::U8,
        }),
    )
        .encode(env))
}

#[rustler::nif(schedule = "DirtyCpu")]
fn not_equal<'a>(
    env: Env<'a>,
    a: ResourceArc<BufResource>,
    b: ResourceArc<BufResource>,
) -> NifResult<Term<'a>> {
    Ok((
        atoms::ok(),
        ResourceArc::new(BufResource {
            data: binary_op_bool(&a.data, a.dtype, &b.data, b.dtype, |x, y| {
                (x - y).abs() >= f64::EPSILON
            }),
            shape: a.shape.clone(),
            dtype: DType::U8,
        }),
    )
        .encode(env))
}

#[rustler::nif(schedule = "DirtyCpu")]
fn greater<'a>(
    env: Env<'a>,
    a: ResourceArc<BufResource>,
    b: ResourceArc<BufResource>,
) -> NifResult<Term<'a>> {
    Ok((
        atoms::ok(),
        ResourceArc::new(BufResource {
            data: binary_op_bool(&a.data, a.dtype, &b.data, b.dtype, |x, y| x > y),
            shape: a.shape.clone(),
            dtype: DType::U8,
        }),
    )
        .encode(env))
}

#[rustler::nif(schedule = "DirtyCpu")]
fn less<'a>(
    env: Env<'a>,
    a: ResourceArc<BufResource>,
    b: ResourceArc<BufResource>,
) -> NifResult<Term<'a>> {
    Ok((
        atoms::ok(),
        ResourceArc::new(BufResource {
            data: binary_op_bool(&a.data, a.dtype, &b.data, b.dtype, |x, y| x < y),
            shape: a.shape.clone(),
            dtype: DType::U8,
        }),
    )
        .encode(env))
}

#[rustler::nif(schedule = "DirtyCpu")]
fn greater_equal<'a>(
    env: Env<'a>,
    a: ResourceArc<BufResource>,
    b: ResourceArc<BufResource>,
) -> NifResult<Term<'a>> {
    Ok((
        atoms::ok(),
        ResourceArc::new(BufResource {
            data: binary_op_bool(&a.data, a.dtype, &b.data, b.dtype, |x, y| x >= y),
            shape: a.shape.clone(),
            dtype: DType::U8,
        }),
    )
        .encode(env))
}

#[rustler::nif(schedule = "DirtyCpu")]
fn less_equal<'a>(
    env: Env<'a>,
    a: ResourceArc<BufResource>,
    b: ResourceArc<BufResource>,
) -> NifResult<Term<'a>> {
    Ok((
        atoms::ok(),
        ResourceArc::new(BufResource {
            data: binary_op_bool(&a.data, a.dtype, &b.data, b.dtype, |x, y| x <= y),
            shape: a.shape.clone(),
            dtype: DType::U8,
        }),
    )
        .encode(env))
}

#[rustler::nif(schedule = "DirtyCpu")]
fn logical_and<'a>(
    env: Env<'a>,
    a: ResourceArc<BufResource>,
    b: ResourceArc<BufResource>,
) -> NifResult<Term<'a>> {
    Ok((
        atoms::ok(),
        ResourceArc::new(BufResource {
            data: binary_op_bool(&a.data, a.dtype, &b.data, b.dtype, |x, y| {
                x != 0.0 && y != 0.0
            }),
            shape: a.shape.clone(),
            dtype: DType::U8,
        }),
    )
        .encode(env))
}

#[rustler::nif(schedule = "DirtyCpu")]
fn logical_or<'a>(
    env: Env<'a>,
    a: ResourceArc<BufResource>,
    b: ResourceArc<BufResource>,
) -> NifResult<Term<'a>> {
    Ok((
        atoms::ok(),
        ResourceArc::new(BufResource {
            data: binary_op_bool(&a.data, a.dtype, &b.data, b.dtype, |x, y| {
                x != 0.0 || y != 0.0
            }),
            shape: a.shape.clone(),
            dtype: DType::U8,
        }),
    )
        .encode(env))
}

#[rustler::nif(schedule = "DirtyCpu")]
fn logical_xor<'a>(
    env: Env<'a>,
    a: ResourceArc<BufResource>,
    b: ResourceArc<BufResource>,
) -> NifResult<Term<'a>> {
    Ok((
        atoms::ok(),
        ResourceArc::new(BufResource {
            data: binary_op_bool(&a.data, a.dtype, &b.data, b.dtype, |x, y| {
                (x != 0.0) ^ (y != 0.0)
            }),
            shape: a.shape.clone(),
            dtype: DType::U8,
        }),
    )
        .encode(env))
}

// ═══════════════════════════════════════════════════════════════
// Unary ops
// ═══════════════════════════════════════════════════════════════

macro_rules! unary_nif {
    ($name:ident, $op:expr) => {
        #[rustler::nif(schedule = "DirtyCpu")]
        fn $name<'a>(env: Env<'a>, a: ResourceArc<BufResource>) -> NifResult<Term<'a>> {
            Ok((
                atoms::ok(),
                ResourceArc::new(BufResource {
                    data: unary_op(&a.data, a.dtype, $op),
                    shape: a.shape.clone(),
                    dtype: a.dtype,
                }),
            )
                .encode(env))
        }
    };
}

unary_nif!(negate, |x| -x);
unary_nif!(abs_tensor, |x| x.abs());
unary_nif!(exp, |x| x.exp());
unary_nif!(log, |x| if x > 0.0 { x.ln() } else { f64::NAN });
unary_nif!(sqrt, |x| x.sqrt());
unary_nif!(sin, |x| x.sin());
unary_nif!(cos, |x| x.cos());
unary_nif!(tan, |x| x.tan());
unary_nif!(sigmoid, |x| 1.0 / (1.0 + (-x).exp()));
unary_nif!(relu, |x| if x > 0.0 { x } else { 0.0 });
unary_nif!(expm1, |x| x.exp_m1());
unary_nif!(log1p, |x| (1.0 + x).ln());
unary_nif!(cosh, |x| x.cosh());
unary_nif!(sinh, |x| x.sinh());
unary_nif!(tanh, |x| x.tanh());
unary_nif!(acos, |x| x.acos());
unary_nif!(asin, |x| x.asin());
unary_nif!(atan, |x| x.atan());
unary_nif!(acosh, |x| x.acosh());
unary_nif!(asinh, |x| x.asinh());
unary_nif!(atanh, |x| x.atanh());
unary_nif!(rsqrt, |x| 1.0 / x.sqrt());
unary_nif!(cbrt, |x| x.cbrt());
unary_nif!(ceil_tensor, |x| x.ceil());
unary_nif!(floor_tensor, |x| x.floor());
unary_nif!(round_tensor, |x| x.round());

#[rustler::nif(schedule = "DirtyCpu")]
fn sign_tensor<'a>(env: Env<'a>, a: ResourceArc<BufResource>) -> NifResult<Term<'a>> {
    Ok((
        atoms::ok(),
        ResourceArc::new(BufResource {
            data: unary_op(&a.data, a.dtype, |x| {
                if x > 0.0 {
                    1.0
                } else if x < 0.0 {
                    -1.0
                } else {
                    0.0
                }
            }),
            shape: a.shape.clone(),
            dtype: a.dtype,
        }),
    )
        .encode(env))
}

fn erf_approx(x: f64) -> f64 {
    let s = if x < 0.0 { -1.0 } else { 1.0 };
    let ax = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * ax);
    let p = 1.061405429 * t - 1.453152027;
    let p = p * t + 1.421413741;
    let p = p * t - 0.284496736;
    let p = p * t + 0.254829592;
    s * (1.0 - p * t * (-ax * ax).exp())
}

#[rustler::nif(schedule = "DirtyCpu")]
fn erf<'a>(env: Env<'a>, a: ResourceArc<BufResource>) -> NifResult<Term<'a>> {
    Ok((
        atoms::ok(),
        ResourceArc::new(BufResource {
            data: unary_op(&a.data, a.dtype, erf_approx),
            shape: a.shape.clone(),
            dtype: a.dtype,
        }),
    )
        .encode(env))
}

#[rustler::nif(schedule = "DirtyCpu")]
fn erfc<'a>(env: Env<'a>, a: ResourceArc<BufResource>) -> NifResult<Term<'a>> {
    Ok((
        atoms::ok(),
        ResourceArc::new(BufResource {
            data: unary_op(&a.data, a.dtype, |x| 1.0 - erf_approx(x)),
            shape: a.shape.clone(),
            dtype: a.dtype,
        }),
    )
        .encode(env))
}

#[rustler::nif(schedule = "DirtyCpu")]
fn erf_inv<'a>(env: Env<'a>, a: ResourceArc<BufResource>) -> NifResult<Term<'a>> {
    Ok((
        atoms::ok(),
        ResourceArc::new(BufResource {
            data: unary_op(&a.data, a.dtype, |x| {
                let mut y = 0.0f64;
                for _ in 0..50 {
                    let e = erf_approx(y) - x;
                    let d = 2.0 / std::f64::consts::PI.sqrt() * (-y * y).exp();
                    if d.abs() < 1e-15 {
                        break;
                    }
                    y -= e / d;
                }
                y
            }),
            shape: a.shape.clone(),
            dtype: a.dtype,
        }),
    )
        .encode(env))
}

#[rustler::nif(schedule = "DirtyCpu")]
fn bitwise_not<'a>(env: Env<'a>, a: ResourceArc<BufResource>) -> NifResult<Term<'a>> {
    Ok((
        atoms::ok(),
        ResourceArc::new(BufResource {
            data: unary_op_int(&a.data, a.dtype, |x| !x, |x| (!(x as i64)) as f64),
            shape: a.shape.clone(),
            dtype: a.dtype,
        }),
    )
        .encode(env))
}

#[rustler::nif(schedule = "DirtyCpu")]
fn conjugate<'a>(env: Env<'a>, a: ResourceArc<BufResource>) -> NifResult<Term<'a>> {
    Ok((
        atoms::ok(),
        ResourceArc::new(BufResource {
            data: a.data.clone(),
            shape: a.shape.clone(),
            dtype: a.dtype,
        }),
    )
        .encode(env))
}

#[rustler::nif(schedule = "DirtyCpu")]
fn count_leading_zeros<'a>(env: Env<'a>, a: ResourceArc<BufResource>) -> NifResult<Term<'a>> {
    let data = match a.dtype {
        DType::S32 => {
            let va: &[i32] = bytemuck::cast_slice(&a.data);
            let v: Vec<i32> = va.iter().map(|x| x.leading_zeros() as i32).collect();
            bytemuck::cast_slice(&v).to_vec()
        }
        DType::S64 => {
            let va: &[i64] = bytemuck::cast_slice(&a.data);
            let v: Vec<i64> = va.iter().map(|x| x.leading_zeros() as i64).collect();
            bytemuck::cast_slice(&v).to_vec()
        }
        DType::U32 => {
            let va: &[u32] = bytemuck::cast_slice(&a.data);
            let v: Vec<u32> = va.iter().map(|x| x.leading_zeros() as u32).collect();
            bytemuck::cast_slice(&v).to_vec()
        }
        DType::U8 => {
            let v: Vec<u8> = a.data.iter().map(|x| x.leading_zeros() as u8).collect();
            v
        }
        _ => {
            let v = to_f64(&a.data, a.dtype);
            from_f64(
                v.iter()
                    .map(|x| (*x as u64).leading_zeros() as f64)
                    .collect(),
                a.dtype,
            )
        }
    };
    Ok((
        atoms::ok(),
        ResourceArc::new(BufResource {
            data,
            shape: a.shape.clone(),
            dtype: a.dtype,
        }),
    )
        .encode(env))
}

#[rustler::nif(schedule = "DirtyCpu")]
fn population_count<'a>(env: Env<'a>, a: ResourceArc<BufResource>) -> NifResult<Term<'a>> {
    let data = match a.dtype {
        DType::S32 => {
            let va: &[i32] = bytemuck::cast_slice(&a.data);
            let v: Vec<i32> = va.iter().map(|x| x.count_ones() as i32).collect();
            bytemuck::cast_slice(&v).to_vec()
        }
        DType::S64 => {
            let va: &[i64] = bytemuck::cast_slice(&a.data);
            let v: Vec<i64> = va.iter().map(|x| x.count_ones() as i64).collect();
            bytemuck::cast_slice(&v).to_vec()
        }
        DType::U32 => {
            let va: &[u32] = bytemuck::cast_slice(&a.data);
            let v: Vec<u32> = va.iter().map(|x| x.count_ones() as u32).collect();
            bytemuck::cast_slice(&v).to_vec()
        }
        DType::U8 => {
            let v: Vec<u8> = a.data.iter().map(|x| x.count_ones() as u8).collect();
            v
        }
        _ => {
            let v = to_f64(&a.data, a.dtype);
            from_f64(
                v.iter().map(|x| (*x as u64).count_ones() as f64).collect(),
                a.dtype,
            )
        }
    };
    Ok((
        atoms::ok(),
        ResourceArc::new(BufResource {
            data,
            shape: a.shape.clone(),
            dtype: a.dtype,
        }),
    )
        .encode(env))
}

#[rustler::nif(schedule = "DirtyCpu")]
fn real<'a>(env: Env<'a>, a: ResourceArc<BufResource>) -> NifResult<Term<'a>> {
    Ok((
        atoms::ok(),
        ResourceArc::new(BufResource {
            data: a.data.clone(),
            shape: a.shape.clone(),
            dtype: a.dtype,
        }),
    )
        .encode(env))
}

#[rustler::nif(schedule = "DirtyCpu")]
fn imag<'a>(env: Env<'a>, a: ResourceArc<BufResource>) -> NifResult<Term<'a>> {
    let n = a.num_elements();
    Ok((
        atoms::ok(),
        ResourceArc::new(BufResource {
            data: vec![0u8; n * a.dtype.size_in_bytes()],
            shape: a.shape.clone(),
            dtype: a.dtype,
        }),
    )
        .encode(env))
}

#[rustler::nif(schedule = "DirtyCpu")]
fn is_nan<'a>(env: Env<'a>, a: ResourceArc<BufResource>) -> NifResult<Term<'a>> {
    Ok((
        atoms::ok(),
        ResourceArc::new(BufResource {
            data: unary_op_bool(&a.data, a.dtype, |x| x.is_nan()),
            shape: a.shape.clone(),
            dtype: DType::U8,
        }),
    )
        .encode(env))
}

#[rustler::nif(schedule = "DirtyCpu")]
fn is_infinity<'a>(env: Env<'a>, a: ResourceArc<BufResource>) -> NifResult<Term<'a>> {
    Ok((
        atoms::ok(),
        ResourceArc::new(BufResource {
            data: unary_op_bool(&a.data, a.dtype, |x| x.is_infinite()),
            shape: a.shape.clone(),
            dtype: DType::U8,
        }),
    )
        .encode(env))
}

// ═══════════════════════════════════════════════════════════════
// Shape operations
// ═══════════════════════════════════════════════════════════════

#[rustler::nif(schedule = "DirtyCpu")]
fn reshape_tensor<'a>(
    env: Env<'a>,
    buf: ResourceArc<BufResource>,
    new_shape: Vec<usize>,
) -> NifResult<Term<'a>> {
    let old_n: usize = buf.shape.iter().product();
    let new_n: usize = new_shape.iter().product();
    if old_n != new_n {
        return Err(Error::RaiseTerm(Box::new(format!(
            "reshape: {} vs {} elements",
            old_n, new_n
        ))));
    }
    Ok((
        atoms::ok(),
        ResourceArc::new(BufResource {
            data: buf.data.clone(),
            shape: new_shape,
            dtype: buf.dtype,
        }),
    )
        .encode(env))
}

#[rustler::nif(schedule = "DirtyCpu")]
fn squeeze_tensor<'a>(
    env: Env<'a>,
    buf: ResourceArc<BufResource>,
    axes: Vec<i64>,
) -> NifResult<Term<'a>> {
    let rank = buf.shape.len() as i64;
    let mut ns = buf.shape.clone();
    let mut sorted = axes.clone();
    sorted.sort_by(|a, b| b.cmp(a));
    for ax in sorted {
        let idx = if ax < 0 { rank + ax } else { ax };
        if idx < 0 || idx >= rank {
            return Err(Error::RaiseTerm(Box::new("squeeze: axis out of range")));
        }
        let idx = idx as usize;
        if ns[idx] == 1 {
            ns.remove(idx);
        }
    }
    if ns.is_empty() {
        ns.push(1);
    }
    Ok((
        atoms::ok(),
        ResourceArc::new(BufResource {
            data: buf.data.clone(),
            shape: ns,
            dtype: buf.dtype,
        }),
    )
        .encode(env))
}

#[rustler::nif(schedule = "DirtyCpu")]
fn broadcast_tensor<'a>(
    env: Env<'a>,
    buf: ResourceArc<BufResource>,
    target_shape: Vec<usize>,
    axes: Vec<usize>,
) -> NifResult<Term<'a>> {
    let in_rank = buf.shape.len();
    let out_rank = target_shape.len();
    if axes.len() != in_rank {
        return Err(Error::RaiseTerm(Box::new(
            "broadcast: axes length mismatch",
        )));
    }
    let total_out: usize = target_shape.iter().product();
    let out_strides = strides_for(&target_shape);
    let in_strides = strides_for(&buf.shape);
    let in_f64 = to_f64(&buf.data, buf.dtype);
    let mut out_vals = vec![0.0f64; total_out];
    for i in 0..total_out {
        let mut coords = vec![0usize; out_rank];
        let mut rem = i;
        for d in 0..out_rank {
            coords[d] = rem / out_strides[d];
            rem %= out_strides[d];
        }
        let mut in_coords = vec![0usize; in_rank];
        for (di, &ax) in axes.iter().enumerate() {
            in_coords[di] = if ax < out_rank && buf.shape[di] == target_shape[ax] {
                coords[ax]
            } else {
                0
            };
        }
        let flat_in: usize = in_coords
            .iter()
            .zip(in_strides.iter())
            .map(|(c, s)| c * s)
            .sum();
        out_vals[i] = in_f64[flat_in];
    }
    Ok((
        atoms::ok(),
        ResourceArc::new(BufResource {
            data: from_f64(out_vals, buf.dtype),
            shape: target_shape,
            dtype: buf.dtype,
        }),
    )
        .encode(env))
}

#[rustler::nif(schedule = "DirtyCpu")]
fn transpose_tensor<'a>(
    env: Env<'a>,
    buf: ResourceArc<BufResource>,
    axes: Vec<usize>,
) -> NifResult<Term<'a>> {
    let rank = buf.shape.len();
    if axes.len() != rank {
        return Err(Error::RaiseTerm(Box::new(
            "transpose: axes length mismatch",
        )));
    }
    let new_shape: Vec<usize> = axes.iter().map(|&a| buf.shape[a]).collect();
    let total: usize = buf.shape.iter().product();
    let in_strides = strides_for(&buf.shape);
    let out_strides = strides_for(&new_shape);
    let in_f64 = to_f64(&buf.data, buf.dtype);
    let mut out_vals = vec![0.0f64; total];
    for i in 0..total {
        let mut in_coords = vec![0usize; rank];
        let mut rem = i;
        for d in 0..rank {
            in_coords[axes[d]] = rem / out_strides[d];
            rem %= out_strides[d];
        }
        let flat_in: usize = in_coords
            .iter()
            .zip(in_strides.iter())
            .map(|(c, s)| c * s)
            .sum();
        out_vals[i] = in_f64[flat_in];
    }
    Ok((
        atoms::ok(),
        ResourceArc::new(BufResource {
            data: from_f64(out_vals, buf.dtype),
            shape: new_shape,
            dtype: buf.dtype,
        }),
    )
        .encode(env))
}

#[rustler::nif(schedule = "DirtyCpu")]
fn pad_tensor<'a>(
    env: Env<'a>,
    buf: ResourceArc<BufResource>,
    pad_ref: ResourceArc<BufResource>,
    padding_config: Vec<(i64, i64, i64)>,
) -> NifResult<Term<'a>> {
    let rank = buf.shape.len();
    if padding_config.len() != rank {
        return Err(Error::RaiseTerm(Box::new("pad: config length mismatch")));
    }
    let pv = to_f64(&pad_ref.data, pad_ref.dtype);
    let pad_value = if pv.is_empty() { 0.0 } else { pv[0] };
    let new_shape: Vec<usize> = buf
        .shape
        .iter()
        .zip(padding_config.iter())
        .map(|(&s, &(lo, hi, interior))| {
            let lo = if lo < 0 { 0 } else { lo as usize };
            let hi = if hi < 0 { 0 } else { hi as usize };
            s + lo + hi + s.saturating_sub(1) * (if interior < 0 { 0 } else { interior as usize })
        })
        .collect();
    let total_out: usize = new_shape.iter().product();
    let out_strides = strides_for(&new_shape);
    let in_strides = strides_for(&buf.shape);
    let in_f64 = to_f64(&buf.data, buf.dtype);
    let mut out_vals = vec![pad_value; total_out];
    let total_in: usize = buf.shape.iter().product();
    for i in 0..total_in {
        let mut in_coords = vec![0usize; rank];
        let mut rem = i;
        for d in 0..rank {
            in_coords[d] = rem / in_strides[d];
            rem %= in_strides[d];
        }
        let mut out_coords = vec![0usize; rank];
        let mut acc = 0usize;
        for d in 0..rank {
            let lo = if padding_config[d].0 < 0 {
                0
            } else {
                padding_config[d].0 as usize
            };
            out_coords[d] = in_coords[d] + lo + acc;
            acc += in_coords[d]
                * (if padding_config[d].2 < 0 {
                    0
                } else {
                    padding_config[d].2 as usize
                });
        }
        let flat_out: usize = out_coords
            .iter()
            .zip(out_strides.iter())
            .map(|(c, s)| c * s)
            .sum();
        if flat_out < total_out {
            out_vals[flat_out] = in_f64[i];
        }
    }
    Ok((
        atoms::ok(),
        ResourceArc::new(BufResource {
            data: from_f64(out_vals, buf.dtype),
            shape: new_shape,
            dtype: buf.dtype,
        }),
    )
        .encode(env))
}

#[rustler::nif(schedule = "DirtyCpu")]
fn reverse_tensor<'a>(
    env: Env<'a>,
    buf: ResourceArc<BufResource>,
    axes: Vec<usize>,
) -> NifResult<Term<'a>> {
    let rank = buf.shape.len();
    let total: usize = buf.shape.iter().product();
    let in_strides = strides_for(&buf.shape);
    let in_f64 = to_f64(&buf.data, buf.dtype);
    let mut out_vals = vec![0.0f64; total];
    for i in 0..total {
        let mut coords = vec![0usize; rank];
        let mut rem = i;
        for d in 0..rank {
            coords[d] = rem / in_strides[d];
            rem %= in_strides[d];
        }
        let mut in_coords = coords.clone();
        for &ax in &axes {
            if ax < rank {
                in_coords[ax] = buf.shape[ax] - 1 - coords[ax];
            }
        }
        let flat_in: usize = in_coords
            .iter()
            .zip(in_strides.iter())
            .map(|(c, s)| c * s)
            .sum();
        out_vals[i] = in_f64[flat_in];
    }
    Ok((
        atoms::ok(),
        ResourceArc::new(BufResource {
            data: from_f64(out_vals, buf.dtype),
            shape: buf.shape.clone(),
            dtype: buf.dtype,
        }),
    )
        .encode(env))
}

#[rustler::nif(schedule = "DirtyCpu")]
fn slice_tensor<'a>(
    env: Env<'a>,
    buf: ResourceArc<BufResource>,
    starts: Vec<usize>,
    lengths: Vec<usize>,
    strides: Vec<usize>,
) -> NifResult<Term<'a>> {
    let rank = buf.shape.len();
    if starts.len() != rank || lengths.len() != rank || strides.len() != rank {
        return Err(Error::RaiseTerm(Box::new(
            "slice: parameter length mismatch",
        )));
    }
    let out_shape = lengths.clone();
    let total_out: usize = out_shape.iter().product();
    let out_strides = strides_for(&out_shape);
    let in_strides = strides_for(&buf.shape);
    let in_f64 = to_f64(&buf.data, buf.dtype);
    let mut out_vals = vec![0.0f64; total_out];
    for i in 0..total_out {
        let mut out_coords = vec![0usize; rank];
        let mut rem = i;
        for d in 0..rank {
            out_coords[d] = rem / out_strides[d];
            rem %= out_strides[d];
        }
        let in_coords: Vec<usize> = (0..rank)
            .map(|d| starts[d] + out_coords[d] * strides[d])
            .collect();
        let flat_in: usize = in_coords
            .iter()
            .zip(in_strides.iter())
            .map(|(c, s)| c * s)
            .sum();
        out_vals[i] = in_f64[flat_in];
    }
    Ok((
        atoms::ok(),
        ResourceArc::new(BufResource {
            data: from_f64(out_vals, buf.dtype),
            shape: out_shape,
            dtype: buf.dtype,
        }),
    )
        .encode(env))
}

#[rustler::nif(schedule = "DirtyCpu")]
fn concatenate_tensors<'a>(
    env: Env<'a>,
    bufs: Vec<ResourceArc<BufResource>>,
    axis: usize,
) -> NifResult<Term<'a>> {
    if bufs.is_empty() {
        return Err(Error::RaiseTerm(Box::new("concat: empty")));
    }
    let dtype = bufs[0].dtype;
    let rank = bufs[0].shape.len();
    for b in &bufs {
        if b.shape.len() != rank {
            return Err(Error::RaiseTerm(Box::new("concat: rank mismatch")));
        }
    }
    let out_axis: usize = bufs.iter().map(|b| b.shape[axis]).sum();
    let mut out_shape = bufs[0].shape.clone();
    out_shape[axis] = out_axis;
    let total_out: usize = out_shape.iter().product();
    let out_strides = strides_for(&out_shape);
    let mut out_vals = vec![0.0f64; total_out];
    let mut offset = 0usize;
    for buf in &bufs {
        let in_strides = strides_for(&buf.shape);
        let in_f64 = to_f64(&buf.data, buf.dtype);
        let total_in: usize = buf.shape.iter().product();
        for i in 0..total_in {
            let mut in_coords = vec![0usize; rank];
            let mut rem = i;
            for d in 0..rank {
                in_coords[d] = rem / in_strides[d];
                rem %= in_strides[d];
            }
            let mut out_coords = in_coords.clone();
            out_coords[axis] += offset;
            let flat_out: usize = out_coords
                .iter()
                .zip(out_strides.iter())
                .map(|(c, s)| c * s)
                .sum();
            out_vals[flat_out] = in_f64[i];
        }
        offset += buf.shape[axis];
    }
    Ok((
        atoms::ok(),
        ResourceArc::new(BufResource {
            data: from_f64(out_vals, dtype),
            shape: out_shape,
            dtype,
        }),
    )
        .encode(env))
}

#[rustler::nif(schedule = "DirtyCpu")]
fn stack_tensors<'a>(
    env: Env<'a>,
    bufs: Vec<ResourceArc<BufResource>>,
    axis: usize,
) -> NifResult<Term<'a>> {
    if bufs.is_empty() {
        return Err(Error::RaiseTerm(Box::new("stack: empty")));
    }
    let dtype = bufs[0].dtype;
    let rank = bufs[0].shape.len();
    for b in &bufs {
        if b.shape != bufs[0].shape {
            return Err(Error::RaiseTerm(Box::new("stack: shape mismatch")));
        }
    }
    let mut out_shape = bufs[0].shape.clone();
    out_shape.insert(axis, bufs.len());
    let out_rank = out_shape.len();
    let total_out: usize = out_shape.iter().product();
    let out_strides = strides_for(&out_shape);
    let mut out_vals = vec![0.0f64; total_out];
    for (si, buf) in bufs.iter().enumerate() {
        let in_strides = strides_for(&buf.shape);
        let in_f64 = to_f64(&buf.data, buf.dtype);
        let total_in: usize = buf.shape.iter().product();
        for i in 0..total_in {
            let mut in_coords = vec![0usize; rank];
            let mut rem = i;
            for d in 0..rank {
                in_coords[d] = rem / in_strides[d];
                rem %= in_strides[d];
            }
            let mut out_coords = vec![0usize; out_rank];
            let mut j = 0;
            for d in 0..out_rank {
                if d == axis {
                    out_coords[d] = si;
                } else {
                    out_coords[d] = in_coords[j];
                    j += 1;
                }
            }
            let flat_out: usize = out_coords
                .iter()
                .zip(out_strides.iter())
                .map(|(c, s)| c * s)
                .sum();
            out_vals[flat_out] = in_f64[i];
        }
    }
    Ok((
        atoms::ok(),
        ResourceArc::new(BufResource {
            data: from_f64(out_vals, dtype),
            shape: out_shape,
            dtype,
        }),
    )
        .encode(env))
}

fn broadcast_shape_simple(a: &[usize], b: &[usize], c: &[usize]) -> Vec<usize> {
    let max_len = a.len().max(b.len()).max(c.len());
    let mut result = vec![0usize; max_len];
    for i in 0..max_len {
        let da = if i < a.len() { a[a.len() - 1 - i] } else { 1 };
        let db = if i < b.len() { b[b.len() - 1 - i] } else { 1 };
        let dc = if i < c.len() { c[c.len() - 1 - i] } else { 1 };
        result[max_len - 1 - i] = da.max(db).max(dc);
    }
    result
}

#[rustler::nif(schedule = "DirtyCpu")]
fn select_tensor<'a>(
    env: Env<'a>,
    pred: ResourceArc<BufResource>,
    on_true: ResourceArc<BufResource>,
    on_false: ResourceArc<BufResource>,
) -> NifResult<Term<'a>> {
    let out_shape = broadcast_shape_simple(&pred.shape, &on_true.shape, &on_false.shape);
    let out_dtype = on_true.dtype;
    let total_out: usize = out_shape.iter().product();
    let out_strides = strides_for(&out_shape);
    let rank = out_shape.len();
    let pred_f = to_f64(&pred.data, pred.dtype);
    let true_f = to_f64(&on_true.data, on_true.dtype);
    let false_f = to_f64(&on_false.data, on_false.dtype);
    let pred_s = strides_for(&pred.shape);
    let true_s = strides_for(&on_true.shape);
    let false_s = strides_for(&on_false.shape);
    let mut out_vals = vec![0.0f64; total_out];
    for i in 0..total_out {
        let mut coords = vec![0usize; rank];
        let mut rem = i;
        for d in 0..rank {
            coords[d] = rem / out_strides[d];
            rem %= out_strides[d];
        }
        let pi = if pred.shape.len() == 1 && pred.shape[0] == 1 {
            0
        } else {
            coords
                .iter()
                .zip(pred_s.iter())
                .enumerate()
                .map(|(d, (c, s))| {
                    (if d < pred.shape.len() && pred.shape[d] == 1 {
                        0
                    } else {
                        *c
                    }) * s
                })
                .sum::<usize>()
        };
        let ti = if on_true.shape.len() == 1 && on_true.shape[0] == 1 {
            0
        } else {
            coords
                .iter()
                .zip(true_s.iter())
                .enumerate()
                .map(|(d, (c, s))| {
                    (if d < on_true.shape.len() && on_true.shape[d] == 1 {
                        0
                    } else {
                        *c
                    }) * s
                })
                .sum::<usize>()
        };
        let fi = if on_false.shape.len() == 1 && on_false.shape[0] == 1 {
            0
        } else {
            coords
                .iter()
                .zip(false_s.iter())
                .enumerate()
                .map(|(d, (c, s))| {
                    (if d < on_false.shape.len() && on_false.shape[d] == 1 {
                        0
                    } else {
                        *c
                    }) * s
                })
                .sum::<usize>()
        };
        out_vals[i] = if pred_f[pi] != 0.0 {
            true_f[ti]
        } else {
            false_f[fi]
        };
    }
    Ok((
        atoms::ok(),
        ResourceArc::new(BufResource {
            data: from_f64(out_vals, out_dtype),
            shape: out_shape,
            dtype: out_dtype,
        }),
    )
        .encode(env))
}

// ═══════════════════════════════════════════════════════════════
// Reductions
// ═══════════════════════════════════════════════════════════════

fn reduce_along_axes(
    data: &[u8],
    dtype: DType,
    shape: &[usize],
    axes: &[usize],
    keep_dims: bool,
    init: f64,
    op: impl Fn(f64, f64) -> f64,
) -> (Vec<u8>, Vec<usize>) {
    if shape.is_empty() {
        return (data.to_vec(), vec![]);
    }
    let v = to_f64(data, dtype);
    let total_in: usize = shape.iter().product();
    if total_in == 0 {
        return (from_f64(vec![init], dtype), vec![1]);
    }
    let in_strides = strides_for(shape);
    let mut out_shape = Vec::new();
    for (i, &s) in shape.iter().enumerate() {
        if !axes.contains(&i) {
            out_shape.push(s);
        } else if keep_dims {
            out_shape.push(1);
        }
    }
    if out_shape.is_empty() {
        out_shape = vec![1];
    }
    let out_total: usize = out_shape.iter().product();
    let out_strides = strides_for(&out_shape);
    let mut result = vec![init; out_total];
    for idx in 0..total_in {
        let mut remainder = idx;
        let mut out_idx = 0usize;
        for d in 0..shape.len() {
            let coord = remainder / in_strides[d];
            remainder %= in_strides[d];
            if !axes.contains(&d) {
                let out_d = if keep_dims {
                    d
                } else {
                    shape[..d]
                        .iter()
                        .enumerate()
                        .filter(|&(dd, _)| !axes.contains(&dd))
                        .count()
                };
                if out_d < out_strides.len() {
                    out_idx += coord * out_strides[out_d];
                }
            }
        }
        if out_idx < result.len() {
            result[out_idx] = op(result[out_idx], v[idx]);
        }
    }
    (from_f64(result, dtype), out_shape)
}

fn decode_reduction_opts(opts: Term) -> NifResult<(Vec<usize>, bool)> {
    let mut axes = Vec::new();
    let mut keep_dims = false;
    if let Ok(list) = opts.decode::<Vec<Term>>() {
        for item in list {
            if let Ok((ref key, val)) = item.decode::<(String, Term)>() {
                match key.as_str() {
                    "axes" => {
                        axes = val.decode::<Vec<usize>>().unwrap_or_default();
                    }
                    "keep_axes" | "keep_dims" => {
                        keep_dims = val.decode::<bool>().unwrap_or(false);
                    }
                    _ => {}
                }
            }
        }
    }
    Ok((axes, keep_dims))
}

fn decode_axis_opts(opts: Term) -> NifResult<(usize, bool)> {
    let mut axis = 0usize;
    let mut keep_dims = false;
    if let Ok(list) = opts.decode::<Vec<Term>>() {
        for item in list {
            if let Ok((ref key, val)) = item.decode::<(String, Term)>() {
                match key.as_str() {
                    "axis" => {
                        axis = val.decode::<usize>().unwrap_or(0);
                    }
                    "keep_axes" | "keep_dims" => {
                        keep_dims = val.decode::<bool>().unwrap_or(false);
                    }
                    _ => {}
                }
            }
        }
    }
    Ok((axis, keep_dims))
}

fn decode_window_opts(opts: Term) -> NifResult<(Vec<usize>, Vec<usize>, bool)> {
    let mut shape = Vec::new();
    let mut axes = Vec::new();
    let mut keep_dims = false;
    if let Ok(list) = opts.decode::<Vec<Term>>() {
        for item in list {
            if let Ok((ref key, val)) = item.decode::<(String, Term)>() {
                match key.as_str() {
                    "shape" => {
                        shape = val.decode::<Vec<usize>>().unwrap_or_default();
                    }
                    "axes" => {
                        axes = val.decode::<Vec<usize>>().unwrap_or_default();
                    }
                    "keep_axes" | "keep_dims" => {
                        keep_dims = val.decode::<bool>().unwrap_or(false);
                    }
                    _ => {}
                }
            }
        }
    }
    Ok((shape, axes, keep_dims))
}

#[rustler::nif(schedule = "DirtyCpu")]
fn sum_tensor<'a>(env: Env<'a>, buf: ResourceArc<BufResource>, opts: Term) -> NifResult<Term<'a>> {
    let (axes, kd) = decode_reduction_opts(opts)?;
    let (data, shape) =
        reduce_along_axes(&buf.data, buf.dtype, &buf.shape, &axes, kd, 0.0, |a, b| {
            a + b
        });
    Ok((
        atoms::ok(),
        ResourceArc::new(BufResource {
            data,
            shape,
            dtype: buf.dtype,
        }),
    )
        .encode(env))
}

#[rustler::nif(schedule = "DirtyCpu")]
fn product_tensor<'a>(
    env: Env<'a>,
    buf: ResourceArc<BufResource>,
    opts: Term,
) -> NifResult<Term<'a>> {
    let (axes, kd) = decode_reduction_opts(opts)?;
    let (data, shape) =
        reduce_along_axes(&buf.data, buf.dtype, &buf.shape, &axes, kd, 1.0, |a, b| {
            a * b
        });
    Ok((
        atoms::ok(),
        ResourceArc::new(BufResource {
            data,
            shape,
            dtype: buf.dtype,
        }),
    )
        .encode(env))
}

#[rustler::nif(schedule = "DirtyCpu")]
fn reduce_max<'a>(env: Env<'a>, buf: ResourceArc<BufResource>, opts: Term) -> NifResult<Term<'a>> {
    let (axes, kd) = decode_reduction_opts(opts)?;
    let (data, shape) = reduce_along_axes(
        &buf.data,
        buf.dtype,
        &buf.shape,
        &axes,
        kd,
        f64::MIN,
        |a, b| a.max(b),
    );
    Ok((
        atoms::ok(),
        ResourceArc::new(BufResource {
            data,
            shape,
            dtype: buf.dtype,
        }),
    )
        .encode(env))
}

#[rustler::nif(schedule = "DirtyCpu")]
fn reduce_min<'a>(env: Env<'a>, buf: ResourceArc<BufResource>, opts: Term) -> NifResult<Term<'a>> {
    let (axes, kd) = decode_reduction_opts(opts)?;
    let (data, shape) = reduce_along_axes(
        &buf.data,
        buf.dtype,
        &buf.shape,
        &axes,
        kd,
        f64::MAX,
        |a, b| a.min(b),
    );
    Ok((
        atoms::ok(),
        ResourceArc::new(BufResource {
            data,
            shape,
            dtype: buf.dtype,
        }),
    )
        .encode(env))
}

#[rustler::nif(schedule = "DirtyCpu")]
fn all_tensor<'a>(env: Env<'a>, buf: ResourceArc<BufResource>, opts: Term) -> NifResult<Term<'a>> {
    let (axes, kd) = decode_reduction_opts(opts)?;
    let (data, shape) =
        reduce_along_axes(&buf.data, buf.dtype, &buf.shape, &axes, kd, 1.0, |a, b| {
            if a != 0.0 && b != 0.0 {
                1.0
            } else {
                0.0
            }
        });
    Ok((
        atoms::ok(),
        ResourceArc::new(BufResource {
            data,
            shape,
            dtype: buf.dtype,
        }),
    )
        .encode(env))
}

#[rustler::nif(schedule = "DirtyCpu")]
fn any_tensor<'a>(env: Env<'a>, buf: ResourceArc<BufResource>, opts: Term) -> NifResult<Term<'a>> {
    let (axes, kd) = decode_reduction_opts(opts)?;
    let (data, shape) =
        reduce_along_axes(&buf.data, buf.dtype, &buf.shape, &axes, kd, 0.0, |a, b| {
            if a != 0.0 || b != 0.0 {
                1.0
            } else {
                0.0
            }
        });
    Ok((
        atoms::ok(),
        ResourceArc::new(BufResource {
            data,
            shape,
            dtype: buf.dtype,
        }),
    )
        .encode(env))
}

#[rustler::nif(schedule = "DirtyCpu")]
fn argmax_tensor<'a>(
    env: Env<'a>,
    buf: ResourceArc<BufResource>,
    opts: Term,
) -> NifResult<Term<'a>> {
    let (axis, kd) = decode_axis_opts(opts)?;
    let shape = &buf.shape;
    let rank = shape.len();
    if axis >= rank {
        return Err(Error::RaiseTerm(Box::new("argmax: axis out of range")));
    }
    let axis_size = shape[axis];
    let mut out_shape = Vec::new();
    for (i, &s) in shape.iter().enumerate() {
        if i != axis {
            out_shape.push(s);
        } else if kd {
            out_shape.push(1);
        }
    }
    if out_shape.is_empty() {
        out_shape = vec![1];
    }
    let total_out: usize = out_shape.iter().product();
    let out_strides = strides_for(&out_shape);
    let in_strides = strides_for(shape);
    let in_f64 = to_f64(&buf.data, buf.dtype);
    let mut out_data = vec![0u8; total_out * 4];
    let out_i32: &mut [i32] = bytemuck::cast_slice_mut(&mut out_data);
    for i in 0..total_out {
        let mut out_coords = vec![0usize; out_shape.len()];
        let mut rem = i;
        for d in 0..out_shape.len() {
            out_coords[d] = rem / out_strides[d];
            rem %= out_strides[d];
        }
        let mut in_coords = vec![0usize; rank];
        let mut oi = 0;
        for d in 0..rank {
            if d == axis {
                if kd {
                    oi += 1;
                }
            } else {
                in_coords[d] = out_coords[oi];
                oi += 1;
            }
        }
        let mut best_val = f64::MIN;
        let mut best_idx: i32 = 0;
        for j in 0..axis_size {
            in_coords[axis] = j;
            let flat_in: usize = in_coords
                .iter()
                .zip(in_strides.iter())
                .map(|(c, s)| c * s)
                .sum();
            let val = in_f64[flat_in];
            if val > best_val {
                best_val = val;
                best_idx = j as i32;
            }
        }
        out_i32[i] = best_idx;
    }
    Ok((
        atoms::ok(),
        ResourceArc::new(BufResource {
            data: out_data,
            shape: out_shape,
            dtype: DType::S32,
        }),
    )
        .encode(env))
}

#[rustler::nif(schedule = "DirtyCpu")]
fn argmin_tensor<'a>(
    env: Env<'a>,
    buf: ResourceArc<BufResource>,
    opts: Term,
) -> NifResult<Term<'a>> {
    let (axis, kd) = decode_axis_opts(opts)?;
    let shape = &buf.shape;
    let rank = shape.len();
    if axis >= rank {
        return Err(Error::RaiseTerm(Box::new("argmin: axis out of range")));
    }
    let axis_size = shape[axis];
    let mut out_shape = Vec::new();
    for (i, &s) in shape.iter().enumerate() {
        if i != axis {
            out_shape.push(s);
        } else if kd {
            out_shape.push(1);
        }
    }
    if out_shape.is_empty() {
        out_shape = vec![1];
    }
    let total_out: usize = out_shape.iter().product();
    let out_strides = strides_for(&out_shape);
    let in_strides = strides_for(shape);
    let in_f64 = to_f64(&buf.data, buf.dtype);
    let mut out_data = vec![0u8; total_out * 4];
    let out_i32: &mut [i32] = bytemuck::cast_slice_mut(&mut out_data);
    for i in 0..total_out {
        let mut out_coords = vec![0usize; out_shape.len()];
        let mut rem = i;
        for d in 0..out_shape.len() {
            out_coords[d] = rem / out_strides[d];
            rem %= out_strides[d];
        }
        let mut in_coords = vec![0usize; rank];
        let mut oi = 0;
        for d in 0..rank {
            if d == axis {
                if kd {
                    oi += 1;
                }
            } else {
                in_coords[d] = out_coords[oi];
                oi += 1;
            }
        }
        let mut best_val = f64::MAX;
        let mut best_idx: i32 = 0;
        for j in 0..axis_size {
            in_coords[axis] = j;
            let flat_in: usize = in_coords
                .iter()
                .zip(in_strides.iter())
                .map(|(c, s)| c * s)
                .sum();
            let val = in_f64[flat_in];
            if val < best_val {
                best_val = val;
                best_idx = j as i32;
            }
        }
        out_i32[i] = best_idx;
    }
    Ok((
        atoms::ok(),
        ResourceArc::new(BufResource {
            data: out_data,
            shape: out_shape,
            dtype: DType::S32,
        }),
    )
        .encode(env))
}

// ═══════════════════════════════════════════════════════════════
// Window operations
// ═══════════════════════════════════════════════════════════════

fn window_reduce(
    data: &[u8],
    dtype: DType,
    shape: &[usize],
    ws: &[usize],
    axes: &[usize],
    kd: bool,
    init: f64,
    op: impl Fn(f64, f64) -> f64,
) -> (Vec<u8>, Vec<usize>) {
    let rank = shape.len();
    let out_shape: Vec<usize> = (0..rank)
        .map(|d| {
            if axes.contains(&d) {
                shape[d].saturating_sub(ws[d] - 1)
            } else {
                shape[d]
            }
        })
        .collect();
    let out_shape = if kd {
        out_shape
    } else {
        out_shape
            .iter()
            .enumerate()
            .filter(|(i, _)| !axes.contains(i))
            .map(|(_, s)| *s)
            .collect()
    };
    let out_total: usize = out_shape.iter().product();
    let out_strides = strides_for(&out_shape);
    let in_strides = strides_for(shape);
    let in_f64 = to_f64(data, dtype);
    let mut out_vals = vec![init; out_total];
    for i in 0..out_total {
        let mut out_coords = vec![0usize; out_shape.len()];
        let mut rem = i;
        for d in 0..out_shape.len() {
            out_coords[d] = rem / out_strides[d];
            rem %= out_strides[d];
        }
        let mut in_coords = vec![0usize; rank];
        let mut oi = 0;
        for d in 0..rank {
            in_coords[d] = out_coords[oi];
            oi += 1;
        }
        let mut acc = init;
        let n_window: usize = axes.iter().map(|&a| ws[a]).product();
        for w_idx in 0..n_window {
            let mut wc = vec![0usize; axes.len()];
            let mut r = w_idx;
            for k in (0..axes.len()).rev() {
                wc[k] = r % ws[axes[k]];
                r /= ws[axes[k]];
            }
            let mut ic = in_coords.clone();
            for (k, &ax) in axes.iter().enumerate() {
                ic[ax] += wc[k];
            }
            let flat_in: usize = ic.iter().zip(in_strides.iter()).map(|(c, s)| c * s).sum();
            if flat_in < in_f64.len() {
                acc = op(acc, in_f64[flat_in]);
            } else {
                break;
            }
        }
        out_vals[i] = acc;
    }
    (from_f64(out_vals, dtype), out_shape)
}

#[rustler::nif(schedule = "DirtyCpu")]
fn window_sum<'a>(
    env: Env<'a>,
    buf: ResourceArc<BufResource>,
    shape: Vec<usize>,
    opts: Term,
) -> NifResult<Term<'a>> {
    let (ws, axes, kd) = decode_window_opts(opts)?;
    let (data, out_shape) = window_reduce(
        &buf.data,
        buf.dtype,
        &buf.shape,
        &ws,
        &axes,
        kd,
        0.0,
        |a, b| a + b,
    );
    Ok((
        atoms::ok(),
        ResourceArc::new(BufResource {
            data,
            shape: out_shape,
            dtype: buf.dtype,
        }),
    )
        .encode(env))
}

#[rustler::nif(schedule = "DirtyCpu")]
fn window_max<'a>(
    env: Env<'a>,
    buf: ResourceArc<BufResource>,
    shape: Vec<usize>,
    opts: Term,
) -> NifResult<Term<'a>> {
    let (ws, axes, kd) = decode_window_opts(opts)?;
    let (data, out_shape) = window_reduce(
        &buf.data,
        buf.dtype,
        &buf.shape,
        &ws,
        &axes,
        kd,
        f64::MIN,
        |a, b| a.max(b),
    );
    Ok((
        atoms::ok(),
        ResourceArc::new(BufResource {
            data,
            shape: out_shape,
            dtype: buf.dtype,
        }),
    )
        .encode(env))
}

#[rustler::nif(schedule = "DirtyCpu")]
fn window_min<'a>(
    env: Env<'a>,
    buf: ResourceArc<BufResource>,
    shape: Vec<usize>,
    opts: Term,
) -> NifResult<Term<'a>> {
    let (ws, axes, kd) = decode_window_opts(opts)?;
    let (data, out_shape) = window_reduce(
        &buf.data,
        buf.dtype,
        &buf.shape,
        &ws,
        &axes,
        kd,
        f64::MAX,
        |a, b| a.min(b),
    );
    Ok((
        atoms::ok(),
        ResourceArc::new(BufResource {
            data,
            shape: out_shape,
            dtype: buf.dtype,
        }),
    )
        .encode(env))
}

// ═══════════════════════════════════════════════════════════════
// LinAlg / Type conversion / Creation
// ═══════════════════════════════════════════════════════════════

#[rustler::nif(schedule = "DirtyCpu")]
fn dot_tensor<'a>(
    env: Env<'a>,
    a: ResourceArc<BufResource>,
    _c1: Vec<usize>,
    _b1: Vec<usize>,
    b: ResourceArc<BufResource>,
    _c2: Vec<usize>,
    _b2: Vec<usize>,
) -> NifResult<Term<'a>> {
    let a_s = &a.shape;
    let b_s = &b.shape;
    if a_s.len() < 2 || b_s.len() < 2 {
        return Err(Error::RaiseTerm(Box::new("dot: need 2D")));
    }
    let m = a_s[a_s.len() - 2];
    let k = a_s[a_s.len() - 1];
    let k2 = b_s[b_s.len() - 2];
    let n = b_s[b_s.len() - 1];
    if k != k2 {
        return Err(Error::RaiseTerm(Box::new(format!(
            "dot: inner mismatch {} vs {}",
            k, k2
        ))));
    }
    let total_out = m * n;
    let out_dtype = a.dtype;
    let a_f = to_f64(&a.data, a.dtype);
    let b_f = to_f64(&b.data, b.dtype);
    let mut out_vals = vec![0.0f64; total_out];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for l in 0..k {
                sum += a_f[i * k + l] * b_f[l * n + j];
            }
            out_vals[i * n + j] = sum;
        }
    }
    Ok((
        atoms::ok(),
        ResourceArc::new(BufResource {
            data: from_f64(out_vals, out_dtype),
            shape: vec![m, n],
            dtype: out_dtype,
        }),
    )
        .encode(env))
}

#[rustler::nif(schedule = "DirtyCpu")]
fn clip_tensor<'a>(
    env: Env<'a>,
    buf: ResourceArc<BufResource>,
    min_ref: ResourceArc<BufResource>,
    max_ref: ResourceArc<BufResource>,
) -> NifResult<Term<'a>> {
    let min_v = to_f64(&min_ref.data, min_ref.dtype);
    let max_v = to_f64(&max_ref.data, max_ref.dtype);
    let lo = if min_v.is_empty() { f64::MIN } else { min_v[0] };
    let hi = if max_v.is_empty() { f64::MAX } else { max_v[0] };
    Ok((
        atoms::ok(),
        ResourceArc::new(BufResource {
            data: unary_op(&buf.data, buf.dtype, |x| x.clamp(lo, hi)),
            shape: buf.shape.clone(),
            dtype: buf.dtype,
        }),
    )
        .encode(env))
}

#[rustler::nif(schedule = "DirtyCpu")]
fn as_type_tensor<'a>(
    env: Env<'a>,
    buf: ResourceArc<BufResource>,
    dtype_str: String,
) -> NifResult<Term<'a>> {
    let new_dtype = decode_dtype(&dtype_str)?;
    if buf.dtype == new_dtype {
        return Ok((
            atoms::ok(),
            ResourceArc::new(BufResource {
                data: buf.data.clone(),
                shape: buf.shape.clone(),
                dtype: buf.dtype,
            }),
        )
            .encode(env));
    }
    let vals = to_f64(&buf.data, buf.dtype);
    Ok((
        atoms::ok(),
        ResourceArc::new(BufResource {
            data: from_f64(vals, new_dtype),
            shape: buf.shape.clone(),
            dtype: new_dtype,
        }),
    )
        .encode(env))
}

#[rustler::nif(schedule = "DirtyCpu")]
fn constant_tensor<'a>(
    env: Env<'a>,
    shape: Vec<usize>,
    dtype_str: String,
    value: f64,
) -> NifResult<Term<'a>> {
    let dtype = decode_dtype(&dtype_str)?;
    let n: usize = shape.iter().product();
    Ok((
        atoms::ok(),
        ResourceArc::new(BufResource {
            data: from_f64(vec![value; n], dtype),
            shape,
            dtype,
        }),
    )
        .encode(env))
}

#[rustler::nif(schedule = "DirtyCpu")]
fn eye_tensor<'a>(env: Env<'a>, shape: Vec<usize>, dtype_str: String) -> NifResult<Term<'a>> {
    let dtype = decode_dtype(&dtype_str)?;
    if shape.len() < 2 {
        return Err(Error::RaiseTerm(Box::new("eye: need >= 2D")));
    }
    let rows = shape[shape.len() - 2];
    let cols = shape[shape.len() - 1];
    let batch: usize = shape[..shape.len() - 2].iter().product();
    let n = batch * rows * cols;
    let mut vals = vec![0.0f64; n];
    for b in 0..batch {
        for i in 0..rows {
            for j in 0..cols {
                let idx = b * rows * cols + i * cols + j;
                vals[idx] = if i == j { 1.0 } else { 0.0 };
            }
        }
    }
    Ok((
        atoms::ok(),
        ResourceArc::new(BufResource {
            data: from_f64(vals, dtype),
            shape,
            dtype,
        }),
    )
        .encode(env))
}

#[rustler::nif(schedule = "DirtyCpu")]
fn iota_tensor<'a>(
    env: Env<'a>,
    shape: Vec<usize>,
    dtype_str: String,
    axis: usize,
) -> NifResult<Term<'a>> {
    let dtype = decode_dtype(&dtype_str)?;
    let n: usize = shape.iter().product();
    let strides = strides_for(&shape);
    let mut vals = vec![0.0f64; n];
    for i in 0..n {
        let mut coords = vec![0usize; shape.len()];
        let mut rem = i;
        for d in 0..shape.len() {
            coords[d] = rem / strides[d];
            rem %= strides[d];
        }
        vals[i] = coords[axis] as f64;
    }
    Ok((
        atoms::ok(),
        ResourceArc::new(BufResource {
            data: from_f64(vals, dtype),
            shape,
            dtype,
        }),
    )
        .encode(env))
}

// ═══════════════════════════════════════════════════════════════
// Sorting
// ═══════════════════════════════════════════════════════════════

#[rustler::nif(schedule = "DirtyCpu")]
fn sort_tensor<'a>(env: Env<'a>, buf: ResourceArc<BufResource>, opts: Term) -> NifResult<Term<'a>> {
    let (axis, _) = decode_axis_opts(opts).map(|(a, _kd)| (a, false))?;
    let shape = &buf.shape;
    let rank = shape.len();
    if rank == 0 {
        return Ok((
            atoms::ok(),
            ResourceArc::new(BufResource {
                data: buf.data.clone(),
                shape: buf.shape.clone(),
                dtype: buf.dtype,
            }),
        )
            .encode(env));
    }
    let axis = if axis >= rank { 0 } else { axis };
    let axis_size = shape[axis];
    if axis_size <= 1 {
        return Ok((
            atoms::ok(),
            ResourceArc::new(BufResource {
                data: buf.data.clone(),
                shape: buf.shape.clone(),
                dtype: buf.dtype,
            }),
        )
            .encode(env));
    }
    let in_strides = strides_for(shape);
    let in_f64 = to_f64(&buf.data, buf.dtype);
    let total: usize = shape.iter().product();
    let mut out_vals = vec![0.0f64; total];
    let n_slices = total / axis_size;
    for s in 0..n_slices {
        let mut rem = s;
        let mut base_idx = 0usize;
        for d in 0..rank {
            let coord = rem / in_strides[d];
            rem %= in_strides[d];
            if d != axis {
                base_idx += coord * in_strides[d];
            }
        }
        let mut slice_vals: Vec<f64> = (0..axis_size)
            .map(|j| in_f64[base_idx + j * in_strides[axis]])
            .collect();
        slice_vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        for (j, &val) in slice_vals.iter().enumerate() {
            out_vals[base_idx + j * in_strides[axis]] = val;
        }
    }
    Ok((
        atoms::ok(),
        ResourceArc::new(BufResource {
            data: from_f64(out_vals, buf.dtype),
            shape: buf.shape.clone(),
            dtype: buf.dtype,
        }),
    )
        .encode(env))
}

#[rustler::nif(schedule = "DirtyCpu")]
fn argsort_tensor<'a>(
    env: Env<'a>,
    buf: ResourceArc<BufResource>,
    opts: Term,
) -> NifResult<Term<'a>> {
    let (axis, _) = decode_axis_opts(opts).map(|(a, _kd)| (a, false))?;
    let shape = &buf.shape;
    let rank = shape.len();
    if rank == 0 {
        return Ok((
            atoms::ok(),
            ResourceArc::new(BufResource {
                data: vec![0u8; 8],
                shape: buf.shape.clone(),
                dtype: DType::S64,
            }),
        )
            .encode(env));
    }
    let axis = if axis >= rank { 0 } else { axis };
    let axis_size = shape[axis];
    let in_strides = strides_for(shape);
    let in_f64 = to_f64(&buf.data, buf.dtype);
    let total: usize = shape.iter().product();
    let n_slices = total / axis_size;
    let mut out_vals = vec![0i64; total];
    for s in 0..n_slices {
        let mut rem = s;
        let mut base_idx = 0usize;
        for d in 0..rank {
            let coord = rem / in_strides[d];
            rem %= in_strides[d];
            if d != axis {
                base_idx += coord * in_strides[d];
            }
        }
        let mut indexed: Vec<(f64, usize)> = (0..axis_size)
            .map(|j| (in_f64[base_idx + j * in_strides[axis]], j))
            .collect();
        indexed.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        for (j, &(_, orig_idx)) in indexed.iter().enumerate() {
            out_vals[base_idx + j * in_strides[axis]] = orig_idx as i64;
        }
    }
    let out_data = bytemuck::cast_slice(&out_vals).to_vec();
    Ok((
        atoms::ok(),
        ResourceArc::new(BufResource {
            data: out_data,
            shape: buf.shape.clone(),
            dtype: DType::S64,
        }),
    )
        .encode(env))
}

// ═══════════════════════════════════════════════════════════════
// Additional operations (gather, put_slice, bitcast, conv, indexed)
// ═══════════════════════════════════════════════════════════════

#[rustler::nif(schedule = "DirtyCpu")]
fn gather<'a>(
    env: Env<'a>,
    input: ResourceArc<BufResource>,
    indices: ResourceArc<BufResource>,
    opts: Term,
) -> NifResult<Term<'a>> {
    let mut axis = 0usize;
    if let Ok(list) = opts.decode::<Vec<Term>>() {
        for item in list {
            if let Ok((ref key, val)) = item.decode::<(String, Term)>() {
                match key.as_str() {
                    "axes" => {
                        if let Ok(axes) = val.decode::<Vec<usize>>() {
                            axis = axes.first().copied().unwrap_or(0);
                        }
                    }
                    "axis" => {
                        axis = val.decode::<usize>().unwrap_or(0);
                    }
                    _ => {}
                }
            }
        }
    }
    let in_shape = &input.shape;
    let idx_shape = &indices.shape;
    let rank = in_shape.len();
    if axis >= rank {
        return Err(Error::RaiseTerm(Box::new("gather: axis out of range")));
    }
    let axis_size = in_shape[axis];
    let in_strides = strides_for(in_shape);
    let idx_data = to_f64(&indices.data, indices.dtype);
    let in_f64 = to_f64(&input.data, input.dtype);
    let idx_strides = strides_for(idx_shape);
    let num_indices: usize = idx_shape.iter().product();
    // Output shape: indices shape without the last dimension (which corresponds to axes)
    // For simple 1D indices, output shape = indices shape
    // For multi-dimensional indices with last dim = num_axes, output shape = indices shape without last dim
    let num_axes = 1; // We only support single axis for now
    let out_shape: Vec<usize> = if idx_shape.len() > num_axes {
        idx_shape[..idx_shape.len() - num_axes].to_vec()
    } else {
        vec![1]
    };
    let total_out: usize = out_shape.iter().product();
    let out_strides = strides_for(&out_shape);
    let mut out_vals = vec![0.0f64; total_out];
    for i in 0..total_out {
        // Compute output coordinates
        let mut out_coords = vec![0usize; out_shape.len()];
        let mut rem = i;
        for d in 0..out_shape.len() {
            out_coords[d] = rem / out_strides[d];
            rem %= out_strides[d];
        }
        // Compute the flat index into the indices tensor
        let idx_flat = if out_shape.len() == 1 && out_shape[0] == num_indices {
            i
        } else {
            let mut idx_coords = out_coords.clone();
            for _ in 0..num_axes {
                idx_coords.push(0);
            }
            idx_coords
                .iter()
                .zip(idx_strides.iter())
                .map(|(c, s)| c * s)
                .sum()
        };
        let idx_val = if idx_flat < num_indices {
            idx_data[idx_flat] as usize
        } else {
            0
        };
        let idx_val = idx_val.min(axis_size - 1);
        // Compute the flat index into the input tensor
        let flat_in = if rank == 1 {
            idx_val
        } else {
            let mut in_coords = vec![0usize; rank];
            let mut oi = 0;
            for d in 0..rank {
                if d == axis {
                    in_coords[d] = idx_val;
                } else {
                    in_coords[d] = out_coords.get(oi).copied().unwrap_or(0);
                    oi += 1;
                }
            }
            in_coords
                .iter()
                .zip(in_strides.iter())
                .map(|(c, s)| c * s)
                .sum()
        };
        out_vals[i] = in_f64[flat_in];
    }
    Ok((
        atoms::ok(),
        ResourceArc::new(BufResource {
            data: from_f64(out_vals, input.dtype),
            shape: out_shape,
            dtype: input.dtype,
        }),
    )
        .encode(env))
}

#[rustler::nif(schedule = "DirtyCpu")]
fn put_slice<'a>(
    env: Env<'a>,
    buf: ResourceArc<BufResource>,
    starts: Vec<usize>,
    slice: ResourceArc<BufResource>,
) -> NifResult<Term<'a>> {
    let rank = buf.shape.len();
    let in_strides = strides_for(&buf.shape);
    let slice_strides = strides_for(&slice.shape);
    let mut out_data = to_f64(&buf.data, buf.dtype);
    let slice_f64 = to_f64(&slice.data, slice.dtype);
    let total_slice: usize = slice.shape.iter().product();
    for i in 0..total_slice {
        let mut slice_coords = vec![0usize; rank];
        let mut rem = i;
        for d in 0..rank {
            slice_coords[d] = rem / slice_strides[d];
            rem %= slice_strides[d];
        }
        let mut out_idx = 0usize;
        for d in 0..rank {
            out_idx += (starts[d] + slice_coords[d]) * in_strides[d];
        }
        if out_idx < out_data.len() {
            out_data[out_idx] = slice_f64[i];
        }
    }
    Ok((
        atoms::ok(),
        ResourceArc::new(BufResource {
            data: from_f64(out_data, buf.dtype),
            shape: buf.shape.clone(),
            dtype: buf.dtype,
        }),
    )
        .encode(env))
}

#[rustler::nif(schedule = "DirtyCpu")]
fn bitcast_tensor<'a>(
    env: Env<'a>,
    buf: ResourceArc<BufResource>,
    dtype_str: String,
) -> NifResult<Term<'a>> {
    let new_dtype = decode_dtype(&dtype_str)?;
    let old_size = buf.dtype.size_in_bytes();
    let new_size = new_dtype.size_in_bytes();
    if old_size == new_size {
        return Ok((
            atoms::ok(),
            ResourceArc::new(BufResource {
                data: buf.data.clone(),
                shape: buf.shape.clone(),
                dtype: new_dtype,
            }),
        )
            .encode(env));
    }
    if old_size > new_size {
        // Truncate: keep only the bytes that fit
        let new_n = buf.data.len() / new_size;
        let mut new_data = vec![0u8; new_n * new_size];
        for i in 0..new_n {
            new_data[i * new_size..(i + 1) * new_size]
                .copy_from_slice(&buf.data[i * old_size..i * old_size + new_size]);
        }
        let mut new_shape = buf.shape.clone();
        let last = new_shape.len() - 1;
        new_shape[last] = new_shape[last] * old_size / new_size;
        return Ok((
            atoms::ok(),
            ResourceArc::new(BufResource {
                data: new_data,
                shape: new_shape,
                dtype: new_dtype,
            }),
        )
            .encode(env));
    }
    // old_size < new_size: expand with zeros
    let repeat = new_size / old_size;
    let n = buf.data.len() / old_size;
    let mut new_data = vec![0u8; n * new_size];
    for i in 0..n {
        new_data[i * new_size..i * new_size + old_size]
            .copy_from_slice(&buf.data[i * old_size..(i + 1) * old_size]);
    }
    let mut new_shape = buf.shape.clone();
    let last = new_shape.len() - 1;
    new_shape[last] = new_shape[last] / repeat;
    Ok((
        atoms::ok(),
        ResourceArc::new(BufResource {
            data: new_data,
            shape: new_shape,
            dtype: new_dtype,
        }),
    )
        .encode(env))
}

#[rustler::nif(schedule = "DirtyCpu")]
fn conv<'a>(
    env: Env<'a>,
    input: ResourceArc<BufResource>,
    kernel: ResourceArc<BufResource>,
    _opts: Term,
) -> NifResult<Term<'a>> {
    // Simple 2D convolution (no padding, stride=1)
    let in_shape = &input.shape;
    let k_shape = &kernel.shape;
    if in_shape.len() < 2 || k_shape.len() < 2 {
        return Err(Error::RaiseTerm(Box::new("conv: need >= 2D")));
    }
    let in_h = in_shape[in_shape.len() - 2];
    let in_w = in_shape[in_shape.len() - 1];
    let k_h = k_shape[k_shape.len() - 2];
    let k_w = k_shape[k_shape.len() - 1];
    let out_h = in_h.saturating_sub(k_h - 1);
    let out_w = in_w.saturating_sub(k_w - 1);
    if out_h == 0 || out_w == 0 {
        return Err(Error::RaiseTerm(Box::new("conv: kernel larger than input")));
    }
    let batch: usize = in_shape[..in_shape.len() - 2].iter().product();
    let in_f = to_f64(&input.data, input.dtype);
    let k_f = to_f64(&kernel.data, kernel.dtype);
    let mut out_shape = in_shape[..in_shape.len() - 2].to_vec();
    out_shape.push(out_h);
    out_shape.push(out_w);
    let total_out = batch * out_h * out_w;
    let mut out_vals = vec![0.0f64; total_out];
    for b in 0..batch {
        for oh in 0..out_h {
            for ow in 0..out_w {
                let mut sum = 0.0;
                for kh in 0..k_h {
                    for kw in 0..k_w {
                        let ih = oh + kh;
                        let iw = ow + kw;
                        let in_idx = b * in_h * in_w + ih * in_w + iw;
                        let k_idx = kh * k_w + kw;
                        sum += in_f[in_idx] * k_f[k_idx];
                    }
                }
                out_vals[b * out_h * out_w + oh * out_w + ow] = sum;
            }
        }
    }
    Ok((
        atoms::ok(),
        ResourceArc::new(BufResource {
            data: from_f64(out_vals, input.dtype),
            shape: out_shape,
            dtype: input.dtype,
        }),
    )
        .encode(env))
}

#[rustler::nif(schedule = "DirtyCpu")]
fn indexed_add<'a>(
    env: Env<'a>,
    t: ResourceArc<BufResource>,
    indices: ResourceArc<BufResource>,
    updates: ResourceArc<BufResource>,
    _opts: Term,
) -> NifResult<Term<'a>> {
    let mut out_data = to_f64(&t.data, t.dtype);
    let idx_f = to_f64(&indices.data, indices.dtype);
    let upd_f = to_f64(&updates.data, updates.dtype);
    let n = idx_f.len();
    let rank = t.shape.len();
    let in_strides = strides_for(&t.shape);
    for i in 0..n {
        let idx = idx_f[i] as usize;
        if idx < t.shape[0] {
            let offset = idx * in_strides[0];
            let upd_offset = i * in_strides[0];
            for j in 0..in_strides[0] {
                if offset + j < out_data.len() && upd_offset + j < upd_f.len() {
                    out_data[offset + j] += upd_f[upd_offset + j];
                }
            }
        }
    }
    Ok((
        atoms::ok(),
        ResourceArc::new(BufResource {
            data: from_f64(out_data, t.dtype),
            shape: t.shape.clone(),
            dtype: t.dtype,
        }),
    )
        .encode(env))
}

#[rustler::nif(schedule = "DirtyCpu")]
fn indexed_put<'a>(
    env: Env<'a>,
    t: ResourceArc<BufResource>,
    indices: ResourceArc<BufResource>,
    updates: ResourceArc<BufResource>,
    _opts: Term,
) -> NifResult<Term<'a>> {
    let mut out_data = to_f64(&t.data, t.dtype);
    let idx_f = to_f64(&indices.data, indices.dtype);
    let upd_f = to_f64(&updates.data, updates.dtype);
    let n = idx_f.len();
    let in_strides = strides_for(&t.shape);
    for i in 0..n {
        let idx = idx_f[i] as usize;
        if idx < t.shape[0] {
            let offset = idx * in_strides[0];
            let upd_offset = i * in_strides[0];
            for j in 0..in_strides[0] {
                if offset + j < out_data.len() && upd_offset + j < upd_f.len() {
                    out_data[offset + j] = upd_f[upd_offset + j];
                }
            }
        }
    }
    Ok((
        atoms::ok(),
        ResourceArc::new(BufResource {
            data: from_f64(out_data, t.dtype),
            shape: t.shape.clone(),
            dtype: t.dtype,
        }),
    )
        .encode(env))
}

// ═══════════════════════════════════════════════════════════════
// Stubs for complex ops (fallback to BinaryBackend on Elixir side)
// ═══════════════════════════════════════════════════════════════

#[rustler::nif(schedule = "DirtyCpu")]
fn triangular_solve<'a>(
    env: Env<'a>,
    _a: ResourceArc<BufResource>,
    _b: ResourceArc<BufResource>,
    _c: Term,
) -> NifResult<Term<'a>> {
    Err(Error::RaiseTerm(Box::new(
        "triangular_solve: not implemented, use BinaryBackend fallback",
    )))
}
#[rustler::nif(schedule = "DirtyCpu")]
fn fft_tensor<'a>(env: Env<'a>, _a: ResourceArc<BufResource>, _b: Term) -> NifResult<Term<'a>> {
    Err(Error::RaiseTerm(Box::new(
        "fft: not implemented, use BinaryBackend fallback",
    )))
}
#[rustler::nif(schedule = "DirtyCpu")]
fn ifft_tensor<'a>(env: Env<'a>, _a: ResourceArc<BufResource>, _b: Term) -> NifResult<Term<'a>> {
    Err(Error::RaiseTerm(Box::new(
        "ifft: not implemented, use BinaryBackend fallback",
    )))
}
#[rustler::nif(schedule = "DirtyCpu")]
fn window_scatter_max<'a>(
    env: Env<'a>,
    _a: ResourceArc<BufResource>,
    _b: ResourceArc<BufResource>,
    _c: ResourceArc<BufResource>,
    _d: Vec<usize>,
    _e: Term,
) -> NifResult<Term<'a>> {
    Err(Error::RaiseTerm(Box::new(
        "window_scatter_max: not implemented",
    )))
}
#[rustler::nif(schedule = "DirtyCpu")]
fn window_scatter_min<'a>(
    env: Env<'a>,
    _a: ResourceArc<BufResource>,
    _b: ResourceArc<BufResource>,
    _c: ResourceArc<BufResource>,
    _d: Vec<usize>,
    _e: Term,
) -> NifResult<Term<'a>> {
    Err(Error::RaiseTerm(Box::new(
        "window_scatter_min: not implemented",
    )))
}
#[rustler::nif(schedule = "DirtyCpu")]
fn reduce<'a>(
    env: Env<'a>,
    _a: ResourceArc<BufResource>,
    _b: ResourceArc<BufResource>,
    _c: Term,
    _d: Term,
) -> NifResult<Term<'a>> {
    Err(Error::RaiseTerm(Box::new(
        "reduce: not implemented, use BinaryBackend fallback",
    )))
}
#[rustler::nif(schedule = "DirtyCpu")]
fn window_reduce<'a>(
    env: Env<'a>,
    _a: ResourceArc<BufResource>,
    _b: ResourceArc<BufResource>,
    _c: Vec<usize>,
    _d: Term,
    _e: Term,
) -> NifResult<Term<'a>> {
    Err(Error::RaiseTerm(Box::new(
        "window_reduce: not implemented, use BinaryBackend fallback",
    )))
}
#[rustler::nif(schedule = "DirtyCpu")]
fn window_product<'a>(
    env: Env<'a>,
    _a: ResourceArc<BufResource>,
    _b: Vec<usize>,
    _c: Term,
) -> NifResult<Term<'a>> {
    Err(Error::RaiseTerm(Box::new(
        "window_product: not implemented, use BinaryBackend fallback",
    )))
}
