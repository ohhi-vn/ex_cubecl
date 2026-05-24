//! Phase 2 — Media I/O and Transcode NIFs.
//!
//! All functions are stubs that return structured data matching the expected
//! Elixir-side types. When FFmpeg (ffmpeg-next crate) is integrated, these
//! will delegate to actual demux/decode/encode/mux operations.
//!
//! Resource types:
//!   `MediaSource` — opened media file/stream with stream metadata.
//!   `Transcoder`  — encoder context for writing output.

use crate::atoms;
use rustler::{Encoder, Env, Error, NifResult, ResourceArc, Term};

// ── MediaSource resource ─────────────────────────────────────

/// Represents an opened media source (file, RTMP, HLS, camera).
#[derive(Debug, Clone)]
pub struct MediaSource {
    pub path: String,
    pub streams: Vec<StreamInfo>,
}

/// Metadata for a single stream within a media source.
#[derive(Debug, Clone)]
pub struct StreamInfo {
    pub index: usize,
    pub stream_type: StreamType,
    pub codec: String,
    /// Video-only fields
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub fps: Option<f64>,
    /// Audio-only fields
    pub sample_rate: Option<u32>,
    pub channels: Option<u32>,
}

#[derive(Debug, Clone)]
pub enum StreamType {
    Video,
    Audio,
}

impl rustler::Resource for MediaSource {}

// ── Transcoder resource ───────────────────────────────────────

/// Represents an active transcoder (encoder + muxer context).
#[derive(Debug, Clone)]
pub struct Transcoder {
    pub path: String,
    pub video_opts: TranscodeVideoOpts,
    pub audio_opts: TranscodeAudioOpts,
    pub frame_count: u64,
    pub sample_count: u64,
}

/// Video encoding options passed from Elixir.
#[derive(Debug, Clone, Default)]
pub struct TranscodeVideoOpts {
    pub codec: String,
    pub bitrate: String,
    pub fps: u32,
    pub width: u32,
    pub height: u32,
}

/// Audio encoding options passed from Elixir.
#[derive(Debug, Clone, Default)]
pub struct TranscodeAudioOpts {
    pub codec: String,
    pub bitrate: String,
    pub sample_rate: u32,
}

impl rustler::Resource for Transcoder {}

// ── NIF: media_open ───────────────────────────────────────────

pub fn nif_media_open<'a>(env: Env<'a>, path: Term) -> NifResult<Term<'a>> {
    let path_str: String = path.decode()?;

    // Phase 2 stub: return a MediaSource with synthetic stream info.
    // TODO: Replace with ffmpeg-next::format::input(&path_str) and iterate streams.
    let source = MediaSource {
        path: path_str,
        streams: vec![
            StreamInfo {
                index: 0,
                stream_type: StreamType::Video,
                codec: "h264".to_string(),
                width: Some(1920),
                height: Some(1080),
                fps: Some(30.0),
                sample_rate: None,
                channels: None,
            },
            StreamInfo {
                index: 1,
                stream_type: StreamType::Audio,
                codec: "aac".to_string(),
                width: None,
                height: None,
                fps: None,
                sample_rate: Some(44100),
                channels: Some(2),
            },
        ],
    };

    let resource = ResourceArc::<MediaSource>::new(source);
    Ok((atoms::ok(), resource).encode(env))
}

// ── NIF: media_streams ───────────────────────────────────────

pub fn nif_media_streams(env: Env, source: ResourceArc<MediaSource>) -> NifResult<Term> {
    let mut stream_terms = Vec::new();

    for s in &source.streams {
        let map = rustler::types::map::map_new(env);
        let map = map
            .map_put(atoms::index().encode(env), s.index.encode(env))
            .map_err(|e| e)?;

        let type_atom = match s.stream_type {
            StreamType::Video => atoms::video(),
            StreamType::Audio => atoms::audio(),
        };
        let map = map
            .map_put(
                rustler::Atom::from_str(env, "type").unwrap().encode(env),
                type_atom.encode(env),
            )
            .map_err(|e| e)?;

        let map = map
            .map_put(atoms::codec().encode(env), s.codec.as_str().encode(env))
            .map_err(|e| e)?;

        let map = match s.stream_type {
            StreamType::Video => {
                let map = map
                    .map_put(atoms::width().encode(env), s.width.unwrap_or(0).encode(env))
                    .map_err(|e| e)?;
                let map = map
                    .map_put(
                        atoms::height().encode(env),
                        s.height.unwrap_or(0).encode(env),
                    )
                    .map_err(|e| e)?;
                map.map_put(atoms::fps().encode(env), s.fps.unwrap_or(0.0).encode(env))
                    .map_err(|e| e)?
            }
            StreamType::Audio => {
                let map = map
                    .map_put(
                        atoms::sample_rate().encode(env),
                        s.sample_rate.unwrap_or(0).encode(env),
                    )
                    .map_err(|e| e)?;
                map.map_put(
                    atoms::channels().encode(env),
                    s.channels.unwrap_or(0).encode(env),
                )
                .map_err(|e| e)?
            }
        };

        stream_terms.push(map.encode(env));
    }

    Ok((atoms::ok(), stream_terms).encode(env))
}

// ── NIF: media_read_video_frame ──────────────────────────────

pub fn nif_media_read_video_frame<'a>(
    env: Env<'a>,
    _source: ResourceArc<MediaSource>,
) -> NifResult<Term<'a>> {
    // Phase 2 stub: return a synthetic GPU buffer representing a video frame.
    // TODO: Replace with ffmpeg-next decode → GPU buffer upload.
    // Frame format: 1920×1080 YUV420p = 1920*1080*1.5 = 3,110,400 bytes
    let frame_size = 1920 * 1080 * 3 / 2;
    let data = vec![0u8; frame_size];

    use crate::Buffer;
    use crate::DType;
    let buffer = Buffer {
        data,
        shape: vec![frame_size],
        dtype: DType::U8,
    };

    let resource = ResourceArc::<Buffer>::new(buffer);

    // Return { :ok, { video_frame_resource, width, height, format, pts, duration } }
    let frame_map = rustler::types::map::map_new(env);
    let frame_map = frame_map
        .map_put(atoms::handle().encode(env), resource.encode(env))
        .map_err(|e| e)?;
    let frame_map = frame_map
        .map_put(atoms::width().encode(env), 1920u32.encode(env))
        .map_err(|e| e)?;
    let frame_map = frame_map
        .map_put(atoms::height().encode(env), 1080u32.encode(env))
        .map_err(|e| e)?;
    let frame_map = frame_map
        .map_put(
            atoms::format().encode(env),
            rustler::Atom::from_str(env, "yuv420p").unwrap().encode(env),
        )
        .map_err(|e| e)?;
    let frame_map = frame_map
        .map_put(atoms::pts().encode(env), 0u64.encode(env))
        .map_err(|e| e)?;
    let frame_map = frame_map
        .map_put(atoms::duration().encode(env), 33333u64.encode(env))
        .map_err(|e| e)?;

    Ok((atoms::ok(), frame_map.encode(env)).encode(env))
}

// ── NIF: media_read_audio_samples ────────────────────────────

pub fn nif_media_read_audio_samples<'a>(
    env: Env<'a>,
    _source: ResourceArc<MediaSource>,
) -> NifResult<Term<'a>> {
    // Phase 2 stub: return a synthetic GPU buffer representing audio samples.
    // TODO: Replace with ffmpeg-next decode → GPU buffer upload.
    // Format: 1024 frames × 2 channels × f32 = 1024*2*4 = 8192 bytes
    let sample_frames = 1024usize;
    let channels = 2usize;
    let data = vec![0u8; sample_frames * channels * 4];

    use crate::Buffer;
    use crate::DType;
    let buffer = Buffer {
        data,
        shape: vec![sample_frames * channels],
        dtype: DType::F32,
    };

    let resource = ResourceArc::<Buffer>::new(buffer);

    let samples_map = rustler::types::map::map_new(env);
    let samples_map = samples_map
        .map_put(atoms::handle().encode(env), resource.encode(env))
        .map_err(|e| e)?;
    let samples_map = samples_map
        .map_put(atoms::channels().encode(env), channels.encode(env))
        .map_err(|e| e)?;
    let samples_map = samples_map
        .map_put(atoms::sample_rate().encode(env), 48000u32.encode(env))
        .map_err(|e| e)?;
    let samples_map = samples_map
        .map_put(atoms::frames().encode(env), sample_frames.encode(env))
        .map_err(|e| e)?;
    let samples_map = samples_map
        .map_put(atoms::pts().encode(env), 0u64.encode(env))
        .map_err(|e| e)?;

    Ok((atoms::ok(), samples_map.encode(env)).encode(env))
}

// ── NIF: media_close ─────────────────────────────────────────

pub fn nif_media_close(env: Env, _source: ResourceArc<MediaSource>) -> NifResult<Term> {
    // Phase 2 stub: Resource is freed automatically by Rustler when the
    // Elixir-side reference is GC'd. No explicit close needed.
    // TODO: Close FFmpeg context when integrated.
    Ok((atoms::ok()).encode(env))
}

// ── NIF: transcode_start ─────────────────────────────────────

pub fn nif_transcode_start<'a>(env: Env<'a>, path: Term, opts: Term) -> NifResult<Term<'a>> {
    let path_str: String = path.decode()?;
    let opts_map: std::collections::HashMap<String, Term> = opts.decode()?;

    let video_opts = parse_video_opts(env, &opts_map)?;
    let audio_opts = parse_audio_opts(env, &opts_map)?;

    let transcoder = Transcoder {
        path: path_str,
        video_opts,
        audio_opts,
        frame_count: 0,
        sample_count: 0,
    };

    let resource = ResourceArc::<Transcoder>::new(transcoder);
    Ok((atoms::ok(), resource).encode(env))
}

// ── NIF: transcode_write_video ───────────────────────────────

pub fn nif_transcode_write_video(
    env: Env,
    _encoder: ResourceArc<Transcoder>,
    _frame: ResourceArc<crate::Buffer>,
) -> NifResult<Term> {
    // Phase 2 stub: no-op. TODO: GPU buffer download → FFmpeg encode → mux.
    Ok((atoms::ok()).encode(env))
}

// ── NIF: transcode_write_audio ───────────────────────────────

pub fn nif_transcode_write_audio(
    env: Env,
    _encoder: ResourceArc<Transcoder>,
    _samples: ResourceArc<crate::Buffer>,
) -> NifResult<Term> {
    // Phase 2 stub: no-op. TODO: GPU buffer download → FFmpeg encode → mux.
    Ok((atoms::ok()).encode(env))
}

// ── NIF: transcode_finish ────────────────────────────────────

pub fn nif_transcode_finish(env: Env, _encoder: ResourceArc<Transcoder>) -> NifResult<Term> {
    // Phase 2 stub: no-op. TODO: Flush encoder, write trailer, close muxer.
    Ok((atoms::ok()).encode(env))
}

// ── Helpers ──────────────────────────────────────────────────

fn parse_video_opts(
    _env: Env,
    opts: &std::collections::HashMap<String, Term>,
) -> NifResult<TranscodeVideoOpts> {
    let mut v = TranscodeVideoOpts::default();

    if let Some(term) = opts.get("codec") {
        let s: String = term
            .decode()
            .map_err(|_| Error::RaiseTerm(Box::new("transcode_start: invalid video codec")))?;
        v.codec = s;
    }
    if let Some(term) = opts.get("bitrate") {
        let s: String = term
            .decode()
            .map_err(|_| Error::RaiseTerm(Box::new("transcode_start: invalid video bitrate")))?;
        v.bitrate = s;
    }
    if let Some(term) = opts.get("fps") {
        let fps: u32 = term
            .decode()
            .map_err(|_| Error::RaiseTerm(Box::new("transcode_start: invalid video fps")))?;
        v.fps = fps;
    }
    if let Some(term) = opts.get("width") {
        let w: u32 = term
            .decode()
            .map_err(|_| Error::RaiseTerm(Box::new("transcode_start: invalid video width")))?;
        v.width = w;
    }
    if let Some(term) = opts.get("height") {
        let h: u32 = term
            .decode()
            .map_err(|_| Error::RaiseTerm(Box::new("transcode_start: invalid video height")))?;
        v.height = h;
    }

    Ok(v)
}

fn parse_audio_opts(
    _env: Env,
    opts: &std::collections::HashMap<String, Term>,
) -> NifResult<TranscodeAudioOpts> {
    let mut a = TranscodeAudioOpts::default();

    if let Some(term) = opts.get("codec") {
        let s: String = term
            .decode()
            .map_err(|_| Error::RaiseTerm(Box::new("transcode_start: invalid audio codec")))?;
        a.codec = s;
    }
    if let Some(term) = opts.get("bitrate") {
        let s: String = term
            .decode()
            .map_err(|_| Error::RaiseTerm(Box::new("transcode_start: invalid audio bitrate")))?;
        a.bitrate = s;
    }
    if let Some(term) = opts.get("sample_rate") {
        let sr: u32 = term.decode().map_err(|_| {
            Error::RaiseTerm(Box::new("transcode_start: invalid audio sample_rate"))
        })?;
        a.sample_rate = sr;
    }

    Ok(a)
}
