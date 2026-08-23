// Adapted from kvark/dinovision at dc35cdf1c7c910cdd93c5b5362846842ae469a21 (MIT).
//! Binding a HuggingFace DINOv3 checkpoint to the encoder graph's
//! parameters.
//!
//! Three shape conventions differ between the checkpoint and the graph,
//! and all three are handled here rather than in the graph:
//!
//! * `nn.Linear` stores `[out_features, in_features]`; meganeura's
//!   `matmul` wants `[in, out]`. Every projection is transposed.
//! * The patch embedding is a 4D `[out, 3, k, k]` conv weight; it is
//!   flattened to `[out, patch_dim]` and transposed.
//! * `cls_token` and `register_tokens` are separate parameters; the graph
//!   declares one `prefix_tokens` matrix, assembled here.

use meganeura::Session;
use meganeura::data::safetensors::SafeTensorsModel;

use crate::vision::dinov3::Config;
use crate::vision::preprocess::conv_weight_to_matmul;

type Error = Box<dyn std::error::Error>;

/// Checkpoints appear both with and without a `model.` prefix on the
/// encoder layers, depending on whether they were saved from
/// `DINOv3ViTModel` or from the bare backbone. Resolve whichever this one
/// uses once, up front, instead of guessing per tensor.
fn layer_prefix(model: &SafeTensorsModel) -> Result<&'static str, Error> {
    let info = model.tensor_info();
    for candidate in ["layer.", "model.layer."] {
        if info.keys().any(|k| k.starts_with(candidate)) {
            return Ok(candidate);
        }
    }
    Err(format!(
        "no encoder layers found; checkpoint has {} tensors, e.g. {:?}",
        info.len(),
        info.keys().take(5).collect::<Vec<_>>()
    )
    .into())
}

/// Read a tensor under an optional `model.` prefix.
fn embedding_tensor(model: &SafeTensorsModel, name: &str) -> Result<Vec<f32>, Error> {
    if model.tensor_info().contains_key(name) {
        model.tensor_f32_auto(name)
    } else {
        model.tensor_f32_auto(&format!("model.{name}"))
    }
}

/// Upload every parameter the encoder graph declares.
///
/// Fails loudly on a missing or mis-shaped tensor: a silently skipped
/// weight would leave a zero buffer and produce plausible-looking
/// garbage.
pub fn load_encoder(
    session: &mut Session,
    model: &SafeTensorsModel,
    config: &Config,
) -> Result<(), Error> {
    let hidden = config.hidden_size;
    let prefix = layer_prefix(model)?;
    log::info!("checkpoint layer prefix: {prefix:?}");

    // --- Patch embedding: [out, 3, k, k] -> [patch_dim, out] ---
    let conv = embedding_tensor(model, "embeddings.patch_embeddings.weight")?;
    let expected = hidden * config.patch_dim();
    if conv.len() != expected {
        return Err(format!(
            "patch embedding has {} values, expected {expected} \
             (is the checkpoint's patch_size or hidden_size different?)",
            conv.len()
        )
        .into());
    }
    session.set_parameter(
        "embeddings.patch_embeddings.weight",
        &conv_weight_to_matmul(&conv, hidden, config.patch_dim()),
    );
    session.set_parameter(
        "embeddings.patch_embeddings.bias",
        &embedding_tensor(model, "embeddings.patch_embeddings.bias")?,
    );

    // --- Prefix tokens: CLS first, then the registers ---
    //
    // Order matters and is not arbitrary: the reference builds the
    // sequence as cat([cls, registers, patches]), and RoPE identifies
    // prefix tokens purely by their position at the front.
    let cls = embedding_tensor(model, "embeddings.cls_token")?;
    let registers = embedding_tensor(model, "embeddings.register_tokens")?;
    if cls.len() != hidden {
        return Err(format!("cls_token has {} values, expected {hidden}", cls.len()).into());
    }
    if registers.len() != config.num_register_tokens * hidden {
        return Err(format!(
            "register_tokens has {} values, expected {} * {hidden}",
            registers.len(),
            config.num_register_tokens
        )
        .into());
    }
    let mut prefix_tokens = Vec::with_capacity(config.num_prefix_tokens() * hidden);
    prefix_tokens.extend_from_slice(&cls);
    prefix_tokens.extend_from_slice(&registers);
    session.set_parameter("prefix_tokens", &prefix_tokens);

    // --- Encoder layers ---
    for i in 0..config.num_hidden_layers {
        let src = format!("{prefix}{i}");
        let dst = format!("layer.{i}");

        for norm in ["norm1", "norm2"] {
            for part in ["weight", "bias"] {
                session.set_parameter(
                    &format!("{dst}.{norm}.{part}"),
                    &model.tensor_f32_auto(&format!("{src}.{norm}.{part}"))?,
                );
            }
        }

        // K has no bias — `key_bias: false` in the config.
        for (proj, has_bias) in [
            ("q_proj", true),
            ("k_proj", false),
            ("v_proj", true),
            ("o_proj", true),
        ] {
            session.set_parameter(
                &format!("{dst}.attention.{proj}.weight"),
                &model.tensor_f32_auto_transposed(&format!("{src}.attention.{proj}.weight"))?,
            );
            if has_bias {
                session.set_parameter(
                    &format!("{dst}.attention.{proj}.bias"),
                    &model.tensor_f32_auto(&format!("{src}.attention.{proj}.bias"))?,
                );
            }
        }

        for proj in ["up_proj", "down_proj"] {
            session.set_parameter(
                &format!("{dst}.mlp.{proj}.weight"),
                &model.tensor_f32_auto_transposed(&format!("{src}.mlp.{proj}.weight"))?,
            );
            session.set_parameter(
                &format!("{dst}.mlp.{proj}.bias"),
                &model.tensor_f32_auto(&format!("{src}.mlp.{proj}.bias"))?,
            );
        }

        for ls in ["layer_scale1", "layer_scale2"] {
            session.set_parameter(
                &format!("{dst}.{ls}.lambda1"),
                &model.tensor_f32_auto(&format!("{src}.{ls}.lambda1"))?,
            );
        }
    }

    // --- Final norm ---
    for part in ["weight", "bias"] {
        session.set_parameter(
            &format!("norm.{part}"),
            &embedding_tensor(model, &format!("norm.{part}"))?,
        );
    }

    log::info!(
        "loaded DINOv3 weights: {} layers, hidden {hidden}",
        config.num_hidden_layers
    );
    Ok(())
}
