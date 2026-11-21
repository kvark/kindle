use super::*;
use crate::dreamer::{
    ObservationKind,
    networks::{ObservationDecoder, Representation, concat_columns},
};
use meganeura::{data::safetensors::SafeTensorsModel, graph::Op};
use serde_json::{Value, json};
use std::{fs, path::PathBuf};

fn config() -> DreamerConfig {
    let mut c = DreamerConfig::tiny(18);
    c.observation_kind = ObservationKind::Rgb32;
    c.rgb_latent_only = true;
    c.actor_critic_gradient = false;
    c.loss_scales.reconstruction = 100.;
    c.loss_scales.future_prediction = 0.;
    c.loss_scales.policy = 0.;
    c.loss_scales.value = 0.;
    c.loss_scales.replay_value = 0.;
    c.disagreement_bonus = false;
    c
}

#[test]
fn observation_code_config_is_optional_persisted_and_plain_world_only() {
    let mut c = config();
    assert!(c.check().is_ok());
    assert!(validate_config(&c).is_ok());
    let encoded = serde_json::to_value(&c).unwrap();
    assert_eq!(encoded["rgb_latent_only"], true);
    assert_eq!(serde_json::from_value::<DreamerConfig>(encoded).unwrap(), c);
    c.rgb_latent_only = false;
    let legacy = serde_json::to_value(&c).unwrap();
    assert!(legacy.get("rgb_latent_only").is_none());
    assert_eq!(serde_json::from_value::<DreamerConfig>(legacy).unwrap(), c);
    for defect in 0..7 {
        let mut bad = config();
        match defect {
            0 => bad.observation_kind = ObservationKind::Features,
            1 => bad.rgb_frame_pair = true,
            2 => bad.loss_scales.policy = 1.,
            3 => bad.loss_scales.value = 1.,
            4 => bad.loss_scales.replay_value = 1.,
            5 => bad.actor_critic_gradient = true,
            6 => bad.video_encoder = Some(crate::dreamer::VideoEncoder::Frozen),
            _ => unreachable!(),
        }
        assert!(bad.check().unwrap_err().contains("latent-only RGB"));
    }
}

#[test]
fn observation_code_headers_require_v4_and_reject_disguised_architectures() {
    let mut metadata = json!({"format":"kindle-world-only-v4","config":{"rgb_latent_only":true},"pixel_loss":"motion_latent_overshooting"});
    assert_eq!(
        checkpoint_pixel_loss(&metadata).unwrap(),
        PixelLoss::MotionLatentOvershooting
    );
    for version in [
        "kindle-world-only-v1",
        "kindle-world-only-v2",
        "kindle-world-only-v3",
    ] {
        metadata["format"] = json!(version);
        assert!(checkpoint_pixel_loss(&metadata).is_err());
    }
    metadata["format"] = json!("kindle-world-only-v4");
    metadata["config"]["rgb_latent_only"] = json!(false);
    assert!(checkpoint_pixel_loss(&metadata).is_err());
    metadata["config"]["rgb_latent_only"] = json!(true);
    assert_eq!(
        TrainingScope::from_metadata(&metadata, PixelLoss::MotionLatentOvershooting, 16, 0)
            .unwrap(),
        TrainingScope::All
    );
    metadata["training_scope"] = json!("frozen_visual");
    assert!(
        TrainingScope::from_metadata(&metadata, PixelLoss::MotionLatentOvershooting, 16, 0)
            .is_err()
    );
}

fn visual_graph(c: &DreamerConfig, posterior: bool) -> Graph {
    let mut g = Graph::new();
    let b = c.batch_size;
    let s = c.network();
    let h = g.input("probe_h", &[b, s.deter]);
    let prediction = if posterior {
        let representation = Representation::new(&mut g, c);
        let image = g.input("image", &c.observation_shape(b));
        let encoded = representation.encoder.forward(&mut g, image, b);
        representation.posterior.forward(&mut g, h, encoded, b)
    } else {
        let z = g.input("z", &[b * s.stoch, s.classes]);
        let features = concat_columns(&mut g, h, z, b, s.deter, s.stoch * s.classes);
        ObservationDecoder::new(&mut g, c, "world.decoder", c.feature_dim())
            .forward(&mut g, features, b)
    };
    // Expose h itself so the native perturbation proves that the changed input ran.
    g.set_outputs(vec![prediction, h]);
    g
}

#[test]
fn observation_code_posterior_ignores_h_and_parameter_geometry_is_explicit() {
    let c = config();
    let g = visual_graph(&c, true);
    let mut seen = std::collections::HashSet::new();
    let mut pending = vec![g.outputs()[0]];
    while let Some(id) = pending.pop() {
        if seen.insert(id) {
            pending.extend(g.node(id).inputs.iter().copied());
        }
    }
    assert!(
        seen.iter()
            .all(|&id| !matches!(&g.node(id).op,Op::Input{name} if name=="probe_h"))
    );
    let mut large = DreamerConfig::new(18);
    large.observation_kind = ObservationKind::Rgb64;
    large.rgb_latent_only = true;
    large.loss_scales.policy = 0.;
    large.loss_scales.value = 0.;
    large.loss_scales.replay_value = 0.;
    large.actor_critic_gradient = false;
    large.loss_scales.reconstruction = 100.;
    large.loss_scales.future_prediction = 0.;
    large.disagreement_bonus = false;
    large.batch_size = 8;
    large.batch_length = 32;
    large.world_backprop_length = 32;
    let g = world::build_world_only_pixel_training_graph(
        &large,
        16,
        PixelLoss::MotionLatentOvershooting,
    );
    let params = g
        .nodes()
        .iter()
        .filter_map(|n| match &n.op {
            Op::Parameter { name } => Some((name, n.ty.shape.iter().product::<usize>())),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(params.len(), 66);
    assert_eq!(params.iter().map(|x| x.1).sum::<usize>(), 8_065_219);
    assert!(
        params
            .iter()
            .all(|(name, _)| !name.starts_with("world.decoder.sp0."))
    );
    let (plan, _) = meganeura::compile_training_graph(&g);
    for prefix in [
        "world.dynamics.",
        "world.representation.encoder.",
        "world.representation.posterior.",
        "world.decoder.",
        "world.reward.",
        "world.continuation.",
    ] {
        assert!(
            plan.param_buffers
                .iter()
                .any(|(name, buffer)| name.starts_with(prefix)
                    && plan.param_grad_pairs.iter().any(|(p, _)| p == buffer)),
            "{prefix}"
        );
    }
    println!("observation code: 66 parameters, 8065219 scalars, 198 optimizer tensors");
}

#[test]
#[ignore = "requires the declared full-shape observation-code qualification"]
fn observation_code_native_visual_separation() {
    let root = PathBuf::from(std::env::var("KINDLE_OBSERVATION_CODE_ROOT").unwrap());
    let value: Value =
        serde_json::from_slice(&fs::read(root.join("pixel-config.json")).unwrap()).unwrap();
    let config: DreamerConfig = serde_json::from_value(value).unwrap();
    config.validate();
    assert!(config.rgb_latent_only);
    let source = SafeTensorsModel::load(root.join("preflight/initial/world.safetensors")).unwrap();
    assert_eq!(source.tensor_info().len(), 198);
    let input = SafeTensorsModel::load(root.join("fixture.safetensors")).unwrap();
    let image = input.tensor_f32("image").unwrap();
    let s = config.network();
    let b = config.batch_size;
    let gpu = Arc::new(crate::init_gpu_context().unwrap());
    assert_eq!(
        gpu.device_information().device_name,
        std::env::var("KINDLE_EXPECT_DEVICE_NAME").unwrap()
    );
    let ha = vec![0.; b * s.deter];
    let hb = (0..b * s.deter)
        .map(|i| ((i % 19) as f32 - 9.) * 17.)
        .collect::<Vec<_>>();
    let mut invariant = 0;
    let mut output_sizes = Vec::new();
    for posterior in [true, false] {
        let g = visual_graph(&config, posterior);
        let mut session = runtime::build_session(&g, &gpu, Mode::Inference, false);
        let names = session
            .param_names()
            .iter()
            .map(|n| n.to_string())
            .collect::<Vec<_>>();
        for name in &names {
            assert!(!session.has_param_grad(name));
            session.set_parameter(name, &source.tensor_f32(name).unwrap());
        }
        if posterior {
            session.set_input("image", &image);
        } else {
            let z = (0..b * s.stoch * s.classes)
                .map(|i| f32::from(i % s.classes == (i / s.classes) % s.classes))
                .collect::<Vec<_>>();
            session.set_input("z", &z);
        }
        let width = g.node(g.outputs()[0]).ty.shape.iter().product::<usize>();
        let mut previous = Vec::new();
        for h in [&ha, &hb] {
            session.set_input("probe_h", h);
            session.step();
            session.wait();
            let mut actual = vec![0.; width];
            session.read_output_by_index(0, &mut actual);
            assert!(actual.iter().all(|x| x.is_finite()));
            let mut echoed = vec![0.; h.len()];
            session.read_output_by_index(1, &mut echoed);
            assert_eq!(&echoed, h);
            if previous.is_empty() {
                previous = actual;
            } else {
                assert_eq!(actual, previous);
            }
        }
        for name in &names {
            let expected = source.tensor_f32(name).unwrap();
            let mut actual = vec![0.; expected.len()];
            session.read_param(name, &mut actual);
            assert_eq!(actual, expected);
        }
        assert_eq!(session.adam_step_count(), 0);
        invariant += names.len();
        output_sizes.push(width);
    }
    assert_eq!(invariant, 34);
    fs::write(root.join("visual-separation.json"),serde_json::to_vec_pretty(&json!({"status":"complete","posterior_h_invariant":true,"decoder_h_invariant":true,"h_perturbation_echo_exact":true,"visual_parameters_invariant":34,"optimizer_steps":0,"output_sizes":output_sizes})).unwrap()).unwrap();
}
