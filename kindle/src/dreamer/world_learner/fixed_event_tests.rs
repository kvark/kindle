use super::*;
use crate::dreamer::ObservationKind;
use meganeura::graph::Op;
use serde_json::json;
use std::collections::HashSet;

fn config() -> DreamerConfig {
    let mut c = DreamerConfig::tiny(18);
    c.observation_kind = ObservationKind::Rgb32;
    c.batch_length = 32;
    c.world_backprop_length = 32;
    c.world_microbatch_size = Some(c.batch_size);
    c.loss_scales.policy = 0.;
    c.loss_scales.value = 0.;
    c.loss_scales.replay_value = 0.;
    c.disagreement_bonus = false;
    c
}

#[test]
fn fixed_events_bind_explicit_phase_header_and_scope_before_gpu() {
    let loss = PixelLoss::MotionFixedTeacherEvents;
    assert!(loss.motion_attention() && loss.latent_overshooting() && loss.recurrent_energy());
    assert!(loss.fixed_teacher() && loss.fixed_teacher_events());
    assert!(!loss.rollout_events() && !loss.clean_prior() && !loss.predictive());
    assert!(
        matches!(WorldModelLearner::with_pixel_loss(config(),16,loss), Err(e) if e.to_string().contains("explicit source phase"))
    );
    let m = json!({"format":"kindle-world-only-v16","config":config(),"pixel_loss":"motion_fixed_teacher_events",
        "prediction_length":16,"training_scope":"fixed_teacher","phase_start_updates":8192,
        "updates":12288,"teacher_sha256":"0".repeat(64)});
    assert_eq!(checkpoint_pixel_loss(&m).unwrap(), loss);
    assert_eq!(
        TrainingScope::from_metadata(&m, loss, 16, 12288).unwrap(),
        TrainingScope::FixedTeacher {
            start_updates: 8192
        }
    );
    for version in 1..16 {
        let mut bad = m.clone();
        bad["format"] = json!(format!("kindle-world-only-v{version}"));
        assert!(checkpoint_pixel_loss(&bad).is_err());
    }
    let mut bad = m.clone();
    bad["pixel_loss"] = json!("motion_fixed_teacher");
    assert!(checkpoint_pixel_loss(&bad).is_err());
    let mut bad = m.clone();
    bad["training_scope"] = json!("all");
    assert!(TrainingScope::from_metadata(&bad, loss, 16, 12288).is_err());
    let mut bad = m;
    bad["teacher_sha256"] = json!("bad");
    assert!(TrainingScope::from_metadata(&bad, loss, 16, 12288).is_err());
}

#[test]
fn fixed_event_composition_preserves_parameters_unique_inputs_and_gradient_scope() {
    let c = config();
    let old = world::build_world_only_pixel_training_graph(&c, 16, PixelLoss::MotionFixedTeacher);
    let mut g =
        world::build_world_only_pixel_training_graph(&c, 16, PixelLoss::MotionFixedTeacherEvents);
    let parameters = |g: &meganeura::Graph| {
        g.nodes()
            .iter()
            .filter_map(|n| match &n.op {
                Op::Parameter { name } => Some((name.clone(), n.ty.shape.clone())),
                _ => None,
            })
            .collect::<std::collections::BTreeMap<_, _>>()
    };
    assert_eq!(parameters(&old), parameters(&g));
    let inputs = g
        .nodes()
        .iter()
        .filter_map(|n| match &n.op {
            Op::Input { name } => Some(name.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(inputs.len(), inputs.iter().collect::<HashSet<_>>().len());
    for name in [
        "recurrent_first_uniforms_15",
        "recurrent_second_uniforms_15",
        "fixed_teacher_recurrent_weight_15",
    ] {
        assert!(inputs.contains(&name));
    }
    let [reward, continuation] = [
        g.outputs()[world::LOSS_IMAGE_FEEDBACK],
        g.outputs()[world::LOSS_IMAGE_FEEDBACK + 1],
    ];
    let total = g.add(reward, continuation);
    g.set_outputs(vec![total]);
    let b = meganeura::autodiff::differentiate(&g);
    let names = g.nodes().iter().filter_map(|n| match &n.op {
        Op::Parameter { name } => Some(name),
        _ => None,
    });
    let mut active = Vec::new();
    for (name, &id) in names.zip(&b.outputs()[1..]) {
        let zero = matches!(&b.node(id).op,Op::Constant{data} if data.iter().all(|&v|v==0.));
        if name.starts_with("world.representation.") || name.starts_with("world.decoder.") {
            assert!(zero, "unexpected visual gradient: {name}");
        } else if !zero {
            active.push(name.as_str());
        }
    }
    for prefix in [
        "world.dynamics.core.",
        "world.dynamics.prior",
        "world.reward.",
        "world.continuation.",
    ] {
        assert!(
            active.iter().any(|n| n.starts_with(prefix)),
            "missing {prefix}"
        );
    }
}
