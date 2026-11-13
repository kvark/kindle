use super::*;
use crate::dreamer::{ObservationKind, runtime};
use meganeura::{Mode, graph::Op};
use std::{collections::HashSet, sync::Arc};

fn ancestors(g: &Graph, id: NodeId) -> HashSet<NodeId> {
    let mut seen = HashSet::new();
    let mut todo = vec![id];
    while let Some(n) = todo.pop() {
        if seen.insert(n) {
            todo.extend(g.node(n).inputs.iter().copied());
        }
    }
    seen
}

fn fixture() -> (DreamerConfig, Graph, Losses) {
    let mut c = DreamerConfig::tiny(3);
    c.batch_size = 2;
    c.value_bins = 5;
    c.observation_kind = ObservationKind::Rgb32;
    c.loss_scales.policy = 0.;
    c.loss_scales.value = 0.;
    c.loss_scales.replay_value = 0.;
    let mut g = Graph::new();
    let m = WorldModel::with_behavior(&mut g, &c, false);
    let n = c.network();
    let h = g.input("seed_h", &[2, n.deter]);
    let z = g.input("seed_z", &[2 * n.stoch, n.classes]);
    let actions = (0..3)
        .map(|t| g.input(&format!("action{t}"), &[2, 3]))
        .collect::<Vec<_>>();
    let targets = (0..3)
        .map(|t| g.input(&format!("true_h{t}"), &[2, n.deter]))
        .collect::<Vec<_>>();
    let reward = (0..3)
        .map(|t| g.input(&format!("true_reward{t}"), &[2, 5]))
        .collect::<Vec<_>>();
    let continuation = (0..3)
        .map(|t| g.input(&format!("true_continuation{t}"), &[2, 1]))
        .collect::<Vec<_>>();
    let losses = build_with_events(
        &mut g,
        &c,
        &m,
        (h, z),
        &actions,
        &targets,
        "event_keep",
        Some(super::super::pixel_rollout::EventTargets {
            reward: &reward,
            continuation: &continuation,
        }),
    );
    (c, g, losses)
}

#[test]
fn recurrent_event_states_never_depend_on_targets_or_masks_and_sample_the_final_prior() {
    let (_, mut g, l) = fixture();
    assert_eq!(l.states.len(), 6);
    for &state in &l.states {
        assert!(ancestors(&g,state).iter().all(|&n|!matches!(&g.node(n).op,Op::Input{name} if name.starts_with("true_")||name.starts_with("event_keep"))));
    }
    let [r, c] = l.events.unwrap();
    let sum = g.add(r, c);
    let seen = ancestors(&g, sum);
    for stream in ["recurrent_first", "recurrent_second"] {
        assert!(seen.iter().any(
            |&id| matches!(&g.node(id).op,Op::Input{name} if name==&format!("{stream}_uniforms_2"))
        ));
    }
    assert!(l.states.iter().all(|id| seen.contains(id)));
}

#[test]
#[ignore = "requires a separately declared fixed-event native composition qualifier"]
fn recurrent_event_targets_masks_and_lanes_have_causal_native_behavior() {
    let (c, mut g, l) = fixture();
    let events = l.events.unwrap();
    let mut ids = vec![l.energy, events[0], events[1]];
    ids.extend(l.states);
    let widths = ids
        .iter()
        .map(|&n| g.node(n).ty.num_elements())
        .collect::<Vec<_>>();
    g.set_outputs(ids);
    let gpu = Arc::new(crate::init_gpu_context().unwrap());
    let mut s = runtime::build_session(&g, &gpu, Mode::Inference, false);
    runtime::initialize_d3(&mut s, &g, 47);
    s.set_parameter("world.reward.out.bias", &[0.2, -0.1, 0.7, -0.4, 0.3]);
    s.set_parameter("world.continuation.out.bias", &[0.3]);
    let n = c.network();
    let h = (0..2 * n.deter)
        .map(|i| (i % 17) as f32 * 0.01 - 0.08)
        .collect::<Vec<_>>();
    let mut z = vec![0.; 2 * n.stoch * n.classes];
    for i in 0..2 * n.stoch {
        z[i * n.classes + i % n.classes] = 1.;
    }
    s.set_input("seed_z", &z);
    let mut run = |kind| {
        let mut seed = h.clone();
        if kind == 4 {
            for x in &mut seed[n.deter..] {
                *x += 0.5;
            }
        }
        s.set_input("seed_h", &seed);
        for t in 0..3 {
            s.set_input(&format!("action{t}"), &[1., 0., 0., 0., 1., 0.]);
            for stream in ["recurrent_first", "recurrent_second"] {
                s.set_input(
                    &format!("{stream}_uniforms_{t}"),
                    &vec![0.37; 2 * n.stoch * n.classes],
                );
            }
            s.set_input(&format!("true_h{t}"), &vec![0.1; 2 * n.deter]);
            let mut y = vec![0.; 10];
            for lane in 0..2 {
                y[lane * 5 + usize::from(kind == 1)] = 1.;
            }
            s.set_input(&format!("true_reward{t}"), &y);
            s.set_input(
                &format!("true_continuation{t}"),
                &[if kind == 2 {
                    0.
                } else {
                    c.continuation_discount()
                }; 2],
            );
            s.set_input(
                &format!("event_keep_{t}"),
                &[if kind == 3 { 0. } else { 1. }; 2],
            );
        }
        s.step();
        s.wait();
        widths
            .iter()
            .enumerate()
            .map(|(i, &w)| {
                let mut v = vec![0.; w];
                s.read_output_by_index(i, &mut v);
                v
            })
            .collect::<Vec<_>>()
    };
    let a = run(0);
    assert!(a.iter().flatten().all(|x| x.is_finite()));
    assert!(a[1][0] > 0. && a[2][0] > 0.);
    for kind in 1..4 {
        let b = run(kind);
        assert_eq!(a[3..], b[3..]);
        if kind == 1 {
            assert_ne!(a[1], b[1]);
        }
        if kind == 2 {
            assert_ne!(a[2], b[2]);
        }
        if kind == 3 {
            assert_eq!([b[0][0], b[1][0], b[2][0]], [0.; 3]);
        }
    }
    let b = run(4);
    for (x, y) in a[3..].iter().zip(&b[3..]) {
        assert_eq!(x[..n.deter], y[..n.deter]);
    }
}
