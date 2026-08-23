use kindle::{ActionMode, DreamerAgent, DreamerConfig, Environment, ModelSize};
use kindle_gym::grid_world::{ACTION_COUNT, GridWorld};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    let mut arguments = std::env::args().skip(1);
    let dino_checkpoint = arguments
        .next()
        .ok_or("usage: grid_world <DINOv3 model.safetensors> [steps]")?;
    let steps = arguments
        .next()
        .map_or(Ok(2_000), |value| value.parse::<usize>())?;
    let mut config = DreamerConfig::new(ACTION_COUNT);
    config.model_size = ModelSize::Size1M;
    let mut agent = DreamerAgent::new(config, dino_checkpoint, None)?;
    let mut environment = GridWorld::new();
    let first = environment.reset();
    agent.begin_episode(&first);

    for step in 0..steps {
        let mask = environment.action_mask();
        let action = agent.act(ActionMode::Sample, mask.as_deref());
        let transition = environment.step(action);
        let ended = transition.terminated || transition.truncated;
        agent.observe(&transition);
        for report in agent.learn_scheduled(1) {
            println!("{}", serde_json::to_string(&report)?);
        }
        if ended && step + 1 < steps {
            agent.begin_episode(&environment.reset());
        }
    }
    Ok(())
}
