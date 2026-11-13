//! Bounded native capture/acting smoke test using mind-games' Dullahan layer.
//! No reward adapter: this proves transport/acting, not Quake competence.

#[cfg(not(target_os = "linux"))]
fn main() {
    panic!("Vulkan FD capture requires Linux");
}

#[cfg(target_os = "linux")]
mod native {
    use clap::Parser;
    use kindle::{
        ActionMode, DreamerAgent, DreamerConfig, FrameFlags, Reward, vision::capture::CaptureStream,
    };
    use serde_json::json;
    use std::{
        fs::{self, File},
        io::{self, BufRead, BufReader, Write},
        os::unix::process::CommandExt,
        path::PathBuf,
        process::{Child, Command, Stdio},
        thread,
        time::{Duration, Instant},
    };

    #[derive(Parser)]
    pub struct Args {
        #[arg(long)]
        encoder: PathBuf,
        #[arg(long)]
        game: PathBuf,
        /// Directory containing id1/pak0.pak. Existing configs are not copied.
        #[arg(long)]
        data: PathBuf,
        /// Dullahan checkout with a release library and GPU_SYNC v4 support.
        #[arg(long)]
        dullahan: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(long, default_value_t = 128)]
        steps: usize,
        /// Explicit plumbing-only learner preset; normal default is Dreamer12M.
        #[arg(long)]
        tiny_world: bool,
        #[arg(long)]
        learn: bool,
        /// Optional visual audit; excluded from the action-only hot path.
        #[arg(long)]
        snapshot_at: Option<usize>,
    }

    struct OwnedChild(Child);
    impl OwnedChild {
        fn spawn(command: &mut Command) -> io::Result<Self> {
            let parent = std::process::id();
            // Only our children are killed. Parent death also contains them if
            // the external host guard terminates this native runner.
            unsafe {
                command.pre_exec(move || {
                    if libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGKILL) != 0 {
                        return Err(io::Error::last_os_error());
                    }
                    if libc::getppid() as u32 != parent {
                        return Err(io::Error::other("parent exited"));
                    }
                    Ok(())
                });
            }
            Ok(Self(command.spawn()?))
        }
    }
    impl Drop for OwnedChild {
        fn drop(&mut self) {
            let _ = self.0.kill();
            let _ = self.0.wait();
        }
    }

    pub fn run(args: Args) -> Result<(), Box<dyn std::error::Error>> {
        if args.steps == 0 {
            return Err("steps must be positive".into());
        }
        fs::create_dir_all(&args.output)?;
        let root = args.output.canonicalize()?;
        let mut result = File::create_new(root.join("frames.jsonl"))?;
        let mut config = if args.tiny_world {
            DreamerConfig::tiny(7)
        } else {
            DreamerConfig::new(7)
        };
        config.replay_capacity = 4096;
        config.loss_scales.reconstruction = 0.0;
        config.loss_scales.future_prediction = 0.25;
        if args.tiny_world {
            config.batch_size = 2;
            config.batch_length = 8;
            config.world_backprop_length = 8;
        }
        let mut agent = DreamerAgent::new(config, &args.encoder)?;
        let device = agent.core().gpu_device();
        assert!(!device.is_software_emulated);
        if let Ok(expected) = std::env::var("KINDLE_EXPECT_DEVICE_NAME") {
            assert_eq!(device.device_name, expected);
        }
        let budget = agent.core().gpu_memory_budget();
        assert!(budget.budget_bytes.saturating_sub(budget.usage_bytes) >= 2 << 30);
        if args.learn {
            agent.save_checkpoint(root.join("initial"))?;
        }

        let mut display = OwnedChild::spawn(
            Command::new("Xvfb")
                .args([
                    "-displayfd",
                    "1",
                    "-screen",
                    "0",
                    "640x480x24",
                    "-nolisten",
                    "tcp",
                ])
                .stdout(Stdio::piped())
                .stderr(File::create_new(root.join("xvfb.log"))?),
        )?;
        let mut number = String::new();
        BufReader::new(display.0.stdout.take().unwrap()).read_line(&mut number)?;
        let number: u16 = number.trim().parse()?;
        let display_name = format!(":{number}");
        // Private game directory: no existing user configs, demos or saves are modified.
        let game_data = root.join("game");
        fs::create_dir_all(game_data.join("id1"))?;
        std::os::unix::fs::symlink(
            args.data.canonicalize()?.join("id1/pak0.pak"),
            game_data.join("id1/pak0.pak"),
        )?;
        fs::write(
            game_data.join("id1/autoexec.cfg"),
            "bind w +forward\nbind s +back\nbind a +moveleft\nbind d +moveright\nbind Left +left\nbind Right +right\nbind space +attack\nvid_vsync 0\nhost_maxfps 0\nhost_framerate 0.016666667\nmap e1m1\n",
        )?;
        let socket = root.join("capture.sock");
        let shm = format!("kindle-gpu-{}", std::process::id());
        let mut game = OwnedChild::spawn(
            Command::new(args.game.canonicalize()?)
                .current_dir(&game_data)
                .args([
                    "-nosound", "-window", "-width", "640", "-height", "480", "-basedir",
                ])
                .arg(&game_data)
                .arg("-userdir")
                .arg(&game_data)
                .env("DISPLAY", &display_name)
                .env("SDL_VIDEODRIVER", "x11")
                .env("VK_LAYER_PATH", args.dullahan.canonicalize()?)
                .env(
                    "LD_LIBRARY_PATH",
                    args.dullahan.canonicalize()?.join("target/release"),
                )
                .env("VK_INSTANCE_LAYERS", "VK_LAYER_PRIVATE_dullahan")
                .env("VK_LAYER_DULLAHAN_MODE", "opaque")
                .env("VK_LAYER_DULLAHAN_SURFACE_EXTENT", "640x480")
                .env("VK_LAYER_DULLAHAN_SHM_NAME", &shm)
                .env("VK_LAYER_DULLAHAN_GPU_SOCKET", &socket)
                .env("VK_LAYER_DULLAHAN_GPU_SYNC", "1")
                .env("RUST_LOG", "info")
                .stdin(Stdio::null())
                .stdout(File::create_new(root.join("game.stdout"))?)
                .stderr(File::create_new(root.join("game.stderr"))?),
        )?;
        let deadline = Instant::now() + Duration::from_secs(30);
        while !socket.exists() {
            if game.0.try_wait()?.is_some() || Instant::now() > deadline {
                return Err("game did not start capture".into());
            }
            thread::sleep(Duration::from_millis(50));
        }
        // Only this owned Dullahan producer is trusted with the imported allocation.
        let mut capture = unsafe {
            CaptureStream::connect(agent.gpu_context(), &socket, Duration::from_secs(30))?
        };
        let xdo = |words: &[&str]| -> io::Result<()> {
            let status = Command::new("xdotool")
                .env("DISPLAY", &display_name)
                .args(words)
                .status()?;
            if status.success() {
                Ok(())
            } else {
                Err(io::Error::other("game input failed"))
            }
        };
        // SDL creates the private X window before the first swapchain.
        let window = Command::new("xdotool")
            .env("DISPLAY", &display_name)
            .args(["search", "--onlyvisible", "--class", "vkquake"])
            .output()?;
        let window = String::from_utf8(window.stdout)?
            .lines()
            .next()
            .ok_or("missing game window")?
            .to_owned();
        xdo(&["windowfocus", &window])?;
        let keys = ["w", "s", "a", "d", "Left", "Right", "space"];
        let started = Instant::now();
        let mut counts = [0u64; 7];
        let mut held: Option<usize> = None;
        for step in 0..=args.steps {
            let frame = capture.next_frame()?;
            let info = frame.info();
            if args.snapshot_at == Some(step) {
                let rgb = frame.read_rgb8();
                let mut file = File::create_new(root.join("snapshot.ppm"))?;
                write!(file, "P6\n{} {}\n255\n", rgb.width(), rgb.height())?;
                file.write_all(rgb.pixels())?;
            }
            let boundary = FrameFlags {
                is_first: step == 0,
                is_last: step == args.steps,
                is_terminal: false,
            };
            let acting = Instant::now();
            agent.observe_gpu(frame.frame(), boundary, Reward::default());
            if step == args.steps {
                frame.finish(true)?;
                break;
            }
            let action = agent.act(ActionMode::Sample, None);
            let policy_seconds = acting.elapsed().as_secs_f64();
            counts[action] += 1;
            if held != Some(action) {
                let mut words = Vec::new();
                if let Some(previous) = held {
                    words.extend(["keyup", "--delay", "0", keys[previous]]);
                }
                words.extend(["keydown", "--delay", "0", keys[action]]);
                xdo(&words)?;
                held = Some(action);
            }
            frame.finish(false)?;
            let reports = if args.learn {
                agent.learn_scheduled(1)
            } else {
                Vec::new()
            };
            for report in &reports {
                assert!(
                    report.world.total_loss.is_finite() && report.behavior.total_loss.is_finite()
                );
            }
            let memory = agent.core().gpu_memory_budget();
            assert!(memory.budget_bytes.saturating_sub(memory.usage_bytes) >= 2 << 30);
            writeln!(
                result,
                "{}",
                json!({"step":step,"action":action,"width":info.width,"height":info.height,"policy_seconds":policy_seconds,"updates":reports.len(),"world_loss":reports.last().map(|r| r.world.total_loss),"behavior_loss":reports.last().map(|r| r.behavior.total_loss),"memory":memory})
            )?;
        }
        drop(capture);
        drop(game);
        drop(display);
        // The game may be killed before the layer's normal SHM destructor.
        let _ = fs::remove_file(PathBuf::from("/dev/shm").join(shm));
        if args.learn {
            agent.save_checkpoint(root.join("final"))?;
        }
        let summary = json!({"device":device,"steps":args.steps,"actions":counts,"seconds":started.elapsed().as_secs_f64(),"updates":agent.core().learner_step(),"acting_pixel_readbacks":0,"diagnostic_pixel_readbacks":usize::from(args.snapshot_at.is_some_and(|step| step <= args.steps)),"live_feature_readbacks":0,"reward_adapter":false,"competence_claim":false});
        fs::write(
            root.join("result.json"),
            serde_json::to_vec_pretty(&summary)?,
        )?;
        println!("{summary}");
        Ok(())
    }
}

#[cfg(target_os = "linux")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use clap::Parser;
    native::run(native::Args::parse())
}
