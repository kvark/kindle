use super::*;

#[test]
#[ignore = "requires separately guarded RGB96 resize and small-object qualification"]
fn gpu_rgb96_resize_matches_scalar_and_preserves_small_objects() {
    let gpu = Arc::new(crate::init_gpu_context().unwrap());
    let side = 96;
    let elements = 3 * side * side;
    let mut graph = meganeura::Graph::new();
    let input = graph.input("observation", &[2, elements]);
    let output = graph.neg(input);
    graph.set_outputs(vec![output]);
    let mut session = meganeura::build(
        &graph,
        meganeura::SessionConfig {
            mode: meganeura::Mode::Inference,
            gpu: Some(Arc::clone(&gpu)),
            ..Default::default()
        },
    )
    .0;
    session.set_input("observation", &vec![0.125; 2 * elements]);
    let mut pixels = GpuPreprocessor::rgb(Arc::clone(&gpu), 2, side, None);
    let (width, height) = (160, 210);
    for (x, y) in [(0, 0), (158, 0), (0, 206), (158, 206), (79, 103), (31, 180)] {
        let mut rgb = vec![0; width * height * 3];
        for dy in 0..4 {
            for dx in 0..2 {
                let start = ((y + dy) * width + x + dx) * 3;
                rgb[start..start + 3].copy_from_slice(&[255, 160, 64]);
            }
        }
        let resized = rgb_filter::reference(&rgb, width, height, side);
        let mut expected = vec![0.125; 2 * elements];
        for index in 0..side * side {
            for channel in 0..3 {
                expected[elements + channel * side * side + index] =
                    f32::from(resized[index * 3 + channel]) / 255.0 - 0.5;
            }
        }
        assert!(resized.iter().any(|&value| value > 0));
        pixels.cpu_frames(&mut session, &[(1, &rgb, width, height)]);
        let uploaded = tests::check_patches(&mut session, &expected, "RGB96 CPU small object");
        assert!(uploaded[..elements].iter().all(|&value| value == -0.125));
        let peak = uploaded[elements..elements + side * side]
            .iter()
            .map(|&value| 0.5 - value)
            .fold(0.0_f32, f32::max);
        assert!(peak > 1.0 / 255.0, "object disappeared at {x},{y}");
        for format in [PixelFormat::Rgb8, PixelFormat::Rgba8, PixelFormat::Bgra8] {
            let channels = format.channels() as usize;
            let stride = width * channels + 7;
            let base = 5;
            let mut packed = vec![0xE1; (base + stride * height).next_multiple_of(4)];
            for y in 0..height {
                for x in 0..width {
                    for channel in 0..3 {
                        let component = if format == PixelFormat::Bgra8 {
                            2 - channel
                        } else {
                            channel
                        };
                        packed[base + y * stride + x * channels + component] =
                            rgb[(y * width + x) * 3 + channel];
                    }
                }
            }
            let buffer = gpu.create_buffer(gpu::BufferDesc {
                name: "rgb96_test_source",
                size: packed.len() as u64,
                memory: gpu::Memory::Shared,
            });
            unsafe {
                std::ptr::copy_nonoverlapping(packed.as_ptr(), buffer.data(), packed.len());
            }
            let frame = unsafe {
                GpuFrame::from_buffer(
                    &gpu,
                    &buffer,
                    base as u64,
                    FrameLayout::new(width as u32, height as u32, stride as u32, format),
                )
            };
            pixels.gpu_frames(&mut session, &[(1, frame)]);
            assert_eq!(
                tests::check_patches(&mut session, &expected, "RGB96 strided resident source"),
                uploaded
            );
            pixels.wait();
            gpu.destroy_buffer(buffer);
        }
        eprintln!(
            "RGB96 2x4 colored object at {x},{y}: peak RGB8={}, scalar/layout/inactive-stream checks pass",
            peak * 255.0
        );
    }
}
