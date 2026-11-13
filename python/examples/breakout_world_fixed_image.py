"""Fixed full observation filter with direct future RGB supervision."""
import breakout_world_fixed_teacher as phase

if __name__ == '__main__':
    phase.main(pixel_loss=phase.corpus.FIXED_IMAGE_OBJECTIVE)
