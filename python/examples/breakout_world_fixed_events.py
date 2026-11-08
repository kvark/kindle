"""Fixed observation-filter dynamics with explicit future reward/continuation."""
import breakout_world_corpus as corpus
from breakout_world_fixed_teacher import main

if __name__ == '__main__':
    main(pixel_loss=corpus.FIXED_EVENTS_OBJECTIVE)
