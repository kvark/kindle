Kindle agent

Goal: learn to play any game by naturally playing with sparse rewards, like a human does.

Architecture:
  - start with DreamerV3
  - replace the world encoding with LeVJEPA, predict latent space

Extra features:
  - pretrain on video of the game (like in Dreamer 4)
    - learn dynamics, world model, help exploration
  - share experience with other Kindles playing the same game

Infrastructure:
  - all processing should be done on GPU without extra readbacks
  - aiming at low-latency realtime play of the policy
  - as such, game should run on GPU, frames capture on GPU, processed on GPU before feeding the policy, which would run on GPU.
  - only resulting actions (keyboard/mouse/joystick inputs) would be read back on CPU and fed into the game
  - Kindle should run on anything that supports Blade,
    - we'll have different size options

Main challenges today:
  - world modeling is too heavy
  - training takes too long to iterate effectively
  - the LeVJEPA integration concept is not fully confirmed
