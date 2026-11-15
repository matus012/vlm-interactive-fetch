# Third-party components

The MIT licence in `LICENSE` covers the code in this repository only. Nothing below is redistributed here unless
stated.

| component | used for | licence | in this repo? |
|---|---|---|---|
| [MuJoCo](https://github.com/google-deepmind/mujoco) 3.17.0 | physics and rendering | Apache-2.0 | no (pip dependency) |
| [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie), `stanford_tidybot` model, commit `370c43965323c0944b433b202aac5058b5f907a9` | the robot (TidyBot base + Kinova arm); a mast camera is added at load time with MjSpec | MIT, Copyright (c) 2024 Stanford Interactive Perception and Robot Learning Lab (`stanford_tidybot/LICENSE`) | no (fetched into `ext/` by the README's setup step) |
| Qwen3.8-27B | the VLM (weights used off the shelf, never fine-tuned) | Apache-2.0 (the model's `LICENSE` / model card) | no; weights are never redistributed |
| [vLLM](https://github.com/vllm-project/vllm) 0.32.0 | OpenAI-compatible server for the VLM, in an Apptainer image | Apache-2.0 | no |
| Python packages pinned in `uv.lock` (numpy, pillow, openai, pyyaml, pytest, imageio, imageio-ffmpeg and their dependencies) | sim, client, tests, video | permissive (BSD / MIT / Apache-2.0 / HPND / MIT-CMU) | no |
| FFmpeg binary downloaded by `imageio-ffmpeg` | encoding clips (dev dependency only) | LGPL / GPL, depending on the build | no |

Renders in `media/` (the demo video and preview GIF) show the Menagerie TidyBot model and follow its MIT licence.
The simulated house, its generator, the skills, the agent harness, the scorer and the baselines were written for this
project. Container recipes (`hpc/nav116.def`) and batch scripts follow patterns from the author's own earlier
projects; no third-party code was copied into this repository.

Clean-room: public code and public models only. No proprietary, employer or client code or data was used.
