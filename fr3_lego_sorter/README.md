# FR3 LEGO pile sorter

A Franka FR3 with a wrist-mounted RealSense D435 empties a pile of LEGO 2x4 bricks on its own. It looks at the pile,
finds the bricks of the top layer (SAM3 segmentation + depth), classifies their colour, has a vision model (Codex)
check the result, then picks them one after the other. Good bricks go onto a 3 x 2 x n stack of their own colour;
bad bricks go into a bin. Then it looks again, until two scans in a row find nothing.

```
observe pose ──► capture (D435) ──► SAM3 + depth: top-layer bricks ──► colour classifier ──► Codex review
     ▲                                                                                          │
     │                                                                                          ▼
     └──── recover on failure ◄── place: colour stack / bin ◄── pick (MoveIt, straight lines) ◄── top layer, pile area only
```

The full technical description (method, coordinate frames, calibration, motion, every setting) is in
[DOCUMENTATION.md](DOCUMENTATION.md).

## Repository layout

| Path | What it is |
|---|---|
| `src/fr3_pose_pick/` | ROS 2 (Humble) package: the application and the robot side |
| `src/fr3_pose_pick/scripts/sort_pile.py` | **The application**: observe, scan, pick the top layer, repeat, recover |
| `src/fr3_pose_pick/scripts/pose_pick_stacks.py` | Per-colour 3 x 2 x n stacks, bad bricks to the bin |
| `src/fr3_pose_pick/scripts/pose_pick*.py`, `smooth_motion.py`, `trajectory_smoothing.py` | Pick and place: collision-checked straight-line moves, grasp checks |
| `src/fr3_pose_pick/scripts/record_pose.py`, `go_to_pose.py` | Record named hand poses / move to one |
| `src/fr3_pose_pick/config/pose_pick.yaml` | **All settings**, including the standard run (`sort_pile:` section) |
| `src/fr3_pose_pick/config/poses.yaml` | Recorded poses: `observe`, `bin`, `stack_<colour>`, `pile_corner_1/2` |
| `src/fr3_pose_pick/test/` | Unit tests (no robot needed) |
| `vision/` | Camera pipeline: D435 capture, SAM3 top-layer detection, colour classifier, Codex/Claude review |
| `data/calibration/charuco_result.json` | Hand-eye calibration (camera on the hand) |
| `env.sh`, `build.sh` | Environment for every terminal, and the build |

## Requirements

- Ubuntu 22.04, ROS 2 Humble, MoveIt 2.
- [franka_ros2](https://github.com/frankaemika/franka_ros2) with libfranka, built in its own workspace
  (default `~/Documents/Hackathon/hackathon_ros2_ws`; elsewhere: `export FRANKA_WS=<that folder>` before `env.sh`).
- A Franka FR3 with the Franka Hand (FCI enabled) and a RealSense D435 on the hand.
- The camera pipeline's Python environment: a conda env `sam3` with [vision/requirements.txt](vision/requirements.txt),
  a CUDA GPU, and Hugging Face access to `facebook/sam3`. Another interpreter: `export PY=<python>`.
- The Codex CLI, logged in (`codex login`), for the review step.

## Build (once)

```bash
cd ~/Documents/Hackathon/fr3_lego_sorter
./build.sh
```

## Run

Every terminal starts with `source ~/Documents/Hackathon/fr3_lego_sorter/env.sh`. It sources ROS, franka_ros2 and this
workspace, sets `ROS_DOMAIN_ID=77`, and moves to the repository root. The commands below are run from there.

**Terminal 1: the robot, MoveIt and RViz**

```bash
ros2 launch fr3_pose_pick pose_bringup.launch.py robot_ip:=192.168.3.10
```

(`pose_bringup` is `franka_fr3_moveit_config moveit.launch.py` plus the table in the planning scene;
the plain MoveIt launch works too.)

**Terminal 2: once per setup, only when the layout changed.** Hold the hand still at each place and record it:

```bash
ros2 run fr3_pose_pick record_pose.py observe           --config src/fr3_pose_pick/config/pose_pick.yaml   # camera over the pile
ros2 run fr3_pose_pick record_pose.py pile_corner_1     --config src/fr3_pose_pick/config/pose_pick.yaml   # two opposite corners
ros2 run fr3_pose_pick record_pose.py pile_corner_2     --config src/fr3_pose_pick/config/pose_pick.yaml   #   of the pile area
ros2 run fr3_pose_pick record_pose.py stack_red         --config src/fr3_pose_pick/config/pose_pick.yaml   # centre of each stack
ros2 run fr3_pose_pick record_pose.py stack_dark_red    --config src/fr3_pose_pick/config/pose_pick.yaml
ros2 run fr3_pose_pick record_pose.py stack_dark_orange --config src/fr3_pose_pick/config/pose_pick.yaml
ros2 run fr3_pose_pick record_pose.py bin               --config src/fr3_pose_pick/config/pose_pick.yaml   # over the bin
ros2 run fr3_pose_pick record_pose.py --show            --config src/fr3_pose_pick/config/pose_pick.yaml
```

**Terminal 2: the sorter**

```bash
ros2 run fr3_pose_pick pose_pick_stacks.py --config src/fr3_pose_pick/config/pose_pick.yaml --reset    # only if the stacks were emptied
ros2 run fr3_pose_pick sort_pile.py --config src/fr3_pose_pick/config/pose_pick.yaml
```

The standard run settings are in the `sort_pile:` section of `pose_pick.yaml`:

| Setting | Value | Meaning |
|---|---|---|
| `review` | `codex` | Codex checks every scan's colours and masks (`claude`, or `null` for the classifier alone) |
| `per_round` | `15` | Bricks picked per scan; drops to 1 when the pile is seen to move more than `max_shift` |
| `max_failures` | `10` | Failed rounds in a row that are recovered from before the run stops |
| `gui` | `false` | `true`: a window with the last scan and each brick's status |

An option on the command line wins for that run, e.g. `--per-round 5`, `--gui`, `--review none`, `--dry-run`
(look once and print what it would do; the arm only moves to `observe`). `sort_pile.py --help` lists them all.

## Safety

The robot moves on its own. Keep the e-stop in reach. Ctrl+C stops the program.
Only one robot stack may run in the ROS domain; the scripts check this before moving.
After a collision reflex (red light) the run clears the error, restarts the arm controller, takes a held brick to
the bin, and goes on with a new scan, up to `max_failures` times in a row.

## Tests

```bash
source env.sh
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q src/fr3_pose_pick/test
```

Run-time files (`pose_pick_stacks.json`: what is on each stack; `pose_pick_auto_log.json`: every pick) are written to
the repository root; camera captures and pipeline results to `vision/data/`.
