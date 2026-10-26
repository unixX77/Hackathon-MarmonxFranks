# One-layer loose-brick rehearsal

From the project root, run:

```bash
bash scripts/run_one_layer_mock.sh
```

For a smaller trial with Enter-controlled stages:

```bash
bash scripts/run_one_layer_mock.sh --bricks 3 --step
```

`--bricks N` accepts 1 through 30 and uses the first N positions of the existing
one-layer layout, without changing the reference source geometry. The simulated
colour cycle starts red, dark_red, dark_orange, medium_nougat, unknown and repeats.
One brick therefore tests a red placement; three also exercise simulated discard.
Omit `--step` for an automatic run, or omit `--bricks` for the default 30.

In stage mode, Enter approves exactly the displayed simulated stage. Prompts
cover observation, capture, hand opening, approach, lowering, grasping, lifting,
transfer, release, retreat and placement verification. `q`, Ctrl+C at a prompt,
or end-of-input stops before that stage. Stopping while holding preserves a
simulated held brick and marks its reserved destination uncertain; no automatic
release or return occurs. Each new script invocation creates a new mock world.
The effective settings are saved as `config.yaml` in the printed results folder.

This runs the existing Coordinator, Manipulator and Store with MockWorld and
LocalTransport. It opens no ROS connection, robot connection or camera. Each
invocation creates a fresh results directory under `runtime/one-layer-mock-*`,
so previous mission state is preserved. The configuration is
`src/lego_sorter/config/loose_one_layer_mock.yaml`.

The source is one layer of six rows by five columns: 30 loose bricks with the
existing five-colour cycle. Six high-confidence red bricks are placed in a
separate loose destination layer. The other 24 go to simulated discard-box
receipts. Source and destination coordinates are illustrative. Success means
the simulated transaction loop completed, not that physical pick/place works.

The physical one-layer test remains separate. Loose bricks remove the need for
stud extraction, but do not resolve the recorded Franka communication reflex,
the observed approximately 20 mm physical alignment error, or missing measured
source/destination geometry. The main hardware backend requires candidates to
match its configured source layout within 4 mm; arbitrary loose positions are
not automatically supported. Actual trash-box release/fill handling remains
unimplemented. Hardware validation flags and startup checks are unchanged.
