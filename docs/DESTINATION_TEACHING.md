# Teaching destination centres

For the implemented bounded physical runner and current two-corner boundary
teaching commands, follow [PHYSICAL_ONE_LAYER.md](PHYSICAL_ONE_LAYER.md).
The earlier source-grid teaching notes below describe the development sequence;
the physical runner now uses a broad rectangle and measured detections, without
grid-cell matching.

The operator chose the centre of the eventual 6x5 red layer for `place`, and
a separate spot on the same measured table for `discard`. Both capture names
refer to XY region centres, not the location of the first grid cell. Discard
bricks will need separate placement cells; recording a centre does not validate
repeated releases at one point or discard-area capacity.

Once the robot has physically reached a stationary hover above the red layer
centre, run from a ROS-sourced project terminal:

```bash
/usr/bin/python3 -s scripts/record_destination.py --name place
```

Once it has physically reached a stationary hover above the discard-area centre:

```bash
/usr/bin/python3 -s scripts/record_destination.py --name discard
```

The script reads stationary joint feedback, fresh `fr3_link0` to
`fr3_hand_tcp` TF and advancing physical Franka robot time. It contains no
planning or execution clients. It records both destinations and capture history
in `runtime/destination-teaching/destinations.json`, leaving active task configs
and commissioning flags unchanged. Recorded orientation is evidence only;
it does not silently rotate the layout to an arbitrary nonorthogonal yaw.

## Observation pose and use in a mock run

With the actual arm stationary at a camera pose that sees the complete source
region and surrounding table, record:

```bash
/usr/bin/python3 -s scripts/record_destination.py --name observation
```

The same registry retains both existing destinations and adds the full measured
observation XYZ/quaternion, joint readings and an XYZ/yaw representation. The
current mock motion model uses XYZ/yaw and a tool-down orientation; the original
quaternion and tilt remain available for later hardware commissioning.

To explicitly use all three captures in a new simulated trial:

```bash
bash scripts/run_one_layer_mock.sh --bricks 3 --step \
  --teachings runtime/destination-teaching/destinations.json
```

The importer requires all three records with matching base/TCP frames. It uses
place XY as the red-layer centre and discard XY as a separate table-grid centre,
with distinct cells for non-red bricks rather than a simulated single box-drop
point. The known table plane supplies their support Z; captured hover Z is not
used as release Z. It prints imported coordinates and saves the effective
`config.yaml` in the new runtime directory. Missing observation data produces
an error rather than silently falling back to the default. Without `--teachings`,
the mock keeps its original illustrative coordinates. Hardware flags and active
hardware configuration are unchanged.

The cyan wall-clock preview viewer is visualization only. In normal MoveIt
RViz, dragging the interaction marker sets a proposed goal, not the actual
robot pose. Clear the previously observed communication fault before attempting
physical positioning. The recorder does not perform recovery or move the robot.

Use a hover to teach XY; contact with the table is unnecessary. The measured
table support plane is -0.020 m in fr3_link0. Captured TCP Z is saved separately
from this plane, and `release_tcp_z_m` remains unset. The physical TCP-to-finger
relationship, release height, full layout footprints and finger clearances have
not been independently verified. This does not establish physical pick/place
readiness or resolve the prior approximately 20 mm alignment discrepancy.

## Physical-trial setup (separate from the mock runner)

The operator explicitly clarified that the desired next task is physical,
Enter-controlled execution for three loose bricks, initially arranged in one
row of three. The mock runner is not a hardware launcher and must not be
presented as one. The operator confirmed the intended correction remains
-20 mm along the tool-X coordinate used by the relative-shift diagnostic;
that attempted move failed and no permanent correction has been applied to
the main sorter config.

Additional read-only capture labels are available:

- `--name source`: the centre of the source grid. For one row of three this
  is above the middle brick, recorded from a stationary hover. This does not
  imply physical contact or a grasp.
- `--name grasp`: a separately established actual TCP grasp-pose measurement.
  It records TCP height above the measured table as evidence, without claiming
  successful alignment, finger clearance or a successful grasp. Do not teach
  a low/contact pose until its clearance has been established.

The source-grid centre, observation pose and measured grasp/release geometry
are not yet all present in the teaching registry. The stored place/discard Z
values are hover heights, not release heights. A complete physical three-brick
runner has not yet been validated or made ready for execution. No new physical
motion was commanded while adding these capture labels. The new recording
behavior passed all five destination-recording tests.
