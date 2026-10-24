^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Changelog for package franka_gazebo_hardware
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

UNRELEASED
----------

* feat: added a model-based gravity-compensation system plugin
  (``franka_gazebo_hardware/GazeboGravityCompensationSystem``) for gz_ros2_control that
  injects pinocchio-computed gravity torque on the effort-controlled arm joints. Gravity is
  enabled globally in the Gazebo world, so the zero-torque example controllers behave as on
  the real robot instead of collapsing. Split out of the former ``franka_gazebo`` package.
* feat: express the simulated external wrench in the stiffness frame K (``K_F_ext_hat_K``
  and the ``_tcp`` force/torque state interfaces), while ``O_F_ext_hat_K`` stays in the
  base frame; the wrench sign follows the reaction convention (a push in +x reads a
  measured external force of -x).
* feat: export the 16 ``<i>/cartesian_pose_state`` state interfaces so Cartesian-pose
  controllers can activate in simulation.
