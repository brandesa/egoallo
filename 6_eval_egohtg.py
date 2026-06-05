from __future__ import annotations

import dataclasses
import time
from pathlib import Path

import numpy as np
import torch
import viser
import yaml

from egoallo import fncsmpl, fncsmpl_extensions
from egoallo.data.aria_mps import load_point_cloud_and_find_ground
from egoallo.guidance_optimizer_jax import GuidanceMode
from egoallo.hand_detection_structs import (
    CorrespondedAriaHandWristPoseDetections,
    CorrespondedHamerDetections,
)
from egoallo.inference_utils import (
    InferenceInputTransforms,
    InferenceTrajectoryPaths,
    load_denoiser,
)
from egoallo.sampling import run_sampling_with_stitching
from egoallo.transforms import SE3, SO3
from egoallo.vis_helpers import visualize_traj_and_hand_detections

from egoallo_eval.configs import SequenceContext, EVAL_XHALL, EXOMC_XHALL, ALL, ONE, CLEAN
from projectaria_tools.core.data_provider import (
    VrsDataProvider,
    create_vrs_data_provider,
)
from projectaria_tools.core import calibration



@dataclasses.dataclass
class Args:
    seq: str | None = None 
    """Search directory for trajectories. This should generally be laid out as something like:

    traj_dir/
        video.vrs
        egoallo_outputs/
            {date}_{start_index}-{end_index}.npz
            ...
        ...
    """
    checkpoint_dir: Path = Path("./egoallo_checkpoint_april13/checkpoints_3000000/")
    smplh_npz_path: Path = Path("./data/smplh/neutral/model.npz")

    glasses_x_angle_offset: float = 0.0
    """Rotate the CPF poses by some X angle."""
    start_index: int = 0
    """Index within the downsampled trajectory to start inference at."""
    traj_length: int = 128
    """How many timesteps to estimate body motion for."""
    num_samples: int = 1
    """Number of samples to take."""
    guidance_mode: GuidanceMode = "aria_hamer"
    """Which guidance mode to use."""
    guidance_inner: bool = True
    """Whether to apply guidance optimizer between denoising steps. This is
    important if we're doing anything with hands. It can be turned off to speed
    up debugging/experiments, or if we only care about foot skating losses."""
    guidance_post: bool = True
    """Whether to apply guidance optimizer after diffusion sampling."""
    save_traj: bool = True
    """Whether to save the output trajectory, which will be placed under `traj_dir/egoallo_outputs/some_name.npz`."""
    visualize_traj: bool = False
    """Whether to visualize the trajectory after sampling."""


def main(args: Args, seq_ctx: SequenceContext) -> None:
    device = torch.device("cuda")

    config_path = seq_ctx.config_yaml
    cfg = load_config(config_path)

    # Get point cloud + floor.
    _, floor_z = load_point_cloud_and_find_ground(seq_ctx.aria_subject.global_pcd)

    # Read transforms from VRS / MPS, downsampled.
    fps = 30
    transforms = InferenceInputTransforms.load(
        seq_ctx.aria_subject.aria_vrs, seq_ctx.aria_subject.aria_slam_dir, fps=fps
    ).to(device=device)

    #Set traj start and length
    if cfg["traj_start_frame_idx"] is not None:
        args.start_index = int(cfg["traj_start_frame_idx"] * fps / transforms.aria_fps)
        args.traj_length = transforms.Ts_world_cpf.shape[0] - args.start_index
    else:
        ValueError(f"[WARNING]: No traj_start_frame_idx specified in config")

    # INFO printing
    print(f"[INFO]: Starting inference for sequence {args.seq}")
    print(f"[INFO]: Trajectory start index: {args.start_index}")
    print(f"[INFO]: Trajectory length: {args.traj_length}")

    #run in chunks 
    chunk_size = 6000
    joint_positions_cam = torch.empty(
        (args.traj_length-1, 52, 3), device=device
    )  # (time, joints, xyz)
    verts_cam = torch.empty(
        (args.num_samples, args.traj_length-1, 6890, 3), device=device
    )  # (samples, time, vertices, xyz)
    for chunk_start in range(args.start_index, args.start_index + args.traj_length, chunk_size):
        chunk_end = min(chunk_start + chunk_size, args.start_index + args.traj_length)
        print(f"[INFO]: Running inference on frames {chunk_start} to {chunk_end}")

        if chunk_start != args.start_index:
            transforms = InferenceInputTransforms.load(
                seq_ctx.aria_subject.aria_vrs, seq_ctx.aria_subject.aria_slam_dir, fps=fps
            ).to(device=device)


        # Note the off-by-one for Ts_world_cpf, which we need for relative transform computation.
        Ts_world_cpf = (
            SE3(
                transforms.Ts_world_cpf[
                    chunk_start : chunk_end + 1
                ]
            )
            @ SE3.from_rotation(
                SO3.from_x_radians(
                    transforms.Ts_world_cpf.new_tensor(args.glasses_x_angle_offset)
                )
            )
        ).parameters()
        del transforms

        # server = None
        # if args.visualize_traj:
        #     server = viser.ViserServer()
        #     server.gui.configure_theme(dark_mode=True)

        denoiser_network = load_denoiser(args.checkpoint_dir).to(device)
        body_model = fncsmpl.SmplhModel.load(args.smplh_npz_path).to(device)

       

        traj = run_sampling_with_stitching(
            denoiser_network,
            body_model=body_model,
            guidance_mode=args.guidance_mode,
            guidance_inner=args.guidance_inner,
            guidance_post=args.guidance_post,
            Ts_world_cpf=Ts_world_cpf,
            hamer_detections=None,
            aria_detections=None,
            num_samples=args.num_samples,
            device=device,
            floor_z=floor_z,
            guidance_verbose=False,
        )

        T_device_cpf, T_device_camera = get_device_and_camera_transforms(seq_ctx)

        joint_positions_cam_chunk, verts_cam_chunk, faces_chunk = get_joint_traj(Ts_world_cpf[1:, :], traj, body_model, T_device_cpf, T_device_camera)
        # verts_cam_chunk is (S, T, V, 3), assign to corresponding slice
        verts_cam[:, chunk_start-args.start_index:chunk_end-args.start_index] = verts_cam_chunk
        joint_positions_cam[chunk_start-args.start_index:chunk_end-args.start_index] = joint_positions_cam_chunk

    export_joint_traj(seq_ctx.egoallo_data, joint_positions_cam, verts_cam, faces_chunk)
    save_name = (
        time.strftime("%Y%m%d-%H%M%S")
        + f"_{args.start_index}-{args.start_index + args.traj_length}"
    )
    (seq_ctx.egoallo_data.parent / (save_name + "_args.yaml")).write_text(
        yaml.dump(dataclasses.asdict(args))
    )


def get_device_and_camera_transforms(seq_ctx: SequenceContext):
    provider = create_vrs_data_provider(str(seq_ctx.aria_subject.aria_vrs))
    device_calib = provider.get_device_calibration()
    T_device_cpf = SE3(
        torch.from_numpy(
            device_calib.get_transform_device_cpf().to_quat_and_translation()
        )
    )
    
    raw_calib = device_calib.get_camera_calib("camera-rgb") 
    w, h = raw_calib.get_image_size()
        
    # Create the standard pinhole model (Rectified)
    linear_calib = calibration.get_linear_camera_calibration(
        int(w), int(h),
        raw_calib.get_focal_lengths()[0],
        "pinhole",
        raw_calib.get_transform_device_camera(),
    )
    
    # Standardize rotation for head-mounted portrait-to-landscape conversion
    calib = calibration.rotate_camera_calib_cw90deg(linear_calib)
    
    
    T_device_camera = SE3(
        torch.from_numpy(
            calib.get_transform_device_camera().to_quat_and_translation()
        )
    )
    return T_device_cpf, T_device_camera

def load_config(path: Path):
    with open(str(path), "r") as f:
        return yaml.safe_load(f)["generator_config"]["aria_time_sync"]

def get_joint_traj(Ts_world_cpf, traj, body_model, T_device_cpf, T_device_camera):

    betas = traj.betas
    timesteps = betas.shape[1]
    sample_count = betas.shape[0]
    assert betas.shape == (sample_count, timesteps, 16)
    body_quats = SO3.from_matrix(traj.body_rotmats).wxyz
    assert body_quats.shape == (sample_count, timesteps, 21, 4)
    device = body_quats.device
    dtype = body_quats.dtype

    if traj.hand_rotmats is not None:
        hand_quats = SO3.from_matrix(traj.hand_rotmats).wxyz
        left_hand_quats = hand_quats[..., :15, :]
        right_hand_quats = hand_quats[..., 15:30, :]
    else:
        left_hand_quats = None
        right_hand_quats = None

    shaped = body_model.with_shape(torch.mean(betas, dim=1, keepdim=True))
    fk_outputs = shaped.with_pose_decomposed(
        T_world_root=SE3.identity(
            device=device, dtype=body_quats.dtype
        ).parameters(),
        body_quats=body_quats,
        left_hand_quats=left_hand_quats,
        right_hand_quats=right_hand_quats,
    )
    assert Ts_world_cpf.shape == (timesteps, 7)
    T_world_root = fncsmpl_extensions.get_T_world_root_from_cpf_pose(
        # Batch axes of fk_outputs are (num_samples, time).
        # Batch axes of Ts_world_cpf are (time,).
        fk_outputs,
        Ts_world_cpf[None, ...],
    )
    fk_outputs = fk_outputs.with_new_T_world_root(T_world_root)

    # Build SE3 transforms on the same device/dtype.
    Ts_world_cpf_se3 = SE3(Ts_world_cpf.to(device=device, dtype=dtype))
    T_device_cpf_se3 = SE3(T_device_cpf.wxyz_xyz.to(device=device, dtype=dtype))
    T_device_camera_se3 = SE3(T_device_camera.wxyz_xyz.to(device=device, dtype=dtype))

    # World -> camera for each timestep, then camera -> world.
    T_world_camera = Ts_world_cpf_se3 @ T_device_cpf_se3.inverse() @ T_device_camera_se3
    T_camera_world = T_world_camera.inverse()

    # Joint positions in world and camera frames.
    #root 
    root_position_world = fk_outputs.T_world_root[..., 4:7]
    #smpl joints
    joint_positions_no_root_world = fk_outputs.Ts_world_joint[..., 4:7]
    joint_positions_world = torch.cat([root_position_world.unsqueeze(2), joint_positions_no_root_world], dim=2)
    # joint_positions_world = fk_outputs.Ts_world_joint[..., 4:7]
    joint_positions_cam = torch.empty_like(joint_positions_world[0])

    # Iterate over time to apply the corresponding camera transform.
    for t in range(timesteps):
        # Slice underlying parameters to build a per-frame SE3, then apply.
        T_camera_world_t = SE3(T_camera_world.wxyz_xyz[t])
        joint_positions_cam[t] = T_camera_world_t @ joint_positions_world[:, t]

    # Extract per-frame vertices by re-posing the body model.
    # For each timestep, we call with_pose_decomposed() to get the posed vertices.
    verts_list = []
    for t in range(timesteps):
        # Get the pose for all samples at timestep t
        body_quats_t = body_quats[:, t:t+1, :, :]  # (S, 1, 21, 4)
        left_hand_quats_t = left_hand_quats[:, t:t+1, :, :] if left_hand_quats is not None else None
        right_hand_quats_t = right_hand_quats[:, t:t+1, :, :] if right_hand_quats is not None else None
        
        # Get the actual root transform for this timestep (must match joint positions)
        T_world_root_t = fk_outputs.T_world_root[:, t]  # (S, 7)
        
        # Re-pose to get vertices at this frame with the correct root transform
        posed_t = shaped.with_pose_decomposed(
            T_world_root=T_world_root_t,
            body_quats=body_quats_t,
            left_hand_quats=left_hand_quats_t,
            right_hand_quats=right_hand_quats_t,
        )
        # posed_t.lbs() applies linear blend skinning to get deformed vertices in world frame
        mesh_t = posed_t.lbs()
        # mesh_t.verts is (S, V, 3) - deformed vertices at this pose in world frame
        verts_list.append(mesh_t.verts[:,0])

    # Stack across time: (S, T, V, 3)
    # verts_list contains T tensors of shape (S, V, 3), stack on dim=1 for time
    verts_STV = torch.stack(verts_list, dim=1)

    # Now transform to camera frame per-frame
    verts_cam_STV = torch.empty_like(verts_STV, device=device, dtype=dtype)
    for t in range(timesteps):
        T_camera_world_t = SE3(T_camera_world.wxyz_xyz[t])
        # verts_STV[:, t] is (S, V, 3), transform each sample's vertices
        verts_cam_STV[:, t] = T_camera_world_t @ verts_STV[:, t]

    # Get faces from body model (constant across time)
    faces = body_model.faces

    return joint_positions_cam, verts_cam_STV, faces

def export_joint_traj(out_path, joint_positions_cam, verts_cam=None, faces=None):
    # Export to NPZ (move to CPU and NumPy).
    payload = {
        # "joint_positions_world": joint_positions_world.cpu().numpy(force=True),
        "joint_positions": joint_positions_cam.cpu().numpy(force=True),
        "verts": verts_cam.cpu().numpy(force=True) if verts_cam is not None else None,
        "faces": faces.cpu().numpy(force=True) if faces is not None else None,
        # "T_world_camera": T_world_camera.as_matrix().cpu().numpy(force=True),
    }
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **payload)



if __name__ == "__main__":
    import tyro

    seq_root = Path("/home/brandesa/mt/human_terrain_generation")

    if tyro.cli(Args).seq is not None:
        # Run on a single specified sequence.
        seq_ctx = SequenceContext(name=tyro.cli(Args).seq, root=seq_root)
        main(tyro.cli(Args), seq_ctx=seq_ctx)

    else:
        sequences = ONE
        for seq in sequences:
            # Create SequenceContext
            seq_ctx = SequenceContext(name=seq, root=seq_root)

            # try:
            main(tyro.cli(Args), seq_ctx=seq_ctx)
            # except Exception as e:
                # print(f"Error occurred while processing sequence {seq}: {e}")