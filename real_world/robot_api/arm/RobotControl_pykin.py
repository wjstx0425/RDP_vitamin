"""
RobotControl_pykin.py
---------------------
AM2 双臂机械臂的运动学控制层。

设计 (方案 A):
    - URDF 已拆成左右两个单臂文件, 每个的根都是 base_link
    - pykin SingleArm 加载时不传 ab2rb 偏移 (Transform 用单位变换)
    - setup_link_name("base_link", "left-link_arm_7"), 让 IK/FK 链
      从 URDF 根开始, 自动包含 JOINT_L00 的肩部偏移
    - 结果: FK 输出 = ^rbT_eef, IK 输入 = ^rbT_eef, 完全自洽

接口:
    - get_robot_joints() -> dict
    - get_ee_pose()      -> dict, ^rbT_eef 形式 (7-vector wxyz)
    - set_target_JP(...)
    - set_target_CP(target_pose, single_arm_mode=False)
    - execute()
    - stop()

EE 位姿格式 (与旧版兼容):
    7-vector [x, y, z, qw, qx, qy, qz]  (scalar-first 四元数)

可配置项 (优先级: 显式参数 > YAML > 默认值):
    - urdf_path_left/right
    - base_link_left/right (默认 "base_link")
    - ee_link_left/right (默认 "left-link_arm_7" / "right-link_arm_7")
"""

import os
import sys
import io
import re
import time
import shutil
import tempfile
import contextlib
from pathlib import Path

# 让 `python real_world/.../RobotControl_pykin.py` 直接跑也能解析包路径
# 把仓库根加到 sys.path
_REPO_ROOT_FOR_IMPORT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT_FOR_IMPORT))

import numpy as np
import yaml
import xml.etree.ElementTree as ET

import collections
import collections.abc
if not hasattr(collections, "Iterable"):
    collections.Iterable = collections.abc.Iterable

import transforms3d as t3d
import pykin
from pykin.robots.single_arm import SingleArm
from pykin.kinematics import transform as t_utils

from real_world.robot_api.arm.RobotWrapper_typhon import RobotWrapperTyphon


MODULE_DIR = Path(__file__).resolve().parent
REPO_ROOT = MODULE_DIR.parents[2]


# ============================================================
# Helpers
# ============================================================
def _resolve_repo_path(path_like: str) -> str:
    p = Path(path_like)
    return str(p) if p.is_absolute() else str((REPO_ROOT / p).resolve())


def _load_yaml_config(config_path: str) -> dict:
    p = Path(config_path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    if not p.exists():
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def preprocess_urdf(urdf_path: str) -> str:
    """把 ROS package:// URI 替换成相对路径, 写到临时文件返回路径"""
    with open(urdf_path, "r") as f:
        content = f.read()
    urdf_dir = os.path.dirname(os.path.abspath(urdf_path))
    parent_dir = os.path.dirname(urdf_dir)
    package_name = os.path.basename(parent_dir)
    content = re.sub(f"package://{re.escape(package_name)}/", "../", content)
    fd, tmp_path = tempfile.mkstemp(suffix=".urdf", prefix="preprocessed_",
                                    dir=urdf_dir)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(content)
        return tmp_path
    except Exception:
        os.close(fd)
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def prepare_urdf_for_pykin(urdf_path: str) -> tuple[str, str]:
    """
    pykin 的 URDFModel 用 ``pykin/assets/<f_name>`` 读文件; Robot.mesh_path 为
    URDF 所在目录 (dirname), 再拼 URDF 里的 mesh 相对路径. 因此必须把「含 meshes/
    的包目录」整包链到 pykin/assets/vb_vla_assets/<pkg_name>/ 下, 且 staged URDF
    的相对位置与仓库里一致.

    支持布局:
    - 扁平: ``.../assets/AM2_left.urdf`` + ``.../assets/meshes/`` (pkg_dir=assets)
    - ROS: ``.../AM2_left/urdf/AM2_left.urdf`` + ``.../AM2_left/meshes/``

    返回 (相对 pykin/assets 的路径, 预处理临时 URDF 绝对路径)
    """
    src = Path(urdf_path).resolve()
    if not src.is_file():
        raise FileNotFoundError(
            f"URDF not found: {src}\n"
            "Use e.g. real_world/robot_api/assets/AM2_left.urdf "
            "with meshes/ next to it, or pkg/urdf/*.urdf + pkg/meshes/."
        )

    parent = src.parent
    if (parent / "meshes").is_dir():
        pkg_dir = parent
        rel_in_pkg = Path(src.name)
    elif parent.name == "urdf" and (parent.parent / "meshes").is_dir():
        pkg_dir = parent.parent
        rel_in_pkg = Path("urdf") / src.name
    else:
        # 兼容旧默认路径 (pkg/urdf/name.urdf), 即使未检测到 meshes 也按此结构 stage
        pkg_dir = parent.parent
        rel_in_pkg = Path("urdf") / src.name

    pkg_name = pkg_dir.name

    pykin_assets = Path(pykin.__file__).resolve().parent / "assets"
    stage_root = pykin_assets / "vb_vla_assets"
    stage_pkg = stage_root / pkg_name
    stage_urdf = stage_pkg / rel_in_pkg

    stage_root.mkdir(parents=True, exist_ok=True)
    if not stage_pkg.exists():
        try:
            stage_pkg.symlink_to(pkg_dir, target_is_directory=True)
        except OSError:
            shutil.copytree(pkg_dir, stage_pkg)

    if not stage_urdf.exists():
        raise FileNotFoundError(
            f"URDF not staged: {stage_urdf} (pkg_dir={pkg_dir}, rel_in_pkg={rel_in_pkg})"
        )

    preprocessed = Path(preprocess_urdf(str(stage_urdf))).resolve()
    try:
        rel = str(preprocessed.relative_to(pykin_assets))
    except ValueError:
        rel = os.path.relpath(str(preprocessed), str(pykin_assets))
    return rel, str(preprocessed)


# ============================================================
# RobotControl
# ============================================================
class RobotControl:
    """
    AM2 双臂机械臂运动学控制层.

    Args:
        urdf_path_left/right: 单臂 URDF 路径 (拆分后的)
        base_link_left/right: pykin IK/FK 链的根 link 名 (方案 A 推荐 'base_link')
        ee_link_left/right: pykin IK/FK 链的末端 link 名 (推荐 'left-link_arm_7')
        config_path: YAML 配置, 用来集中管理上面这些参数
        robot_wrapper: 直接注入的 wrapper 实例
    """

    def __init__(
        self,
        urdf_path_left: str | None = None,
        urdf_path_right: str | None = None,
        base_link_left: str | None = None,
        base_link_right: str | None = None,
        ee_link_left: str | None = None,
        ee_link_right: str | None = None,
        config_path: str | None = "configs/typhon_am2.yaml",
        robot_wrapper=None,
        left_joint_names: list[str] | None = None,
        right_joint_names: list[str] | None = None,
    ):
        cfg = _load_yaml_config(config_path) if config_path else {}

        def _g(key, default):
            return cfg.get(key, default)

        urdf_left = urdf_path_left or _g(
            "urdf_path_left", "real_world/robot_api/assets/AM2_left.urdf"
        )
        urdf_right = urdf_path_right or _g(
            "urdf_path_right", "real_world/robot_api/assets/AM2_right.urdf"
        )
        base_l = base_link_left or _g("base_link_left", "base_link")
        base_r = base_link_right or _g("base_link_right", "base_link")
        ee_l = ee_link_left or _g("ee_link_left", "left-link_arm_7")
        ee_r = ee_link_right or _g("ee_link_right", "right-link_arm_7")

        self._joint_layout = {
            "left_arm": left_joint_names if left_joint_names is not None else _g(
                "left_joint_names", [f"left-joint_arm_{i}" for i in range(1, 8)]
            ),
            "right_arm": right_joint_names if right_joint_names is not None else _g(
                "right_joint_names", [f"right-joint_arm_{i}" for i in range(1, 8)]
            ),
        }
        expected = 7
        for arm_name, joint_names_for_arm in self._joint_layout.items():
            actual = len(joint_names_for_arm)
            unique = len(set(joint_names_for_arm))
            if actual != expected or unique != expected:
                raise RuntimeError(
                    f"{arm_name} joint layout invalid: "
                    f"expected={expected}, actual={actual}, unique={unique}"
                )

        if robot_wrapper is not None:
            self.robot = robot_wrapper
        else:
            typhon_cfg = _g("typhon", {})
            self.robot = RobotWrapperTyphon(
                base_url=typhon_cfg.get("base_url", "http://192.168.100.100:8081"),
                timeout=typhon_cfg.get("timeout", 5.0),
                auto_enter_control_mode=typhon_cfg.get("auto_enter_control_mode", True),
            )

        # ── 命令缓冲 ──
        self.action_target = dict(
            left_arm=None, right_arm=None,
            left_gripper=None, right_gripper=None,
        )

        # ── pykin URDF 准备 ──
        urdf_l_pykin, urdf_l_tmp = prepare_urdf_for_pykin(_resolve_repo_path(urdf_left))
        urdf_r_pykin, urdf_r_tmp = prepare_urdf_for_pykin(_resolve_repo_path(urdf_right))
        self._temp_urdf_files = [urdf_l_tmp, urdf_r_tmp]

        # ── pykin SingleArm (方案 A: 不传 ab2rb 偏移) ──
        # 单臂 URDF 已经从 base_link 开始, JOINT_L00/R00 在 URDF 内部,
        # pykin 加载后从 base_link 走到配置的 arm_7 末端, 自动包含所有偏移.
        # SingleArm 第二个参数留单位变换, 表示 "URDF 根 = world 原点".
        self.kin_left  = SingleArm(urdf_l_pykin)
        self.kin_right = SingleArm(urdf_r_pykin)

        self.kin_left.setup_link_name(base_l, ee_l)
        self.kin_right.setup_link_name(base_r, ee_r)

        # ── 关节限位 (从 URDF 抽取, 用于 set_target_JP 时 clip 防止超限) ──
        self._joint_limits = {
            "left_arm":  self._extract_joint_limits(_resolve_repo_path(urdf_left),  self._joint_layout["left_arm"]),
            "right_arm": self._extract_joint_limits(_resolve_repo_path(urdf_right), self._joint_layout["right_arm"]),
        }

        self._joint_shape_logged = False
        print(f"[RobotControl] initialized "
              f"(backend=typhon, base_link={base_l}, ee={ee_l})")

    # ============================================================
    # URDF 限位抽取
    # ============================================================
    @staticmethod
    def _extract_joint_limits(
        urdf_path: str,
        joint_names: list[str],
        require_all: bool = True,
    ) -> list[tuple[float, float]]:
        fallback = [(-np.inf, np.inf) for _ in joint_names]
        try:
            tree = ET.parse(urdf_path)
            root = tree.getroot()
        except Exception as e:
            message = f"failed to parse URDF {urdf_path}: {e}"
            if require_all:
                raise RuntimeError(message) from e
            print(f"[WARN] {message}; using unbounded limits")
            return fallback

        joints = {
            joint.attrib["name"]: joint
            for joint in root.findall("joint")
            if joint.attrib.get("name")
        }
        errors = []
        limits = []
        for joint_name in joint_names:
            joint = joints.get(joint_name)
            limit = joint.find("limit") if joint is not None else None
            lower = limit.attrib.get("lower") if limit is not None else None
            upper = limit.attrib.get("upper") if limit is not None else None

            if joint is None:
                errors.append(f"{joint_name}: joint is missing")
                limits.append((-np.inf, np.inf))
                continue
            if lower is None or upper is None:
                errors.append(f"{joint_name}: lower or upper limit is missing")
                limits.append((-np.inf, np.inf))
                continue
            try:
                bounds = (float(lower), float(upper))
            except ValueError:
                errors.append(f"{joint_name}: limit is not numeric ({lower}, {upper})")
                limits.append((-np.inf, np.inf))
                continue
            if not all(np.isfinite(bound) for bound in bounds):
                errors.append(f"{joint_name}: limit is non-finite {bounds}")
                limits.append((-np.inf, np.inf))
                continue
            if bounds[0] >= bounds[1]:
                errors.append(f"{joint_name}: limits are not ordered {bounds}")
                limits.append((-np.inf, np.inf))
                continue
            limits.append(bounds)

        if errors:
            message = f"incomplete joint limits in {urdf_path}: {'; '.join(errors)}"
            if require_all:
                raise RuntimeError(message)
            print(f"[WARN] {message}; using unbounded limits where needed")
        return limits

    def sanitize_joint_targets(self, arm_name: str, joints, strategy: str = "clip"):
        """clip 关节目标到限位内. strategy='skip' 时超限返回 None."""
        arr = np.asarray(joints, dtype=float).copy()
        limits = self._joint_limits.get(arm_name, [])
        if len(arr) != len(limits):
            raise RuntimeError(
                f"{arm_name} joint target/limit dimension mismatch: "
                f"target={len(arr)}, limits={len(limits)}"
            )

        exceeded = False
        for i, (lo, hi) in enumerate(limits):
            if np.isfinite(lo) and arr[i] < lo:
                exceeded = True
                if strategy == "clip":
                    print(f"[WARN] {arm_name}[{i}] {arr[i]:.4f} < {lo:.4f}, clipped")
                    arr[i] = lo
            if np.isfinite(hi) and arr[i] > hi:
                exceeded = True
                if strategy == "clip":
                    print(f"[WARN] {arm_name}[{i}] {arr[i]:.4f} > {hi:.4f}, clipped")
                    arr[i] = hi

        if exceeded and strategy == "skip":
            return None
        return arr.tolist()

    # ============================================================
    # 主接口
    # ============================================================
    def get_robot_joints(self) -> dict[str, list[float]]:
        out = {
            "left_arm":      np.asarray(self.robot.get_joint_angle("left_arm"),     dtype=float).tolist(),
            "right_arm":     np.asarray(self.robot.get_joint_angle("right_arm"),    dtype=float).tolist(),
            "left_gripper":  np.asarray(self.robot.get_joint_angle("left_gripper"), dtype=float).tolist(),
            "right_gripper": np.asarray(self.robot.get_joint_angle("right_gripper"),dtype=float).tolist(),
        }
        if not self._joint_shape_logged:
            print("[INFO] get_robot_joints structure (logged once):")
            for k, v in out.items():
                print(f"  {k}: len={len(v)}")
            self._joint_shape_logged = True
        return out

    def get_ee_pose(self) -> dict[str, np.ndarray]:
        """
        返回左右臂 ee 在 base 系下的位姿.

        方案 A 下 pykin FK 链从 base_link 开始, JOINT_L00/R00 包含在内,
        compute_eef_pose 输出就是 ^rbT_eef, 直接返回, 不做任何额外变换.

        格式: 7-vector [x, y, z, qw, qx, qy, qz]  (scalar-first 四元数)
        """
        joints = self.get_robot_joints()

        fk_l = self.kin_left.forward_kin(np.asarray(joints["left_arm"],  dtype=float))
        fk_r = self.kin_right.forward_kin(np.asarray(joints["right_arm"], dtype=float))

        ee_l = self.kin_left.compute_eef_pose(fk_l)
        ee_r = self.kin_right.compute_eef_pose(fk_r)

        return {
            "left_arm_ee2rb":  np.asarray(ee_l, dtype=float),
            "right_arm_ee2rb": np.asarray(ee_r, dtype=float),
            "left_gripper":  joints["left_gripper"],
            "right_gripper": joints["right_gripper"],
        }

    def set_target_JP(
        self,
        joint_left: np.ndarray,
        joint_right: np.ndarray = None,
        gripper_left: np.ndarray = None,
        gripper_right: np.ndarray = None,
    ):
        """关节空间目标设置. 自动 clip 到关节限位."""
        safe_l = self.sanitize_joint_targets("left_arm", joint_left, "clip")
        self.action_target["left_arm"] = safe_l if safe_l is not None else list(joint_left)
        self.action_target["left_gripper"] = gripper_left

        if joint_right is not None:
            safe_r = self.sanitize_joint_targets("right_arm", joint_right, "clip")
            self.action_target["right_arm"] = safe_r if safe_r is not None else list(joint_right)
        if gripper_right is not None:
            self.action_target["right_gripper"] = gripper_right

    def set_target_CP(self, target_pose: dict, single_arm_mode: bool = False):
        """
        笛卡尔空间目标设置.

        target_pose 里的 *_ee2rb 都是 ^rbT_eef (与 get_ee_pose 输出格式一致).
        方案 A 下 pykin IK 期望的也是 ^rbT_eef, 直接传入.
        """
        joints = self.get_robot_joints()

        # 左臂
        target_l = target_pose["left_arm_ee2rb"]
        joints_l = self._inverse_kin_silent(self.kin_left, joints["left_arm"], target_l)

        if single_arm_mode:
            self.action_target["left_arm"] = joints_l
            self.action_target["left_gripper"] = target_pose.get("left_gripper")
        else:
            target_r = target_pose["right_arm_ee2rb"]
            joints_r = self._inverse_kin_silent(self.kin_right, joints["right_arm"], target_r)
            self.set_target_JP(
                joints_l, joints_r,
                target_pose.get("left_gripper"),
                target_pose.get("right_gripper"),
            )

    @staticmethod
    def _inverse_kin_silent(kin: SingleArm, current_joints, target):
        """静默版 IK, 屏蔽 pykin 的迭代日志."""
        with contextlib.redirect_stdout(io.StringIO()):
            return kin.inverse_kin(np.asarray(current_joints, dtype=float),
                                   target, method="LM", max_iter=100)

    def execute(self):
        for name, joints in self.action_target.items():
            if joints is not None:
                self.robot.set_joint_angle(name, joints)

    def stop(self):
        if hasattr(self.robot, "_robot") and hasattr(self.robot._robot, "shutdown"):
            self.robot._robot.shutdown()
        self._cleanup_temp_files()

    def _cleanup_temp_files(self):
        for f in getattr(self, "_temp_urdf_files", []):
            try:
                if os.path.exists(f):
                    os.remove(f)
            except Exception:
                pass

    def __del__(self):
        self._cleanup_temp_files()


# ============================================================
# 自测
# ============================================================
if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    rc = RobotControl()

    print("\n=== Joints ===")
    js = rc.get_robot_joints()
    for k, v in js.items():
        print(f"  {k}: {np.round(v, 4)}")

    print("\n=== EE Pose ===")
    ee = rc.get_ee_pose()
    for k in ["left_arm_ee2rb", "right_arm_ee2rb"]:
        v = ee[k]
        print(f"  {k}:")
        print(f"    pos:  {np.round(v[:3], 4)}")
        print(f"    quat: {np.round(v[3:], 4)} (wxyz)")
        print(f"    ||quat||: {np.linalg.norm(v[3:]):.6f}")
        print(f"    distance from base: {np.linalg.norm(v[:3]):.4f} m")

    print("\n=== IK ↔ FK self-consistency ===")
    target = ee["left_arm_ee2rb"].copy()
    js_before = np.asarray(js["left_arm"], dtype=float)

    rc.set_target_CP({
        "left_arm_ee2rb":  target,
        "right_arm_ee2rb": ee["right_arm_ee2rb"],
        "left_gripper":  js["left_gripper"],
        "right_gripper": js["right_gripper"],
    })
    js_solved = np.asarray(rc.action_target["left_arm"], dtype=float)
    diff = js_solved - js_before

    print(f"  current joints: {np.round(js_before, 4)}")
    print(f"  IK solved:      {np.round(js_solved, 4)}")
    print(f"  max abs diff:   {np.max(np.abs(diff)):.6f} rad")

    if np.max(np.abs(diff)) < 0.001:
        print("  ✓ IK ↔ FK 自洽")
    else:
        print("  ✗ FAILED — IK 和 FK 坐标系约定不一致")
