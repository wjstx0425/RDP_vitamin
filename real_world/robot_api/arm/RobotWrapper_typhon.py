"""
RobotWrapper_typhon.py
----------------------
Typhon HTTP API 版本的 RobotWrapper, 用于 AM2 双臂机械臂。

接口与 RobotWrapper.py 一致:
    - get_joint_angle(arm)
    - set_joint_angle(arm, position)
    - get_joint_velo(arm)

arm 字符串: "left_arm" | "right_arm" | "left_gripper" | "right_gripper"

/action 端点 schema (基于 api_demo.py L820-827 确认):
    {
      "nxp_ec_left_arm":  [j1, ..., j7],   # 7 floats, 弧度
      "left_gripper":     [pos],           # 1 float, 米
      "nxp_ec_right_arm": [j1, ..., j7],
      "right_gripper":    [pos],
    }

依赖: pip install requests numpy
"""

from __future__ import annotations

import time
import requests
import numpy as np
from typing import Optional


# ─────────────── 硬件名映射 ───────────────
# 上层用 "left_arm"，HTTP API 用 "nxp_ec_left_arm"
ARM_KEY_MAP = {
    "left_arm":      "nxp_ec_left_arm",
    "right_arm":     "nxp_ec_right_arm",
    "left_gripper":  "left_gripper",
    "right_gripper": "right_gripper",
}

# 反向映射，方便在错误信息里把硬件名翻译回来
HW_TO_USER = {v: k for k, v in ARM_KEY_MAP.items()}


class RobotWrapperTyphon:
    """通过 HTTP 与 typhon_backend 通信的 RobotWrapper。"""

    def __init__(
        self,
        base_url: str = "http://192.168.100.100:8081",
        timeout: float = 5.0,
        auto_enter_control_mode: bool = True,
    ):
        """
        Args:
            base_url: typhon_backend 的 HTTP 地址
            timeout: 单次请求超时(秒). 5s 是经验值, 高负载下偶尔会有
                     1-3s 的尾延迟, 给一些余量.
            auto_enter_control_mode: 初始化时自动进入 DUAL_ARM_API_CONTROL.
                     必须进入此模式 /action 才能生效.
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

        # 缓存最近一次发出去的 action body, 因为 /action 必须四个 key 同时发,
        # 上层一次只调用一个 set_joint_angle 时, 用缓存补全其他三个
        self._last_action: dict = {}

        # _robot 这个属性是为了兼容上层调用链:
        #     self.robot._robot.shutdown()
        self._robot = _ShutdownProxy(self)

        if auto_enter_control_mode:
            self.ensure_action_ready()

    # ─────────────── HTTP helpers ───────────────

    def _get(self, path: str, **kwargs):
        """GET 请求, 返回 JSON dict 或 None"""
        try:
            r = requests.get(f"{self.base_url}{path}",
                             timeout=self.timeout, **kwargs)
            r.raise_for_status()
            return r.json() if r.content else {}
        except Exception as e:
            raise RuntimeError(f"Typhon GET {path} failed: {e}") from e

    def _post(self, path: str, body: dict):
        """POST JSON body, 返回 response 或抛 RuntimeError"""
        try:
            r = requests.post(f"{self.base_url}{path}",
                              json=body, timeout=self.timeout)
            r.raise_for_status()
            return r
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else "?"
            text = ""
            if e.response is not None:
                text = (e.response.text or "").strip()
                if len(text) > 1500:
                    text = text[:1500] + "...<truncated>"
            raise RuntimeError(
                f"Typhon POST {path} failed:\n"
                f"  status_code = {status}\n"
                f"  request_body = {body}\n"
                f"  response = {text or '<empty>'}"
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"Typhon POST {path} failed:\n"
                f"  request_body = {body}\n"
                f"  error = {e}"
            ) from e

    # ─────────────── State machine ───────────────

    def get_state(self) -> Optional[str]:
        """读 backend 的 FSM state. 失败返回 None."""
        try:
            data = self._get("/state")
            if isinstance(data, dict):
                return data.get("state")
        except Exception:
            return None
        return None

    def ensure_action_ready(
        self,
        target_state: str = "DUAL_ARM_API_CONTROL",
        switch_command: str = "state_enter_dual_arm_api_control",
        max_wait_s: float = 10.0,
        poll_interval_s: float = 0.2,
    ) -> bool:
        """确保 backend 在能接受 /action 的状态. 不在则切过去."""
        state = self.get_state()
        if state == target_state:
            return True

        print(f"[RobotWrapperTyphon] switching {state} -> {target_state}")
        self._post("/command", {switch_command: ""})

        t0 = time.time()
        while time.time() - t0 < max_wait_s:
            state = self.get_state()
            if state == target_state:
                print(f"[RobotWrapperTyphon] now in {target_state}")
                return True
            time.sleep(poll_interval_s)

        raise RuntimeError(
            f"Failed to enter {target_state} within {max_wait_s}s "
            f"(last state: {state})"
        )

    def enter_standby(self):
        """退出回 STANDBY"""
        try:
            self._post("/command", {"state_enter_standby": ""})
        except Exception as e:
            print(f"[RobotWrapperTyphon] enter_standby skipped: {e}")

    # ─────────────── 兼容旧接口 ───────────────

    def get_joint_angle(self, arm: str) -> np.ndarray:
        """
        读取关节位置.
        - arm 是手臂时返回 7 个 float (弧度)
        - arm 是夹爪时返回 1 个 float (米)
        """
        if arm not in ARM_KEY_MAP:
            raise ValueError(f"Unknown arm: {arm}. Valid: {list(ARM_KEY_MAP)}")
        hw = ARM_KEY_MAP[arm]

        data = self._get("/state", params={"keys": [hw]})
        if hw not in data:
            raise RuntimeError(
                f"{hw} missing from /state response. "
                f"Available: {list(data.keys())}"
            )
        if "position" not in data[hw]:
            raise RuntimeError(f"No 'position' field for {hw}")

        return np.asarray(data[hw]["position"], dtype=np.float64)

    def set_joint_angle(self, arm: str, position: np.ndarray) -> None:
        """
        设置关节目标位置.

        注意: typhon /action 必须四个 key 同时发出去. 上层只调一个 set_joint_angle
        时, 这里会用 _last_action 缓存 + 当前真实状态 补全其他三个 key.
        """
        if arm not in ARM_KEY_MAP:
            raise ValueError(f"Unknown arm: {arm}. Valid: {list(ARM_KEY_MAP)}")
        hw = ARM_KEY_MAP[arm]

        pos = np.asarray(position, dtype=np.float64).flatten().tolist()
        self._last_action[hw] = pos

        body = self._build_action_body()
        self._post("/action", body)

    def get_joint_velo(self, arm: str) -> np.ndarray:
        """读取关节速度. backend 不支持时返回零."""
        if arm not in ARM_KEY_MAP:
            raise ValueError(f"Unknown arm: {arm}")
        hw = ARM_KEY_MAP[arm]

        try:
            data = self._get("/state", params={"keys": [hw]})
            if hw in data and "velocity" in data[hw]:
                return np.asarray(data[hw]["velocity"], dtype=np.float64)
        except Exception:
            pass

        # fallback: 用 position 的 shape 返回零
        try:
            angles = self.get_joint_angle(arm)
            return np.zeros_like(angles)
        except Exception:
            # 终极 fallback
            n = 7 if "arm" in arm else 1
            return np.zeros(n, dtype=np.float64)

    # ─────────────── 内部 helpers ───────────────

    def _build_action_body(self) -> dict:
        """
        构建 /action 请求 body.

        必须四个 key 同时存在, 缺的从 _last_action 拿, 还缺就从当前 /state 补.
        """
        body = {}
        missing = []
        for hw in ARM_KEY_MAP.values():
            if hw in self._last_action:
                body[hw] = self._last_action[hw]
            else:
                missing.append(hw)

        if missing:
            try:
                state = self._get("/state", params={"keys": missing})
                for hw in missing:
                    if hw in state and "position" in state[hw]:
                        body[hw] = state[hw]["position"]
                        # 把它也放进 _last_action, 下次就不用再查了
                        self._last_action[hw] = state[hw]["position"]
            except Exception as e:
                raise RuntimeError(
                    f"Cannot build /action body: missing keys {missing} "
                    f"and failed to fetch from /state: {e}"
                )

        # 最后再校验一次
        for hw in ARM_KEY_MAP.values():
            if hw not in body:
                raise RuntimeError(
                    f"Failed to populate {hw} in /action body. "
                    f"Body so far: {list(body.keys())}"
                )

        return body


class _ShutdownProxy:
    """让 wrapper.robot._robot.shutdown() 这种调用链不报错。"""
    def __init__(self, wrapper: RobotWrapperTyphon):
        self._wrapper = wrapper

    def shutdown(self):
        print("[RobotWrapperTyphon] shutdown -> entering STANDBY")
        self._wrapper.enter_standby()


# ─────────────── 自测 ───────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_url", default="http://192.168.100.100:8081")
    parser.add_argument("--readonly", action="store_true",
                        help="只读测试, 不切到 DUAL_ARM_API_CONTROL")
    args = parser.parse_args()

    rw = RobotWrapperTyphon(
        base_url=args.base_url,
        auto_enter_control_mode=not args.readonly,
    )

    print(f"\nState: {rw.get_state()}")

    print("\n=== get_joint_angle ===")
    for arm in ARM_KEY_MAP:
        try:
            angles = rw.get_joint_angle(arm)
            print(f"  {arm:15s}: {np.round(angles, 4)}")
        except Exception as e:
            print(f"  {arm:15s}: ERROR - {e}")