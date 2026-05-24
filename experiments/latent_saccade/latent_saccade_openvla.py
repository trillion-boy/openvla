"""
latent_saccade_openvla.py

OpenVLA용 Latent Saccade (Post-RMSNorm variant).

UniVLA postnorm 버전과 메커니즘 동일:
  token IDs → embed_tokens → RMSNorm × weight → Q,K,V
                                        ↑ weight survives into attention

UniVLA와의 핵심 차이
--------------------
UniVLA (Emu3):
  visual token = 이산 ID (vis_start ≤ id ≤ vis_end)
  위치 식별:  input_ids 값을 스캔해서 visual 위치 찾음
  레이어 경로: model.model.layers

OpenVLA (LLaMA):
  visual token = 연속 임베딩 (projector 출력)
  위치 식별:  항상 positions [1, 1+num_patches) — ID 스캔 불필요
  레이어 경로: model.llm_backbone.llm.model.layers
  시퀀스 구조: [BOS(pos 0)] [patch_0..patch_255(pos 1..256)] [text_1..text_N(pos 257..)]

변경된 부분
-----------
  _find_decoder_layers : 레이어 경로 변경
  _build_seq_weight    : ID 스캔 → positional slice [1, 1+num_patches)
  __init__             : Emu3 파라미터 제거, OpenVLA 모델 래핑
  step()               : Emu3 파이프라인 → OpenVLA predict_action() 파이프라인

변경되지 않은 부분
-----------------
  hook 핸들러 함수 (output * w.view(1, seq_len, 1))
  _register_postnorm_hooks 구조
  _current_seq_weight 메커니즘
  DINO 검출 및 _build_weight_map
  Saccade state machine

Recommended weights (postnorm과 동일):
  bg_weight=1.0, place_src_weight=1.1, fovea_weight=1.3

Usage
-----
  model = load_openvla(...)           # OpenVLA 모델 로드
  saccade = LatentSaccadeOpenVLAInference(
      model=model, unnorm_key="bridge_orig",
      bg_weight=1.0, place_src_weight=1.1, fovea_weight=1.3,
  )
  saccade.reset()
  action = saccade.step(image_np, instruction)   # np.ndarray (7,)
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image as PIL_Image


# ---------------------------------------------------------------------------
# Saccade State Machine
# ---------------------------------------------------------------------------

class SaccadeStateMachine:
    """
    Grasp / Place 2-phase state machine.

    state="grasp"  → fovea on source object
    state="place"  → fovea on destination object
    Transition: gripper close count ≥ consecutive_close_required
                AND grasp_steps ≥ min_grasp_steps
    """

    def __init__(
        self,
        min_grasp_steps: int = 15,
        consecutive_close_required: int = 3,
        min_place_steps: int = 8,
        max_grasp_steps: int = 60,
    ):
        self.min_grasp_steps = min_grasp_steps
        self.consecutive_close_required = consecutive_close_required
        self.min_place_steps = min_place_steps
        self.max_grasp_steps = max_grasp_steps

        self.source_noun: str = ""
        self.dest_noun: str = ""
        self.state: str = "grasp"
        self._close_count: int = 0
        self._grasp_steps: int = 0

    @property
    def current_target(self) -> str:
        return self.source_noun if self.state == "grasp" else self.dest_noun

    def update(self, gripper_norm: float) -> bool:
        """
        Update state from gripper value.
        gripper_norm: 0.0=open, 1.0=closed  (computed as (1-g)/2 from raw action)
        Returns True if state just transitioned grasp→place.

        OpenVLA bridge_orig gripper: g=1.0=open, g=0.0=close
        → gripper_norm = (1-0)/2 = 0.5  so threshold is >= 0.5 (not >)
        """
        if self.state == "grasp":
            self._grasp_steps += 1
            if gripper_norm >= 0.5:   # >= catches g=0.0 → gripper_norm=0.5
                self._close_count += 1
            else:
                self._close_count = 0

            # Normal transition: gripper held closed long enough
            if (
                self._grasp_steps >= self.min_grasp_steps
                and self._close_count >= self.consecutive_close_required
            ):
                self.state = "place"
                print(
                    f"[LatentSaccade] grasp→place  (gripper_close trigger, "
                    f"steps={self._grasp_steps})",
                    flush=True,
                )
                return True

            # Timeout: force place phase so episode doesn't stall forever
            if self.max_grasp_steps > 0 and self._grasp_steps >= self.max_grasp_steps:
                self.state = "place"
                print(
                    f"[LatentSaccade] grasp→place  (timeout at {self._grasp_steps} steps, "
                    f"close_count={self._close_count})",
                    flush=True,
                )
                return True
        return False

    def reset(self):
        self.state = "grasp"
        self._close_count = 0
        self._grasp_steps = 0


# ---------------------------------------------------------------------------
# GroundingDINO Detector
# ---------------------------------------------------------------------------

class GroundingDINODetector:
    """
    GroundingDINO wrapper using HuggingFace transformers.
    Requires: transformers >= 4.38, groundingdino weights accessible via HF Hub.
    """

    def __init__(
        self,
        model_id: str = "IDEA-Research/grounding-dino-tiny",
        box_threshold: float = 0.15,
        text_threshold: float = 0.15,
        device: str = "cuda",
    ):
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
        self.model.eval()
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.device = device

    def detect(
        self, image_np: np.ndarray, text: str
    ) -> List[Tuple[np.ndarray, float]]:
        """
        Returns [(bbox_xyxy_pixels, score), ...] sorted by score descending.
        bbox_xyxy_pixels: [x1, y1, x2, y2] in original image pixel coordinates.
        text should end with '.' for GroundingDINO.
        """
        if not text:
            return []
        if not text.endswith("."):
            text = text + "."
        pil_image = PIL_Image.fromarray(image_np)
        inputs = self.processor(
            images=pil_image, text=text, return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[pil_image.size[::-1]],
        )[0]
        boxes = results["boxes"].cpu().numpy()
        scores = results["scores"].cpu().numpy()
        detections = sorted(zip(boxes, scores), key=lambda x: -x[1])
        return detections

    @staticmethod
    def extract_source_dest_nouns(instruction: str) -> Tuple[str, str]:
        """
        Regex-based extraction for standard manipulation instructions.
        e.g. "put the spoon on the towel" → ("spoon", "towel")
             "stack the green block on the yellow block" → ("green block", "yellow block")
             "put the eggplant in the basket" → ("eggplant", "basket")
        """
        inst = instruction.lower().strip()
        pattern = (
            r"(?:put|place|move|stack|push)\s+"
            r"(?:the\s+)?(.+?)\s+"
            r"(?:on(?:\s+top\s+of)?|in(?:to)?|onto|inside)\s+"
            r"(?:the\s+)?(.+)"
        )
        m = re.match(pattern, inst)
        if m:
            src = m.group(1).strip().rstrip(".,")
            dst = m.group(2).strip().rstrip(".,")
            return src, dst
        return "", ""


# ---------------------------------------------------------------------------
# Main inference class
# ---------------------------------------------------------------------------

class LatentSaccadeOpenVLAInference:
    """
    Latent Saccade for OpenVLA (post-RMSNorm variant).

    Mechanism (identical to UniVLA postnorm):
      Registers persistent forward hooks on input_layernorm of every LLaMA
      decoder layer.  During the prefill forward pass (seq_len > 1), the hook
      multiplies hidden_states by a (seq_len,) weight tensor.

      Weight tensor:
        pos 0             (BOS)       → 1.0
        pos 1..num_patches (visual)   → spatial weight from DINO detection
        pos num_patches+1.. (text)    → 1.0

    Visual position identification:
      UniVLA: scans input_ids for vis_start ≤ id ≤ vis_end
      OpenVLA: fixed positional slice [1, 1+num_patches)  ← key change
    """

    def __init__(
        self,
        model,                                   # loaded OpenVLA (OpenVLA instance)
        processor=None,                          # HF AutoProcessor (required for HF-loaded models)
        unnorm_key: Optional[str] = None,
        device: str = "cuda",
        dino_model: str = "IDEA-Research/grounding-dino-tiny",
        dino_cache_steps: int = 5,
        box_threshold: float = 0.15,
        text_threshold: float = 0.15,
        bbox_margin: int = 2,
        bg_weight: float = 1.0,
        place_src_weight: float = 1.1,
        fovea_weight: float = 1.3,
        min_grasp_steps: int = 15,
        consecutive_close_required: int = 3,
        min_place_steps: int = 8,
        max_grasp_steps: int = 60,
        enable_latent_mask: bool = True,
        dino_debug_dir: Optional[str] = None,
    ):
        self.model = model
        self.processor = processor
        self.device = device
        self._unnorm_key = unnorm_key
        self._bg_weight = bg_weight
        self._place_src_weight = place_src_weight
        self._fovea_weight = fovea_weight
        self._enable_latent_mask = enable_latent_mask
        self._dino_cache_steps = dino_cache_steps
        self._bbox_margin = bbox_margin
        self._dino_debug_dir = dino_debug_dir

        # ── Visual patch config ───────────────────────────────────────────
        # num_patches: native Prismatic has .num_patches property;
        # HF-loaded PrismaticVisionBackbone exposes it via featurizer.patch_embed.num_patches
        vb = model.vision_backbone
        if hasattr(vb, "num_patches"):
            self.num_patches: int = vb.num_patches
        elif hasattr(vb, "featurizer") and hasattr(vb.featurizer, "patch_embed"):
            self.num_patches: int = vb.featurizer.patch_embed.num_patches
        else:
            self.num_patches: int = 256  # openvla-7b default (224px / patch14 → 16×16)
        self._grid_size: int = int(round(self.num_patches ** 0.5))
        assert self._grid_size ** 2 == self.num_patches, (
            f"num_patches={self.num_patches} is not a perfect square; "
            "update _build_weight_map if using non-square patch grids."
        )

        # ── Saccade state machine ─────────────────────────────────────────
        self.saccade = SaccadeStateMachine(
            min_grasp_steps=min_grasp_steps,
            consecutive_close_required=consecutive_close_required,
            min_place_steps=min_place_steps,
            max_grasp_steps=max_grasp_steps,
        )

        # ── GroundingDINO detector ────────────────────────────────────────
        self.detector = GroundingDINODetector(
            model_id=dino_model,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=device,
        )

        # ── Internal state ────────────────────────────────────────────────
        self._current_instruction: Optional[str] = None
        # weight_1d: (num_patches,) spatial weights, set before each generate()
        # Hook builds the full (seq_len,) weight tensor on-the-fly from this.
        # This removes the need to compute text length / call get_prompt_builder().
        self._current_weight_1d: Optional[torch.Tensor] = None
        self._ln_hook_handles: List = []
        self._fovea_bbox_cache = None
        self._secondary_bbox_cache = None
        self._cache_step: int = 0

        # ── Register post-RMSNorm hooks ───────────────────────────────────
        self._register_postnorm_hooks()

    # ── Layer discovery ───────────────────────────────────────────────────

    def _find_decoder_layers(self):
        """
        OpenVLA layer path (changed from UniVLA):
          UniVLA: model.model.layers  (or model.language_model.model.layers)
          OpenVLA: model.llm_backbone.llm.model.layers
            llm_backbone  : LLaMa2LLMBackbone (HFCausalLLMBackbone)
            llm           : LlamaForCausalLM
            model         : LlamaModel
            layers        : ModuleList[LlamaDecoderLayer]
        """
        candidates = [
            lambda m: m.llm_backbone.llm.model.layers,   # OpenVLA (LLaMA)
            lambda m: m.llm_backbone.model.layers,
            lambda m: m.model.layers,
            lambda m: m.language_model.model.layers,
        ]
        for fn in candidates:
            try:
                layers = fn(self.model)
                if layers is not None and len(layers) > 0:
                    return layers
            except AttributeError:
                continue
        raise RuntimeError(
            "[LatentSaccade] Cannot find decoder layers. "
            "Tried: llm_backbone.llm.model.layers, llm_backbone.model.layers, "
            "model.layers, language_model.model.layers"
        )

    def _find_layernorm(self, layer):
        """
        LlamaDecoderLayer also has input_layernorm — same attribute name as Emu3.
        No change needed vs UniVLA postnorm version.
        """
        for attr in ("input_layernorm", "ln_1", "layer_norm_1", "norm1"):
            if hasattr(layer, attr):
                return getattr(layer, attr)
        raise RuntimeError(
            f"[LatentSaccade] Cannot find input_layernorm in {type(layer).__name__}. "
            f"Norm-like attrs: {[a for a in dir(layer) if 'norm' in a.lower() or 'ln' in a.lower()]}"
        )

    # ── Hook registration ─────────────────────────────────────────────────

    def _register_postnorm_hooks(self):
        """
        Hook logic identical to UniVLA postnorm in structure.

        Key simplification vs UniVLA:
          UniVLA pre-builds a (seq_len,) tensor outside the hook.
          Here the hook builds the weight tensor on-the-fly from _current_weight_1d,
          so we never need to call get_prompt_builder() or measure text length.
          Visual positions are always [1, 1+num_patches) regardless of text length.
        """
        layers = self._find_decoder_layers()

        for layer in layers:
            ln = self._find_layernorm(layer)

            def _make_hook(self_ref):
                def _hook(module, inp, output):
                    if not self_ref._enable_latent_mask:
                        return output
                    if self_ref._current_weight_1d is None:
                        return output
                    # Skip single-token autoregressive steps (KV cache)
                    if output.shape[1] <= 1:
                        return output

                    seq_len = output.shape[1]
                    # Build full weight vector on-the-fly:
                    #   pos 0              : BOS  → 1.0
                    #   pos 1..num_patches : visual patches → spatial weight
                    #   pos num_patches+1..: text → 1.0
                    w = torch.ones(seq_len, dtype=output.dtype, device=output.device)
                    w1d = self_ref._current_weight_1d.to(
                        dtype=output.dtype, device=output.device
                    )
                    n_vis = min(w1d.shape[0], self_ref.num_patches, seq_len - 1)
                    w[1 : 1 + n_vis] = w1d[:n_vis]

                    out = output.clone()
                    out = out * w.view(1, seq_len, 1)
                    return out
                return _hook

            handle = ln.register_forward_hook(_make_hook(self))
            self._ln_hook_handles.append(handle)

        print(
            f"[LatentSaccade] Registered post-RMSNorm hooks on "
            f"{len(self._ln_hook_handles)} LLaMA decoder layers  "
            f"(num_patches={self.num_patches}, grid={self._grid_size}×{self._grid_size})"
        )

    # ── DINO detection ────────────────────────────────────────────────────

    def _get_bboxes(
        self, image: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Returns (fovea_bbox, secondary_bbox) with DINO cache.

        When detection returns None (e.g. robot arm occludes object), keep
        the last valid bbox rather than propagating None through the weight map.
        """
        if self._cache_step % self._dino_cache_steps == 0:
            target = self.saccade.current_target
            secondary = (
                self.saccade.source_noun
                if self.saccade.state == "place"
                else self.saccade.dest_noun
            )

            if target:
                dets = self.detector.detect(image, target)
                if dets:
                    self._fovea_bbox_cache = dets[0][0]   # update only on success
                # else: keep previous valid bbox as fallback

            if secondary and secondary != target:
                dets = self.detector.detect(image, secondary)
                if dets:
                    self._secondary_bbox_cache = dets[0][0]
                # else: keep previous valid bbox as fallback

        self._cache_step += 1
        return self._fovea_bbox_cache, self._secondary_bbox_cache

    # ── Spatial weight map ────────────────────────────────────────────────

    def _build_weight_map(
        self,
        image: np.ndarray,
        fovea_bbox: Optional[np.ndarray],
        secondary_bbox: Optional[np.ndarray],
    ) -> torch.Tensor:
        """
        Build (num_patches,) spatial weight vector.

        Maps image pixel bboxes → ViT patch grid (grid_size × grid_size).
        Identical logic to UniVLA postnorm; grid_size adapts to ViT config.
        """
        H, W = image.shape[:2]
        g = self._grid_size

        grid = torch.full((g, g), self._bg_weight, dtype=torch.float32)

        def _bbox_to_grid(bbox):
            if bbox is None:
                return None
            x1, y1, x2, y2 = bbox
            x1 = max(0.0, x1 - self._bbox_margin)
            y1 = max(0.0, y1 - self._bbox_margin)
            x2 = min(float(W), x2 + self._bbox_margin)
            y2 = min(float(H), y2 + self._bbox_margin)
            c1 = max(0, int(x1 / W * g))
            r1 = max(0, int(y1 / H * g))
            c2 = min(g, int(np.ceil(x2 / W * g)))
            r2 = min(g, int(np.ceil(y2 / H * g)))
            if r2 <= r1 or c2 <= c1:
                return None
            return r1, c1, r2, c2

        sec_region = _bbox_to_grid(secondary_bbox)
        if sec_region:
            r1, c1, r2, c2 = sec_region
            grid[r1:r2, c1:c2] = self._place_src_weight

        fov_region = _bbox_to_grid(fovea_bbox)
        if fov_region:
            r1, c1, r2, c2 = fov_region
            grid[r1:r2, c1:c2] = self._fovea_weight

        if self._dino_debug_dir is not None:
            self._save_debug_image(image, fovea_bbox, secondary_bbox, grid)

        return grid.view(-1)   # (num_patches,)

    # ── step: main inference with latent saccade ──────────────────────────

    def step(self, image: np.ndarray, goal: str) -> np.ndarray:
        """
        Run one inference step.

        Changes vs UniVLA postnorm step():
          - No Emu3 tokenization / VQ image encoding / text length computation
          - Stores weight_1d only; hook builds full weight tensor on-the-fly
          - Uses OpenVLA predict_action() directly

        Returns:
          action (np.ndarray, shape (7,)): unnormalized continuous action
            [dx, dy, dz, drx, dry, drz, gripper]
        """
        # ── 1. Sync instruction → saccade nouns ──────────────────────────
        if goal != self._current_instruction:
            self._current_instruction = goal
            src, dst = GroundingDINODetector.extract_source_dest_nouns(goal)
            self.saccade.source_noun = src
            self.saccade.dest_noun = dst
            print(f"[LatentSaccade] Instruction → src='{src}'  dst='{dst}'")

        # ── 2. DINO detection → spatial weight map ────────────────────────
        fovea_bbox, secondary_bbox = self._get_bboxes(image)
        if self._enable_latent_mask:
            weight_1d = self._build_weight_map(image, fovea_bbox, secondary_bbox)
        else:
            weight_1d = None

        n_fovea = int((weight_1d >= self._fovea_weight).sum()) if weight_1d is not None else 0
        n_src = (
            int(((weight_1d >= self._place_src_weight) & (weight_1d < self._fovea_weight)).sum())
            if weight_1d is not None else 0
        )
        n_bg = int((weight_1d < self._place_src_weight).sum()) if weight_1d is not None else 0
        print(
            f"[LatentSaccade] phase={self.saccade.state}  "
            f"target='{self.saccade.current_target}'  "
            f"fovea={n_fovea}  src={n_src}  bg={n_bg}  "
            f"fovea_bbox={fovea_bbox}"
        )

        # ── 3. Store weight_1d → hooks read it during generate() ─────────
        # (hooks build the full seq weight on-the-fly; no text length needed)
        self._current_weight_1d = weight_1d

        # ── 4. predict_action (hook fires on prefill, skips AR steps) ─────
        pil_image = PIL_Image.fromarray(image)
        try:
            if self.processor is not None:
                # HF-loaded model: processor builds input_ids + pixel_values
                prompt = f"In: What action should the robot take to {goal.lower()}?\nOut:"
                inputs = self.processor(prompt, pil_image, return_tensors="pt")
                inputs = {
                    k: (v.to(self.device, dtype=torch.bfloat16) if torch.is_floating_point(v) else v.to(self.device))
                    for k, v in inputs.items()
                }
                action = self.model.predict_action(
                    **inputs, unnorm_key=self._unnorm_key, do_sample=False
                )
            else:
                # Native Prismatic load: predict_action(image, instruction, unnorm_key)
                action = self.model.predict_action(
                    pil_image, goal, unnorm_key=self._unnorm_key
                )
        finally:
            self._current_weight_1d = None   # always clear after generate

        # ── 5. Update saccade state from gripper output ───────────────────
        # OpenVLA bridge action[-1]: ~+1.0=open, ~-1.0=close
        # gripper_norm: 0.0=open, 1.0=closed  (identical to UniVLA formula)
        g = float(action[-1])
        gripper_norm = (1.0 - g) / 2.0
        print(
            f"[LatentSaccade-dbg] g={g:.2f}  gripper_norm={gripper_norm:.2f}  "
            f"close_count={self.saccade._close_count}  "
            f"grasp_steps={self.saccade._grasp_steps}",
            flush=True,
        )
        transitioned = self.saccade.update(gripper_norm)
        if transitioned:
            self._fovea_bbox_cache = None
            self._secondary_bbox_cache = None
            self._cache_step = 0
            print("[LatentSaccade] State transition: grasp → place", flush=True)

        return action

    # ── Reset / Cleanup ───────────────────────────────────────────────────

    def reset(self):
        """Reset per-episode state (call at episode start)."""
        self.saccade.reset()
        self._current_instruction = None
        self._current_weight_1d = None
        self._fovea_bbox_cache = None
        self._secondary_bbox_cache = None
        self._cache_step = 0

    def __del__(self):
        for handle in getattr(self, "_ln_hook_handles", []):
            handle.remove()

    # ── Debug helpers ─────────────────────────────────────────────────────

    def _save_debug_image(self, image, fovea_bbox, secondary_bbox, grid):
        """Save annotated debug image showing detected bboxes and weight grid."""
        import os
        from PIL import ImageDraw, ImageFont
        os.makedirs(self._dino_debug_dir, exist_ok=True)
        pil = PIL_Image.fromarray(image).copy()
        draw = ImageDraw.Draw(pil)
        if fovea_bbox is not None:
            x1, y1, x2, y2 = fovea_bbox
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            draw.text((x1, y1 - 12), f"fovea ({self._fovea_weight})", fill="red")
        if secondary_bbox is not None:
            x1, y1, x2, y2 = secondary_bbox
            draw.rectangle([x1, y1, x2, y2], outline="blue", width=2)
            draw.text((x1, y1 - 12), f"secondary ({self._place_src_weight})", fill="blue")
        step_idx = self._cache_step
        save_path = os.path.join(self._dino_debug_dir, f"step_{step_idx:05d}.png")
        pil.save(save_path)
