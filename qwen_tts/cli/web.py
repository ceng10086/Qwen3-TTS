# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
"""
Unified local Web UI for Qwen3-TTS.

- One page supports CustomVoice / VoiceDesign / Base (voice clone)
- Supports switching checkpoints at runtime (load / unload)
"""

import argparse
import gc
import inspect
import os
import tempfile
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import numpy as np
import soundfile as sf
import torch

from .. import Qwen3TTSModel, VoiceClonePromptItem


def _title_case_display(s: str) -> str:
    s = (s or "").strip()
    s = s.replace("_", " ")
    return " ".join([w[:1].upper() + w[1:] if w else "" for w in s.split()])


def _build_choices_and_map(items: Optional[List[str]]) -> Tuple[List[str], Dict[str, str]]:
    if not items:
        return [], {}
    display = [_title_case_display(x) for x in items]
    mapping = {d: r for d, r in zip(display, items)}
    return display, mapping


def _dtype_from_str(s: str) -> torch.dtype:
    s = (s or "").strip().lower()
    if s in ("bf16", "bfloat16"):
        return torch.bfloat16
    if s in ("fp16", "float16", "half"):
        return torch.float16
    if s in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unsupported torch dtype: {s}. Use bfloat16/float16/float32.")


def _normalize_audio(wav, eps=1e-12, clip=True):
    x = np.asarray(wav)

    if np.issubdtype(x.dtype, np.integer):
        info = np.iinfo(x.dtype)
        if info.min < 0:
            y = x.astype(np.float32) / max(abs(info.min), info.max)
        else:
            mid = (info.max + 1) / 2.0
            y = (x.astype(np.float32) - mid) / mid
    elif np.issubdtype(x.dtype, np.floating):
        y = x.astype(np.float32)
        m = np.max(np.abs(y)) if y.size else 0.0
        if m > 1.0 + 1e-6:
            y = y / (m + eps)
    else:
        raise TypeError(f"Unsupported dtype: {x.dtype}")

    if clip:
        y = np.clip(y, -1.0, 1.0)

    if y.ndim > 1:
        y = np.mean(y, axis=-1).astype(np.float32)
    return y


def _audio_to_tuple(audio: Any) -> Optional[Tuple[np.ndarray, int]]:
    if audio is None:
        return None

    if isinstance(audio, tuple) and len(audio) == 2 and isinstance(audio[0], int):
        sr, wav = audio
        wav = _normalize_audio(wav)
        return wav, int(sr)

    if isinstance(audio, dict) and "sampling_rate" in audio and "data" in audio:
        sr = int(audio["sampling_rate"])
        wav = _normalize_audio(audio["data"])
        return wav, sr

    return None


def _wav_to_gradio_audio(wav: np.ndarray, sr: int) -> Tuple[int, np.ndarray]:
    wav = _normalize_audio(wav, clip=True)
    wav_i16 = (wav * 32767.0).round().astype(np.int16)
    return sr, wav_i16


def _detect_model_kind(tts: Qwen3TTSModel) -> str:
    mt = getattr(tts.model, "tts_model_type", None)
    if mt in ("custom_voice", "voice_design", "base"):
        return mt
    raise ValueError(f"Unknown Qwen-TTS model type: {mt}")


def _mkdirp(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p


def _data_dir(default: str) -> str:
    return os.environ.get("QWEN_TTS_DATA_DIR", default)


@dataclass
class LoadedModel:
    ckpt: str
    device: str
    dtype: str
    flash_attn: bool
    kind: str
    lang_choices_disp: List[str]
    lang_map: Dict[str, str]
    spk_choices_disp: List[str]
    spk_map: Dict[str, str]
    tts: Qwen3TTSModel


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="qwen-tts-web",
        description="Launch unified Qwen3-TTS Web UI (with model switching).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--ip", default="0.0.0.0", help="Server bind IP for Gradio.")
    p.add_argument("--port", type=int, default=8000, help="Server port for Gradio.")
    p.add_argument("--share", action="store_true", help="Create a public Gradio link.")
    p.add_argument("--concurrency", type=int, default=8, help="Gradio queue concurrency.")

    p.add_argument("--ssl-certfile", default=None, help="Path to SSL certificate file for HTTPS (optional).")
    p.add_argument("--ssl-keyfile", default=None, help="Path to SSL key file for HTTPS (optional).")
    p.add_argument("--no-ssl-verify", dest="ssl_verify", action="store_false", help="Disable SSL verify (self-signed).")
    p.set_defaults(ssl_verify=True)

    p.add_argument("--device", default=os.environ.get("QWEN_TTS_DEVICE", "cpu"), help="Default device (cpu/cuda:0).")
    p.add_argument(
        "--dtype",
        default=os.environ.get("QWEN_TTS_DTYPE", "float32"),
        choices=["bfloat16", "bf16", "float16", "fp16", "float32", "fp32"],
        help="Default torch dtype when loading.",
    )
    p.add_argument(
        "--flash-attn/--no-flash-attn",
        dest="flash_attn",
        default=(os.environ.get("QWEN_TTS_FLASH_ATTN", "0").strip() not in ("0", "false", "no")),
        action=argparse.BooleanOptionalAction,
        help="Enable FlashAttention-2 when loading (GPU only).",
    )

    p.add_argument(
        "--default-model",
        default=os.environ.get("QWEN_TTS_DEFAULT_MODEL", "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"),
        help="Default model id/path pre-filled in the UI.",
    )
    p.add_argument(
        "--models",
        default=os.environ.get(
            "QWEN_TTS_MODELS",
            ",".join(
                [
                    "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
                    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
                    "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
                    "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
                    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
                ]
            ),
        ),
        help="Comma-separated model ids/paths for dropdown.",
    )
    p.add_argument(
        "--autoload/--no-autoload",
        dest="autoload",
        default=(os.environ.get("QWEN_TTS_AUTOLOAD", "1").strip() not in ("0", "false", "no")),
        action=argparse.BooleanOptionalAction,
        help="Auto-load the default model when the UI opens.",
    )
    return p


def build_ui(args: argparse.Namespace) -> gr.Blocks:
    default_models = [m.strip() for m in (args.models or "").split(",") if m.strip()]
    default_model = args.default_model.strip() if args.default_model else (default_models[0] if default_models else "")

    outputs_dir = _mkdirp(os.path.join(_data_dir("/data"), "outputs"))
    voices_dir = _mkdirp(os.path.join(_data_dir("/data"), "voices"))

    def _unload(lm: Optional[LoadedModel]) -> None:
        if lm is None:
            return
        try:
            del lm.tts
        except Exception:
            pass
        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    def _save_wav(wav: np.ndarray, sr: int, prefix: str) -> str:
        ts = time.strftime("%Y%m%d_%H%M%S")
        fd, path = tempfile.mkstemp(prefix=f"{prefix}_{ts}_", suffix=".wav", dir=outputs_dir)
        os.close(fd)
        sf.write(path, np.asarray(wav, dtype=np.float32), int(sr))
        return path

    theme = gr.themes.Soft(font=[gr.themes.GoogleFont("Source Sans Pro"), "Arial", "sans-serif"])
    css = ".gradio-container {max-width: none !important;}"
    launch_sig = inspect.signature(gr.Blocks.launch)
    blocks_kwargs: Dict[str, Any] = {}
    launch_extras: Dict[str, Any] = {}
    if "theme" in launch_sig.parameters:
        launch_extras["theme"] = theme
    else:
        blocks_kwargs["theme"] = theme
    if "css" in launch_sig.parameters:
        launch_extras["css"] = css
    else:
        blocks_kwargs["css"] = css

    with gr.Blocks(**blocks_kwargs) as demo:
        gr.Markdown("# Qwen3-TTS Web UI\nLoad/switch models, then generate audio in the tabs below.")

        model_state = gr.State(value=None)

        with gr.Row():
            with gr.Column(scale=2):
                model_pick = gr.Dropdown(
                    label="Model Presets (可选模型)",
                    choices=default_models,
                    value=(default_model if default_model in default_models else (default_models[0] if default_models else None)),
                    interactive=True,
                    allow_custom_value=False,
                )
                model_id = gr.Textbox(
                    label="Model ID / Path (模型ID或本地路径)",
                    value=default_model,
                    placeholder="e.g. Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
                )
                with gr.Row():
                    device_in = gr.Textbox(label="Device (设备)", value=args.device, placeholder="cpu / cuda:0")
                    dtype_in = gr.Dropdown(
                        label="Dtype (精度)",
                        choices=["float32", "float16", "bfloat16"],
                        value=("bfloat16" if args.dtype in ("bfloat16", "bf16") else ("float16" if args.dtype in ("float16", "fp16") else "float32")),
                        interactive=True,
                    )
                    flash_in = gr.Checkbox(label="FlashAttention-2", value=bool(args.flash_attn))

                with gr.Row():
                    load_btn = gr.Button("Load / Switch Model (加载/切换模型)", variant="primary")
                    unload_btn = gr.Button("Unload (卸载)", variant="secondary")
                status = gr.Textbox(label="Status (状态)", lines=3)

            with gr.Column(scale=3):
                with gr.Accordion("Generation Settings (生成参数)", open=False):
                    max_new_tokens = gr.Number(label="max_new_tokens", value=2048, precision=0)
                    temperature = gr.Number(label="temperature", value=None)
                    top_k = gr.Number(label="top_k", value=None, precision=0)
                    top_p = gr.Number(label="top_p", value=None)
                    repetition_penalty = gr.Number(label="repetition_penalty", value=None)
                save_output = gr.Checkbox(label=f"Save output wav to {outputs_dir}", value=True)

        with gr.Tabs():
            with gr.Tab("Custom Voice (定制音色)"):
                with gr.Group(visible=True) as custom_group:
                    with gr.Row():
                        with gr.Column(scale=2):
                            text_cv = gr.Textbox(label="Text", lines=4, placeholder="Enter text to synthesize.")
                            with gr.Row():
                                lang_cv = gr.Dropdown(label="Language", choices=["Auto"], value="Auto", interactive=True)
                                spk_cv = gr.Dropdown(label="Speaker", choices=[], value=None, interactive=True)
                            instruct_cv = gr.Textbox(label="Instruction (optional)", lines=2)
                            gen_cv = gr.Button("Generate", variant="primary")
                        with gr.Column(scale=3):
                            audio_cv = gr.Audio(label="Output Audio", type="numpy")
                            file_cv = gr.File(label="Saved wav (if enabled)")
                            err_cv = gr.Textbox(label="Status", lines=2)

            with gr.Tab("Voice Design (音色设计)"):
                with gr.Group(visible=True) as vd_group:
                    with gr.Row():
                        with gr.Column(scale=2):
                            text_vd = gr.Textbox(label="Text", lines=4)
                            lang_vd = gr.Dropdown(label="Language", choices=["Auto"], value="Auto", interactive=True)
                            design_vd = gr.Textbox(label="Voice design instruction", lines=3)
                            gen_vd = gr.Button("Generate", variant="primary")
                        with gr.Column(scale=3):
                            audio_vd = gr.Audio(label="Output Audio", type="numpy")
                            file_vd = gr.File(label="Saved wav (if enabled)")
                            err_vd = gr.Textbox(label="Status", lines=2)

            with gr.Tab("Voice Clone (声音克隆 / Base 模型)"):
                with gr.Group(visible=True) as vc_group:
                    with gr.Tabs():
                        with gr.Tab("Quick Clone (快速克隆)"):
                            with gr.Row():
                                with gr.Column(scale=2):
                                    ref_audio = gr.Audio(label="Reference Audio", type="numpy")
                                    ref_text = gr.Textbox(label="Reference Text (optional when x-vector only)", lines=2)
                                    xvec_only = gr.Checkbox(label="Use x-vector only", value=False)
                                    text_vc = gr.Textbox(label="Target Text", lines=4)
                                    lang_vc = gr.Dropdown(label="Language", choices=["Auto"], value="Auto", interactive=True)
                                    gen_vc = gr.Button("Generate", variant="primary")
                                with gr.Column(scale=3):
                                    audio_vc = gr.Audio(label="Output Audio", type="numpy")
                                    file_vc = gr.File(label="Saved wav (if enabled)")
                                    err_vc = gr.Textbox(label="Status", lines=2)

                        with gr.Tab("Save / Load Voice (保存/加载音色)"):
                            with gr.Row():
                                with gr.Column(scale=2):
                                    ref_audio_s = gr.Audio(label="Reference Audio", type="numpy")
                                    ref_text_s = gr.Textbox(label="Reference Text", lines=2)
                                    xvec_only_s = gr.Checkbox(label="Use x-vector only", value=False)
                                    save_voice_btn = gr.Button("Save Voice File", variant="primary")
                                    prompt_file_out = gr.File(label="Voice File (.pt)")
                                    err_vc2 = gr.Textbox(label="Status", lines=2)

                                with gr.Column(scale=2):
                                    prompt_file_in = gr.File(label="Upload Voice File (.pt)")
                                    text_vc2 = gr.Textbox(label="Target Text", lines=4)
                                    lang_vc2 = gr.Dropdown(label="Language", choices=["Auto"], value="Auto", interactive=True)
                                    gen_vc2 = gr.Button("Generate", variant="primary")

                                with gr.Column(scale=3):
                                    audio_vc2 = gr.Audio(label="Output Audio", type="numpy")
                                    file_vc2 = gr.File(label="Saved wav (if enabled)")

        def _gen_kwargs(mn, t, tk, tp, rp) -> Dict[str, Any]:
            out: Dict[str, Any] = {}
            if mn is not None and str(mn).strip() != "":
                out["max_new_tokens"] = int(mn)
            if t is not None and str(t).strip() != "":
                out["temperature"] = float(t)
            if tk is not None and str(tk).strip() != "":
                out["top_k"] = int(tk)
            if tp is not None and str(tp).strip() != "":
                out["top_p"] = float(tp)
            if rp is not None and str(rp).strip() != "":
                out["repetition_penalty"] = float(rp)
            return out

        def sync_model_dropdown(preset: Optional[str]) -> str:
            return preset or ""

        model_pick.change(sync_model_dropdown, inputs=[model_pick], outputs=[model_id])

        def do_unload(lm: Optional[LoadedModel]):
            _unload(lm)
            return None, "Unloaded."

        unload_btn.click(do_unload, inputs=[model_state], outputs=[model_state, status])

        def do_load(lm: Optional[LoadedModel], ckpt: str, device: str, dtype_s: str, flash: bool):
            try:
                ckpt = (ckpt or "").strip()
                if not ckpt:
                    return lm, "Model ID/Path is required."

                device = (device or "").strip() or "cpu"
                dtype_s = (dtype_s or "").strip() or "float32"
                dtype = _dtype_from_str(dtype_s)
                attn_impl = None
                flash = bool(flash)
                if flash:
                    if "cuda" not in device.lower():
                        return lm, "FlashAttention-2 requires a CUDA device (e.g. cuda:0)."
                    if dtype not in (torch.float16, torch.bfloat16):
                        return lm, "FlashAttention-2 requires dtype float16/bfloat16."
                    attn_impl = "flash_attention_2"

                if lm is not None and (lm.ckpt, lm.device, lm.dtype, lm.flash_attn) == (ckpt, device, dtype_s, flash):
                    return lm, f"Already loaded: {ckpt} ({lm.kind})"

                _unload(lm)
                tts = Qwen3TTSModel.from_pretrained(
                    ckpt,
                    device_map=device,
                    dtype=dtype,
                    attn_implementation=attn_impl,
                )
                kind = _detect_model_kind(tts)

                supported_langs_raw = None
                if callable(getattr(tts.model, "get_supported_languages", None)):
                    supported_langs_raw = tts.model.get_supported_languages()
                supported_spks_raw = None
                if callable(getattr(tts.model, "get_supported_speakers", None)):
                    supported_spks_raw = tts.model.get_supported_speakers()

                lang_choices_disp, lang_map = _build_choices_and_map([x for x in (supported_langs_raw or [])])
                spk_choices_disp, spk_map = _build_choices_and_map([x for x in (supported_spks_raw or [])])

                new_lm = LoadedModel(
                    ckpt=ckpt,
                    device=device,
                    dtype=dtype_s,
                    flash_attn=bool(flash),
                    kind=kind,
                    lang_choices_disp=["Auto"] + lang_choices_disp,
                    lang_map={"Auto": "Auto", **lang_map},
                    spk_choices_disp=spk_choices_disp,
                    spk_map=spk_map,
                    tts=tts,
                )
                return new_lm, f"Loaded: {ckpt} ({kind})"
            except Exception as e:
                return lm, f"{type(e).__name__}: {e}"

        load_btn.click(do_load, inputs=[model_state, model_id, device_in, dtype_in, flash_in], outputs=[model_state, status])

        if args.autoload and default_model:
            demo.load(
                do_load,
                inputs=[model_state, model_id, device_in, dtype_in, flash_in],
                outputs=[model_state, status],
            )

        def _update_after_load(lm: Optional[LoadedModel]):
            if lm is None:
                return (
                    gr.update(choices=["Auto"], value="Auto"),
                    gr.update(choices=[], value=None),
                    gr.update(choices=["Auto"], value="Auto"),
                    gr.update(choices=["Auto"], value="Auto"),
                    gr.update(choices=["Auto"], value="Auto"),
                    gr.update(visible=True),
                    gr.update(visible=True),
                    gr.update(visible=True),
                )

            lang_update = gr.update(choices=lm.lang_choices_disp, value="Auto")
            spk_update = gr.update(choices=lm.spk_choices_disp, value=(lm.spk_choices_disp[0] if lm.spk_choices_disp else None))

            return (
                lang_update,
                spk_update,
                lang_update,
                lang_update,
                lang_update,
                gr.update(visible=(lm.kind == "custom_voice")),
                gr.update(visible=(lm.kind == "voice_design")),
                gr.update(visible=(lm.kind == "base")),
            )

        model_state.change(
            _update_after_load,
            inputs=[model_state],
            outputs=[lang_cv, spk_cv, lang_vd, lang_vc, lang_vc2, custom_group, vd_group, vc_group],
        )

        def run_custom_voice(lm: Optional[LoadedModel], text: str, lang_disp: str, spk_disp: str, instruct: str, mn, t, tk, tp, rp, save: bool):
            try:
                if lm is None:
                    return None, None, "Please load a model first."
                if lm.kind != "custom_voice":
                    return None, None, f"Loaded model is {lm.kind}, please switch to a CustomVoice model."
                if not text or not text.strip():
                    return None, None, "Text is required."
                language = lm.lang_map.get(lang_disp, "Auto")
                speaker = lm.spk_map.get(spk_disp, spk_disp)
                kwargs = _gen_kwargs(mn, t, tk, tp, rp)
                wavs, sr = lm.tts.generate_custom_voice(
                    text=text.strip(),
                    language=language,
                    speaker=speaker,
                    instruct=(instruct or "").strip() or None,
                    **kwargs,
                )
                saved = _save_wav(wavs[0], sr, "custom_voice") if save else None
                return _wav_to_gradio_audio(wavs[0], sr), saved, "Finished."
            except Exception as e:
                return None, None, f"{type(e).__name__}: {e}"

        gen_cv.click(
            run_custom_voice,
            inputs=[model_state, text_cv, lang_cv, spk_cv, instruct_cv, max_new_tokens, temperature, top_k, top_p, repetition_penalty, save_output],
            outputs=[audio_cv, file_cv, err_cv],
        )

        def run_voice_design(lm: Optional[LoadedModel], text: str, lang_disp: str, design: str, mn, t, tk, tp, rp, save: bool):
            try:
                if lm is None:
                    return None, None, "Please load a model first."
                if lm.kind != "voice_design":
                    return None, None, f"Loaded model is {lm.kind}, please switch to a VoiceDesign model."
                if not text or not text.strip():
                    return None, None, "Text is required."
                if not design or not design.strip():
                    return None, None, "Voice design instruction is required."
                language = lm.lang_map.get(lang_disp, "Auto")
                kwargs = _gen_kwargs(mn, t, tk, tp, rp)
                wavs, sr = lm.tts.generate_voice_design(
                    text=text.strip(),
                    language=language,
                    instruct=design.strip(),
                    **kwargs,
                )
                saved = _save_wav(wavs[0], sr, "voice_design") if save else None
                return _wav_to_gradio_audio(wavs[0], sr), saved, "Finished."
            except Exception as e:
                return None, None, f"{type(e).__name__}: {e}"

        gen_vd.click(
            run_voice_design,
            inputs=[model_state, text_vd, lang_vd, design_vd, max_new_tokens, temperature, top_k, top_p, repetition_penalty, save_output],
            outputs=[audio_vd, file_vd, err_vd],
        )

        def run_voice_clone(lm: Optional[LoadedModel], ref_aud, ref_txt: str, use_xvec: bool, text: str, lang_disp: str, mn, t, tk, tp, rp, save: bool):
            try:
                if lm is None:
                    return None, None, "Please load a model first."
                if lm.kind != "base":
                    return None, None, f"Loaded model is {lm.kind}, please switch to a Base model."
                at = _audio_to_tuple(ref_aud)
                if at is None:
                    return None, None, "Reference audio is required."
                if not text or not text.strip():
                    return None, None, "Target text is required."
                if (not use_xvec) and (not ref_txt or not ref_txt.strip()):
                    return None, None, "Reference text is required when x-vector only is NOT enabled."

                language = lm.lang_map.get(lang_disp, "Auto")
                kwargs = _gen_kwargs(mn, t, tk, tp, rp)
                wavs, sr = lm.tts.generate_voice_clone(
                    text=text.strip(),
                    language=language,
                    ref_audio=at,
                    ref_text=(ref_txt.strip() if ref_txt else None),
                    x_vector_only_mode=bool(use_xvec),
                    **kwargs,
                )
                saved = _save_wav(wavs[0], sr, "voice_clone") if save else None
                return _wav_to_gradio_audio(wavs[0], sr), saved, "Finished."
            except Exception as e:
                return None, None, f"{type(e).__name__}: {e}"

        gen_vc.click(
            run_voice_clone,
            inputs=[model_state, ref_audio, ref_text, xvec_only, text_vc, lang_vc, max_new_tokens, temperature, top_k, top_p, repetition_penalty, save_output],
            outputs=[audio_vc, file_vc, err_vc],
        )

        def save_prompt(lm: Optional[LoadedModel], ref_aud, ref_txt: str, use_xvec: bool):
            try:
                if lm is None:
                    return None, "Please load a model first."
                if lm.kind != "base":
                    return None, f"Loaded model is {lm.kind}, please switch to a Base model."
                at = _audio_to_tuple(ref_aud)
                if at is None:
                    return None, "Reference audio is required."
                if (not use_xvec) and (not ref_txt or not ref_txt.strip()):
                    return None, "Reference text is required when x-vector only is NOT enabled."

                items = lm.tts.create_voice_clone_prompt(
                    ref_audio=at,
                    ref_text=(ref_txt.strip() if ref_txt else None),
                    x_vector_only_mode=bool(use_xvec),
                )
                payload = {"items": [asdict(it) for it in items]}

                fd, out_path = tempfile.mkstemp(prefix="voice_clone_prompt_", suffix=".pt", dir=voices_dir)
                os.close(fd)
                torch.save(payload, out_path)
                return out_path, "Finished."
            except Exception as e:
                return None, f"{type(e).__name__}: {e}"

        save_voice_btn.click(save_prompt, inputs=[model_state, ref_audio_s, ref_text_s, xvec_only_s], outputs=[prompt_file_out, err_vc2])

        def load_prompt_and_gen(lm: Optional[LoadedModel], file_obj, text: str, lang_disp: str, mn, t, tk, tp, rp, save: bool):
            try:
                if lm is None:
                    return None, None, "Please load a model first."
                if lm.kind != "base":
                    return None, None, f"Loaded model is {lm.kind}, please switch to a Base model."
                if file_obj is None:
                    return None, None, "Voice file is required."
                if not text or not text.strip():
                    return None, None, "Target text is required."

                path = getattr(file_obj, "name", None) or getattr(file_obj, "path", None) or str(file_obj)
                payload = torch.load(path, map_location="cpu", weights_only=True)
                if not isinstance(payload, dict) or "items" not in payload:
                    return None, None, "Invalid voice file."

                items_raw = payload["items"]
                if not isinstance(items_raw, list) or len(items_raw) == 0:
                    return None, None, "Empty voice file."

                items: List[VoiceClonePromptItem] = []
                for d in items_raw:
                    if not isinstance(d, dict):
                        return None, None, "Invalid item in voice file."
                    ref_code = d.get("ref_code", None)
                    if ref_code is not None and not torch.is_tensor(ref_code):
                        ref_code = torch.tensor(ref_code)
                    ref_spk = d.get("ref_spk_embedding", None)
                    if ref_spk is None:
                        return None, None, "Missing ref_spk_embedding in voice file."
                    if not torch.is_tensor(ref_spk):
                        ref_spk = torch.tensor(ref_spk)

                    items.append(
                        VoiceClonePromptItem(
                            ref_code=ref_code,
                            ref_spk_embedding=ref_spk,
                            x_vector_only_mode=bool(d.get("x_vector_only_mode", False)),
                            icl_mode=bool(d.get("icl_mode", not bool(d.get("x_vector_only_mode", False)))),
                            ref_text=d.get("ref_text", None),
                        )
                    )

                language = lm.lang_map.get(lang_disp, "Auto")
                kwargs = _gen_kwargs(mn, t, tk, tp, rp)
                wavs, sr = lm.tts.generate_voice_clone(
                    text=text.strip(),
                    language=language,
                    voice_clone_prompt=items,
                    **kwargs,
                )
                saved = _save_wav(wavs[0], sr, "voice_clone") if save else None
                return _wav_to_gradio_audio(wavs[0], sr), saved, "Finished."
            except Exception as e:
                return None, None, f"{type(e).__name__}: {e}"

        gen_vc2.click(
            load_prompt_and_gen,
            inputs=[model_state, prompt_file_in, text_vc2, lang_vc2, max_new_tokens, temperature, top_k, top_p, repetition_penalty, save_output],
            outputs=[audio_vc2, file_vc2, err_vc2],
        )

        gr.Markdown(
            """
**Notes**
- Model switching unloads the previous model to reduce memory pressure.
- If you need microphone input (especially remotely), browsers may require HTTPS.
"""
        )

    setattr(demo, "_qwen_launch_extras", launch_extras)
    return demo


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    demo = build_ui(args)

    data_dir = os.environ.get("QWEN_TTS_DATA_DIR", "/data")
    allowed_paths = [
        os.path.join(data_dir, "outputs"),
        os.path.join(data_dir, "voices"),
        os.path.join(data_dir, "gradio-tmp"),
    ]

    launch_kwargs: Dict[str, Any] = dict(
        server_name=args.ip,
        server_port=args.port,
        share=bool(args.share),
        ssl_verify=True if args.ssl_verify else False,
        allowed_paths=allowed_paths,
    )
    launch_kwargs.update(getattr(demo, "_qwen_launch_extras", {}) or {})
    if args.ssl_certfile is not None:
        launch_kwargs["ssl_certfile"] = args.ssl_certfile
    if args.ssl_keyfile is not None:
        launch_kwargs["ssl_keyfile"] = args.ssl_keyfile

    demo.queue(default_concurrency_limit=int(args.concurrency)).launch(**launch_kwargs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
