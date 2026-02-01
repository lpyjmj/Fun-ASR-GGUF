import time
import ctypes
import numpy as np
from typing import List, Tuple, Optional, Dict, Any

from .. import llama
from ..nano_ctc import decode_ctc, align_timestamps
from ..nano_onnx import encode_audio
from ..utils import vprint
from ..nano_dataclass import DecodeResult, Timings, RecognitionStream
from ..display import DisplayReporter
from .model_manager import ModelManager

class CTCDecoder:
    """负责 CTC 推理和热词匹配"""
    def __init__(self, models: ModelManager):
        self.models = models

    def decode(self, enc_output: np.ndarray, enable_ctc: bool, max_hotwords: int) -> Tuple[List, List[str]]:
        if not enable_ctc or self.models.ctc_sess is None:
            return [], []

        ctc_logits = self.models.ctc_sess.run(None, {"enc_output": enc_output})[0]
        ctc_text, ctc_results = decode_ctc(ctc_logits, self.models.ctc_id2token)

        hotwords = []
        if self.models.corrector and self.models.corrector.hotwords and ctc_text:
            res = self.models.corrector.correct(ctc_text, k=max_hotwords)
            candidates = set()
            for _, hw, _ in res.matchs: candidates.add(hw)
            for _, hw, _ in res.similars: candidates.add(hw)
            hotwords = list(candidates)
            
        return ctc_results, hotwords

class LLMDecoder:
    """负责 LLM 推理循环"""
    def __init__(self, models: ModelManager):
        self.models = models
        self.stop_tokens = [151643, 151645]

    def decode(
        self,
        full_embd: np.ndarray,
        n_input_tokens: int,
        n_predict: int,
        stream_output: bool = False,
        reporter: Optional[DisplayReporter] = None,
        temperature: float = 0.3,
        top_p: float = 1.0,
        top_k: int = 50
    ) -> Tuple[str, int, float, float]:
        
        t_inject_start = time.perf_counter()
        
        # 1. Inject
        self.models.ctx.clear_kv_cache()
        
        batch_embd = llama.LlamaBatch(n_input_tokens, full_embd.shape[1], 1)
        batch_embd.set_embd(full_embd)
        batch_embd.struct.token = ctypes.cast(None, ctypes.POINTER(llama.llama_token))

        ret = self.models.ctx.decode(batch_embd)
        if ret != 0: raise RuntimeError(f"Decode failed (ret={ret})")
        
        t_inject = time.perf_counter() - t_inject_start

        # 2. Generation Loop
        t_gen_start = time.perf_counter()
        batch_text = llama.LlamaBatch(1, 0, 1)

        current_pos = n_input_tokens
        asr_decoder = llama.ASRStreamDecoder(self.models.vocab, reporter if stream_output else None)
        
        with llama.LlamaSampler(temperature=temperature, top_k=top_k, top_p=top_p) as smpl:
            for _ in range(n_predict):
                # 使用面向对象接口采样
                token_id = smpl.sample(self.models.ctx, -1)

                if token_id == self.models.eos_token or token_id in self.stop_tokens:
                    break

                # 集成化处理新 Token：解码字节 -> 拼接 -> 实时汇报
                asr_decoder.push(token_id)

                if self.models.ctx.decode_token(batch_text, token_id, current_pos) != 0:
                    break
                current_pos += 1
        
        asr_decoder.flush()

        # batch_text 会由 __del__ 自动释放
        t_gen = time.perf_counter() - t_gen_start
        
        return asr_decoder.generated_text, asr_decoder.tokens_generated, t_inject, t_gen

class StreamDecoder:
    """协调完整流程的解码器"""
    def __init__(self, models: ModelManager):
        self.models = models
        self.ctc_decoder = CTCDecoder(models)
        self.llm_decoder = LLMDecoder(models)

    def decode_stream(
        self,
        stream: RecognitionStream,
        language: Optional[str] = None,
        context: Optional[str] = None,
        verbose: bool = True,
        reporter: Optional[DisplayReporter] = None,
        temperature: float = 0.3,
        top_p: float = 1.0,
        top_k: int = 50
    ) -> DecodeResult:
        
        timings = Timings()
        
        # 1. Encode
        if reporter: reporter.print("\n[2] 音频编码...")
        t_s = time.perf_counter()
        audio_embd, enc_output = encode_audio(stream.audio_data, self.models.encoder_sess)
        timings.encode = time.perf_counter() - t_s
        if reporter: reporter.print(f"    耗时: {timings.encode*1000:.2f}ms")

        # 2. CTC
        if reporter: reporter.print("\n[3] CTC 解码...")
        t_s = time.perf_counter()
        ctc_results, hotwords = self.ctc_decoder.decode(
            enc_output, 
            self.models.config.enable_ctc, 
            self.models.config.max_hotwords
        )
        timings.ctc = time.perf_counter() - t_s
        
        if verbose and ctc_results:
            ctc_text = "".join([r.text for r in ctc_results])
            if reporter:
                reporter.print(f"    CTC: {ctc_text}")
                if hotwords: reporter.print(f"    热词: {hotwords}")
        if reporter: reporter.print(f"    耗时: {timings.ctc*1000:.2f}ms")

        # 3. Prompt
        if reporter: reporter.print("\n[4] 准备 Prompt...")
        t_s = time.perf_counter()
        p_embd, s_embd, n_p, n_s, p_text = self.models.prompt_builder.build_prompt(hotwords, language, context)
        timings.prepare = time.perf_counter() - t_s
        
        if verbose and reporter:
            reporter.print("-" * 15 + " Prefix Prompt " + "-" * 15 + "\n" + p_text + "\n" + "-" * 40)
        if reporter:
            reporter.print(f"    Prefix: {n_p} tokens")
            reporter.print(f"    Suffix: {n_s} tokens")

        # 4. LLM
        if reporter:
            reporter.print("\n[5] LLM 解码...")
            reporter.print("=" * 70)
        
        full_embd = np.concatenate([p_embd, audio_embd.astype(np.float32), s_embd], axis=0)
        text, n_gen, t_inj, t_gen = self.llm_decoder.decode(
            full_embd, full_embd.shape[0], self.models.config.n_predict, 
            stream_output=verbose, reporter=reporter,
            temperature=temperature, top_p=top_p, top_k=top_k
        )
        text = text.strip()
        timings.inject = t_inj
        timings.llm_generate = t_gen
        
        if reporter: reporter.print("\n" + "=" * 70)

        # 5. Align
        if reporter: reporter.print("\n[6] 时间戳对齐")
        t_s = time.perf_counter()
        aligned = None
        timestamps = []
        tokens = []
        if ctc_results:
            aligned = align_timestamps(ctc_results, text)
            if aligned:
                tokens = [seg['char'] for seg in aligned]
                timestamps = [seg['start'] for seg in aligned]
        timings.align = time.perf_counter() - t_s
        
        if reporter and aligned:
            reporter.print(f"    对齐耗时: {timings.align*1000:.2f}ms")
            preview = " ".join([f"{r['char']}({r['start']:.2f}s)" for r in aligned[:10]])
            if len(aligned) > 10: preview += " ..."
            reporter.print(f"    结果预览: {preview}")

        # Set stream result
        stream.set_result(text=text, timestamps=timestamps, tokens=tokens)
        
        return DecodeResult(
            text=text, ctc_results=ctc_results, aligned=aligned,
            audio_embd=audio_embd, n_prefix=n_p, n_suffix=n_s,
            n_gen=n_gen, timings=timings, hotwords=hotwords
        )
