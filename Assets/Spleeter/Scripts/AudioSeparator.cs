using System;
using System.Buffers;
using System.Collections.Generic;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;
using MathNet.Numerics;
using MathNet.Numerics.IntegralTransforms;
using UnityEngine;

/// <summary>
/// 音频分离器 —— 对齐 Spleeter 参考实现，深度性能优化版 v4
///
/// ═══════════════════════════════════════════════════════════════
/// v4 相比 v3 新增优化：
///
/// 1. SIMD Vector&lt;float&gt; 加速
///    - Wiener 掩码：4-8 个 float 并行运算（AVX/SSE）
///    - 幅度谱 sqrt(r²+i²)：同样向量化
///    - 补集掩码 1-x：向量化减法
///
/// 2. Wiener 数学优化
///    - mask_vocals + mask_accomp ≡ 1（数学恒等式）
///    - 只算 vocals 掩码，accomp = 1 - vocals，省 50% Wiener 计算
///
/// 3. 4 色并行 IFFT+加窗+OLA
///    - N_FFT/HOP = 4，帧分 4 色，同色帧不重叠
///    - 每色内 Parallel.For 安全写 output[]（无竞争）
///    - 消除 perFrameData[][]，消除 Phase-B 串行瓶颈
///
/// 4. 预分配 STFT 结果缓冲
///    - 4 个 64MB 数组从每次 new 改为实例级复用
///    - 消除 ~256MB/次的 GC 压力
///
/// 5. 预计算 ISTFT 窗 = hann × (2/3)
///    - 消除独立的全数组 ×2/3 遍历
///
/// 6. 直接解交错到零填充数组
///    - 省去 ch0Raw/ch1Raw 中间数组 + 2 次 Array.Copy
///
/// 7. STFT 消除中间 frame[] 数组
///    - 窗口乘法 + Complex32 构造合并为单循环
///
/// 8. 高频置零（对齐 Python mask_extension="zeros"）
///    - 比"原样透传"更简单且更快（赋零 vs 读+构造）
///
/// 9. 分阶段计时日志
///
/// ═══════════════════════════════════════════════════════════════
/// 未来可探索（需外部依赖）：
///   - FFTW (FFTWSharp)：FFT 2-5× 加速
///   - Unity Burst + Jobs：数值计算 5-10× 加速
///   - TensorRT EP：ONNX 推理加速
///   - OrtIOBinding：ONNX 零拷贝 I/O
///   - R2C FFT：实数输入 FFT 比 C2C 快 ~1.5×
/// ═══════════════════════════════════════════════════════════════
/// </summary>
public class AudioSeparator : MonoBehaviour
{
    private OnnxModel _vocalsModel;
    private OnnxModel _accompanimentModel;

    // ── 参数 ──────────────────────────────────────────────────────────────
    private const int N_FFT        = 4096;
    private const int HOP_LENGTH   = 1024;
    private const int NUM_BINS     = 2049;   // N_FFT/2 + 1
    private const int MODEL_BINS   = 1024;   // F
    private const int CHUNK_SIZE   = 512;    // T
    private const int OLA_COLORS   = 4;      // N_FFT / HOP_LENGTH
    private const float EPSILON    = 1e-10f;
    private const float WINDOW_COMPENSATION_FACTOR = 2f / 3f;

    private int _sampleRate = 44100;

    // ── 预计算窗口 ─────────────────────────────────────────────────────────
    private float[] _hannWindowSTFT;   // Periodic Hann（正向 STFT 用）
    private float[] _hannWindowISTFT;  // Hann × (2/3)（逆向 ISTFT 用，含补偿因子）

    // ── 预分配 STFT 结果缓冲（复用，消除 GC）─────────────────────────────
    private float[] _stftReal0, _stftImag0;
    private float[] _stftReal1, _stftImag1;

    private int _originalSamplesPerChannel;

    // ── 模型输入 / Wiener 掩码缓冲（复用）─────────────────────────────────
    private float[] _modelInputFlat;
    private float[] _vocalsMaskBuf;
    private float[] _accompMaskBuf;

    // ═══════════════════════════════════════════════════════════════════════
    // 初始化
    // ═══════════════════════════════════════════════════════════════════════
    public void Initialize(string vocalsModelPath, string accompanimentModelPath)
    {
        try
        {
            _vocalsModel        = new OnnxModel(vocalsModelPath);
            _accompanimentModel = new OnnxModel(accompanimentModelPath);

            // 预计算窗口
            _hannWindowSTFT  = CreatePeriodicHannWindow(N_FFT);
            _hannWindowISTFT = new float[N_FFT];
            for (int i = 0; i < N_FFT; i++)
                _hannWindowISTFT[i] = _hannWindowSTFT[i] * WINDOW_COMPENSATION_FACTOR;

            Debug.Log("AudioSeparator 初始化成功（v4 SIMD + 4色OLA + 数学优化版）");
            Debug.Log($"  SIMD 加速: {Vector.IsHardwareAccelerated}  Vector<float>.Count: {Vector<float>.Count}");
        }
        catch (Exception ex)
        {
            Debug.LogError($"初始化失败: {ex.Message}");
            throw;
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 主入口
    // ═══════════════════════════════════════════════════════════════════════
    public Dictionary<string, float[]> Separate(float[] waveform)
    {
        if (_vocalsModel == null || _accompanimentModel == null)
            throw new InvalidOperationException("分离器未初始化");

        try
        {
            var swTotal = System.Diagnostics.Stopwatch.StartNew();
            float dur = (waveform.Length / 2) / (float)_sampleRate;

            // ── 1. 解交错 + 前导零填充（合并为一步）────────────────────────
            _originalSamplesPerChannel = waveform.Length / 2;
            int paddedLen = N_FFT + _originalSamplesPerChannel;
            float[] ch0 = new float[paddedLen];
            float[] ch1 = new float[paddedLen];
            // 前 N_FFT 个样本已是 0（new 自动清零），直接从 N_FFT 开始写
            DeinterleaveIntoPadded(waveform, ch0, ch1, _originalSamplesPerChannel, N_FFT);

            var sw1 = System.Diagnostics.Stopwatch.StartNew();

            // ── 2. 并行双声道 STFT ─────────────────────────────────────────
            int numFrames = 0;
            int nf1;
            Parallel.Invoke(
                () => ComputeStftParallel(ch0, ref _stftReal0, ref _stftImag0, out numFrames),
                () => ComputeStftParallel(ch1, ref _stftReal1, ref _stftImag1, out nf1)
            );
            sw1.Stop();
            Debug.Log($"[STFT ] {numFrames} 帧  {sw1.ElapsedMilliseconds}ms");

            // ── 3. 构建模型输入（SIMD 幅度谱）──────────────────────────────
            int padding      = (CHUNK_SIZE - numFrames % CHUNK_SIZE) % CHUNK_SIZE;
            int paddedFrames = numFrames + padding;
            int numSplits    = paddedFrames / CHUNK_SIZE;
            int flatSize     = 2 * numSplits * CHUNK_SIZE * MODEL_BINS;

            if (_modelInputFlat == null || _modelInputFlat.Length < flatSize)
                _modelInputFlat = new float[flatSize];
            else
                Array.Clear(_modelInputFlat, 0, flatSize);

            var sw2 = System.Diagnostics.Stopwatch.StartNew();
            FillModelInputSimd(_modelInputFlat, numFrames, numSplits);
            sw2.Stop();
            Debug.Log($"[INPUT ] splits={numSplits} pad={padding}  {sw2.ElapsedMilliseconds}ms");

            // ── 4. 双模型并行推理 ──────────────────────────────────────────
            var sw3 = System.Diagnostics.Stopwatch.StartNew();
            float[] vocalsOut = null, accompOut = null;
            Parallel.Invoke(
                () => vocalsOut = _vocalsModel.RunFlatNoCopy(_modelInputFlat, 2, numSplits, CHUNK_SIZE, MODEL_BINS),
                () => accompOut = _accompanimentModel.RunFlatNoCopy(_modelInputFlat, 2, numSplits, CHUNK_SIZE, MODEL_BINS)
            );
            sw3.Stop();
            Debug.Log($"[ONNX ] 2 模型推理  {sw3.ElapsedMilliseconds}ms");

            // ── 5. Wiener 掩码（SIMD + 数学优化：只算 vocals，accomp=1-vocals）
            var sw4 = System.Diagnostics.Stopwatch.StartNew();
            EnsureBuf(ref _vocalsMaskBuf, flatSize);
            EnsureBuf(ref _accompMaskBuf, flatSize);

            ComputeWienerMaskSimd(vocalsOut, accompOut, _vocalsMaskBuf, flatSize);
            DeriveComplementaryMaskSimd(_vocalsMaskBuf, _accompMaskBuf, flatSize);
            sw4.Stop();
            Debug.Log($"[MASK ] Wiener+补集  {sw4.ElapsedMilliseconds}ms");

            // ── 6. 并行 ISTFT（4 色并行 IFFT+加窗+OLA）─────────────────────
            var sw5 = System.Diagnostics.Stopwatch.StartNew();
            float[] vocalsAudio = null, accompAudio = null;
            Parallel.Invoke(
                () => vocalsAudio = ReconstructStereo(_vocalsMaskBuf, numSplits, numFrames),
                () => accompAudio = ReconstructStereo(_accompMaskBuf, numSplits, numFrames)
            );
            sw5.Stop();
            Debug.Log($"[ISTFT] 4色并行 OLA  {sw5.ElapsedMilliseconds}ms");

            swTotal.Stop();
            Debug.Log($"✓ 分离完成！耗时={swTotal.ElapsedMilliseconds}ms  时长={dur:F2}s  RTF={swTotal.ElapsedMilliseconds / 1000f / dur:F3}");

            return new Dictionary<string, float[]>(2)
            {
                { "vocals",        vocalsAudio },
                { "accompaniment", accompAudio }
            };
        }
        catch (Exception ex)
        {
            Debug.LogError($"分离错误: {ex.Message}\n{ex.StackTrace}");
            throw;
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 解交错 → 直接写入零填充数组的目标位置
    // ═══════════════════════════════════════════════════════════════════════
    private static void DeinterleaveIntoPadded(float[] src, float[] ch0, float[] ch1,
                                                int n, int destOffset)
    {
        const int BLOCK = 4096;
        int blocks = n / BLOCK;
        Parallel.For(0, blocks, b =>
        {
            int s = b * BLOCK;
            for (int i = s; i < s + BLOCK; i++)
            {
                ch0[destOffset + i] = src[i * 2];
                ch1[destOffset + i] = src[i * 2 + 1];
            }
        });
        for (int i = blocks * BLOCK; i < n; i++)
        {
            ch0[destOffset + i] = src[i * 2];
            ch1[destOffset + i] = src[i * 2 + 1];
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // STFT — Parallel.For 逐帧并行，预分配缓冲，无中间 frame[]
    // ═══════════════════════════════════════════════════════════════════════
    private void ComputeStftParallel(float[] signal,
                                     ref float[] realBuf, ref float[] imagBuf,
                                     out int numFrames)
    {
        numFrames = (signal.Length + HOP_LENGTH - 1) / HOP_LENGTH;
        int needed = numFrames * NUM_BINS;

        if (realBuf == null || realBuf.Length < needed)
            realBuf = new float[needed];
        if (imagBuf == null || imagBuf.Length < needed)
            imagBuf = new float[needed];

        float[] hann  = _hannWindowSTFT;
        float[] realL = realBuf;
        float[] imagL = imagBuf;
        int     sigLen = signal.Length;

        Parallel.For(0, numFrames, fi =>
        {
            Complex32[] fft = ArrayPool<Complex32>.Shared.Rent(N_FFT);
            try
            {
                int offset  = fi * HOP_LENGTH;
                int baseIdx = fi * NUM_BINS;
                int copyLen = Math.Min(N_FFT, Math.Max(0, sigLen - offset));

                // 窗口乘法 + Complex32 构造合并（无中间 frame[]）
                for (int i = 0; i < copyLen; i++)
                    fft[i] = new Complex32(signal[offset + i] * hann[i], 0f);
                for (int i = copyLen; i < N_FFT; i++)
                    fft[i] = Complex32.Zero;

                Fourier.Forward(fft, FourierOptions.Matlab);

                for (int k = 0; k < NUM_BINS; k++)
                {
                    realL[baseIdx + k] = fft[k].Real;
                    imagL[baseIdx + k] = fft[k].Imaginary;
                }
            }
            finally
            {
                ArrayPool<Complex32>.Shared.Return(fft);
            }
        });
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 构建模型输入 — SIMD 幅度谱
    // ═══════════════════════════════════════════════════════════════════════
    private void FillModelInputSimd(float[] dst, int numFrames, int numSplits)
    {
        int strideCh    = numSplits * CHUNK_SIZE * MODEL_BINS;
        int strideSplit = CHUNK_SIZE * MODEL_BINS;
        int vecSize     = Vector<float>.Count;
        bool useSimd    = Vector.IsHardwareAccelerated && MODEL_BINS >= vecSize;
        int simdBins    = useSimd ? MODEL_BINS - (MODEL_BINS % vecSize) : 0;

        Parallel.For(0, 2, ch =>
        {
            float[] r  = ch == 0 ? _stftReal0 : _stftReal1;
            float[] im = ch == 0 ? _stftImag0 : _stftImag1;
            int chBase = ch * strideCh;

            for (int s = 0; s < numSplits; s++)
            {
                int splitBase = chBase + s * strideSplit;
                for (int fi = 0; fi < CHUNK_SIZE; fi++)
                {
                    int gf      = s * CHUNK_SIZE + fi;
                    int dstBase = splitBase + fi * MODEL_BINS;
                    if (gf >= numFrames) continue; // padding 帧保持零

                    int srcBase = gf * NUM_BINS;

                    if (useSimd)
                    {
                        // SIMD: 4-8 个 bin 并行 sqrt
                        int k = 0;
                        for (; k < simdBins; k += vecSize)
                        {
                            var rv  = new Vector<float>(r,  srcBase + k);
                            var iv  = new Vector<float>(im, srcBase + k);
                            var mag = Vector.SquareRoot(rv * rv + iv * iv);
                            mag.CopyTo(dst, dstBase + k);
                        }
                        // 尾部
                        for (; k < MODEL_BINS; k++)
                        {
                            float rv = r[srcBase + k], iv = im[srcBase + k];
                            dst[dstBase + k] = MathF.Sqrt(rv * rv + iv * iv);
                        }
                    }
                    else
                    {
                        for (int k = 0; k < MODEL_BINS; k++)
                        {
                            float rv = r[srcBase + k], iv = im[srcBase + k];
                            dst[dstBase + k] = MathF.Sqrt(rv * rv + iv * iv);
                        }
                    }
                }
            }
        });
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Wiener 掩码 — SIMD 向量化
    // mask = (src² + ε/2) / (src² + other² + ε)
    // ═══════════════════════════════════════════════════════════════════════
    private static void ComputeWienerMaskSimd(float[] src, float[] other, float[] mask, int len)
    {
        int vecSize  = Vector<float>.Count;
        bool useSimd = Vector.IsHardwareAccelerated && len >= vecSize;
        int simdLen  = useSimd ? len - (len % vecSize) : 0;

        if (useSimd)
        {
            var vEps     = new Vector<float>(EPSILON);
            var vEpsHalf = new Vector<float>(EPSILON * 0.5f);

            int i = 0;
            for (; i < simdLen; i += vecSize)
            {
                var s  = new Vector<float>(src, i);
                var o  = new Vector<float>(other, i);
                var ss = s * s;
                var oo = o * o;
                var result = (ss + vEpsHalf) / (ss + oo + vEps);
                result.CopyTo(mask, i);
            }
            // 尾部
            for (; i < len; i++)
            {
                float ss = src[i] * src[i];
                float oo = other[i] * other[i];
                mask[i] = (ss + EPSILON * 0.5f) / (ss + oo + EPSILON);
            }
        }
        else
        {
            // 标量 x8 展开
            int i = 0;
            for (; i <= len - 8; i += 8)
            {
                mask[i]   = WienerVal(src[i],   other[i]);
                mask[i+1] = WienerVal(src[i+1], other[i+1]);
                mask[i+2] = WienerVal(src[i+2], other[i+2]);
                mask[i+3] = WienerVal(src[i+3], other[i+3]);
                mask[i+4] = WienerVal(src[i+4], other[i+4]);
                mask[i+5] = WienerVal(src[i+5], other[i+5]);
                mask[i+6] = WienerVal(src[i+6], other[i+6]);
                mask[i+7] = WienerVal(src[i+7], other[i+7]);
            }
            for (; i < len; i++)
                mask[i] = WienerVal(src[i], other[i]);
        }
    }

    /// <summary>
    /// 推导补集掩码：mask_accomp = 1 - mask_vocals
    /// 数学依据：Wiener 掩码满足 mask_v + mask_a ≡ 1
    /// </summary>
    private static void DeriveComplementaryMaskSimd(float[] src, float[] dst, int len)
    {
        int vecSize  = Vector<float>.Count;
        bool useSimd = Vector.IsHardwareAccelerated && len >= vecSize;
        int simdLen  = useSimd ? len - (len % vecSize) : 0;

        if (useSimd)
        {
            var vOne = Vector<float>.One;
            int i = 0;
            for (; i < simdLen; i += vecSize)
            {
                var s = new Vector<float>(src, i);
                (vOne - s).CopyTo(dst, i);
            }
            for (; i < len; i++)
                dst[i] = 1f - src[i];
        }
        else
        {
            int i = 0;
            for (; i <= len - 8; i += 8)
            {
                dst[i]   = 1f - src[i];
                dst[i+1] = 1f - src[i+1];
                dst[i+2] = 1f - src[i+2];
                dst[i+3] = 1f - src[i+3];
                dst[i+4] = 1f - src[i+4];
                dst[i+5] = 1f - src[i+5];
                dst[i+6] = 1f - src[i+6];
                dst[i+7] = 1f - src[i+7];
            }
            for (; i < len; i++)
                dst[i] = 1f - src[i];
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float WienerVal(float s, float o)
    {
        float ss = s * s, oo = o * o;
        return (ss + EPSILON * 0.5f) / (ss + oo + EPSILON);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 立体声重构
    // ═══════════════════════════════════════════════════════════════════════
    private float[] ReconstructStereo(float[] mask, int numSplits, int numFrames)
    {
        float[] ch0Out = null, ch1Out = null;
        Parallel.Invoke(
            () => ch0Out = ApplyMaskAndISTFT_4Color(mask, 0, numSplits, numFrames),
            () => ch1Out = ApplyMaskAndISTFT_4Color(mask, 1, numSplits, numFrames)
        );

        int n = ch0Out.Length;
        float[] stereo = new float[n * 2];
        Interleave(ch0Out, ch1Out, stereo, n);
        return stereo;
    }

    private static void Interleave(float[] ch0, float[] ch1, float[] dst, int n)
    {
        const int BLOCK = 4096;
        int blocks = n / BLOCK;
        Parallel.For(0, blocks, b =>
        {
            int s = b * BLOCK;
            for (int i = s; i < s + BLOCK; i++)
            {
                dst[i * 2]     = ch0[i];
                dst[i * 2 + 1] = ch1[i];
            }
        });
        for (int i = blocks * BLOCK; i < n; i++)
        {
            dst[i * 2]     = ch0[i];
            dst[i * 2 + 1] = ch1[i];
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // ISTFT — 4 色并行 IFFT + 加窗 + OLA
    //
    // 原理：N_FFT/HOP = 4096/1024 = 4
    //   帧 {c, c+4, c+8, ...} 的输出区间互不重叠
    //   → 同色帧可安全并行写入 output[]
    //   → 4 色顺序执行（4 轮并行），消除串行 Phase-B
    //
    // 合并优化：IFFT → 加窗 → OLA 在同一并行任务中完成
    //   → 无需 perFrameData[][] 中间存储
    //   → 无需额外 ArrayPool<float> 租用
    // ═══════════════════════════════════════════════════════════════════════
    private float[] ApplyMaskAndISTFT_4Color(float[] mask, int ch,
                                              int numSplits, int numFrames)
    {
        float[] stftR = ch == 0 ? _stftReal0 : _stftReal1;
        float[] stftI = ch == 0 ? _stftImag0 : _stftImag1;

        int maskChOffset    = ch * numSplits * CHUNK_SIZE * MODEL_BINS;
        int maskSplitStride = CHUNK_SIZE * MODEL_BINS;

        int fullOutputLength = (numFrames - 1) * HOP_LENGTH + N_FFT;
        float[] output = new float[fullOutputLength]; // 自动清零
        float[] istftWin = _hannWindowISTFT; // 含 2/3 补偿

        for (int color = 0; color < OLA_COLORS; color++)
        {
            int colorStart = color;
            int colorCount = (numFrames - color + OLA_COLORS - 1) / OLA_COLORS;

            Parallel.For(0, colorCount, j =>
            {
                int fi = colorStart + j * OLA_COLORS;
                if (fi >= numFrames) return;

                Complex32[] ifft = ArrayPool<Complex32>.Shared.Rent(N_FFT);
                try
                {
                    int splitIdx   = fi / CHUNK_SIZE;
                    int inSplitIdx = fi % CHUNK_SIZE;
                    int stftBase   = fi * NUM_BINS;
                    int maskBase   = maskChOffset
                                   + splitIdx * maskSplitStride
                                   + inSplitIdx * MODEL_BINS;

                    // ── 低频：掩码 × STFT ────────────────────────────────
                    if (splitIdx < numSplits)
                    {
                        for (int k = 0; k < MODEL_BINS; k++)
                        {
                            float m = mask[maskBase + k];
                            ifft[k] = new Complex32(
                                stftR[stftBase + k] * m,
                                stftI[stftBase + k] * m);
                        }
                    }
                    else
                    {
                        for (int k = 0; k < MODEL_BINS; k++)
                            ifft[k] = new Complex32(stftR[stftBase + k], stftI[stftBase + k]);
                    }

                    // ── 高频：置零（对齐 Python mask_extension="zeros"）───
                    for (int k = MODEL_BINS; k < NUM_BINS; k++)
                        ifft[k] = Complex32.Zero;

                    // ── 共轭对称填充 ──────────────────────────────────────
                    for (int k = NUM_BINS; k < N_FFT; k++)
                    {
                        int conj = N_FFT - k;
                        ifft[k] = conj < NUM_BINS
                            ? Complex32.Conjugate(ifft[conj])
                            : Complex32.Zero;
                    }

                    // ── IFFT ──────────────────────────────────────────────
                    Fourier.Inverse(ifft, FourierOptions.Matlab);

                    // ── 加窗 + OLA（写入 output[]，同色帧无竞争）──────────
                    int offset = fi * HOP_LENGTH;
                    int end    = Math.Min(N_FFT, fullOutputLength - offset);
                    for (int i = 0; i < end; i++)
                        output[offset + i] += ifft[i].Real * istftWin[i];
                }
                finally
                {
                    ArrayPool<Complex32>.Shared.Return(ifft);
                }
            });
        }

        // ── 裁剪 output[N_FFT .. N_FFT + originalSamples] ──────────────
        float[] cropped = new float[_originalSamplesPerChannel];
        int copyLen = Math.Min(_originalSamplesPerChannel, fullOutputLength - N_FFT);
        if (copyLen > 0)
            Array.Copy(output, N_FFT, cropped, 0, copyLen);
        return cropped;
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 辅助
    // ═══════════════════════════════════════════════════════════════════════
    private static void EnsureBuf(ref float[] buf, int minLen)
    {
        if (buf == null || buf.Length < minLen)
            buf = new float[minLen];
    }

    private static float[] CreatePeriodicHannWindow(int N)
    {
        float[] w     = new float[N];
        double  scale = 2.0 * Math.PI / N;
        for (int i = 0; i < N; i++)
            w[i] = 0.5f * (1f - (float)Math.Cos(scale * i));
        return w;
    }

    public Dictionary<string, float[]> SeparateFromFile(string audioPath)
    {
        float[] waveform = Util.LoadWavFile(audioPath, ref _sampleRate);
        return Separate(waveform);
    }

    public void Dispose()
    {
        _vocalsModel?.Dispose();
        _accompanimentModel?.Dispose();
    }

    private void OnDestroy() => Dispose();
}
