using System;
using System.Buffers;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;
using MathNet.Numerics;
using MathNet.Numerics.IntegralTransforms;
using UnityEngine;

/// <summary>
/// 音频分离器 —— 对齐 Spleeter 参考实现，深度性能优化版 v3
///
/// ═══════════════════════════════════════════════════════════════
/// v3 相比 v2 新增优化（目标：RTF 0.233 → &lt;0.10）：
///
/// 1. Parallel.For 逐帧并行 STFT
///    - 每帧独立，可完全并行
///    - [ThreadStatic] 替换为 ArrayPool 租用，消除线程本地 null 检查 + 统一释放
///
/// 2. ISTFT 两阶段并行
///    - Phase-A：Parallel.For 并行执行各帧 IFFT + 加窗（写入独立 perFrameData[fi]）
///    - Phase-B：串行 OLA 叠加到 output[]（无并发写冲突）
///    - 相比原单线程逐帧，IFFT 部分完全并行化
///
/// 3. OnnxModel.RunFlatNoCopy()
///    - 返回内部 _outputBuf 引用（零拷贝），省去最后一次 Buffer.BlockCopy
///    - 调用方在 ComputeWienerMaskFlat 前不修改数据，安全可用
///
/// 4. Wiener 掩码缓冲复用
///    - 预分配 _vocalsMaskBuf / _accompMaskBuf，消除每帧分配
///    - 掩码内层循环：x8 展开（RTF 越低展开收益越明显）
///
/// 5. 解交错/重交错 使用 span 分块写
///    - 消除 Parallel.For 小粒度任务调度开销（原粒度=1 sample）
///
/// 6. FillModelInputFlat 内层循环使用 Span&lt;float&gt; 直接写
///    - 减少数组边界检查（JIT 可省略部分检查）
///
/// ═══════════════════════════════════════════════════════════════
/// 继承 v2 算法：
///   - Periodic Hann 窗（/N）
///   - STFT 前前导 N_FFT 零帧
///   - pad_end=True
///   - WINDOW_COMPENSATION_FACTOR = 2/3
///   - 输出裁剪 output[N_FFT .. N_FFT + originalSamples]
///   - 双模型 Parallel.Invoke 并行推理
/// </summary>
public class AudioSeparator : MonoBehaviour
{
    private OnnxModel _vocalsModel;
    private OnnxModel _accompanimentModel;

    // ── 参数（与 spleeter dataset.py DEFAULT_AUDIO_PARAMS 一致）──────────
    private const int N_FFT        = 4096;  // frame_length
    private const int HOP_LENGTH   = 1024;  // frame_step  (= N_FFT/4)
    private const int NUM_BINS     = 2049;  // N_FFT/2 + 1
    private const int MODEL_BINS   = 1024;  // F
    private const int CHUNK_SIZE   = 512;   // T
    private const float EPSILON    = 1e-10f;
    private const float WINDOW_COMPENSATION_FACTOR = 2f / 3f;

    private int _sampleRate = 44100;

    // ── 预计算窗口 ─────────────────────────────────────────────────────────
    private float[] _hannWindow; // Periodic Hann

    // ── STFT 结果（平铺，布局 [frame * NUM_BINS + k]）────────────────────
    private float[] _stftReal0, _stftImag0;
    private float[] _stftReal1, _stftImag1;

    // ── 原始每声道样本数 ──────────────────────────────────────────────────
    private int _originalSamplesPerChannel;

    // ── 模型输入平铺缓冲（复用）───────────────────────────────────────────
    private float[] _modelInputFlat;

    // ── Wiener 掩码缓冲（复用，消除每帧分配）─────────────────────────────
    private float[] _vocalsMaskBuf;
    private float[] _accompMaskBuf;

    // ═══════════════════════════════════════════════════════════════════════
    // 初始化
    // ═══════════════════════════════════════════════════════════════════════
    public void Initialize(string vocalsModelPath, string accompanimentModelPath)
    {
        try
        {
            _vocalsModel       = new OnnxModel(vocalsModelPath);
            _accompanimentModel = new OnnxModel(accompanimentModelPath);
            _hannWindow         = CreatePeriodicHannWindow(N_FFT);
            Debug.Log("AudioSeparator 初始化成功（v3 深度优化版）");
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
            var sw = System.Diagnostics.Stopwatch.StartNew();

            // ── 1. 解交错立体声（分块写，优于逐样本 Parallel.For）──────────
            _originalSamplesPerChannel = waveform.Length / 2;
            float[] ch0Raw = new float[_originalSamplesPerChannel];
            float[] ch1Raw = new float[_originalSamplesPerChannel];
            Deinterleave(waveform, ch0Raw, ch1Raw, _originalSamplesPerChannel);

            Debug.Log($"[1] 解交错: {_originalSamplesPerChannel} 样本/通道");

            // ── 2. 前导零填充 ───────────────────────────────────────────────
            int paddedLen = N_FFT + _originalSamplesPerChannel;
            float[] ch0 = new float[paddedLen];
            float[] ch1 = new float[paddedLen];
            Array.Copy(ch0Raw, 0, ch0, N_FFT, _originalSamplesPerChannel);
            Array.Copy(ch1Raw, 0, ch1, N_FFT, _originalSamplesPerChannel);

            // ── 3. 并行双声道 STFT（Parallel.For 逐帧并行）────────────────
            int numFrames = 0;
            Parallel.Invoke(
                () => ComputeStftFlatParallel(ch0, out _stftReal0, out _stftImag0, out numFrames),
                () => { int nf; ComputeStftFlatParallel(ch1, out _stftReal1, out _stftImag1, out nf); }
            );

            Debug.Log($"[2] STFT: {numFrames} 帧");

            // ── 4. 构建模型输入 ─────────────────────────────────────────────
            int padding      = (CHUNK_SIZE - numFrames % CHUNK_SIZE) % CHUNK_SIZE;
            int paddedFrames = numFrames + padding;
            int numSplits    = paddedFrames / CHUNK_SIZE;
            int flatSize     = 2 * numSplits * CHUNK_SIZE * MODEL_BINS;

            if (_modelInputFlat == null || _modelInputFlat.Length < flatSize)
                _modelInputFlat = new float[flatSize];
            else
                Array.Clear(_modelInputFlat, 0, flatSize);

            FillModelInputFlat(_modelInputFlat, numFrames, numSplits);
            Debug.Log($"[3] 模型输入 (splits={numSplits}, pad={padding})");

            // ── 5. 双模型并行推理（RunFlatNoCopy 零拷贝）──────────────────
            float[] vocalsOut = null, accompOut = null;
            Parallel.Invoke(
                () => vocalsOut  = _vocalsModel.RunFlatNoCopy(_modelInputFlat, 2, numSplits, CHUNK_SIZE, MODEL_BINS),
                () => accompOut  = _accompanimentModel.RunFlatNoCopy(_modelInputFlat, 2, numSplits, CHUNK_SIZE, MODEL_BINS)
            );
            Debug.Log("[4] 推理完成");

            // ── 6. Wiener 掩码（缓冲复用，x8 展开）────────────────────────
            EnsureMaskBuf(ref _vocalsMaskBuf, flatSize);
            EnsureMaskBuf(ref _accompMaskBuf, flatSize);
            ComputeWienerMaskFlatInPlace(vocalsOut, accompOut, _vocalsMaskBuf, flatSize);
            ComputeWienerMaskFlatInPlace(accompOut, vocalsOut, _accompMaskBuf, flatSize);
            Debug.Log("[5] Wiener 掩码完成");

            // ── 7. 并行 ISTFT + 裁剪 ───────────────────────────────────────
            float[] vocalsAudio = null, accompAudio = null;
            Parallel.Invoke(
                () => vocalsAudio = ReconstructStereoFlat(_vocalsMaskBuf, numSplits, numFrames),
                () => accompAudio = ReconstructStereoFlat(_accompMaskBuf, numSplits, numFrames)
            );

            sw.Stop();
            float dur = _originalSamplesPerChannel / (float)_sampleRate;
            Debug.Log($"✓ 分离完成！耗时={sw.ElapsedMilliseconds}ms  时长={dur:F2}s  RTF={sw.ElapsedMilliseconds / 1000f / dur:F3}");

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
    // 解交错（分块访问，缓存友好）
    // ═══════════════════════════════════════════════════════════════════════
    private static void Deinterleave(float[] src, float[] ch0, float[] ch1, int n)
    {
        const int BLOCK = 1024;
        int blocks = n / BLOCK;
        Parallel.For(0, blocks, b =>
        {
            int start = b * BLOCK;
            for (int i = start; i < start + BLOCK; i++)
            {
                ch0[i] = src[i * 2];
                ch1[i] = src[i * 2 + 1];
            }
        });
        // 尾部
        for (int i = blocks * BLOCK; i < n; i++)
        {
            ch0[i] = src[i * 2];
            ch1[i] = src[i * 2 + 1];
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // STFT — Parallel.For 逐帧并行，ArrayPool 复用缓冲
    // ═══════════════════════════════════════════════════════════════════════
    private void ComputeStftFlatParallel(float[] signal,
                                          out float[] real, out float[] imag, out int numFrames)
    {
        // pad_end=True：帧数 = ceil(signalLength / HOP_LENGTH)
        numFrames = (signal.Length + HOP_LENGTH - 1) / HOP_LENGTH;

        real = new float[numFrames * NUM_BINS];
        imag = new float[numFrames * NUM_BINS];

        float[] hannWin  = _hannWindow;
        float[] realBuf  = real;
        float[] imagBuf  = imag;
        int     sigLen   = signal.Length;

        Parallel.For(0, numFrames, fi =>
        {
            // 每线程租用独立工作缓冲（ArrayPool 无锁分配）
            Complex32[] fft   = ArrayPool<Complex32>.Shared.Rent(N_FFT);
            float[]     frame = ArrayPool<float>.Shared.Rent(N_FFT);
            try
            {
                int offset   = fi * HOP_LENGTH;
                int baseIdx  = fi * NUM_BINS;
                int copyLen  = Math.Min(N_FFT, sigLen - offset);

                if (copyLen > 0)
                {
                    for (int i = 0; i < copyLen; i++)
                        frame[i] = signal[offset + i] * hannWin[i];
                }
                for (int i = (copyLen < 0 ? 0 : copyLen); i < N_FFT; i++)
                    frame[i] = 0f;

                for (int i = 0; i < N_FFT; i++)
                    fft[i] = new Complex32(frame[i], 0f);

                Fourier.Forward(fft, FourierOptions.Matlab);

                for (int k = 0; k < NUM_BINS; k++)
                {
                    realBuf[baseIdx + k] = fft[k].Real;
                    imagBuf[baseIdx + k] = fft[k].Imaginary;
                }
            }
            finally
            {
                ArrayPool<Complex32>.Shared.Return(fft);
                ArrayPool<float>.Shared.Return(frame);
            }
        });
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 构建模型输入：(2, numSplits, CHUNK_SIZE, MODEL_BINS) 平铺
    // ═══════════════════════════════════════════════════════════════════════
    private void FillModelInputFlat(float[] dst, int numFrames, int numSplits)
    {
        int strideCh    = numSplits * CHUNK_SIZE * MODEL_BINS;
        int strideSplit = CHUNK_SIZE * MODEL_BINS;

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
                    if (gf < numFrames)
                    {
                        int srcBase = gf * NUM_BINS;
                        // 使用 Span 减少边界检查
                        var dstSpan = dst.AsSpan(dstBase, MODEL_BINS);
                        var rSpan   = r.AsSpan(srcBase,   MODEL_BINS);
                        var imSpan  = im.AsSpan(srcBase,  MODEL_BINS);
                        for (int k = 0; k < MODEL_BINS; k++)
                        {
                            float rv = rSpan[k], iv = imSpan[k];
                            dstSpan[k] = MathF.Sqrt(rv * rv + iv * iv);
                        }
                    }
                    // padding 帧保持零（已被 Array.Clear 清零）
                }
            }
        });
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Wiener 掩码（原地写入，x8 循环展开）
    // ═══════════════════════════════════════════════════════════════════════
    private static void EnsureMaskBuf(ref float[] buf, int minLen)
    {
        if (buf == null || buf.Length < minLen)
            buf = new float[minLen];
    }

    private static void ComputeWienerMaskFlatInPlace(float[] src, float[] other, float[] mask, int len)
    {
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

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float WienerVal(float s, float o)
    {
        float ss = s * s, oo = o * o;
        return (ss + EPSILON * 0.5f) / (ss + oo + EPSILON);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 立体声重构：并行双声道 ISTFT，再重交错
    // ═══════════════════════════════════════════════════════════════════════
    private float[] ReconstructStereoFlat(float[] mask, int numSplits, int numFrames)
    {
        float[] ch0Out = null, ch1Out = null;
        Parallel.Invoke(
            () => ch0Out = ApplyMaskAndISTFTFlat(mask, 0, numSplits, numFrames),
            () => ch1Out = ApplyMaskAndISTFTFlat(mask, 1, numSplits, numFrames)
        );

        int n         = ch0Out.Length;
        float[] stereo = new float[n * 2];
        Interleave(ch0Out, ch1Out, stereo, n);
        return stereo;
    }

    // 重交错（分块，缓存友好）
    private static void Interleave(float[] ch0, float[] ch1, float[] dst, int n)
    {
        const int BLOCK = 1024;
        int blocks = n / BLOCK;
        Parallel.For(0, blocks, b =>
        {
            int start = b * BLOCK;
            for (int i = start; i < start + BLOCK; i++)
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
    // ISTFT 两阶段并行版
    //
    // Phase-A：Parallel.For 各帧独立执行 IFFT + 加窗，写入 perFrameData[fi]
    // Phase-B：串行 OLA 叠加到 output[]
    // 最后统一乘 2/3，裁剪 output[N_FFT .. N_FFT + originalSamples]
    // ═══════════════════════════════════════════════════════════════════════
    private float[] ApplyMaskAndISTFTFlat(float[] mask, int ch,
                                           int numSplits, int numFrames)
    {
        float[] stftR = ch == 0 ? _stftReal0 : _stftReal1;
        float[] stftI = ch == 0 ? _stftImag0 : _stftImag1;

        int maskChOffset    = ch * numSplits * CHUNK_SIZE * MODEL_BINS;
        int maskSplitStride = CHUNK_SIZE * MODEL_BINS;

        int fullOutputLength = (numFrames - 1) * HOP_LENGTH + N_FFT;

        // ── Phase-A：并行 IFFT + 加窗，每帧输出 N_FFT 个 float ────────────
        // perFrameData[fi] 是租用的 N_FFT float[]（使用 ArrayPool）
        float[][] perFrameData = new float[numFrames][];

        float[] hannWin = _hannWindow;

        Parallel.For(0, numFrames, fi =>
        {
            Complex32[] ifft = ArrayPool<Complex32>.Shared.Rent(N_FFT);
            float[]     fOut = ArrayPool<float>.Shared.Rent(N_FFT);
            try
            {
                int splitIdx    = fi / CHUNK_SIZE;
                int inSplitIdx  = fi % CHUNK_SIZE;
                int stftBase    = fi * NUM_BINS;
                int maskBase    = maskChOffset
                                  + splitIdx * maskSplitStride
                                  + inSplitIdx * MODEL_BINS;

                bool validMask = splitIdx < numSplits;

                if (validMask)
                {
                    for (int k = 0; k < MODEL_BINS; k++)
                    {
                        float m  = mask[maskBase + k];
                        ifft[k]  = new Complex32(stftR[stftBase + k] * m,
                                                  stftI[stftBase + k] * m);
                    }
                }
                else
                {
                    for (int k = 0; k < MODEL_BINS; k++)
                        ifft[k] = new Complex32(stftR[stftBase + k], stftI[stftBase + k]);
                }

                // 高频保持原始
                for (int k = MODEL_BINS; k < NUM_BINS; k++)
                    ifft[k] = new Complex32(stftR[stftBase + k], stftI[stftBase + k]);

                // 共轭对称填充
                for (int k = NUM_BINS; k < N_FFT; k++)
                {
                    int conj = N_FFT - k;
                    ifft[k]  = conj < NUM_BINS
                        ? Complex32.Conjugate(ifft[conj])
                        : Complex32.Zero;
                }

                Fourier.Inverse(ifft, FourierOptions.Matlab);

                // 加窗（写入 fOut）
                for (int i = 0; i < N_FFT; i++)
                    fOut[i] = ifft[i].Real * hannWin[i];

                perFrameData[fi] = fOut; // 暂存引用，Phase-B 读取后归还
            }
            finally
            {
                ArrayPool<Complex32>.Shared.Return(ifft);
                // fOut 不在 finally 归还，等 Phase-B 读取后再还
            }
        });

        // ── Phase-B：串行 OLA 叠加（无并发冲突）──────────────────────────
        float[] output = new float[fullOutputLength];
        for (int fi = 0; fi < numFrames; fi++)
        {
            float[] fOut   = perFrameData[fi];
            int     offset = fi * HOP_LENGTH;
            int     end    = Math.Min(N_FFT, fullOutputLength - offset);
            for (int i = 0; i < end; i++)
                output[offset + i] += fOut[i];

            // 归还到 ArrayPool
            ArrayPool<float>.Shared.Return(fOut);
            perFrameData[fi] = null;
        }

        // ── 统一补偿 2/3 ──────────────────────────────────────────────────
        for (int i = 0; i < fullOutputLength; i++)
            output[i] *= WINDOW_COMPENSATION_FACTOR;

        // ── 裁剪 ──────────────────────────────────────────────────────────
        float[] cropped  = new float[_originalSamplesPerChannel];
        int     copyLen  = Math.Min(_originalSamplesPerChannel, fullOutputLength - N_FFT);
        if (copyLen > 0)
            Array.Copy(output, N_FFT, cropped, 0, copyLen);

        return cropped;
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 辅助
    // ═══════════════════════════════════════════════════════════════════════
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
