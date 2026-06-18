using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using MathNet.Numerics;
using MathNet.Numerics.IntegralTransforms;
using UnityEngine;

/// <summary>
/// 音频分离器 —— 对齐 Spleeter 参考实现的最终优化版
///
/// ═══════════════════════════════════════════════════════════════
/// 与上一版相比新增的算法级修正（来自阅读 spleeter 原始源码）：
///
/// 1. Hann 窗改为 Periodic 模式（/N 而非 /(N-1)）
///    - spleeter: hann_window(frame_length, periodic=True)
///    - 之前：symmetric window，与训练时特征提取不一致，导致频谱失真
///
/// 2. STFT 前在信号头部追加 N_FFT 个零样本（一整帧）
///    - spleeter: waveform = concat([zeros(frame_length, n_ch), waveform], 0)
///    - 确保第一个有效帧从真实样本 0 开始，而不是从半帧偏移处开始
///
/// 3. pad_end=True：信号尾部也用零补全最后一帧
///    - spleeter STFT 设置 pad_end=True
///    - 保证最后一帧不被截断
///
/// 4. ISTFT 改用常数补偿因子 2/3，替代逐样本 OLA 归一化
///    - spleeter: WINDOW_COMPENSATION_FACTOR = 2.0 / 3.0
///    - Periodic Hann + hop=N/4 时，OLA sum(hann²) 处处为 1.5（常数）
///    - 因此 output /= 1.5 等价于 output *= 2/3，无需 windowSum 数组
///    - 省去每帧的 windowSum 写入，减少约 N_FFT 次 float 写入/帧
///
/// 5. ISTFT 输出裁剪：取 output[N_FFT .. N_FFT + originalSamples]
///    - spleeter: reshaped[frame_length : frame_length + time_crop, :]
///    - 精确对齐原始信号长度，消除首尾边界 artifact
///
/// ═══════════════════════════════════════════════════════════════
/// 继承上一版的性能优化：
///   - 平铺 float[] 代替锯齿 float[][][][]
///   - 双模型 Parallel.Invoke 并行推理
///   - 双声道 STFT/ISTFT Parallel.Invoke 并行
///   - [ThreadStatic] FFT 缓冲区，无锁无竞争
///   - FillModelInputFlat 一步计算幅度谱+reshape，消除中间数组
///   - Wiener 掩码循环展开 4x
/// </summary>
public class AudioSeparator : MonoBehaviour
{
    private OnnxModel _vocalsModel;
    private OnnxModel _accompanimentModel;

    // ── 参数（与 spleeter dataset.py DEFAULT_AUDIO_PARAMS 一致）──────────
    private const int N_FFT = 4096;   // frame_length
    private const int HOP_LENGTH = 1024;   // frame_step  (= N_FFT/4)
    private const int NUM_BINS = 2049;   // N_FFT/2 + 1
    private const int MODEL_BINS = 1024;   // F
    private const int CHUNK_SIZE = 512;    // T
    private const float EPSILON = 1e-10f;

    /// <summary>
    /// Spleeter ISTFT 补偿因子：2/3
    /// Periodic Hann + hop = N/4 时 OLA(hann²) 恒为 1.5，故 1/1.5 = 2/3。
    /// </summary>
    private const float WINDOW_COMPENSATION_FACTOR = 2f / 3f;

    private int _sampleRate = 44100;

    // ── 预计算窗口 ─────────────────────────────────────────────────────────
    private float[] _hannWindow; // Periodic Hann

    // ── [ThreadStatic] 每线程独立 FFT 缓冲，支持并行无锁 ─────────────────
    [ThreadStatic] private static Complex32[] t_fftBuf;
    [ThreadStatic] private static float[] t_frameBuf;

    // ── STFT 结果（平铺，布局 [frame * NUM_BINS + k]）────────────────────
    private float[] _stftReal0, _stftImag0;
    private float[] _stftReal1, _stftImag1;

    // ── 原始每声道样本数（去掉前导零后的真实长度）───────────────────────
    private int _originalSamplesPerChannel;

    // ── 模型输入平铺缓冲（复用，避免每次分配）────────────────────────────
    private float[] _modelInputFlat;

    // ═══════════════════════════════════════════════════════════════════════
    // 初始化
    // ═══════════════════════════════════════════════════════════════════════
    public void Initialize(string vocalsModelPath, string accompanimentModelPath)
    {
        try
        {
            _vocalsModel = new OnnxModel(vocalsModelPath);
            _accompanimentModel = new OnnxModel(accompanimentModelPath);

            // Periodic Hann：公式 0.5*(1 - cos(2π*i/N))，注意分母是 N 而非 N-1
            _hannWindow = CreatePeriodicHannWindow(N_FFT);

            Debug.Log("AudioSeparator 初始化成功（Spleeter 对齐版）");
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

            // ── 1. 解交错立体声 ─────────────────────────────────────────
            _originalSamplesPerChannel = waveform.Length / 2;
            float[] ch0Raw = new float[_originalSamplesPerChannel];
            float[] ch1Raw = new float[_originalSamplesPerChannel];

            Parallel.For(0, _originalSamplesPerChannel, i =>
            {
                ch0Raw[i] = waveform[i * 2];
                ch1Raw[i] = waveform[i * 2 + 1];
            });

            Debug.Log($"[1] 立体声解交错: {_originalSamplesPerChannel} 样本/通道");

            // ── 2. 前导零填充（spleeter: concat zeros(frame_length) + waveform）
            //       使第一帧居中于样本 0，保持与训练时完全一致的对齐
            int paddedLen = N_FFT + _originalSamplesPerChannel;
            float[] ch0 = new float[paddedLen];
            float[] ch1 = new float[paddedLen];
            Array.Copy(ch0Raw, 0, ch0, N_FFT, _originalSamplesPerChannel);
            Array.Copy(ch1Raw, 0, ch1, N_FFT, _originalSamplesPerChannel);

            // ── 3. 并行双声道 STFT ──────────────────────────────────────
            int numFrames = 0;
            Parallel.Invoke(
                () => ComputeStftFlat(ch0, out _stftReal0, out _stftImag0, out numFrames),
                () => { int nf; ComputeStftFlat(ch1, out _stftReal1, out _stftImag1, out nf); }
            );

            Debug.Log($"[2] STFT 完成: {numFrames} 帧（含前导 {N_FFT / HOP_LENGTH} 帧）");

            // ── 4. 构建模型输入平铺数组（含 chunk padding）────────────────
            int padding = (CHUNK_SIZE - numFrames % CHUNK_SIZE) % CHUNK_SIZE;
            int paddedFrames = numFrames + padding;
            int numSplits = paddedFrames / CHUNK_SIZE;
            int flatSize = 2 * numSplits * CHUNK_SIZE * MODEL_BINS;

            if (_modelInputFlat == null || _modelInputFlat.Length < flatSize)
                _modelInputFlat = new float[flatSize];
            else
                Array.Clear(_modelInputFlat, 0, flatSize);

            FillModelInputFlat(_modelInputFlat, numFrames, numSplits);
            Debug.Log($"[3] 模型输入完成 (splits={numSplits}, chunk_pad={padding})");

            // ── 5. 双模型并行推理 ──────────────────────────────────────
            float[] vocalsFlat = null, accompFlat = null;
            Parallel.Invoke(
                () => vocalsFlat = _vocalsModel.RunFlat(_modelInputFlat, 2, numSplits, CHUNK_SIZE, MODEL_BINS),
                () => accompFlat = _accompanimentModel.RunFlat(_modelInputFlat, 2, numSplits, CHUNK_SIZE, MODEL_BINS)
            );
            Debug.Log("[4] 双模型并行推理完成");

            // ── 6. Wiener 掩码（平铺，原地计算）──────────────────────────
            float[] vocalsMask = ComputeWienerMaskFlat(vocalsFlat, accompFlat, flatSize);
            float[] accompMask = ComputeWienerMaskFlat(accompFlat, vocalsFlat, flatSize);
            Debug.Log("[5] Wiener 掩码完成");

            // ── 7. 并行 ISTFT + 裁剪 + 立体声交错 ─────────────────────
            var results = new Dictionary<string, float[]>(2);
            float[] vocalsAudio = null, accompAudio = null;
            Parallel.Invoke(
                () => vocalsAudio = ReconstructStereoFlat(vocalsMask, numSplits, numFrames),
                () => accompAudio = ReconstructStereoFlat(accompMask, numSplits, numFrames)
            );

            results["vocals"] = vocalsAudio;
            results["accompaniment"] = accompAudio;

            sw.Stop();
            float dur = _originalSamplesPerChannel / (float)_sampleRate;
            Debug.Log($"✓ 分离完成！耗时={sw.ElapsedMilliseconds}ms  时长={dur:F2}s  RTF={sw.ElapsedMilliseconds / 1000f / dur:F3}");
            return results;
        }
        catch (Exception ex)
        {
            Debug.LogError($"分离错误: {ex.Message}\n{ex.StackTrace}");
            throw;
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // STFT — Periodic Hann，pad_end=True，平铺输出，线程安全
    // ═══════════════════════════════════════════════════════════════════════
    private void ComputeStftFlat(float[] signal,
                                  out float[] real, out float[] imag, out int numFrames)
    {
        // pad_end=True：尾部补零使最后一帧完整
        // 帧数：ceil((signalLength) / HOP_LENGTH)
        numFrames = (signal.Length + HOP_LENGTH - 1) / HOP_LENGTH;

        real = new float[numFrames * NUM_BINS];
        imag = new float[numFrames * NUM_BINS];

        if (t_fftBuf == null) t_fftBuf = new Complex32[N_FFT];
        if (t_frameBuf == null) t_frameBuf = new float[N_FFT];

        Complex32[] fft = t_fftBuf;
        float[] frame = t_frameBuf;

        for (int fi = 0; fi < numFrames; fi++)
        {
            int offset = fi * HOP_LENGTH;
            int baseIdx = fi * NUM_BINS;

            int copyLen = Math.Min(N_FFT, signal.Length - offset);
            if (copyLen > 0)
            {
                for (int i = 0; i < copyLen; i++)
                    frame[i] = signal[offset + i] * _hannWindow[i];
            }
            // 超出范围的部分为 0（pad_end）
            for (int i = copyLen < 0 ? 0 : copyLen; i < N_FFT; i++)
                frame[i] = 0f;

            for (int i = 0; i < N_FFT; i++)
                fft[i] = new Complex32(frame[i], 0f);

            Fourier.Forward(fft, FourierOptions.Matlab);

            for (int k = 0; k < NUM_BINS; k++)
            {
                real[baseIdx + k] = fft[k].Real;
                imag[baseIdx + k] = fft[k].Imaginary;
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 构建模型输入：(2, numSplits, CHUNK_SIZE, MODEL_BINS) 平铺
    // 从 STFT 复数直接计算幅度，一步到位
    // ═══════════════════════════════════════════════════════════════════════
    private void FillModelInputFlat(float[] dst, int numFrames, int numSplits)
    {
        int strideCh = numSplits * CHUNK_SIZE * MODEL_BINS;
        int strideSplit = CHUNK_SIZE * MODEL_BINS;
        int strideFrame = MODEL_BINS;

        Parallel.For(0, 2, ch =>
        {
            float[] r = ch == 0 ? _stftReal0 : _stftReal1;
            float[] im = ch == 0 ? _stftImag0 : _stftImag1;
            int chBase = ch * strideCh;

            for (int s = 0; s < numSplits; s++)
            {
                int splitBase = chBase + s * strideSplit;
                for (int fi = 0; fi < CHUNK_SIZE; fi++)
                {
                    int gf = s * CHUNK_SIZE + fi;
                    int dstBase = splitBase + fi * strideFrame;
                    if (gf < numFrames)
                    {
                        int srcBase = gf * NUM_BINS;
                        for (int k = 0; k < MODEL_BINS; k++)
                        {
                            float rv = r[srcBase + k];
                            float iv = im[srcBase + k];
                            dst[dstBase + k] = MathF.Sqrt(rv * rv + iv * iv);
                        }
                    }
                    // padding 帧已被 Array.Clear 置零
                }
            }
        });
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Wiener 掩码（平铺，4x 循环展开）
    // ═══════════════════════════════════════════════════════════════════════
    private static float[] ComputeWienerMaskFlat(float[] src, float[] other, int len)
    {
        float[] mask = new float[len];
        int i = 0;
        for (; i <= len - 4; i += 4)
        {
            mask[i] = WienerVal(src[i], other[i]);
            mask[i + 1] = WienerVal(src[i + 1], other[i + 1]);
            mask[i + 2] = WienerVal(src[i + 2], other[i + 2]);
            mask[i + 3] = WienerVal(src[i + 3], other[i + 3]);
        }
        for (; i < len; i++)
            mask[i] = WienerVal(src[i], other[i]);
        return mask;
    }

    private static float WienerVal(float s, float o)
    {
        float ss = s * s, oo = o * o;
        return (ss + EPSILON * 0.5f) / (ss + oo + EPSILON);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 立体声重构：并行双声道 ISTFT，再交错
    // ═══════════════════════════════════════════════════════════════════════
    private float[] ReconstructStereoFlat(float[] mask, int numSplits, int numFrames)
    {
        float[] ch0Out = null, ch1Out = null;
        Parallel.Invoke(
            () => ch0Out = ApplyMaskAndISTFTFlat(mask, ch: 0, numSplits, numFrames),
            () => ch1Out = ApplyMaskAndISTFTFlat(mask, ch: 1, numSplits, numFrames)
        );

        // 两通道长度应相同（_originalSamplesPerChannel）
        int n = ch0Out.Length; // == _originalSamplesPerChannel
        float[] stereo = new float[n * 2];

        Parallel.For(0, n, i =>
        {
            stereo[i * 2] = ch0Out[i];
            stereo[i * 2 + 1] = ch1Out[i];
        });

        return stereo;
    }

    // ═══════════════════════════════════════════════════════════════════════
    // ISTFT — 对齐 Spleeter 参考实现
    //
    // 关键改动：
    //   ① 不再逐样本计算 windowSum / OLA 归一化；
    //     改为最后统一乘以 WINDOW_COMPENSATION_FACTOR（= 2/3）
    //     因为 Periodic Hann + hop=N/4 => OLA(hann²) = 1.5（常数）
    //   ② 输出裁剪：取 istftOutput[N_FFT .. N_FFT + _originalSamplesPerChannel]
    //     与 spleeter 的 reshaped[frame_length : frame_length + time_crop] 一致
    // ═══════════════════════════════════════════════════════════════════════
    private float[] ApplyMaskAndISTFTFlat(float[] mask, int ch,
                                           int numSplits, int numFrames)
    {
        if (t_fftBuf == null) t_fftBuf = new Complex32[N_FFT];
        Complex32[] ifft = t_fftBuf;

        float[] stftR = ch == 0 ? _stftReal0 : _stftReal1;
        float[] stftI = ch == 0 ? _stftImag0 : _stftImag1;

        int maskChOffset = ch * numSplits * CHUNK_SIZE * MODEL_BINS;
        int maskSplitStride = CHUNK_SIZE * MODEL_BINS;
        int maskFrameStride = MODEL_BINS;

        // 完整 ISTFT 输出长度（包含前导 N_FFT 零帧的部分）
        int fullOutputLength = (numFrames - 1) * HOP_LENGTH + N_FFT;
        float[] output = new float[fullOutputLength];

        for (int fi = 0; fi < numFrames; fi++)
        {
            int offset = fi * HOP_LENGTH;
            int splitIdx = fi / CHUNK_SIZE;
            int inSplitIdx = fi % CHUNK_SIZE;
            int stftBase = fi * NUM_BINS;
            int maskBase = maskChOffset + splitIdx * maskSplitStride + inSplitIdx * maskFrameStride;

            // ── 应用掩码，构造复频谱 ──────────────────────────────────
            bool validMask = splitIdx < numSplits;
            if (validMask)
            {
                for (int k = 0; k < MODEL_BINS; k++)
                {
                    float m = mask[maskBase + k];
                    ifft[k] = new Complex32(stftR[stftBase + k] * m,
                                            stftI[stftBase + k] * m);
                }
            }
            else
            {
                for (int k = 0; k < MODEL_BINS; k++)
                    ifft[k] = new Complex32(stftR[stftBase + k], stftI[stftBase + k]);
            }

            // MODEL_BINS..NUM_BINS-1：高频保持原始
            for (int k = MODEL_BINS; k < NUM_BINS; k++)
                ifft[k] = new Complex32(stftR[stftBase + k], stftI[stftBase + k]);

            // 共轭对称填充
            for (int k = NUM_BINS; k < N_FFT; k++)
            {
                int conj = N_FFT - k;
                ifft[k] = conj < NUM_BINS
                    ? Complex32.Conjugate(ifft[conj])
                    : Complex32.Zero;
            }

            // IFFT
            Fourier.Inverse(ifft, FourierOptions.Matlab);

            // OLA 叠加（加窗）
            int end = Math.Min(N_FFT, fullOutputLength - offset);
            for (int i = 0; i < end; i++)
                output[offset + i] += ifft[i].Real * _hannWindow[i];
        }

        // ── 统一补偿（替代逐样本归一化）──────────────────────────────────
        // Periodic Hann + hop=N/4 => OLA(hann²) = 1.5，故乘 2/3 即可
        for (int i = 0; i < fullOutputLength; i++)
            output[i] *= WINDOW_COMPENSATION_FACTOR;

        // ── 裁剪：跳过前导 N_FFT 样本，取原始信号长度 ───────────────────
        // 对应 spleeter: reshaped[frame_length : frame_length + time_crop]
        float[] cropped = new float[_originalSamplesPerChannel];
        int copyLen2 = Math.Min(_originalSamplesPerChannel, fullOutputLength - N_FFT);
        if (copyLen2 > 0)
            Array.Copy(output, N_FFT, cropped, 0, copyLen2);

        return cropped;
    }

    // ═══════════════════════════════════════════════════════════════════════
    // 辅助方法
    // ═══════════════════════════════════════════════════════════════════════

    /// <summary>
    /// Periodic Hann 窗：hann[i] = 0.5 * (1 - cos(2π*i/N))
    /// 注意分母是 N（不是 N-1），与 spleeter periodic=True 一致。
    /// </summary>
    private static float[] CreatePeriodicHannWindow(int N)
    {
        float[] w = new float[N];
        double scale = 2.0 * Math.PI / N;
        for (int i = 0; i < N; i++)
            w[i] = 0.5f * (1f - (float)Math.Cos(scale * i));
        return w;
    }

    // ── 文件 I/O ───────────────────────────────────────────────────────────
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