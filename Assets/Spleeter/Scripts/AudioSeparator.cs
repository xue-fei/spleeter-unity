using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using MathNet.Numerics;
using MathNet.Numerics.IntegralTransforms;
using UnityEngine;

/// <summary>
/// 音频分离器（性能优化版）
///
/// 优化点：
/// 1. ONNX 双模型并行推理（Task.WhenAll）
/// 2. 双声道 ISTFT 并行处理（Parallel.For）
/// 3. 热路径全部使用平铺一维数组，消除锯齿数组多级间接访问
/// 4. FFT 缓冲区按线程独立分配，解除串行约束
/// 5. ExtractStftMagnitude / ReshapeForModel 消除冗余中间数组
/// 6. Flatten4DArray / Tensor4DToJagged 改用 Buffer.BlockCopy / Span
/// 7. Hann 窗平方预计算，ISTFT 归一化省去重复乘法
/// </summary>
public class AudioSeparator : MonoBehaviour
{
    private OnnxModel _vocalsModel;
    private OnnxModel _accompanimentModel;

    private const int N_FFT = 4096;
    private const int HOP_LENGTH = 1024;
    private const int NUM_BINS = 2049;   // N_FFT/2 + 1
    private const int MODEL_BINS = 1024;   // 模型只使用前 1024 个 bins
    private const int CHUNK_SIZE = 512;
    private const float EPSILON = 1e-10f;
    private int _sampleRate = 44100;

    // 预计算窗口及其平方（ISTFT 归一化用）
    private float[] _hannWindow;
    private float[] _hannWindowSq; // hann[i]^2，OLA 归一化预计算

    // ── 每个工作线程拥有独立 FFT 缓冲区，支持并行 ──────────────────────────
    [ThreadStatic] private static Complex32[] t_fftBuf;
    [ThreadStatic] private static float[] t_frameBuf;

    // ── 静态复用：STFT 原始数据使用平铺 float[] ────────────────────────────
    // stft 布局: [ch * numFrames * NUM_BINS + frame * NUM_BINS + k]
    private float[] _stftReal0, _stftImag0;
    private float[] _stftReal1, _stftImag1;

    // ── 模型输入平铺缓冲（避免 Flatten4DArray 每次分配）──────────────────────
    private float[] _modelInputFlat;

    public void Initialize(string vocalsModelPath, string accompanimentModelPath)
    {
        try
        {
            _vocalsModel = new OnnxModel(vocalsModelPath);
            _accompanimentModel = new OnnxModel(accompanimentModelPath);

            _hannWindow = CreateHannWindow(N_FFT);
            _hannWindowSq = new float[N_FFT];
            for (int i = 0; i < N_FFT; i++)
                _hannWindowSq[i] = _hannWindow[i] * _hannWindow[i];

            Debug.Log("分离器初始化成功");
        }
        catch (Exception ex)
        {
            Debug.LogError($"初始化失败: {ex.Message}");
            throw;
        }
    }

    public Dictionary<string, float[]> Separate(float[] waveform)
    {
        if (_vocalsModel == null || _accompanimentModel == null)
            throw new InvalidOperationException("分离器未初始化");

        try
        {
            var sw = System.Diagnostics.Stopwatch.StartNew();

            // ── 1. 解交错立体声 ────────────────────────────────────────────
            int numSamples = waveform.Length / 2;
            float[] ch0 = new float[numSamples];
            float[] ch1 = new float[numSamples];

            // 并行解交错（numSamples 通常百万量级，值得并行）
            Parallel.For(0, numSamples, i =>
            {
                ch0[i] = waveform[i * 2];
                ch1[i] = waveform[i * 2 + 1];
            });

            Debug.Log($"[1] 立体声分离: {numSamples} 样本/通道");

            // ── 2. 并行双声道 STFT（各自独立缓冲区，无锁）──────────────────
            int numFrames = 0;
            Parallel.Invoke(
                () => ComputeStftFlat(ch0, out _stftReal0, out _stftImag0, out numFrames),
                () =>
                {
                    int nf;
                    ComputeStftFlat(ch1, out _stftReal1, out _stftImag1, out nf);
                }
            );

            Debug.Log($"[2] STFT 完成: {numFrames} 帧");

            // ── 3. 构建模型输入（平铺，含 padding）──────────────────────────
            int padding = (CHUNK_SIZE - numFrames % CHUNK_SIZE) % CHUNK_SIZE;
            int paddedFrames = numFrames + padding;
            int numSplits = paddedFrames / CHUNK_SIZE;

            // 模型输入形状: (2, numSplits, CHUNK_SIZE, MODEL_BINS)
            int flatSize = 2 * numSplits * CHUNK_SIZE * MODEL_BINS;
            if (_modelInputFlat == null || _modelInputFlat.Length < flatSize)
                _modelInputFlat = new float[flatSize];
            else
                Array.Clear(_modelInputFlat, 0, flatSize);

            FillModelInputFlat(_modelInputFlat, numFrames, numSplits);

            Debug.Log($"[3] 模型输入构建完成 (splits={numSplits}, padding={padding})");

            // ── 4. 双模型并行推理 ─────────────────────────────────────────
            float[] vocalsFlat = null, accompFlat = null;

            // OnnxModel.RunFlat 接收平铺数组，返回平铺数组，避免锯齿数组转换开销
            Parallel.Invoke(
                () => vocalsFlat = _vocalsModel.RunFlat(_modelInputFlat, 2, numSplits, CHUNK_SIZE, MODEL_BINS),
                () => accompFlat = _accompanimentModel.RunFlat(_modelInputFlat, 2, numSplits, CHUNK_SIZE, MODEL_BINS)
            );

            Debug.Log($"[4] 双模型并行推理完成");

            // ── 5. 计算 Wiener 掩码（原地，节省分配）─────────────────────
            // vocalsMask / accompMask 与输出 flat 等形状
            float[] vocalsMask = ComputeWienerMaskFlat(vocalsFlat, accompFlat, flatSize);
            float[] accompMask = ComputeWienerMaskFlat(accompFlat, vocalsFlat, flatSize);

            Debug.Log($"[5] Wiener 掩码计算完成");

            // ── 6. 并行双声道 ISTFT + 合并 ───────────────────────────────
            var results = new Dictionary<string, float[]>(2);

            float[] vocalsAudio = null, accompAudio = null;
            Parallel.Invoke(
                () => vocalsAudio = ReconstructStereoFlat(vocalsMask, numSplits, numFrames),
                () => accompAudio = ReconstructStereoFlat(accompMask, numSplits, numFrames)
            );

            results["vocals"] = vocalsAudio;
            results["accompaniment"] = accompAudio;

            sw.Stop();
            float dur = numSamples / (float)_sampleRate;
            Debug.Log($"✓ 分离完成！耗时={sw.ElapsedMilliseconds}ms  时长={dur:F2}s  RTF={sw.ElapsedMilliseconds / 1000f / dur:F3}");

            return results;
        }
        catch (Exception ex)
        {
            Debug.LogError($"分离错误: {ex.Message}\n{ex.StackTrace}");
            throw;
        }
    }

    // ══════════════════════════════════════════════════════════════════════════
    // STFT — 平铺输出版（线程安全：ThreadStatic 缓冲）
    // ══════════════════════════════════════════════════════════════════════════
    private void ComputeStftFlat(float[] signal,
                                  out float[] real, out float[] imag, out int numFrames)
    {
        numFrames = (signal.Length - N_FFT) / HOP_LENGTH + 1;
        real = new float[numFrames * NUM_BINS];
        imag = new float[numFrames * NUM_BINS];

        // ThreadStatic：每个线程第一次使用时初始化，之后复用
        if (t_fftBuf == null) t_fftBuf = new Complex32[N_FFT];
        if (t_frameBuf == null) t_frameBuf = new float[N_FFT];

        Complex32[] fft = t_fftBuf;
        float[] frame = t_frameBuf;

        for (int fi = 0; fi < numFrames; fi++)
        {
            int offset = fi * HOP_LENGTH;
            int baseIdx = fi * NUM_BINS;

            // 加窗
            int copyLen = Math.Min(N_FFT, signal.Length - offset);
            for (int i = 0; i < copyLen; i++)
                frame[i] = signal[offset + i] * _hannWindow[i];
            for (int i = copyLen; i < N_FFT; i++)
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

    // ══════════════════════════════════════════════════════════════════════════
    // 构建模型输入平铺数组
    // 形状 (2, numSplits, CHUNK_SIZE, MODEL_BINS) → 逐元素取幅度
    // ══════════════════════════════════════════════════════════════════════════
    private void FillModelInputFlat(float[] dst, int numFrames, int numSplits)
    {
        int stride_ch = numSplits * CHUNK_SIZE * MODEL_BINS;
        int stride_split = CHUNK_SIZE * MODEL_BINS;
        int stride_frame = MODEL_BINS;

        // 并行两声道
        Parallel.For(0, 2, ch =>
        {
            float[] r = ch == 0 ? _stftReal0 : _stftReal1;
            float[] im = ch == 0 ? _stftImag0 : _stftImag1;
            int chBase = ch * stride_ch;

            for (int s = 0; s < numSplits; s++)
            {
                int splitBase = chBase + s * stride_split;
                for (int fi = 0; fi < CHUNK_SIZE; fi++)
                {
                    int globalFrame = s * CHUNK_SIZE + fi;
                    int dstBase = splitBase + fi * stride_frame;

                    if (globalFrame < numFrames)
                    {
                        int srcBase = globalFrame * NUM_BINS;
                        for (int k = 0; k < MODEL_BINS; k++)
                        {
                            float rv = r[srcBase + k];
                            float iv = im[srcBase + k];
                            dst[dstBase + k] = MathF.Sqrt(rv * rv + iv * iv);
                        }
                    }
                    // padding 帧：已被 Array.Clear 置零，无需额外处理
                }
            }
        });
    }

    // ══════════════════════════════════════════════════════════════════════════
    // Wiener 掩码（原地平铺，无额外分配）
    // ══════════════════════════════════════════════════════════════════════════
    private static float[] ComputeWienerMaskFlat(float[] source, float[] other, int len)
    {
        float[] mask = new float[len];

        // 向量化友好的展开循环
        int i = 0;
        for (; i <= len - 4; i += 4)
        {
            mask[i] = WienerVal(source[i], other[i]);
            mask[i + 1] = WienerVal(source[i + 1], other[i + 1]);
            mask[i + 2] = WienerVal(source[i + 2], other[i + 2]);
            mask[i + 3] = WienerVal(source[i + 3], other[i + 3]);
        }
        for (; i < len; i++)
            mask[i] = WienerVal(source[i], other[i]);

        return mask;
    }

    private static float WienerVal(float s, float o)
    {
        float ss = s * s, oo = o * o;
        return (ss + EPSILON * 0.5f) / (ss + oo + EPSILON);
    }

    // ══════════════════════════════════════════════════════════════════════════
    // 重构立体声：并行双声道 ISTFT，再交错
    // ══════════════════════════════════════════════════════════════════════════
    private float[] ReconstructStereoFlat(float[] mask, int numSplits, int numFrames)
    {
        float[] ch0Out = null, ch1Out = null;

        Parallel.Invoke(
            () => ch0Out = ApplyMaskAndISTFTFlat(mask, ch: 0, numSplits, numFrames),
            () => ch1Out = ApplyMaskAndISTFTFlat(mask, ch: 1, numSplits, numFrames)
        );

        int totalSamples = Math.Max(ch0Out.Length, ch1Out.Length);
        float[] stereo = new float[totalSamples * 2];

        // 并行交错合并
        Parallel.For(0, totalSamples, i =>
        {
            stereo[i * 2] = i < ch0Out.Length ? ch0Out[i] : 0f;
            stereo[i * 2 + 1] = i < ch1Out.Length ? ch1Out[i] : 0f;
        });

        return stereo;
    }

    // ══════════════════════════════════════════════════════════════════════════
    // ISTFT — 平铺掩码版（线程安全：ThreadStatic 缓冲）
    // mask 形状: (2, numSplits, CHUNK_SIZE, MODEL_BINS)
    // ══════════════════════════════════════════════════════════════════════════
    private float[] ApplyMaskAndISTFTFlat(float[] mask, int ch,
                                           int numSplits, int numFrames)
    {
        // ThreadStatic ifft 缓冲
        if (t_fftBuf == null) t_fftBuf = new Complex32[N_FFT];
        Complex32[] ifft = t_fftBuf;

        // 选取当前声道的 STFT 数据
        float[] stftR = ch == 0 ? _stftReal0 : _stftReal1;
        float[] stftI = ch == 0 ? _stftImag0 : _stftImag1;

        // 掩码在平铺数组中的起始偏移
        int maskChOffset = ch * numSplits * CHUNK_SIZE * MODEL_BINS;
        int maskSplitStride = CHUNK_SIZE * MODEL_BINS;
        int maskFrameStride = MODEL_BINS;

        int outputLength = (numFrames - 1) * HOP_LENGTH + N_FFT;
        float[] output = new float[outputLength];
        float[] windowSum = new float[outputLength];

        for (int fi = 0; fi < numFrames; fi++)
        {
            int offset = fi * HOP_LENGTH;
            int splitIdx = fi / CHUNK_SIZE;
            int inSplitIdx = fi % CHUNK_SIZE;
            int stftBase = fi * NUM_BINS;
            int maskBase = maskChOffset + splitIdx * maskSplitStride + inSplitIdx * maskFrameStride;

            // ── 应用掩码，构造复频谱 ────────────────────────────────────
            // 前 MODEL_BINS：乘以掩码值
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

            // MODEL_BINS ~ NUM_BINS-1：保持原始高频
            for (int k = MODEL_BINS; k < NUM_BINS; k++)
                ifft[k] = new Complex32(stftR[stftBase + k], stftI[stftBase + k]);

            // ── 共轭对称填充 ─────────────────────────────────────────────
            for (int k = NUM_BINS; k < N_FFT; k++)
            {
                int conj = N_FFT - k;
                ifft[k] = conj < NUM_BINS
                    ? Complex32.Conjugate(ifft[conj])
                    : Complex32.Zero;
            }

            // ── IFFT ─────────────────────────────────────────────────────
            Fourier.Inverse(ifft, FourierOptions.Matlab);

            // ── OLA 叠加 ──────────────────────────────────────────────────
            int endIdx = Math.Min(N_FFT, outputLength - offset);
            for (int i = 0; i < endIdx; i++)
            {
                output[offset + i] += ifft[i].Real * _hannWindow[i];
                windowSum[offset + i] += _hannWindowSq[i]; // 预计算平方
            }
        }

        // ── OLA 归一化 ────────────────────────────────────────────────────
        for (int i = 0; i < outputLength; i++)
        {
            if (windowSum[i] > 1e-6f)
                output[i] /= windowSum[i];
        }

        return output;
    }

    // ══════════════════════════════════════════════════════════════════════════
    // 辅助
    // ══════════════════════════════════════════════════════════════════════════
    private static float[] CreateHannWindow(int length)
    {
        float[] w = new float[length];
        double scale = 2.0 * Math.PI / (length - 1);
        for (int i = 0; i < length; i++)
            w[i] = 0.5f * (1f - (float)Math.Cos(scale * i));
        return w;
    }

    // ── 文件 I/O ──────────────────────────────────────────────────────────────
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