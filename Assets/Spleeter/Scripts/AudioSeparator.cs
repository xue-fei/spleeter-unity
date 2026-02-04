using System;
using System.Collections.Generic;
using System.IO;
using MathNet.Numerics;
using MathNet.Numerics.IntegralTransforms;
using UnityEngine;

/// <summary>
/// 修复音频重构问题的音频分离器
/// </summary>
public class AudioSeparator : MonoBehaviour
{
    private OnnxModel _vocalsModel;
    private OnnxModel _accompanimentModel;

    private const int N_FFT = 4096;
    private const int HOP_LENGTH = 1024;
    private const int NUM_BINS = 2049; // N_FFT/2 + 1 = 2049
    private const int MODEL_BINS = 1024; // 模型只使用前1024个bins
    private const int CHUNK_SIZE = 512;
    private const float EPSILON = 1e-10f;
    private int _sampleRate = 44100;

    // 性能优化：预分配缓冲区
    private float[] _hannWindow;
    private Complex32[] _fftBuffer;
    private Complex32[] _ifftBuffer;
    private float[] _frameBuffer;

    public void Initialize(string vocalsModelPath, string accompanimentModelPath)
    {
        try
        {
            _vocalsModel = new OnnxModel(vocalsModelPath);
            _accompanimentModel = new OnnxModel(accompanimentModelPath);

            // 预分配缓冲区
            _hannWindow = CreateHannWindow(N_FFT);
            _fftBuffer = new Complex32[N_FFT];
            _ifftBuffer = new Complex32[N_FFT];
            _frameBuffer = new float[N_FFT];

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
        {
            throw new InvalidOperationException("分离器未初始化");
        }

        try
        {
            var stopwatch = System.Diagnostics.Stopwatch.StartNew();

            // 分离立体声
            int numSamples = waveform.Length / 2;
            float[][] waveformStereo = new float[2][];
            waveformStereo[0] = new float[numSamples];
            waveformStereo[1] = new float[numSamples];

            for (int i = 0; i < numSamples; i++)
            {
                waveformStereo[0][i] = waveform[i * 2];
                waveformStereo[1][i] = waveform[i * 2 + 1];
            }

            Debug.Log($"[1] 立体声分离完成: {numSamples} 样本每通道");

            // 计算STFT
            StftResult[] stftResults = new StftResult[2];
            stftResults[0] = ComputeStft(waveformStereo[0]);
            stftResults[1] = ComputeStft(waveformStereo[1]);

            int numFrames = stftResults[0].NumFrames;
            Debug.Log($"[2] STFT计算完成 - {numFrames} 帧");

            // 提取幅度谱
            float[][][] stftData = ExtractStftMagnitude(stftResults);

            // 填充到512的倍数
            int padding = (CHUNK_SIZE - (numFrames % CHUNK_SIZE)) % CHUNK_SIZE;
            int paddedFrames = numFrames + padding;

            if (padding > 0)
            {
                stftData = PadStftData(stftData, padding);
            }

            Debug.Log($"[3] 幅度谱提取完成，填充 {padding} 帧, 总帧数: {paddedFrames}");

            // 重新形成输入
            float[][][][] modelInput = ReshapeForModel(stftData);
            Debug.Log($"[4] 模型输入转换完成");

            // 运行模型
            var vocalsSpec = _vocalsModel.Run(modelInput);
            var accompanimentSpec = _accompanimentModel.Run(modelInput);

            Debug.Log($"[5] 模型推理完成");

            // 计算掩码 - 使用C++代码中的Wiener滤波公式
            float[][][][] vocalsMask = ComputeMaskWiener(vocalsSpec, accompanimentSpec);
            float[][][][] accompanimentMask = ComputeMaskWiener(accompanimentSpec, vocalsSpec);

            Debug.Log($"[6] Wiener掩码计算完成");

            // 重构音频
            var results = new Dictionary<string, float[]>();

            Debug.Log($"[7] 开始重构音频...");

            // 使用原始的未填充帧数
            results["vocals"] = ReconstructAudioFixed(vocalsMask, stftResults, numFrames);
            Debug.Log($"[8] 人声重构完成");

            results["accompaniment"] = ReconstructAudioFixed(accompanimentMask, stftResults, numFrames);
            Debug.Log($"[9] 伴奏重构完成");

            stopwatch.Stop();
            float audioDuration = numSamples / (float)_sampleRate;
            float rtf = stopwatch.ElapsedMilliseconds / 1000f / audioDuration;

            Debug.Log($"✓ 分离完成！");
            Debug.Log($"  总耗时: {stopwatch.ElapsedMilliseconds}ms");
            Debug.Log($"  音频时长: {audioDuration:F2}s");
            Debug.Log($"  RTF: {rtf:F3}");

            return results;
        }
        catch (Exception ex)
        {
            Debug.LogError($"分离过程错误: {ex.Message}\n{ex.StackTrace}");
            throw;
        }
    }

    /// <summary>
    /// 计算STFT
    /// </summary>
    private StftResult ComputeStft(float[] signal)
    {
        int numFrames = (signal.Length - N_FFT) / HOP_LENGTH + 1;
        float[] realPart = new float[numFrames * NUM_BINS];
        float[] imagPart = new float[numFrames * NUM_BINS];

        for (int frameIdx = 0; frameIdx < numFrames; frameIdx++)
        {
            int offset = frameIdx * HOP_LENGTH;

            // 提取帧并应用窗口
            for (int i = 0; i < N_FFT; i++)
            {
                int sampleIdx = offset + i;
                _frameBuffer[i] = sampleIdx < signal.Length ?
                    signal[sampleIdx] * _hannWindow[i] : 0f;
            }

            // 执行FFT
            for (int i = 0; i < N_FFT; i++)
                _fftBuffer[i] = new Complex32(_frameBuffer[i], 0);

            Fourier.Forward(_fftBuffer, FourierOptions.Matlab);

            // 存储结果
            int baseIdx = frameIdx * NUM_BINS;
            for (int k = 0; k < NUM_BINS; k++)
            {
                realPart[baseIdx + k] = _fftBuffer[k].Real;
                imagPart[baseIdx + k] = _fftBuffer[k].Imaginary;
            }
        }

        return new StftResult
        {
            Real = realPart,
            Imag = imagPart,
            NumFrames = numFrames
        };
    }

    /// <summary>
    /// 修正的音频重构函数
    /// </summary>
    private float[] ReconstructAudioFixed(float[][][][] mask, StftResult[] stftResults, int originalNumFrames)
    {
        float[][] reconstructed = new float[2][];

        for (int ch = 0; ch < 2; ch++)
        {
            reconstructed[ch] = ApplyMaskAndISTFT(mask[ch], stftResults[ch], originalNumFrames);
        }

        // 交错成立体声
        int totalSamples = Math.Max(reconstructed[0].Length, reconstructed[1].Length);
        float[] stereo = new float[totalSamples * 2];

        for (int i = 0; i < totalSamples; i++)
        {
            stereo[i * 2] = i < reconstructed[0].Length ? reconstructed[0][i] : 0f;
            stereo[i * 2 + 1] = i < reconstructed[1].Length ? reconstructed[1][i] : 0f;
        }

        return stereo;
    }

    /// <summary>
    /// 应用掩码并执行ISTFT
    /// </summary>
    private float[] ApplyMaskAndISTFT(float[][][] mask, StftResult stft, int numFrames)
    {
        // 计算输出长度
        int outputLength = (numFrames - 1) * HOP_LENGTH + N_FFT;
        float[] output = new float[outputLength];
        float[] windowSum = new float[outputLength];

        int numSplits = mask.Length;
        int numBins = NUM_BINS;

        for (int frameIdx = 0; frameIdx < numFrames; frameIdx++)
        {
            int offset = frameIdx * HOP_LENGTH;
            int splitIdx = frameIdx / CHUNK_SIZE;
            int inSplitIdx = frameIdx % CHUNK_SIZE;
            int stftIdx = frameIdx * numBins;

            // 1. 准备频谱（应用掩码）
            // 注意：模型输出的掩码只对应前MODEL_BINS个频率
            for (int k = 0; k < NUM_BINS; k++)
            {
                if (k < MODEL_BINS && splitIdx < numSplits && inSplitIdx < mask[splitIdx].Length)
                {
                    // 应用掩码到前MODEL_BINS个频率
                    float maskVal = mask[splitIdx][inSplitIdx][k];
                    _ifftBuffer[k] = new Complex32(
                        stft.Real[stftIdx + k] * maskVal,
                        stft.Imag[stftIdx + k] * maskVal
                    );
                }
                else if (k < NUM_BINS)
                {
                    // 高频部分保持原样（不应用掩码）
                    _ifftBuffer[k] = new Complex32(stft.Real[stftIdx + k], stft.Imag[stftIdx + k]);
                }
            }

            // 2. 填充共轭对称部分
            // 对于N_FFT=4096，NUM_BINS=2049，需要填充2048-4095的共轭对称部分
            for (int k = NUM_BINS; k < N_FFT; k++)
            {
                int conjIdx = N_FFT - k;
                if (conjIdx < NUM_BINS)
                {
                    _ifftBuffer[k] = Complex32.Conjugate(_ifftBuffer[conjIdx]);
                }
                else
                {
                    _ifftBuffer[k] = Complex32.Zero;
                }
            }

            // 3. 执行IFFT
            Fourier.Inverse(_ifftBuffer, FourierOptions.Matlab);

            // 4. 应用窗口并叠加（重叠相加）
            for (int i = 0; i < N_FFT && offset + i < outputLength; i++)
            {
                float sample = _ifftBuffer[i].Real * _hannWindow[i];
                output[offset + i] += sample;
                windowSum[offset + i] += _hannWindow[i] * _hannWindow[i];
            }
        }

        // 5. 补偿重叠相加（归一化）
        for (int i = 0; i < outputLength; i++)
        {
            if (windowSum[i] > 1e-6f)
            {
                output[i] /= windowSum[i];
            }
        }

        return output;
    }

    /// <summary>
    /// 使用Wiener滤波计算掩码（与C++代码一致）
    /// </summary>
    private float[][][][] ComputeMaskWiener(float[][][][] source, float[][][][] other)
    {
        int dim0 = source.Length; // 2
        int dim1 = source[0].Length; // num_splits
        int dim2 = source[0][0].Length; // 512
        int dim3 = source[0][0][0].Length; // 1024

        float[][][][] mask = new float[dim0][][][];

        for (int i = 0; i < dim0; i++)
        {
            mask[i] = new float[dim1][][];
            for (int j = 0; j < dim1; j++)
            {
                mask[i][j] = new float[dim2][];
                for (int k = 0; k < dim2; k++)
                {
                    mask[i][j][k] = new float[dim3];
                    for (int l = 0; l < dim3; l++)
                    {
                        float sourceMag = source[i][j][k][l];
                        float otherMag = other[i][j][k][l];

                        // Wiener滤波公式：mask = (source^2 + ε/2) / (source^2 + other^2 + ε)
                        float sourceSq = sourceMag * sourceMag;
                        float otherSq = otherMag * otherMag;
                        float sum = sourceSq + otherSq + EPSILON;
                        mask[i][j][k][l] = (sourceSq + EPSILON / 2f) / sum;
                    }
                }
            }
        }

        return mask;
    }

    #region 辅助方法

    private float[] CreateHannWindow(int length)
    {
        float[] window = new float[length];
        for (int i = 0; i < length; i++)
        {
            // 标准的汉宁窗公式
            window[i] = 0.5f * (1 - Mathf.Cos(2 * Mathf.PI * i / (length - 1)));
        }
        return window;
    }

    private float[][][] ExtractStftMagnitude(StftResult[] stftResults)
    {
        float[][][] result = new float[2][][];

        for (int ch = 0; ch < 2; ch++)
        {
            int numFrames = stftResults[ch].NumFrames;
            result[ch] = new float[numFrames][];

            for (int i = 0; i < numFrames; i++)
            {
                result[ch][i] = new float[MODEL_BINS];
                int idx = i * NUM_BINS;

                // 只提取前MODEL_BINS个频率的幅度谱
                for (int k = 0; k < MODEL_BINS; k++)
                {
                    if (k < NUM_BINS)
                    {
                        float r = stftResults[ch].Real[idx + k];
                        float imag = stftResults[ch].Imag[idx + k];
                        result[ch][i][k] = Mathf.Sqrt(r * r + imag * imag);
                    }
                    else
                    {
                        result[ch][i][k] = 0f;
                    }
                }
            }
        }

        return result;
    }

    private float[][][] PadStftData(float[][][] data, int padding)
    {
        int numFrames = data[0].Length;
        int newFrames = numFrames + padding;
        float[][][] padded = new float[2][][];

        for (int ch = 0; ch < 2; ch++)
        {
            padded[ch] = new float[newFrames][];
            Array.Copy(data[ch], 0, padded[ch], 0, numFrames);

            for (int i = numFrames; i < newFrames; i++)
            {
                padded[ch][i] = new float[MODEL_BINS];
            }
        }

        return padded;
    }

    private float[][][][] ReshapeForModel(float[][][] data)
    {
        int numFrames = data[0].Length;
        int numSplits = numFrames / CHUNK_SIZE;

        // 确保至少有一个split
        if (numSplits == 0) numSplits = 1;

        float[][][][] result = new float[2][][][];

        for (int ch = 0; ch < 2; ch++)
        {
            result[ch] = new float[numSplits][][];

            for (int s = 0; s < numSplits; s++)
            {
                result[ch][s] = new float[CHUNK_SIZE][];

                for (int i = 0; i < CHUNK_SIZE; i++)
                {
                    result[ch][s][i] = new float[MODEL_BINS];
                    int frameIdx = s * CHUNK_SIZE + i;

                    if (frameIdx < numFrames)
                    {
                        Array.Copy(data[ch][frameIdx], 0, result[ch][s][i], 0, MODEL_BINS);
                    }
                }
            }
        }

        return result;
    }

    #endregion

    #region 文件I/O方法

    public Dictionary<string, float[]> SeparateFromFile(string audioPath)
    {
        try
        {
            float[] waveform = LoadWavFile(audioPath);
            return Separate(waveform);
        }
        catch (Exception ex)
        {
            Debug.LogError($"文件分离失败: {ex.Message}\n{ex.StackTrace}");
            throw;
        }
    }

    private float[] LoadWavFile(string path)
    {
        byte[] fileBytes = File.ReadAllBytes(path);

        _sampleRate = BitConverter.ToInt32(fileBytes, 24);
        int channels = BitConverter.ToInt16(fileBytes, 22);
        int dataSize = BitConverter.ToInt32(fileBytes, 40);

        int sampleCount = dataSize / (channels * sizeof(short));
        float[] samples = new float[sampleCount * channels];
        int dataOffset = 44;

        for (int i = 0; i < sampleCount * channels; i++)
        {
            short sample = BitConverter.ToInt16(fileBytes, dataOffset + i * 2);
            samples[i] = sample / 32768f;
        }

        return samples;
    }

    public void SaveToFile(Dictionary<string, float[]> sources, string outputDir)
    {
        try
        {
            if (!Directory.Exists(outputDir))
                Directory.CreateDirectory(outputDir);

            foreach (var kvp in sources)
            {
                string outputPath = Path.Combine(outputDir, $"{kvp.Key}.wav");
                SaveWavFile(outputPath, kvp.Value, _sampleRate);
                Debug.Log($"已保存: {outputPath}");
            }
        }
        catch (Exception ex)
        {
            Debug.LogError($"保存失败: {ex.Message}");
            throw;
        }
    }

    private void SaveWavFile(string path, float[] samples, int sampleRate)
    {
        int channels = 2;
        int sampleCount = samples.Length / channels;
        int byteRate = sampleRate * channels * 2;

        using (var writer = new BinaryWriter(File.Create(path)))
        {
            writer.Write(new char[] { 'R', 'I', 'F', 'F' });
            writer.Write(36 + sampleCount * channels * 2);
            writer.Write(new char[] { 'W', 'A', 'V', 'E' });
            writer.Write(new char[] { 'f', 'm', 't', ' ' });
            writer.Write(16);
            writer.Write((short)1);
            writer.Write((short)channels);
            writer.Write(sampleRate);
            writer.Write(byteRate);
            writer.Write((short)(channels * 2));
            writer.Write((short)16);
            writer.Write(new char[] { 'd', 'a', 't', 'a' });
            writer.Write(sampleCount * channels * 2);

            foreach (float sample in samples)
            {
                short pcm = (short)Mathf.Clamp(sample * 32767f, -32768, 32767);
                writer.Write(pcm);
            }
        }
    }

    public void Dispose()
    {
        _vocalsModel?.Dispose();
        _accompanimentModel?.Dispose();
    }

    private void OnDestroy()
    {
        Dispose();
    }

    #endregion
}