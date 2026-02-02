using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

/// <summary>
/// 音频分离器 - 基于ONNX运行时
/// </summary>
public class AudioSeparator : MonoBehaviour
{
    private OnnxModel _vocalsModel;
    private OnnxModel _accompanimentModel;

    private const int N_FFT = 4096;
    private const int HOP_LENGTH = 1024;
    private const int N_BINS = 1024; // 保留的频率箱数
    private const int STFT_HEIGHT = 512; // 每个split的高度
    private const int STFT_WIDTH = 1024; // 频率维度
    private const float EPSILON = 1e-10f;
    private int _sampleRate = 44100;

    /// <summary>
    /// 初始化分离器
    /// </summary>
    public void Initialize(string vocalsModelPath, string accompanimentModelPath)
    {
        try
        {
            _vocalsModel = new OnnxModel(vocalsModelPath);
            _accompanimentModel = new OnnxModel(accompanimentModelPath);
            Debug.Log("分离器初始化成功");
        }
        catch (Exception ex)
        {
            Debug.LogError($"初始化失败: {ex.Message}");
            throw;
        }
    }

    /// <summary>
    /// 从文件分离音频
    /// </summary>
    public Dictionary<string, float[]> SeparateFromFile(string audioPath)
    {
        try
        {
            float[] waveform = LoadWavFile(audioPath);
            return Separate(waveform);
        }
        catch (Exception ex)
        {
            Debug.LogError($"文件分离失败: {ex.Message}");
            throw;
        }
    }

    /// <summary>
    /// 分离音频
    /// </summary>
    public Dictionary<string, float[]> Separate(float[] waveform)
    {
        if (_vocalsModel == null || _accompanimentModel == null)
        {
            throw new InvalidOperationException("分离器未初始化");
        }

        try
        {
            var stopwatch = System.Diagnostics.Stopwatch.StartNew();

            // 确保立体声
            int numSamples = waveform.Length / 2;
            float[][] waveformStereo = new float[2][];
            waveformStereo[0] = new float[numSamples];
            waveformStereo[1] = new float[numSamples];

            for (int i = 0; i < numSamples; i++)
            {
                waveformStereo[0][i] = waveform[i * 2];
                waveformStereo[1][i] = waveform[i * 2 + 1];
            }

            // 执行STFT
            StftResult[] stftResults = new StftResult[2];
            stftResults[0] = ComputeStft(waveformStereo[0]);
            stftResults[1] = ComputeStft(waveformStereo[1]);

            Debug.Log($"STFT 帧数: {stftResults[0].NumFrames}");

            // 提取幅度谱 - 返回4D数组 (2, numFrames, N_BINS)
            float[][][] stftData = ExtractStftMagnitude(stftResults);

            // 填充到512的倍数
            int numFrames = stftData[0].Length;
            int padding = (512 - (numFrames % 512)) % 512;
            Debug.Log($"填充: {padding}");

            if (padding > 0)
            {
                stftData = PadStftData(stftData, padding);
            }

            // 重新形成输入 (2, num_splits, 512, 1024)
            float[][][][] modelInput = ReshapeForModel(stftData);

            // 运行模型
            float[][][][] vocalsSpec = _vocalsModel.Run(modelInput);
            float[][][][] accompanimentSpec = _accompanimentModel.Run(modelInput);

            // 计算掩码
            float[][][][] vocalsRatio = ComputeMask(vocalsSpec, accompanimentSpec);
            float[][][][] accompanimentRatio = ComputeMask(accompanimentSpec, vocalsSpec);

            // 逆STFT并保存
            var results = new Dictionary<string, float[]>();
            results["vocals"] = ReconstructAudio(vocalsRatio, stftResults, stftData[0].Length);
            results["accompaniment"] = ReconstructAudio(accompanimentRatio, stftResults, stftData[0].Length);

            stopwatch.Stop();
            float audioDuration = numSamples / (float)_sampleRate;
            Debug.Log($"耗时: {stopwatch.ElapsedMilliseconds}ms, RTF: {stopwatch.ElapsedMilliseconds / 1000f / audioDuration:F3}");

            return results;
        }
        catch (Exception ex)
        {
            Debug.LogError($"分离过程错误: {ex.Message}");
            throw;
        }
    }

    /// <summary>
    /// 计算STFT
    /// </summary>
    private StftResult ComputeStft(float[] signal)
    {
        int numFrames = (signal.Length - N_FFT) / HOP_LENGTH + 1;
        float[] realPart = new float[numFrames * (N_FFT / 2 + 1)];
        float[] imagPart = new float[numFrames * (N_FFT / 2 + 1)];

        float[] window = CreateHannWindow(N_FFT);

        for (int frameIdx = 0; frameIdx < numFrames; frameIdx++)
        {
            int offset = frameIdx * HOP_LENGTH;

            // 提取帧并应用窗口
            float[] frame = new float[N_FFT];
            Array.Copy(signal, offset, frame, 0, N_FFT);

            for (int i = 0; i < N_FFT; i++)
            {
                frame[i] *= window[i];
            }

            // 计算FFT (简化版 - 使用System.Numerics)
            Complex[] fftResult = SimpleFFT(frame);

            // 存储结果
            for (int k = 0; k <= N_FFT / 2; k++)
            {
                int idx = frameIdx * (N_FFT / 2 + 1) + k;
                realPart[idx] = fftResult[k].Real;
                imagPart[idx] = fftResult[k].Imag;
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
    /// 简单的FFT实现 (基数2)
    /// </summary>
    private Complex[] SimpleFFT(float[] input)
    {
        int n = input.Length;
        Complex[] result = new Complex[n];

        for (int k = 0; k < n; k++)
        {
            result[k] = Complex.Zero;
            for (int m = 0; m < n; m++)
            {
                float angle = -2f * Mathf.PI * k * m / n;
                result[k] += input[m] * new Complex(Mathf.Cos(angle), Mathf.Sin(angle));
            }
        }

        return result;
    }

    /// <summary>
    /// 创建Hann窗口
    /// </summary>
    private float[] CreateHannWindow(int size)
    {
        float[] window = new float[size];
        for (int i = 0; i < size; i++)
        {
            window[i] = 0.5f * (1 - Mathf.Cos(2f * Mathf.PI * i / (size - 1)));
        }
        return window;
    }

    /// <summary>
    /// 提取STFT幅度 - 返回 (2, numFrames, N_BINS)
    /// </summary>
    private float[][][] ExtractStftMagnitude(StftResult[] stftResults)
    {
        float[][][] result = new float[2][][];

        for (int ch = 0; ch < 2; ch++)
        {
            int numFrames = stftResults[ch].NumFrames;
            result[ch] = new float[numFrames][];

            for (int i = 0; i < numFrames; i++)
            {
                result[ch][i] = new float[N_BINS];

                for (int k = 0; k < N_BINS; k++)
                {
                    int idx = i * (N_FFT / 2 + 1) + k;

                    float real = stftResults[ch].Real[idx];
                    float imag = stftResults[ch].Imag[idx];
                    float magnitude = Mathf.Sqrt(real * real + imag * imag);

                    result[ch][i][k] = magnitude;
                }
            }
        }

        return result;
    }

    /// <summary>
    /// 填充STFT数据 - 处理 (2, numFrames, N_BINS)
    /// </summary>
    private float[][][] PadStftData(float[][][] data, int padding)
    {
        int numFrames = data[0].Length;
        int newFrames = numFrames + padding;
        float[][][] padded = new float[2][][];

        for (int ch = 0; ch < 2; ch++)
        {
            padded[ch] = new float[newFrames][];

            for (int i = 0; i < numFrames; i++)
            {
                padded[ch][i] = data[ch][i];
            }

            for (int i = numFrames; i < newFrames; i++)
            {
                padded[ch][i] = new float[N_BINS];
                for (int k = 0; k < N_BINS; k++)
                {
                    padded[ch][i][k] = 0f;
                }
            }
        }

        return padded;
    }

    /// <summary>
    /// 重新形成为模型输入 (2, num_splits, 512, 1024)
    /// 输入: (2, numFrames, N_BINS)
    /// </summary>
    private float[][][][] ReshapeForModel(float[][][] data)
    {
        int numFrames = data[0].Length;
        int numSplits = numFrames / STFT_HEIGHT;
        float[][][][] result = new float[2][][][];

        for (int ch = 0; ch < 2; ch++)
        {
            result[ch] = new float[numSplits][][];

            for (int s = 0; s < numSplits; s++)
            {
                result[ch][s] = new float[STFT_HEIGHT][];

                for (int i = 0; i < STFT_HEIGHT; i++)
                {
                    result[ch][s][i] = new float[STFT_WIDTH];
                    int frameIdx = s * STFT_HEIGHT + i;

                    for (int k = 0; k < STFT_WIDTH; k++)
                    {
                        if (k < N_BINS)
                        {
                            result[ch][s][i][k] = data[ch][frameIdx][k];
                        }
                        else
                        {
                            result[ch][s][i][k] = 0f;
                        }
                    }
                }
            }
        }

        return result;
    }

    /// <summary>
    /// 计算掩码
    /// </summary>
    private float[][][][] ComputeMask(float[][][][] source, float[][][][] other)
    {
        float[][][][] mask = new float[2][][][];

        for (int ch = 0; ch < 2; ch++)
        {
            mask[ch] = new float[source[ch].Length][][];

            for (int s = 0; s < source[ch].Length; s++)
            {
                mask[ch][s] = new float[STFT_HEIGHT][];

                for (int i = 0; i < STFT_HEIGHT; i++)
                {
                    mask[ch][s][i] = new float[STFT_WIDTH];

                    for (int k = 0; k < STFT_WIDTH; k++)
                    {
                        float sourceMag = source[ch][s][i][k];
                        float otherMag = other[ch][s][i][k];
                        float sum = sourceMag * sourceMag + otherMag * otherMag + EPSILON;
                        mask[ch][s][i][k] = (sourceMag * sourceMag + EPSILON / 2f) / sum;
                    }
                }
            }
        }

        return mask;
    }

    /// <summary>
    /// 重构音频
    /// </summary>
    private float[] ReconstructAudio(float[][][][] mask, StftResult[] stftResults, int originalNumFrames)
    {
        float[][] reconstructed = new float[2][];

        for (int ch = 0; ch < 2; ch++)
        {
            int numBins = N_FFT / 2 + 1;
            float[] real = new float[originalNumFrames * numBins];
            float[] imag = new float[originalNumFrames * numBins];

            for (int i = 0; i < originalNumFrames; i++)
            {
                int splitIdx = i / STFT_HEIGHT;
                int inSplitIdx = i % STFT_HEIGHT;

                for (int k = 0; k < N_BINS; k++)
                {
                    int idx = i * numBins + k;
                    float maskVal = mask[ch][splitIdx][inSplitIdx][k];

                    real[idx] = maskVal * stftResults[ch].Real[idx];
                    imag[idx] = maskVal * stftResults[ch].Imag[idx];
                }

                for (int k = N_BINS; k < numBins; k++)
                {
                    int idx = i * numBins + k;
                    real[idx] = stftResults[ch].Real[idx];
                    imag[idx] = stftResults[ch].Imag[idx];
                }
            }

            StftResult maskedResult = new StftResult
            {
                Real = real,
                Imag = imag,
                NumFrames = originalNumFrames
            };

            reconstructed[ch] = ComputeIstft(maskedResult);
        }

        // 交错成立体声
        int totalSamples = reconstructed[0].Length;
        float[] stereo = new float[totalSamples * 2];
        for (int i = 0; i < totalSamples; i++)
        {
            stereo[i * 2] = reconstructed[0][i];
            stereo[i * 2 + 1] = reconstructed[1][i];
        }

        return stereo;
    }

    /// <summary>
    /// 计算逆STFT
    /// </summary>
    private float[] ComputeIstft(StftResult stftResult)
    {
        int numFrames = stftResult.NumFrames;
        int signalLength = (numFrames - 1) * HOP_LENGTH + N_FFT;
        float[] signal = new float[signalLength];

        float[] window = CreateHannWindow(N_FFT);

        for (int frameIdx = 0; frameIdx < numFrames; frameIdx++)
        {
            int offset = frameIdx * HOP_LENGTH;

            // 执行逆FFT
            float[] frame = new float[N_FFT];

            for (int k = 0; k <= N_FFT / 2; k++)
            {
                int idx = frameIdx * (N_FFT / 2 + 1) + k;
                Complex bin = new Complex(stftResult.Real[idx], stftResult.Imag[idx]);

                for (int n = 0; n < N_FFT; n++)
                {
                    float angle = 2f * Mathf.PI * k * n / N_FFT;
                    frame[n] += bin.Real * Mathf.Cos(angle) - bin.Imag * Mathf.Sin(angle);
                }
            }

            // 归一化和窗口
            for (int i = 0; i < N_FFT; i++)
            {
                frame[i] /= N_FFT;
                frame[i] *= window[i];
            }

            // 叠加到输出
            for (int i = 0; i < N_FFT; i++)
            {
                if (offset + i < signalLength)
                {
                    signal[offset + i] += frame[i];
                }
            }
        }

        return signal;
    }

    /// <summary>
    /// 加载WAV文件
    /// </summary>
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

    /// <summary>
    /// 保存WAV文件
    /// </summary>
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

    private void OnDestroy()
    {
        _vocalsModel?.Dispose();
        _accompanimentModel?.Dispose();
    }
}