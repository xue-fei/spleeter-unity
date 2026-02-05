using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

public class Util  
{
    public static float[] LoadWavFile(string path,ref int _sampleRate)
    {
        byte[] fileBytes = File.ReadAllBytes(path);

        // === 1. 验证 RIFF/WAVE 基础结构 ===
        if (fileBytes.Length < 12)
            throw new InvalidDataException($"WAV文件过小 ({fileBytes.Length}字节)，无法读取RIFF头");

        string riff = System.Text.Encoding.ASCII.GetString(fileBytes, 0, 4);
        if (riff != "RIFF")
            throw new InvalidDataException($"无效文件头: '{riff}' (应为'RIFF')");

        string wave = System.Text.Encoding.ASCII.GetString(fileBytes, 8, 4);
        if (wave != "WAVE")
            throw new InvalidDataException($"无效格式标识: '{wave}' (应为'WAVE')");

        // === 2. 遍历 chunks 动态查找 fmt 和 data ===
        int pos = 12; // 跳过 RIFF header (12字节)
        int channels = 0;
        int sampleRate = 0;
        int bitsPerSample = 0;
        int blockAlign = 0;
        int dataSize = 0;
        int dataOffset = 0;

        while (pos < fileBytes.Length - 8)
        {
            // 读取 chunk ID (4字节)
            string chunkId = System.Text.Encoding.ASCII.GetString(fileBytes, pos, 4);
            // 读取 chunk 大小 (4字节, 小端序)
            int chunkSize = BitConverter.ToInt32(fileBytes, pos + 4);
            pos += 8; // 跳过 chunk header

            // 防御性检查：chunkSize 不能为负或超出文件范围
            if (chunkSize < 0 || pos + chunkSize > fileBytes.Length)
            {
                Debug.LogWarning($"跳过无效 chunk '{chunkId}' (大小={chunkSize}, 偏移={pos - 8})");
                break;
            }

            // 处理 fmt chunk
            if (chunkId == "fmt ")
            {
                if (chunkSize < 16)
                    throw new InvalidDataException($"fmt chunk 过小 ({chunkSize}字节)，至少需要16字节");

                ushort audioFormat = BitConverter.ToUInt16(fileBytes, pos);
                if (audioFormat != 1) // 1 = PCM
                    throw new NotSupportedException($"仅支持PCM格式 (格式代码=1)，当前格式: {audioFormat}");

                channels = BitConverter.ToInt16(fileBytes, pos + 2);
                sampleRate = BitConverter.ToInt32(fileBytes, pos + 4);
                blockAlign = BitConverter.ToInt16(fileBytes, pos + 12);
                bitsPerSample = BitConverter.ToInt16(fileBytes, pos + 14);

                _sampleRate = sampleRate;
                Debug.Log($"✓ WAV格式: {channels}声道, {sampleRate}Hz, {bitsPerSample}位, BlockAlign={blockAlign}");
            }
            // 处理 data chunk (找到后立即退出)
            else if (chunkId == "data")
            {
                dataSize = chunkSize;
                dataOffset = pos;
                Debug.Log($"✓ 找到data chunk: 大小={dataSize}字节, 偏移={dataOffset}");
                break; // data chunk 通常在最后
            }
            // 跳过其他 chunks (LIST/fact/cue等)
            else
            {
                Debug.Log($"  跳过 chunk '{chunkId}' (大小={chunkSize}字节)");
            }

            // 跳过 chunk 数据 + 2字节对齐 (WAV规范要求偶数大小)
            pos += chunkSize;
            if (chunkSize % 2 != 0)
                pos++; // 跳过填充字节
        }

        // === 3. 验证关键参数 ===
        if (dataOffset == 0)
            throw new InvalidDataException("未找到'data' chunk，文件可能损坏或非标准WAV格式");

        if (channels == 0 || sampleRate == 0 || bitsPerSample == 0)
            throw new InvalidDataException("未找到有效的'fmt ' chunk，无法确定音频格式");

        if (bitsPerSample != 16)
            throw new NotSupportedException($"仅支持16位PCM WAV，当前为{bitsPerSample}位");

        // === 4. 计算样本数 ===
        int bytesPerSample = bitsPerSample / 8;  // 每个采样点的字节数 (16位=2字节)
        int totalSamples = dataSize / bytesPerSample; // 总采样点数（所有声道）
        int samplesPerChannel = totalSamples / channels;

        float durationSec = samplesPerChannel / (float)sampleRate;
        Debug.Log($"✓ 音频信息: {samplesPerChannel:N0}样本/通道 | 时长: {durationSec:F2}秒 | 总样本: {totalSamples:N0}");

        // === 5. 读取PCM样本 (小端序16位) ===
        float[] samples = new float[totalSamples];
        for (int i = 0; i < totalSamples; i++)
        {
            int bytePos = dataOffset + i * bytesPerSample;  // ✓ 使用 bytesPerSample
            if (bytePos + bytesPerSample - 1 >= fileBytes.Length)
            {
                Debug.LogWarning($"⚠️ 提前到达文件末尾 (位置{bytePos})，剩余样本填充为0");
                break;
            }

            // 小端序: 低字节在前
            short sample = (short)(fileBytes[bytePos] | (fileBytes[bytePos + 1] << 8));
            samples[i] = sample / 32768f; // 归一化到 [-1, 1]
        }

        // === 6. 声道处理 ===
        if (channels == 1)
        {
            // 单声道 → 立体声 (复制到左右声道)
            Debug.Log("✓ 单声道转立体声");
            float[] stereo = new float[samplesPerChannel * 2];
            for (int i = 0; i < samplesPerChannel; i++)
            {
                stereo[i * 2] = samples[i];
                stereo[i * 2 + 1] = samples[i];
            }
            return stereo;
        }
        else if (channels == 2)
        {
            // 标准立体声 → 直接返回
            return samples;
        }
        else
        {
            // 多声道 (>2) → 仅保留前2声道 (左/右)
            Debug.LogWarning($"⚠️ 多声道音频 ({channels}声道)，仅保留前2声道");
            float[] stereo = new float[samplesPerChannel * 2];
            for (int i = 0; i < samplesPerChannel; i++)
            {
                stereo[i * 2] = samples[i * channels];     // 左声道 (第1声道)
                stereo[i * 2 + 1] = samples[i * channels + 1]; // 右声道 (第2声道)
            }
            return stereo;
        }
    }


    public static void SaveToFile(Dictionary<string, float[]> sources, string outputDir, int sampleRate)
    {
        try
        {
            if (!Directory.Exists(outputDir))
            {
                Directory.CreateDirectory(outputDir);
            }
            foreach (var kvp in sources)
            {
                string outputPath = Path.Combine(outputDir, $"{kvp.Key}.wav");
                Util.SaveWavFile(outputPath, kvp.Value, sampleRate);
                Debug.Log($"✓ 已保存: {outputPath}");
            }
        }
        catch (Exception ex)
        {
            Debug.LogError($"保存失败: {ex.Message}");
            throw;
        }
    }


    public static void SaveWavFile(string path, float[] samples, int sampleRate)
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
}