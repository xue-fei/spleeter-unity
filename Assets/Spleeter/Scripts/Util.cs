using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

public class Util  
{
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