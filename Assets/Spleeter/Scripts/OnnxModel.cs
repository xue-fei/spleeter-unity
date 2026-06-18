using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using UnityEngine;

/// <summary>
/// ONNX 模型包装类（性能优化版）
///
/// RunFlat()：接收/返回平铺 float[]，通过 Memory&lt;float&gt; 零拷贝构造 DenseTensor，
/// 避免 Flatten4DArray / Tensor4DToJagged 的全量内存分配。
/// Run()：原有锯齿数组接口，向后兼容。
/// </summary>
public class OnnxModel : IDisposable
{
    private InferenceSession _session;
    private string _inputName;
    private string _outputName;

    // 复用输出缓冲，避免每次推理分配
    private float[] _outputBuf;

    public OnnxModel(string modelPath)
    {
        try
        {
            var opts = new SessionOptions
            {
                InterOpNumThreads = 1,
                IntraOpNumThreads = Environment.ProcessorCount,
                GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
            };
            opts.AppendExecutionProvider_CUDA(0); // 按需开启

            _session = new InferenceSession(modelPath, opts);
            _inputName = _session.InputMetadata.Keys.First();
            _outputName = _session.OutputMetadata.Keys.First();

            Debug.Log($"---------- 模型加载: {modelPath} ----------");
            foreach (var kv in _session.InputMetadata)
                Debug.Log($"输入: {kv.Key}, 形状: [{string.Join(", ", kv.Value.Dimensions)}]");
            foreach (var kv in _session.OutputMetadata)
                Debug.Log($"输出: {kv.Key}, 形状: [{string.Join(", ", kv.Value.Dimensions)}]");
            Debug.Log("--------------------");
        }
        catch (Exception ex)
        {
            Debug.LogError($"模型加载失败: {ex.Message}");
            throw;
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // RunFlatNoCopy — 零拷贝高性能接口（推荐）
    //
    // 返回内部 _outputBuf 的引用，省去末尾 Buffer.BlockCopy。
    // 调用方在下次推理前不得修改返回值，若需要修改请用 RunFlat()。
    // ═══════════════════════════════════════════════════════════════════════
    public float[] RunFlatNoCopy(float[] inputFlat, int dim0, int dim1, int dim2, int dim3)
    {
        try
        {
            int total = dim0 * dim1 * dim2 * dim3;

            if (_outputBuf == null || _outputBuf.Length < total)
                _outputBuf = new float[total];

            var inputMem    = new Memory<float>(inputFlat, 0, total);
            var inputTensor = new DenseTensor<float>(inputMem, new[] { dim0, dim1, dim2, dim3 });

            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor(_inputName, inputTensor)
            };

            using (var results = _session.Run(inputs))
            {
                var outTensor = results.First(r => r.Name == _outputName).AsTensor<float>();

                if (outTensor is DenseTensor<float> dense)
                {
                    dense.Buffer.Span.Slice(0, total).CopyTo(
                        new Span<float>(_outputBuf, 0, total));
                }
                else
                {
                    int idx = 0;
                    for (int i = 0; i < dim0; i++)
                        for (int j = 0; j < dim1; j++)
                            for (int k = 0; k < dim2; k++)
                                for (int l = 0; l < dim3; l++)
                                    _outputBuf[idx++] = outTensor[i, j, k, l];
                }
            }

            // 直接返回内部缓冲引用（无额外拷贝）
            return _outputBuf;
        }
        catch (Exception ex)
        {
            Debug.LogError($"RunFlatNoCopy 推理失败: {ex.Message}");
            throw;
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // RunFlat — 安全接口（返回独立副本，向后兼容）
    // inputFlat 布局: (dim0, dim1, dim2, dim3) 行优先
    // ═══════════════════════════════════════════════════════════════════════
    public float[] RunFlat(float[] inputFlat, int dim0, int dim1, int dim2, int dim3)
    {
        float[] noCopy = RunFlatNoCopy(inputFlat, dim0, dim1, dim2, dim3);
        int     total  = dim0 * dim1 * dim2 * dim3;
        float[] result = new float[total];
        Buffer.BlockCopy(noCopy, 0, result, 0, total * sizeof(float));
        return result;
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Run — 原有锯齿数组接口（向后兼容）
    // ═══════════════════════════════════════════════════════════════════════
    public float[][][][] Run(float[][][][] input)
    {
        int d0 = input.Length;
        int d1 = input[0].Length;
        int d2 = input[0][0].Length;
        int d3 = input[0][0][0].Length;
        float[] flat = Flatten4DArray(input, d0, d1, d2, d3);
        float[] outFlat = RunFlat(flat, d0, d1, d2, d3);
        return Unflatten4DArray(outFlat, d0, d1, d2, d3);
    }

    // ── 内部辅助 ───────────────────────────────────────────────────────────
    private static float[] Flatten4DArray(float[][][][] input, int d0, int d1, int d2, int d3)
    {
        float[] flat = new float[d0 * d1 * d2 * d3];
        int idx = 0;
        for (int i = 0; i < d0; i++)
            for (int j = 0; j < d1; j++)
                for (int k = 0; k < d2; k++)
                {
                    Array.Copy(input[i][j][k], 0, flat, idx, d3);
                    idx += d3;
                }
        return flat;
    }

    private static float[][][][] Unflatten4DArray(float[] flat, int d0, int d1, int d2, int d3)
    {
        var result = new float[d0][][][];
        int idx = 0;
        for (int i = 0; i < d0; i++)
        {
            result[i] = new float[d1][][];
            for (int j = 0; j < d1; j++)
            {
                result[i][j] = new float[d2][];
                for (int k = 0; k < d2; k++)
                {
                    result[i][j][k] = new float[d3];
                    Array.Copy(flat, idx, result[i][j][k], 0, d3);
                    idx += d3;
                }
            }
        }
        return result;
    }

    public void Dispose() => _session?.Dispose();
}

// ── 公用结构 ────────────────────────────────────────────────────────────────

/// <summary>STFT 结果（供需要原始 STFT 数据的代码使用）</summary>
public struct StftResult
{
    public float[] Real;
    public float[] Imag;
    public int NumFrames;
}

/// <summary>简单复数结构（兼容旧代码）</summary>
public struct Complex
{
    public float Real, Imag;

    public Complex(float real, float imag) { Real = real; Imag = imag; }
    public static Complex Zero => new Complex(0, 0);

    public static Complex operator +(Complex a, Complex b) =>
        new Complex(a.Real + b.Real, a.Imag + b.Imag);
    public static Complex operator -(Complex a, Complex b) =>
        new Complex(a.Real - b.Real, a.Imag - b.Imag);
    public static Complex operator *(Complex a, Complex b) =>
        new Complex(a.Real * b.Real - a.Imag * b.Imag,
                    a.Real * b.Imag + a.Imag * b.Real);
    public static Complex operator *(float s, Complex c) =>
        new Complex(s * c.Real, s * c.Imag);
    public static Complex operator *(Complex c, float s) =>
        new Complex(c.Real * s, c.Imag * s);
}