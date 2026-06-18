using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using UnityEngine;

/// <summary>
/// ONNX 模型包装类（性能优化版）
///
/// 新增 RunFlat()：直接接收/返回平铺 float[]，
/// 消除 Flatten4DArray 和 Tensor4DToJagged 的全量内存拷贝与大量 new[]。
/// 原有 Run() 接口保留以向后兼容。
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
                IntraOpNumThreads = Environment.ProcessorCount, // 充分利用多核
                GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
            };
            // 可选：开启 CUDA EP
            opts.AppendExecutionProvider_CUDA(0);

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

    // ══════════════════════════════════════════════════════════════════════════
    // RunFlat — 高性能接口
    // 输入/输出均为平铺 float[]，形状信息通过参数传入
    // inputFlat 布局: (dim0, dim1, dim2, dim3) 行优先
    // 返回同形状的平铺输出
    // ══════════════════════════════════════════════════════════════════════════
    public float[] RunFlat(float[] inputFlat, int dim0, int dim1, int dim2, int dim3)
    {
        try
        {
            int totalIn = dim0 * dim1 * dim2 * dim3;
            int totalOut = totalIn; // 输出形状与输入相同

            // 复用输出缓冲
            if (_outputBuf == null || _outputBuf.Length < totalOut)
                _outputBuf = new float[totalOut];

            // 直接用 Memory<float> 包装，零拷贝构造 DenseTensor
            var inputMem = new Memory<float>(inputFlat, 0, totalIn);
            var inputTensor = new DenseTensor<float>(inputMem, new[] { dim0, dim1, dim2, dim3 });

            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor(_inputName, inputTensor)
            };

            using (var results = _session.Run(inputs))
            {
                var outTensor = results.First(r => r.Name == _outputName).AsTensor<float>();

                // 使用 Buffer.BlockCopy 路径：若 DenseTensor 底层是连续数组则零额外分配
                if (outTensor is DenseTensor<float> dense)
                {
                    var span = dense.Buffer.Span;
                    span.Slice(0, totalOut).CopyTo(new Span<float>(_outputBuf, 0, totalOut));
                }
                else
                {
                    // 回退：逐元素拷贝
                    int idx = 0;
                    for (int i = 0; i < dim0; i++)
                        for (int j = 0; j < dim1; j++)
                            for (int k = 0; k < dim2; k++)
                                for (int l = 0; l < dim3; l++)
                                    _outputBuf[idx++] = outTensor[i, j, k, l];
                }
            }

            // 返回与 totalOut 等长的切片（避免调用方越界）
            float[] result = new float[totalOut];
            Buffer.BlockCopy(_outputBuf, 0, result, 0, totalOut * sizeof(float));
            return result;
        }
        catch (Exception ex)
        {
            Debug.LogError($"RunFlat 推理失败: {ex.Message}");
            throw;
        }
    }

    // ══════════════════════════════════════════════════════════════════════════
    // Run — 原有锯齿数组接口（向后兼容）
    // ══════════════════════════════════════════════════════════════════════════
    public float[][][][] Run(float[][][][] input)
    {
        try
        {
            int d0 = input.Length;
            int d1 = input[0].Length;
            int d2 = input[0][0].Length;
            int d3 = input[0][0][0].Length;

            float[] flat = Flatten4DArray(input, d0, d1, d2, d3);
            float[] outFlat = RunFlat(flat, d0, d1, d2, d3);
            return Unflatten4DArray(outFlat, d0, d1, d2, d3);
        }
        catch (Exception ex)
        {
            Debug.LogError($"Run 推理失败: {ex.Message}");
            throw;
        }
    }

    // ══════════════════════════════════════════════════════════════════════════
    // 内部辅助
    // ══════════════════════════════════════════════════════════════════════════
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

    private static float[][][][] Tensor4DToJagged(Tensor<float> tensor)
    {
        var dims = tensor.Dimensions;
        int d0 = dims[0], d1 = dims[1], d2 = dims[2], d3 = dims[3];

        if (tensor is DenseTensor<float> dense)
            return Unflatten4DArray(dense.Buffer.ToArray(), d0, d1, d2, d3);

        // 回退
        var result = new float[d0][][][];
        for (int i = 0; i < d0; i++)
        {
            result[i] = new float[d1][][];
            for (int j = 0; j < d1; j++)
            {
                result[i][j] = new float[d2][];
                for (int k = 0; k < d2; k++)
                {
                    result[i][j][k] = new float[d3];
                    for (int l = 0; l < d3; l++)
                        result[i][j][k][l] = tensor[i, j, k, l];
                }
            }
        }
        return result;
    }

    public void Dispose() => _session?.Dispose();
}

/// <summary>STFT 结果</summary>
public struct StftResult
{
    public float[] Real;
    public float[] Imag;
    public int NumFrames;
}

/// <summary>简单复数（兼容旧代码）</summary>
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