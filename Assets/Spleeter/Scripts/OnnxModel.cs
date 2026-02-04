using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using UnityEngine;

/// <summary>
/// ONNX模型包装类 
/// </summary>
public class OnnxModel : IDisposable
{
    private InferenceSession _session;
    private string _modelPath;

    public OnnxModel(string modelPath)
    {
        try
        {
            _modelPath = modelPath;
            var sessionOptions = new SessionOptions
            {
                InterOpNumThreads = 1,
                IntraOpNumThreads = 1,
                GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL
            };

            _session = new InferenceSession(modelPath, sessionOptions);

            Debug.Log($"---------- 模型加载: {modelPath} ----------");
            foreach (var input in _session.InputMetadata)
            {
                Debug.Log($"输入: {input.Key}, 形状: [{string.Join(", ", input.Value.Dimensions)}]");
            }
            foreach (var output in _session.OutputMetadata)
            {
                Debug.Log($"输出: {output.Key}, 形状: [{string.Join(", ", output.Value.Dimensions)}]");
            }
            Debug.Log("--------------------");
        }
        catch (Exception ex)
        {
            Debug.LogError($"模型加载失败: {ex.Message}");
            throw;
        }
    }

    /// <summary>
    /// 运行推理
    /// 输入: (2, num_splits, 512, 1024)
    /// 输出: (2, num_splits, 512, 1024)
    /// </summary>
    public float[][][][] Run(float[][][][] input)
    {
        try
        {
            var inputTensor = new DenseTensor<float>(Flatten4DArray(input), new[] { 2, input[0].Length, 512, 1024 });
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor(_session.InputMetadata.Keys.First(), inputTensor)
            };

            using (var results = _session.Run(inputs))
            {
                var outputName = _session.OutputMetadata.Keys.First();
                var outputTensor = results.First(r => r.Name == outputName).AsTensor<float>();
                return Tensor4DToJagged(outputTensor);
            }
        }
        catch (Exception ex)
        {
            Debug.LogError($"推理失败: {ex.Message}");
            throw;
        }
    }

    /// <summary>
    /// 将4D锯齿数组转换为平铺数组供ONNX使用
    /// </summary>
    private float[] Flatten4DArray(float[][][][] input)
    {
        int dim0 = input.Length;
        int dim1 = input[0].Length;
        int dim2 = input[0][0].Length;
        int dim3 = input[0][0][0].Length;

        float[] flattened = new float[dim0 * dim1 * dim2 * dim3];
        int index = 0;

        for (int i = 0; i < dim0; i++)
        {
            for (int j = 0; j < dim1; j++)
            {
                for (int k = 0; k < dim2; k++)
                {
                    for (int l = 0; l < dim3; l++)
                    {
                        flattened[index++] = input[i][j][k][l];
                    }
                }
            }
        }

        return flattened;
    }

    private float[][][][] Tensor4DToJagged(Tensor<float> tensor)
    {
        var dims = tensor.Dimensions;
        float[][][][] result = new float[dims[0]][][][];

        for (int i = 0; i < dims[0]; i++)
        {
            result[i] = new float[dims[1]][][];
            for (int j = 0; j < dims[1]; j++)
            {
                result[i][j] = new float[dims[2]][];
                for (int k = 0; k < dims[2]; k++)
                {
                    result[i][j][k] = new float[dims[3]];
                    for (int l = 0; l < dims[3]; l++)
                    {
                        // 使用索引访问而不是long[]
                        result[i][j][k][l] = tensor[(int)i, (int)j, (int)k, (int)l];
                    }
                }
            }
        }
        return result;
    }

    public void Dispose()
    {
        _session?.Dispose();
    }
}

/// <summary>
/// STFT结果结构体
/// </summary>
public struct StftResult
{
    public float[] Real;
    public float[] Imag;
    public int NumFrames;
}

// 简单的复数类
public struct Complex
{
    public float Real;
    public float Imag;

    public Complex(float real, float imag)
    {
        Real = real;
        Imag = imag;
    }

    public static Complex Zero => new Complex(0, 0);

    public static Complex operator +(Complex a, Complex b)
    {
        return new Complex(a.Real + b.Real, a.Imag + b.Imag);
    }

    public static Complex operator *(float scalar, Complex c)
    {
        return new Complex(scalar * c.Real, scalar * c.Imag);
    }

    // ✓ 复数乘法 - 最重要！
    public static Complex operator *(Complex a, Complex b)
    {
        float realPart = a.Real * b.Real - a.Imag * b.Imag;
        float imagPart = a.Real * b.Imag + a.Imag * b.Real;
        return new Complex(realPart, imagPart);
    }

    // ✓ 标量乘法（反向）
    public static Complex operator *(Complex c, float scalar)
    {
        return new Complex(c.Real * scalar, c.Imag * scalar);
    }

    // ✓ 减法
    public static Complex operator -(Complex a, Complex b)
    {
        return new Complex(a.Real - b.Real, a.Imag - b.Imag);
    }
}