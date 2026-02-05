using UnityEngine;
using System.Collections;

public class SeparatorExample : MonoBehaviour
{
    private AudioSeparator separator;
    private bool isProcessing = false;

    public void Start()
    {
        // 初始化线程管理器
        if (Loom.Current == null)
        {
            Loom.Initialize();
        }

        separator = gameObject.AddComponent<AudioSeparator>();
        string modelPath1 = Application.streamingAssetsPath + "/2stems/vocals.fp16.onnx";
        string modelPath2 = Application.streamingAssetsPath + "/2stems/accompaniment.fp16.onnx";

        Debug.Log("=== 开始初始化分离器 ===");
        Debug.Log($"模型路径1: {modelPath1}");
        Debug.Log($"模型路径2: {modelPath2}");

        // 检查模型文件是否存在
        if (!System.IO.File.Exists(modelPath1))
        {
            Debug.LogError($"❌ 模型文件不存在: {modelPath1}");
            return;
        }

        if (!System.IO.File.Exists(modelPath2))
        {
            Debug.LogError($"❌ 模型文件不存在: {modelPath2}");
            return;
        }

        Debug.Log("✓ 模型文件存在");

        // 在后台线程初始化模型
        Loom.RunAsync(() =>
        {
            try
            {
                Debug.Log(">> [后台线程] 开始加载模型...");
                separator.Initialize(modelPath1, modelPath2);
                Debug.Log(">> [后台线程] 模型加载完成");

                // 回到主线程进行分离操作
                Loom.QueueOnMainThread(() =>
                {
                    Debug.Log("<< [主线程] 准备执行音频分离");
                    StartCoroutine(PerformSeparation());
                });
            }
            catch (System.Exception ex)
            {
                Debug.LogError($"❌ [后台线程] 初始化失败: {ex.Message}\n{ex.StackTrace}");
                Loom.QueueOnMainThread(() =>
                {
                    Debug.LogError("分离器初始化失败，请检查模型文件和ONNX运行时");
                });
            }
        });
    }

    /// <summary>
    /// 在协程中执行分离操作，避免阻塞
    /// </summary>
    private IEnumerator PerformSeparation()
    {
        if (isProcessing)
        {
            Debug.LogWarning("⚠ 正在处理中，请等待...");
            yield break;
        }

        isProcessing = true;

        string audioPath = Application.dataPath + "/qi-feng-le-zh.wav";
        string outputDir = Application.dataPath + "/SeparatedAudio/";

        Debug.Log($"\n=== 开始音频分离 ===");
        Debug.Log($"输入文件: {audioPath}");
        Debug.Log($"输出目录: {outputDir}");

        // 检查音频文件
        if (!System.IO.File.Exists(audioPath))
        {
            Debug.LogError($"❌ 音频文件不存在: {audioPath}");
            isProcessing = false;
            yield break;
        }

        Debug.Log("✓ 音频文件存在");
        Loom.RunAsync(() =>
        {
            Debug.Log(">> [后台线程] 开始加载音频文件...");
            var sources = separator.SeparateFromFile(audioPath);
            Debug.Log($">> [后台线程] 分离完成，获得 {sources.Count} 个音频源");
            Loom.QueueOnMainThread(() =>
            {
                Debug.Log("<< [主线程] 开始保存文件");
                Util.SaveToFile(sources, outputDir, 44100);
                Debug.Log($"✓ 分离完成！文件已保存至: {outputDir}");
                isProcessing = false;
            });
        });
    }
}