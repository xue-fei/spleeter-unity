using UnityEngine;

public class SeparatorExample : MonoBehaviour
{
    public void Start()
    {
        Loom.Initialize();

        AudioSeparator separator = gameObject.AddComponent<AudioSeparator>();
        string modelPath1 = Application.streamingAssetsPath + "/2stems/vocals.onnx";
        string modelPath2 = Application.streamingAssetsPath + "/2stems/accompaniment.onnx";

        Loom.RunAsync(() =>
        {
            separator.Initialize(
            modelPath1,
            modelPath2);

            Loom.QueueOnMainThread(() =>
            {
                var sources = separator.SeparateFromFile(Application.dataPath + "/qi-feng-le-zh.wav");
                separator.SaveToFile(sources, Application.dataPath + "/");
            });
        });
    }
}