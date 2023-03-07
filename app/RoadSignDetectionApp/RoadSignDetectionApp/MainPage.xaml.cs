using Android.App;
using Java.Util;
using Microsoft.Maui.Storage;
using RoadSignDetectionApp.Model;
using System.ComponentModel;
using ZXing.Net.Maui.Controls;

namespace RoadSignDetectionApp;

public partial class MainPage : ContentPage
{

	public MainPage()
	{
		InitializeComponent();

	}

	private async Task<byte[]> CaptureFrameAsync()
	{
		var frame = await Screenshot.Default.CaptureAsync();

		if (frame != null)
		{
            using (MemoryStream ms = new MemoryStream())
            {
                await frame.CopyToAsync(ms);

                return ms.ToArray();
            }
        }
        return null;

	}

    private void OnCameraView_Loaded(object sender, EventArgs e)
    {
            Task.Run(() =>
            {
                while (true)
                {
                    RunAgainstFrame();
                    Task.Delay(1000);

                }
            });

    }

    private async void RunAgainstFrame()
    {
#if ANDROID
        byte[] frame = await CaptureFrameAsync();

        if (frame != null)
        {
            List<SignClassificationModel> result = TensorFlowClassifier.Classify(frame);

            if (MainThread.IsMainThread)
            {
                UpdateTestLabels(result);
            }
            else
            {
                MainThread.BeginInvokeOnMainThread(() => UpdateTestLabels(result));
            }
        }

#endif
    }

    private void UpdateTestLabels(List<SignClassificationModel> result)
    {
        FiftyKphProbLabel.Text = result[0].Probability.ToString();
        EightyKphProbLabel.Text = result[1].Probability.ToString();
        WarningProbLabel.Text = result[2].Probability.ToString();
    }

    
}

