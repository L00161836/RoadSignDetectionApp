
using Java.Nio;
using Microsoft.Maui.Storage;
using RoadSignDetectionApp.Model;
using System.ComponentModel;
using ZXing.Net.Maui.Controls;
using ZXing.Net.Maui.Readers;

namespace RoadSignDetectionApp;

public partial class MainPage : ContentPage
{

	public MainPage()
	{
		InitializeComponent();
        CameraView.FrameReady += CameraView_FrameReady;

	}

    private void CameraView_FrameReady(object sender, ZXing.Net.Maui.CameraFrameBufferEventArgs e)
    {
#if ANDROID
        PixelBufferHolder pixelBufferHolder = e.Data;
        ByteBuffer byteBuffer = pixelBufferHolder.Data;

        ByteBuffer resized = ByteBuffer.Allocate(388800);
        byteBuffer.Flip();
        resized.Put(byteBuffer);

        List<SignClassificationModel> result = TensorFlowClassifier.Classify(resized);

        if (MainThread.IsMainThread)
        {
            UpdateTestLabels(result);
        }
        else
        {
            MainThread.BeginInvokeOnMainThread(() => UpdateTestLabels(result));
        }

#endif
    }

    //    private async Task<byte[]> CaptureFrameAsync()
    //	{
    //		var frame = await Screenshot.Default.CaptureAsync();

    //		if (frame != null)
    //		{
    //            using (MemoryStream ms = new MemoryStream())
    //            {
    //                await frame.CopyToAsync(ms);

    //                return ms.ToArray();
    //            }
    //        }
    //        return null;

    //	}

    //    private void OnCameraView_Loaded(object sender, EventArgs e)
    //    {
    //            Task.Run(() =>
    //            {
    //                while (true)
    //                {
    //                    RunAgainstFrame();
    //                    Task.Delay(1000);

    //                }
    //            });

    //    }

    //    private async void RunAgainstFrame()
    //    {
    //#if ANDROID
    //        byte[] frame = await CaptureFrameAsync();

    //        if (frame != null)
    //        {
    //            List<SignClassificationModel> result = TensorFlowClassifier.Classify(frame);

    //            if (MainThread.IsMainThread)
    //            {
    //                UpdateTestLabels(result);
    //            }
    //            else
    //            {
    //                MainThread.BeginInvokeOnMainThread(() => UpdateTestLabels(result));
    //            }
    //        }

    //#endif
    //    }

    private void UpdateTestLabels(List<SignClassificationModel> result)
    {
        FiftyKphProbLabel.Text = result[0].Probability.ToString();
        EightyKphProbLabel.Text = result[1].Probability.ToString();
        WarningProbLabel.Text = result[2].Probability.ToString();
    }


}

