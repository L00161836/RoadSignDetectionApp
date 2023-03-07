using Android.App;
using Microsoft.Maui.Storage;
using RoadSignDetectionApp.Model;
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
		IScreenshotResult frame = await CameraView.CaptureAsync();

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
        System.Timers.Timer timer = new System.Timers.Timer();
        timer.Interval = 1000;
        timer.Elapsed += new System.Timers.ElapsedEventHandler(RunAgainstFrame);
        timer.Start();


    }

    private async void RunAgainstFrame(object sender, System.Timers.ElapsedEventArgs e)
    {
#if ANDROID
        var frame = await CaptureFrameAsync();

        if (frame != null)
        {
            List<SignClassificationModel> result = TensorFlowClassifier.Classify(frame);
            Console.WriteLine(result[0]);
            MainThread.BeginInvokeOnMainThread(() => SignNameLabel.Text = result[0].Probability.ToString());
        }

#endif
    }

    
}

