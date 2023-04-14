#if ANDROID
using Java.Nio;
using Java.Util;
using Microsoft.Maui.Storage;
using Plugin.LocalNotification;
using RoadSignDetectionApp.Model;
using System.ComponentModel;
using System.Timers;
using ZXing.Net.Maui.Controls;
using ZXing.Net.Maui.Readers;
#endif

using Plugin.LocalNotification;

namespace RoadSignDetectionApp;

public partial class MainPage : ContentPage
{
    private System.Timers.Timer timer;


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

        //byte[] b = new byte[byteBuffer.Remaining()];
        //byteBuffer.Get(b);

        List<SignClassificationModel> result = TensorFlowClassifier.Classify(byteBuffer);

        if (MainThread.IsMainThread)
        {
            UpdateSignNameLabel(result);
        }
        else
        {
            MainThread.BeginInvokeOnMainThread(() => UpdateSignNameLabel(result));
        }

#endif
    }
    private async void UpdateSignNameLabel(List<SignClassificationModel> result)
    {
        if (result[0].Probability > 0.38f || result[1].Probability > 0.38f || result[2].Probability > 0.98f)
        {
            SignNameBoxView.BackgroundColor = Color.FromArgb("#FCB9B8");
            SignNameLabel.TextColor = Color.FromArgb("#1E4072");

            if (result[0].Probability > 0.50f)
            {
                SignNameLabel.Text = "50 KPH Zone";
                if (SoundButton.Source.Equals("sound_on_icon.svg"))
                {
                    await TextToSpeech.SpeakAsync("This is a 50 kilometre per hour zone");
                }
            }
            else if (result[1].Probability > 0.50f)
            {
                SignNameLabel.Text = "80 KPH Zone";
                if (SoundButton.Source.Equals("sound_on_icon.svg"))
                {
                    await TextToSpeech.SpeakAsync("This is a 80 kilometre per hour zone");
                }
            }
            else
            {
                SignNameLabel.Text = "Roadworks";
                if (SoundButton.Source.Equals("sound_on_icon.svg"))
                {
                    await TextToSpeech.SpeakAsync("Roadworks Ahead");
                }
            }

            var request = new NotificationRequest
            {
                NotificationId = 100,
                Title = "Road Sign Detection",
                Subtitle = "New Sign Detected",
                Description = SignNameLabel.Text,
                BadgeNumber = 42,
                Schedule = new NotificationRequestSchedule
                {
                    NotifyTime = DateTime.Now.AddSeconds(5),
                    NotifyRepeatInterval = TimeSpan.FromDays(1)
                }
            };

            await LocalNotificationCenter.Current.Show(request);
        }
    }

    private void OnSoundButtonChanged(object sender, EventArgs e)
    {
        if (SoundButton.Source.Equals("sound_on_icon.svg"))
        {
            SoundButton.Source = "sound_off_icon.svg";
        }
        else
        {
            SoundButton.Source = "sound_on_icon.svg";
        }
    }
}
