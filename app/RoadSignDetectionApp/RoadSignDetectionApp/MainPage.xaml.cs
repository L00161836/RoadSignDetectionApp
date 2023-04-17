#if ANDROID
using Java.Nio;
using Java.Util;
#endif
using Microsoft.Maui.Storage;
using Plugin.LocalNotification;
using RoadSignDetectionApp.Model;
using System.ComponentModel;
using System.Timers;
using ZXing.Net.Maui.Controls;
using ZXing.Net.Maui.Readers;
using Plugin.LocalNotification;
using ZXing.Net.Maui.Readers;

namespace RoadSignDetectionApp;

public partial class MainPage : ContentPage
{

    public MainPage()
    {
        InitializeComponent();
        CameraView.FrameReady += CameraView_FrameReady;

    }

    /*This takes advantage of the implemented XZing Camera feed on the main page and its FrameReady event, that occurs after every frame.
      Hence, this was the natural place to run the model code. This function takes the results of this and passes them to the UI
      update function */
    private void CameraView_FrameReady(object sender, ZXing.Net.Maui.CameraFrameBufferEventArgs e)
    {
        PixelBufferHolder pixelBufferHolder = e.Data;
#if ANDROID
        ByteBuffer byteBuffer = pixelBufferHolder.Data;
#elif IOS
        Byte[] byteBuffer = PixelBufferHolder.Data;
#endif
        /* Code from a previous attempt to resize the image for the model, see GetResizedByteBuffer function in TensorFlowClassifer
           for the more complex and fully working version */

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
    }

    /* This function updates the UI when the inputted model result reaches a certain degree of accuracy with a given sign. */
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

    //This mutes or unmutes the SpeakAsync function within the previous function.
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
