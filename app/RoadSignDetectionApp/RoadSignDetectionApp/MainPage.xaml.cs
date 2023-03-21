﻿#if ANDROID
using Java.Nio;
using Java.Util;
using Microsoft.Maui.Storage;
using RoadSignDetectionApp.Model;
using System.ComponentModel;
using ZXing.Net.Maui.Controls;
using ZXing.Net.Maui.Readers;
#endif

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

    //private void UpdateTestLabels(List<SignClassificationModel> result)
    //{
    //    if(result != null)
    //    {
    //        FiftyKphProbLabel.Text = result[0].Probability.ToString();
    //        EightyKphProbLabel.Text = result[1].Probability.ToString();
    //        WarningProbLabel.Text = result[2].Probability.ToString();
    //    }

    //}

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
                if(SoundButton.Source.Equals("sound_on_icon.svg"))
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

            Thread.Sleep(4000);

            SignNameBoxView.BackgroundColor = Color.FromArgb("#1E4072");
            SignNameLabel.TextColor = Color.FromArgb("#FCB9B8");
            SignNameLabel.Text = "";

        }
    }

    private void OnSoundButtonChanged(object sender, EventArgs e)
    {
        if(SoundButton.Source.Equals("sound_on_icon.svg"))
        {
            SoundButton.Source = "sound_off_icon.svg";
        }
        else
        {
            SoundButton.Source = "sound_on_icon.svg";
        }
    }
}

