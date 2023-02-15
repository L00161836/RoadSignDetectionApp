using Microsoft.Maui.Storage;

namespace RoadSignDetectionApp;

public partial class MainPage : ContentPage
{
	int count = 0;

	public MainPage()
	{
		InitializeComponent();
	}

	public async FileResult TakePhoto()
	{
		if (MediaPicker.Default.IsCaptureSupported)
		{
			FileResult photo = await MediaPicker.Default.CapturePhotoAsync();

			return photo;

		}
	}
}

