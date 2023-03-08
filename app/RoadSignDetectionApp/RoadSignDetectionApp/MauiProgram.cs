using Microsoft.Extensions.DependencyInjection.Extensions;
using Microsoft.Extensions.Logging;
using ZXing.Net.Maui;
using ZXing.Net.Maui.Controls;

namespace RoadSignDetectionApp;

public static class MauiProgram
{
	public static MauiApp CreateMauiApp()
	{
		var builder = MauiApp.CreateBuilder();
		builder
			.UseMauiApp<App>()
			.UseBarcodeReader()
			.ConfigureMauiHandlers(h =>
			{
				h.AddHandler(typeof
					(ZXing.Net.Maui.Controls.CameraBarcodeReaderView),
					typeof(CameraBarcodeReaderViewHandler));
                h.AddHandler(typeof
                    (ZXing.Net.Maui.Controls.CameraView),
                    typeof(CameraViewHandler));
                h.AddHandler(typeof
                    (ZXing.Net.Maui.Controls.BarcodeGeneratorView),
                    typeof(BarcodeGeneratorViewHandler));
            })
			.ConfigureFonts(fonts =>
			{
				fonts.AddFont("OpenSans-Regular.ttf", "OpenSansRegular");
				fonts.AddFont("OpenSans-Semibold.ttf", "OpenSansSemibold");
			});

#if DEBUG
		builder.Logging.AddDebug();
#endif

		return builder.Build();
	}
}
