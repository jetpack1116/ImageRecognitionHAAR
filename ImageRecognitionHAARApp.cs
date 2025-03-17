using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using System;

namespace ImageRecognitionHAAR
{
    class ImageRecognitionHAARApp
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Starting HAAR-based Image Recognition Application...");

            RecognizeObjectUsingHAAR();

            Console.WriteLine("Application finished.");
        }

        static void RecognizeObjectUsingHAAR()
        {
            // Load the built-in HAAR cascade for face detection
            var faceCascade = new CascadeClassifier("Data/haarcascade_frontalface_default.xml");

            // Capture video via default camera
            using (var capture = new VideoCapture(0))
            {
                if (!capture.IsOpened)
                {
                    Console.WriteLine("Failed to open the camera.");
                    return;
                }

                while (true)
                {
                    // Mat object to hold the frame
                    Mat frame = new Mat();

                    // Capture a frame from the camera
                    capture.Read(frame);

                    // Convert the frame into grayscale
                    Mat grayFrame = new Mat();
                    CvInvoke.CvtColor(frame, grayFrame, ColorConversion.Bgr2Gray);

                    // Detect faces
                    var faces = faceCascade.DetectMultiScale(grayFrame, 1.1, 10,
                        new System.Drawing.Size(30, 30), new System.Drawing.Size(300, 300));

                    // Draw rectangle around faces
                    foreach (var face in faces)
                    {
                        CvInvoke.Rectangle(frame, face, new MCvScalar(0, 255, 0), 2); // Green rectangle
                    }

                    // Display the frame in a window
                    CvInvoke.Imshow("Face Detection", frame);

                    // Break the loop if a key is pressed
                    if (CvInvoke.WaitKey(1) >= 0) break;
                }
            }
        }
    }
}
