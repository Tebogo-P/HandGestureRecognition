package za.ac.cput.gestureappnetbeans;

import com.github.sarxos.webcam.Webcam;
import com.github.sarxos.webcam.WebcamResolution;
import javafx.application.Application;
import javafx.application.Platform;
import javafx.embed.swing.SwingFXUtils;
import javafx.scene.Scene;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.scene.layout.BorderPane;
import javafx.stage.Stage;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfInt4;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import nu.pattern.OpenCV;

import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

/**
 * A complete, self-contained JavaFX application for real-time hand gesture
 * recognition.
 *
 * This application captures video from the default webcam, isolates the hand
 * using skin-color thresholding, and recognizes "Closed Fist" vs. "Open Hand"
 * by counting convexity defects.
 *
 * Dependencies:
 * - JavaFX
 * - OpenCV (core jar + native library)
 * - com.github.sarxos:webcam-capture
 */
public class HandGestureRecognition extends Application {

    private ImageView imageView;
    private Stage stage;

    private Webcam webcam;
    private ScheduledExecutorService timer;
    private boolean cameraActive = false;


    private static final Scalar SKIN_COLOR_LOWER = new Scalar(0, 48, 80);
    private static final Scalar SKIN_COLOR_UPPER = new Scalar(20, 255, 255);
    private static final double MIN_CONTOUR_AREA = 10000; // Min area to be considered a hand

    public static void main(String[] args) {
        try {
            nu.pattern.OpenCV.loadLocally();
            
        } catch (Exception e) {
            System.err.println("Native code library failed to load.\n" + e);
            System.exit(1);
        }
        launch(args);
    }

    @Override
    public void start(Stage primaryStage) {
        this.stage = primaryStage;
        this.imageView = new ImageView();
        this.imageView.setPreserveRatio(true);

        BorderPane root = new BorderPane(this.imageView);
        root.setCenter(this.imageView);
        
        Scene scene = new Scene(root, 640, 480);
        primaryStage.setTitle("Hand Gesture Recognition");
        primaryStage.setScene(scene);
        primaryStage.show();

        primaryStage.setOnCloseRequest(e -> shutdown());

        initWebcam();
        startVideoCapture();
    }


    private void initWebcam() {
        this.webcam = Webcam.getDefault();
        if (this.webcam == null) {
            System.err.println("No webcam found.");
            Platform.exit();
            return;
        }
        this.webcam.setViewSize(WebcamResolution.VGA.getSize());
    }


    private void startVideoCapture() {
        if (!this.webcam.isOpen()) {
            this.webcam.open();
        }
        this.cameraActive = true;
        this.timer = Executors.newSingleThreadScheduledExecutor();

        this.timer.scheduleAtFixedRate(this::grabFrame, 0, 33, TimeUnit.MILLISECONDS);
    }


    private void grabFrame() {
        if (!this.cameraActive) {
            return;
        }

        BufferedImage bImage = this.webcam.getImage();
        if (bImage == null) {
            return;
        }

        Mat frame = bufferedImageToMat(bImage);

        Mat processedFrame = processFrame(frame);

        Image fxImage = matToFxImage(processedFrame);

        Platform.runLater(() -> {
            this.imageView.setImage(fxImage);
            this.stage.sizeToScene();
        });

        frame.release();
    }


    private Mat processFrame(Mat bgrFrame) {
        Mat hsvMat = new Mat();
        Mat mask = new Mat();
        Mat hierarchy = new Mat();

        try {
            Imgproc.cvtColor(bgrFrame, hsvMat, Imgproc.COLOR_BGR2HSV);

            Core.inRange(hsvMat, SKIN_COLOR_LOWER, SKIN_COLOR_UPPER, mask);

            Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(7, 7));
            Imgproc.morphologyEx(mask, mask, Imgproc.MORPH_OPEN, kernel);

            Imgproc.morphologyEx(mask, mask, Imgproc.MORPH_CLOSE, kernel);

            List<MatOfPoint> contours = new ArrayList<>();
            Imgproc.findContours(mask, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

            MatOfPoint largestContour = findLargestContour(contours);

            if (largestContour != null) {
                MatOfInt hullIndices = new MatOfInt();
                Imgproc.convexHull(largestContour, hullIndices, false);

                Point[] contourPoints = largestContour.toArray();
                Point[] hullPointsArr = new Point[hullIndices.rows()];
                List<Integer> hullIndicesList = hullIndices.toList();
                for (int i = 0; i < hullIndicesList.size(); i++) {
                    hullPointsArr[i] = contourPoints[hullIndicesList.get(i)];
                }
                MatOfPoint hullPoints = new MatOfPoint(hullPointsArr);

                MatOfInt4 defects = new MatOfInt4();
                Imgproc.convexityDefects(largestContour, hullIndices, defects);

                int defectCount = 0;
                List<Integer> defectsList = defects.toList();

                for (int i = 0; i < defectsList.size(); i += 4) {
                    Point startPt = contourPoints[defectsList.get(i)];
                    Point endPt = contourPoints[defectsList.get(i + 1)];
                    Point farPt = contourPoints[defectsList.get(i + 2)];
                    
                    double depth = defectsList.get(i + 3) / 256.0; 

                    if (depth > 20) {
                        defectCount++;
                        Imgproc.circle(bgrFrame, farPt, 5, new Scalar(0, 0, 255), -1);
                    }
                }

                String gesture = "Unknown";
                if (defectCount <= 1) {
                    gesture = "Closed Fist";
                }
                else if (defectCount == 4) {
                    gesture = "Open Hand";
                }

                Imgproc.putText(bgrFrame, "GESTURE: " + gesture, new Point(10, 50),
                    Imgproc.FONT_HERSHEY_SIMPLEX, 1, new Scalar(0, 255, 0), 2);

                Imgproc.drawContours(bgrFrame, Arrays.asList(largestContour), -1, new Scalar(255, 0, 0), 2);
                Imgproc.drawContours(bgrFrame, Arrays.asList(hullPoints), -1, new Scalar(0, 255, 0), 2);

                hullIndices.release();
                hullPoints.release();
                defects.release();
                largestContour.release();
            }
        } catch (Exception e) {
            System.err.println("Error during image processing: " + e.getMessage());
            e.printStackTrace();
        } finally {
            hsvMat.release();
            mask.release();
            hierarchy.release();
        }

        return bgrFrame;
    }

    private MatOfPoint findLargestContour(List<MatOfPoint> contours) {
        double maxArea = 0;
        MatOfPoint largestContour = null;

        for (MatOfPoint contour : contours) {
            double area = Imgproc.contourArea(contour);
            if (area > maxArea && area > MIN_CONTOUR_AREA) {
                maxArea = area;
                largestContour = contour;
            } else {
                contour.release();
            }
        }
        return largestContour;
    }

    private void shutdown() {
        this.cameraActive = false;

        if (this.timer != null && !this.timer.isShutdown()) {
            try {
                this.timer.shutdown();
                this.timer.awaitTermination(1000, TimeUnit.MILLISECONDS);
            } catch (InterruptedException e) {
                System.err.println("Error shutting down executor: " + e.getMessage());
            }
        }

        if (this.webcam != null && this.webcam.isOpen()) {
            this.webcam.close();
        }
    }

    private Image matToFxImage(Mat mat) {
        BufferedImage bImage = matToBufferedImage(mat);
        return SwingFXUtils.toFXImage(bImage, null);
    }

    private BufferedImage matToBufferedImage(Mat mat) {
        int type = BufferedImage.TYPE_BYTE_GRAY;
        if (mat.channels() > 1) {
            type = BufferedImage.TYPE_3BYTE_BGR;
        }
        
        int bufferSize = mat.channels() * mat.cols() * mat.rows();
        byte[] buffer = new byte[bufferSize];
        mat.get(0, 0, buffer); // Get all pixels in one go

        BufferedImage image = new BufferedImage(mat.cols(), mat.rows(), type);
        final byte[] targetPixels = ((DataBufferByte) image.getRaster().getDataBuffer()).getData();
        System.arraycopy(buffer, 0, targetPixels, 0, buffer.length);

        return image;
    }


    private Mat bufferedImageToMat(BufferedImage bi) {
        BufferedImage bgrImage;
        
        if (bi.getType() != BufferedImage.TYPE_3BYTE_BGR) {
            bgrImage = new BufferedImage(bi.getWidth(), bi.getHeight(), BufferedImage.TYPE_3BYTE_BGR);
            Graphics2D g = bgrImage.createGraphics();
            g.drawImage(bi, 0, 0, null);
            g.dispose();
        } else {
            bgrImage = bi;
        }

        byte[] pixels = ((DataBufferByte) bgrImage.getRaster().getDataBuffer()).getData();
        
        Mat mat = new Mat(bgrImage.getHeight(), bgrImage.getWidth(), CvType.CV_8UC3);
        mat.put(0, 0, pixels);
        
        return mat;
    }
}
