
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.util.Random;

public class ImageDrawer {

    private JFrame mainFrame;
    private MultiLayerNetwork nn; // The neural network.

    private BufferedImage originalImage;
    private BufferedImage generatedImage;
    private int[] pixels;          // array of pixels of original image.
    private int[] pixelsGenerated; // array of pixels to generate image.

    private INDArray xyOut; //x,y grid to calculate the output image. Needs to be calculated once, then re-used.

    private Random r = new Random();


    private void init() throws IOException {

        mainFrame = new JFrame("Image drawer example");//creating instance of JFrame
        mainFrame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

        originalImage = ImageIO.read(getClass().getResource("Mona_Lisa_2.png"));
        generatedImage = new BufferedImage(originalImage.getWidth(), originalImage.getHeight(), originalImage.getType());

        int width = originalImage.getWidth();
        int height = originalImage.getHeight();
        pixels = originalImage.getRGB(0, 0, width, height, null, 0, width);
        pixelsGenerated = new int[pixels.length];
        generatedImage.setRGB(0, 0, width, height, pixels, 0, width);

        final JLabel originalLabel = new JLabel(new ImageIcon(originalImage));
        final JLabel generatedLabel = new JLabel(new ImageIcon(generatedImage
        ));
        originalLabel.setBounds(0,0, width, height);
        generatedLabel.setBounds(width, 0, width, height);//x axis, y axis, width, height

        mainFrame.add(originalLabel);
        mainFrame.add(generatedLabel);

        mainFrame.setSize(2*width, height +25);
        mainFrame.setLayout(null);//using no layout managers
        mainFrame.setVisible(true);//making the frame visible

        nn = createNN();
        xyOut = calcGrid();

        SwingUtilities.invokeLater(this::onCalc);
    }

    public static void main(String[] args) throws IOException {
        ImageDrawer imageDrawer = new ImageDrawer();
        imageDrawer.init();
    }

    /**
     * Build the Neural network.
     */
    private static MultiLayerNetwork createNN() {
        int seed = 2345;
        double learningRate = 0.01;
        int numInputs = 2;   // x and y.
        int numHiddenNodes = 500;
        int numOutputs = 3 ; //R, G and B value.

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(learningRate))
                .list()
                .layer(new DenseLayer.Builder().nIn(numInputs).nOut(numHiddenNodes)
                        .activation(Activation.LEAKYRELU)
                        .build())
                .layer(new DenseLayer.Builder().nOut(numHiddenNodes )
                        .activation(Activation.LEAKYRELU)
                        .build())
                .layer(new DenseLayer.Builder().nOut(numHiddenNodes )
                        .activation(Activation.LEAKYRELU)
                        .build())
                .layer(new DenseLayer.Builder().nOut(numHiddenNodes )
                        .activation(Activation.LEAKYRELU)
                        .build())
                .layer(new DenseLayer.Builder().nOut(numHiddenNodes )
                        .activation(Activation.LEAKYRELU)
                        .build())
                .layer( new OutputLayer.Builder(LossFunctions.LossFunction.L2)
                        .activation(Activation.IDENTITY)
                        .nOut(numOutputs).build())
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        return net;
    }

    /**
     * Training the NN and updating the current graphical output.
     */
    private void onCalc(){
        int batchSize = 3000;
        int numBatches = 100;
        for (int i =0; i< numBatches; i++){
            DataSet ds = generateDataSet(batchSize);
            nn.fit(ds);
        }
        drawImage();
        mainFrame.invalidate();
        mainFrame.repaint();

        SwingUtilities.invokeLater(this::onCalc);
    }

    /**
     * Process a javafx Image to be consumed by DeepLearning4J.
     *
     * @param batchSize number of sample points to take out of the image.
     * @return DeepLearning4J DataSet.
     */
    private DataSet generateDataSet(int batchSize) {
        int w = originalImage.getWidth();
        int h = originalImage.getHeight();

        INDArray xy = Nd4j.zeros(batchSize, 2);
        INDArray out = Nd4j.zeros(batchSize, 3);
        float [] RGBColorComponents = new float[3];

        for (int index = 0; index < batchSize; index++) {
            int i = r.nextInt(w);
            int j = r.nextInt(h);
            double xp = scaleXY(i,w);
            double yp = scaleXY(j,h);

            int indexPixel = j * w + i;
            Color c = new Color(pixels[indexPixel]);

            xy.put(index, 0, xp); //2 inputs. x and y.
            xy.put(index, 1, yp);

            c.getRGBColorComponents(RGBColorComponents);
            out.put(index, 0, RGBColorComponents[0]);  //3 outputs. the RGB values as 0-1 floats.
            out.put(index, 1, RGBColorComponents[1]);
            out.put(index, 2, RGBColorComponents[2]);
        }
        return new DataSet(xy, out);
    }

    /**
     * Make the Neural network draw the image.
     */
    private void drawImage() {
        int w = originalImage.getWidth();
        int h = originalImage.getHeight();

        INDArray out = nn.output(xyOut);

        for (int i = 0; i < w; i++) {
            for (int j = 0; j < h; j++) {
                int index = i + w * j;
                float red = capNNOutput(out.getFloat(index, 0));
                float  green = capNNOutput(out.getFloat(index, 1));
                float  blue = capNNOutput(out.getFloat(index, 2));

                Color c = new Color(red, green, blue);
                pixelsGenerated[index] = c.getRGB();
            }
        }

        generatedImage.setRGB(0, 0, w, h, pixelsGenerated, 0, w);
    }


    /**
     * The x,y grid to calculate the NN output. Only needs to be calculated once.
     */
    private INDArray calcGrid(){
        int w = originalImage.getWidth();
        int h = originalImage.getHeight();
        int numPoints = h * w;
        INDArray res = Nd4j.zeros(numPoints, 2);

        //TODO: There is a more elegant way to do this with Nd4j.
        for (int i = 0; i < w; i++) {
            double xp = scaleXY(i,w);
            for (int j = 0; j < h; j++) {
                int index = i + w * j;
                double yp = scaleXY(j,h);

                res.put(index, 0, xp); //2 inputs. x and y.
                res.put(index, 1, yp);
            }
        }

        return res;
    }

    /**
     * Make sure the color values are >=0 and <=1
     */
    private static float capNNOutput(float x) {
        float tmp = (x<0.0f) ? 0.0f : x;
        return (tmp > 1.0f) ? 1.0f : tmp;
    }

    /**
     * scale x,y points
     */
    private static double scaleXY(int i, int maxI){
        return (double) i / (double) (maxI - 1) -0.5;
    }
}
