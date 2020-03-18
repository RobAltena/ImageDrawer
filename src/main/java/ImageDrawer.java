
import org.datavec.image.loader.Java2DNativeImageLoader;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.conditions.Conditions;
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
    private JLabel generatedLabel;

    private int[] pixels;          // array of pixels of original image.
    private INDArray xPixels; // x coordinates of the pixels for the NN.
    private INDArray yPixels; // y coordinates of the pixels for the NN.

    private INDArray xyOut; //x,y grid to calculate the output image. Needs to be calculated once, then re-used.

    private Random r = new Random();
    Java2DNativeImageLoader j2dNil;


    private void init() throws IOException {

        mainFrame = new JFrame("Image drawer example");//creating instance of JFrame
        mainFrame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

        originalImage = ImageIO.read(getClass().getResource("Mona_Lisa_2.png"));
        //start with a blank image of the same size as the original.
        BufferedImage generatedImage = new BufferedImage(originalImage.getWidth(), originalImage.getHeight(), originalImage.getType());

        int width = originalImage.getWidth();
        int height = originalImage.getHeight();
        pixels = originalImage.getRGB(0, 0, width, height, null, 0, width);

        final JLabel originalLabel = new JLabel(new ImageIcon(originalImage));
        generatedLabel = new JLabel(new ImageIcon(generatedImage));

        originalLabel.setBounds(0,0, width, height);
        generatedLabel.setBounds(width, 0, width, height);//x axis, y axis, width, height

        mainFrame.add(originalLabel);
        mainFrame.add(generatedLabel);

        mainFrame.setSize(2*width, height +25);
        mainFrame.setLayout(null);//using no layout managers
        mainFrame.setVisible(true);//making the frame visible

        j2dNil = new Java2DNativeImageLoader();
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
        double learningRate = 0.001;
        int numInputs = 2;   // x and y.
        int numHiddenNodes = 1000;
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
        int batchSize = 10000;
        int numBatches = 10;
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

        INDArray xindex = Nd4j.rand(batchSize).muli(w).castTo(DataType.UINT32);
        INDArray yindex = Nd4j.rand(batchSize).muli(h).castTo(DataType.UINT32);

        for (int index = 0; index < batchSize; index++) {
            int i = xindex.getInt(index);
            int j = yindex.getInt(index);
            double xp = xPixels.getDouble(i); //scaleXY(i,w);
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

        INDArray out = nn.output(xyOut); // The raw NN output.
        BooleanIndexing.replaceWhere(out, 0.0, Conditions.lessThan(0.0)); // Cjip between 0 and 1.
        BooleanIndexing.replaceWhere(out, 1.0, Conditions.greaterThan(1.0));
        out = out.mul(255).castTo(DataType.BYTE); //convert to bytes.

        INDArray r = out.getColumn(0); //Extract the individual color layers.
        INDArray g = out.getColumn(1);
        INDArray b = out.getColumn(2);

        INDArray imgArr = Nd4j.vstack(b, g, r).reshape(3, h, w); // recombine the colors and reshape to image size.

        BufferedImage img = j2dNil.asBufferedImage(imgArr); //update the UI.
        generatedLabel.setIcon(new ImageIcon(img));
    }

    /**
     * The x,y grid to calculate the NN output. Only needs to be calculated once.
     */
    private INDArray calcGrid(){
        int w = originalImage.getWidth();
        int h = originalImage.getHeight();

        xPixels = Nd4j.linspace(-0.5, 0.5, w, DataType.DOUBLE);
        yPixels = Nd4j.linspace(-0.5, 0.5, h, DataType.DOUBLE);
        INDArray [] mesh = Nd4j.meshgrid(xPixels, yPixels);
        return Nd4j.vstack(mesh[0].ravel(), mesh[1].ravel()).transpose();
    }

    /**
     * scale x,y points
     */
    private static double scaleXY(int i, int maxI){
        return (double) i / (double) (maxI - 1) -0.5;
    }
}
