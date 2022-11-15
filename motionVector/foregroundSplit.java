import java.awt.*;
import java.awt.image.*;
import java.io.*;
import javax.imageio.ImageIO;
import javax.swing.*;
import java.lang.Exception;
import java.lang.Thread;
import java.util.*;

public class foregroundSplit {
	String foreGroundDir, backGroundDir;
	int width, height; // default image width and height
	double inf = Double.POSITIVE_INFINITY;
	int mode = 0;
	int macroSize = 16;
	//
	ArrayList<String> fFramesPath, bFramesPath;
	// Display
	JFrame frame;
	GridBagLayout gLayout;
	GridBagConstraints c;
	BufferedImage[] outImgs;

	// imgPath,subSamplingY,subSamplingU,subSamplingV,Sw,Sh,A
	public foregroundSplit(String foreGroundDir, String backGroundDir, int mode, int width, int height) {
		assert width> 0: "Width should be positive int";
		assert height> 0: "Width should be positive int";
		assert mode == 1 || mode == 0 : "mode should be either 0 or 1";
		this.foreGroundDir = foreGroundDir;
		this.backGroundDir = backGroundDir;
		this.width = width;
		this.height = height;
		this.mode = mode;
		// getPaths
		fFramesPath = getPaths(foreGroundDir);
		bFramesPath = getPaths(backGroundDir);
		outImgs = new BufferedImage[fFramesPath.size()];
		// display
		frame = new JFrame();
		gLayout = new GridBagLayout();
		frame.getContentPane().setLayout(gLayout);
		c = new GridBagConstraints();
		c.fill = GridBagConstraints.HORIZONTAL;
		c.anchor = GridBagConstraints.CENTER;
		c.weightx = 0.5;
		c.gridx = 0;
		c.gridy = 0;
		c.fill = GridBagConstraints.HORIZONTAL;
		c.gridx = 0;
		c.gridy = 1;
		getMotionVectors();
		// process();
	}
	private void getMotionVectors(){
		BufferedImage Img1 = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
		BufferedImage Img2 = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
		readImageRGB(fFramesPath.get(0),Img1);
		readImageRGB(fFramesPath.get(10),Img2);
		showImg(Img1);
		showImg(Img2);

		// macroblock
		int nRow = height/macroSize;
		int nCol = width/macroSize;
		double[][][] motionVectors = new double[nRow][nCol][2];
		double[][] motionVectorMADs = new double[nRow][nCol];

		int k = macroSize;
		for(int r = 0; r < nRow; r++)
			for(int c = 0; c < nCol; c++){
				motionVectors[r][c] = new double[] {0,0};
				motionVectorMADs[r][c] = inf;
				double error = inf;
				for(int i= -k; i<k; i++)
					for(int j= -k; j<k; j++){
						error = MAD(Img1,Img2,i,j,r,c);
						if(error < motionVectorMADs[r][c]){
							motionVectorMADs[r][c] = error;
							motionVectors[r][c] = new double[] {i,j};
						}
					}
			}
		System.out.println(Arrays.deepToString(motionVectors));
	}
	private double MAD(BufferedImage Img1,BufferedImage Img2,int i,int j,int r,int c){ //mean absolute difference, textbook p.233
		int p = c*macroSize;
		int q = r*macroSize;
		double tmp = 0;
		for(int x=0; x < macroSize; x++)
			for(int y=0; y < macroSize; y++){
				if (p+i+x < 0 || p+i+x >= width || q+j+y < 0 || q+j+y >= height) 
					return inf;
				if(p+x>=width || q+y>=height)
					System.out.println(p+x);
				Color c1 = new Color(Img1.getRGB(p+x, q+y));
				Color c2 = new Color(Img2.getRGB(p+i+x, q+j+y));
				tmp += Math.abs(c1.getRGB()-c2.getRGB());
			}
		return tmp;
	}

	private ArrayList<String> getPaths(String dir){
		File directory = new File(dir);
		File[] rgbFiles = directory.listFiles();
		ArrayList<String> framesPath = new ArrayList<String>();
		for (int i = 0; i < rgbFiles.length; i++)
		framesPath.add(rgbFiles[i].getPath());
		Collections.sort(framesPath);
		// System.out.print(framesPath);
		return framesPath;
	}

	private void readImageRGB(String imgPath, BufferedImage img)
	{
		File file = new File(imgPath);
			try{
				RandomAccessFile raf = new RandomAccessFile(file, "r");
				raf.seek(0);
				int frameLength = width*height*3;
				long len = frameLength;
				byte[] bytes = new byte[(int) len];
				raf.read(bytes);
				raf.close();
				int ind = 0;
				for(int y = 0; y < height; y++)
					for(int x = 0; x < width; x++){
						int r = bytes[ind] & 0xff;
						int g = bytes[ind+height*width] & 0xff;
						int b = bytes[ind+height*width*2] & 0xff; 
						int pixel = 0xff000000 | ((r & 0xff) << 16) | ((g & 0xff) << 8) | (b & 0xff);
						img.setRGB(x, y, pixel);	
						ind++;
					}
			}
			catch (FileNotFoundException e){
				e.printStackTrace();
			} 
			catch (IOException e) {
				e.printStackTrace();
			}
			catch (Exception e) {
				// catching the exception
				System.out.println(e);
			}
	}

	public void replaceGreen(BufferedImage fImg, BufferedImage bImg){
		for(int y = 0; y < height; y++)
			for(int x = 0; x < width; x++){
				Color c = new Color(fImg.getRGB(x, y));
				//RGB to HSB https://stackoverflow.com/questions/2399150/convert-rgb-value-to-hsv
				float[] hsv = new float[3];
				int r = c.getRed(), g = c.getGreen(), b = c.getBlue();
				Color.RGBtoHSB(r, g, b, hsv);
				hsv[0]*=360;
				hsv[1]*=100;//s
				hsv[2]*=100;//v
				// System.out.println(hsv[0] + ", "+ hsv[1] + ", " + hsv[2]);
				/*green candidate
				 * 95.1923, 65.822784, 61.960785
				 * 131.78572, 70.292885, 93.725494
				 */
				// System.out.println(c.getRed() + ", "+ c.getGreen() + ", " + c.getBlue());

				// https://stackoverflow.com/questions/2810970/how-to-remove-a-green-screen-portrait-background
				if(hsv[0]>=360 || hsv[0]<0 || hsv[1] < 0 || hsv[1] > 100 || hsv[2] < 0 || hsv[2] > 100)
					System.out.println("illegal HSV ="+ hsv[0] + ", "+ hsv[1] + ", " + hsv[2]);

				float hG = 100, hRange = 50, hG2 = 100, hRange2 = 10; //100 +- 50
				if (hsv[0] >= hG - hRange && hsv[0] < hG + hRange && hsv[1] >= 30  && hsv[2] >= 30 ) {
					fImg.setRGB(x, y, bImg.getRGB(x,y));
				}else if(hsv[0] >= hG2 - hRange2 && hsv[0] < hG2 + hRange2){
					hsv[1] = 0; //set satuation to 0
					hsv[2] = 100; // set light to 100
					int newGreen = Color.HSBtoRGB(hsv[0] / 360, hsv[1]/100, hsv[2]/100);
					// fImg.setRGB(x, y, newGreen);
				}
			}
	}
	
	public void substraction2(BufferedImage fImg, BufferedImage bImg, double[][]avgR,double[][]avgG,double[][]avgB, int i){
		for(int y = 0; y < height; y++)
			for(int x = 0; x < width; x++){
				Color c = new Color(fImg.getRGB(x, y));
				int r = c.getRed();
				int g = c.getGreen();
				int b = c.getBlue(); 
				double pixelsY = 0.299*r + 0.587*g + 0.114*b;;
				double pixelsU = 0.596*r + (-0.274)*g + (-0.322)*b;
				double pixelsV = 0.211*r + (-0.523)*g + (0.312)*b;
				double diff3 = Math.pow(c.getRed() - avgR[y][x],2) + Math.pow(c.getBlue() - avgB[y][x],2) + Math.pow(c.getGreen() - avgG[y][x],2);
				if(diff3 < 300){
					int greenColor = new Color(0f, 1f, 0f).getRGB();
					// fImg.setRGB(x, y, greenColor);
					fImg.setRGB(x, y, bImg.getRGB(x,y));
				}
				double alpha = 0.1;
				avgR[y][x] = (avgR[y][x] * alpha + (1- alpha) * c.getRed());
				avgG[y][x] = (avgG[y][x] * alpha + (1- alpha) * c.getGreen());
				avgB[y][x] = (avgB[y][x] * alpha + (1- alpha) * c.getBlue());
			}
	}

	public void process(){
		System.out.println("(Mode: "+this.mode+") Processing...");
		double[][] avgR = new double[height][width];
		double[][] avgG = new double[height][width];
		double[][] avgB = new double[height][width];
		for(int y = 0; y < height; y++)
			for(int x = 0; x < width; x++){
				avgR[y][x] = 0;
				avgG[y][x] = 0;
				avgB[y][x] = 0;
			}
		for(int i=0; i < fFramesPath.size(); i++){
			BufferedImage fImg = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
			BufferedImage bImg = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
			readImageRGB(fFramesPath.get(i),fImg);
			readImageRGB(bFramesPath.get(i),bImg);
			if (this.mode == 1){
				replaceGreen(fImg,bImg);
			}else if (this.mode == 0) substraction2(fImg,bImg,avgR,avgG,avgB,i);
			else{System.out.println("Not a valid mode");}
			outImgs[i] = fImg;
			// try{
			// 	Thread.sleep(41); // 1000ms/24fps = 41ms
			// 	showImg(outImgs[i]);
			// }catch (Exception e) {System.out.println(e);}
			// save to output folder
			// try{
			// 	File directory = new File("output/");
			// 	if (! directory.exists()){
			// 		directory.mkdir();
			// 	}
			// 	String[] parts = this.foreGroundDir.split("/");
			// 	String fileName = parts[parts.length - 1];
			// 	// ImageIO.write(fImg,"jpeg", new File("output/"+ fileName + "_mode_" + this.mode + '_' + i + ".jpg"));
			// }catch(Exception exception){
			// 	System.out.println("exception ImageIO");
			// }
		}
	}

	public void playVideo(){ 
		System.out.println("Start play in 24 FPS");
		for(int i=1; i < fFramesPath.size(); i++){
			try{
				Thread.sleep(41); // 1000ms/24fps = 41ms
				showImg(outImgs[i]);
			}catch (Exception e) {System.out.println(e);}
		}
		System.out.println("The end. To exit please press ctrl+c");
	}

	public void showImg(BufferedImage imgOne){
		JFrame frame = new JFrame();
		GridBagLayout gLayout = new GridBagLayout();
		frame.getContentPane().setLayout(gLayout);
		JLabel lbIm1 = new JLabel(new ImageIcon(imgOne));

		GridBagConstraints c = new GridBagConstraints();
		c.fill = GridBagConstraints.HORIZONTAL;
		c.anchor = GridBagConstraints.CENTER;
		c.weightx = 0.5;
		c.gridx = 0;
		c.gridy = 0;

		c.fill = GridBagConstraints.HORIZONTAL;
		c.gridx = 0;
		c.gridy = 1;
		frame.getContentPane().add(lbIm1, c);

		frame.pack();
		frame.setVisible(true);
	}

	public static void main(String[] args) {	
		// String foreGroundDir = args[0];
        // String backGroundDir = args[1];
        // int mode = Integer.parseInt(args[2]);
		String foreGroundDir = "C:\\hw2RGB\\subtraction\\background_subtraction_2";
		String backGroundDir = "C:\\hw2RGB\\input\\background_static_1";
		int mode = 0;

		int width = 640;
		int height = 480;
		foregroundSplit cvt = new foregroundSplit(foreGroundDir,backGroundDir,mode,width,height);
		// cvt.playVideo();
	}

}
