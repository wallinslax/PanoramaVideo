import java.awt.*;
import java.awt.image.*;
import java.io.*;
import javax.imageio.ImageIO;
import javax.swing.*;
import java.lang.Exception;
import java.lang.Thread;
import java.util.*;

public class foregroundSplit {
	String foreGroundDir;
	int width, height, nFrame; // default image width and height
	double inf = Double.POSITIVE_INFINITY;
	int macroSize = 16;
	//
	ArrayList<String> fFramesPath, bFramesPath;
	// Display
	JFrame frame;
	GridBagLayout gLayout;
	GridBagConstraints c;
	BufferedImage[] inImgs, foreImgs, backImgs, outImgs;
	int nRow, nCol;
	int[][][][] motionVectors;// frame, hieght, width, (dx,dy)
	double[][][] motionVectorMADs;// frame, hieght, width, value
 
	// imgPath,subSamplingY,subSamplingU,subSamplingV,Sw,Sh,A
	public foregroundSplit(String foreGroundDir) {
		this.foreGroundDir = foreGroundDir;
		String[] parts = foreGroundDir.split("_");
		this.nFrame = Integer.parseInt( parts[parts.length - 1] );
		this.height = Integer.parseInt( parts[parts.length - 2] );
		this.width = Integer.parseInt( parts[parts.length - 3] );
		// getPaths
		fFramesPath = getPaths(foreGroundDir);

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

		// load input images
		inImgs = new BufferedImage[fFramesPath.size()];
		for(int i=0; i < fFramesPath.size(); i++){
			inImgs[i] = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
			readImageRGB(fFramesPath.get(i),inImgs[i]);
		}
		// this.height = this.height/macroSize * macroSize;
		// this.width = this.width/macroSize * macroSize;

		// Motion Vectors of each macroblock per frame 
		// there are nRow * nCol macroblocks
		this.nRow = height/macroSize;
		this.nCol = width/macroSize;
		this.motionVectors = new int[nFrame][nRow][nCol][2];
		this.motionVectorMADs = new double[nFrame][nRow][nCol];
		// foreground background
		foreImgs = new BufferedImage[fFramesPath.size()];
		backImgs = new BufferedImage[fFramesPath.size()];
		getMotionVectors();
	
	}

	private void getForegroundPerFrame(int fIdx){
		// getForeground
		foreImgs[fIdx] = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
		for(int y = 0; y < height; y++) // init foreImgs to all black
			for(int x = 0; x < width; x++)
				foreImgs[fIdx].setRGB(x, y,0);

		for(int r = 0; r < nRow; r++)
			for(int c = 0; c < nCol; c++){
				int mv_x = motionVectors[fIdx][r][c][0];
				int mv_y = motionVectors[fIdx][r][c][1];
				int base_x = c*macroSize;
				int base_y = r*macroSize;
				if(mv_x != 0 || mv_y !=0) // motion vector is non zero if foreground XXXXXX
					for(int y = 0; y < macroSize; y++)
						for(int x = 0; x < macroSize; x++)
							foreImgs[fIdx].setRGB(base_x + x, base_y + y, inImgs[fIdx].getRGB(base_x + x, base_y + y) );
			}
		showImg(foreImgs[fIdx]);
	}

	private void getMotionVectors(){
		for(int fIdx=1; fIdx<nFrame;fIdx++){
			getMotionVectorsPerFrame(fIdx);
			getForegroundPerFrame(fIdx);
		}
	}

	private void getMotionVectorsPerFrame(int fIdx){
		int k = macroSize;
		for(int r = 0; r < nRow; r++)
			for(int c = 0; c < nCol; c++){
				motionVectors[fIdx][r][c] = new int[] {0,0};
				motionVectorMADs[fIdx][r][c] = inf;
				double error = inf;
				for(int vec_x = -k; vec_x < k; vec_x++)
					for(int vec_y = -k; vec_y < k; vec_y++){
						error = MAD(inImgs[fIdx], inImgs[fIdx-1],vec_x,vec_y,r,c);
						if(error < motionVectorMADs[fIdx][r][c]){
							motionVectorMADs[fIdx][r][c] = error;
							motionVectors[fIdx][r][c] = new int[] {vec_x,vec_y};
						}
					}
			}
		// System.out.println(Arrays.deepToString(motionVectors[fIdx]));
	}

	private double MAD(BufferedImage curImg,BufferedImage prvImg,int vec_x,int vec_y,int r,int c){ //mean absolute difference, textbook p.233
		int base_x = c*macroSize;
		int base_y = r*macroSize;
		double subError = 0;
		for(int x=0; x < macroSize; x++)
			for(int y=0; y < macroSize; y++){
				if (base_x + x + vec_x < 0 || base_x + x + vec_x >= width || 
				base_y + y + vec_y < 0 || base_y + y + vec_y >= height) 
					return inf;
				if(base_x + x >= width || base_y + y >= height)
					System.out.println(base_x + x);
				Color cur_c = new Color(curImg.getRGB(base_x + x, base_y + y));
				Color prv_c = new Color(prvImg.getRGB(base_x + x + vec_x, base_y + y + vec_y));
				int red = prv_c.getRed();
				int gre = prv_c.getGreen();
				int blu = prv_c.getBlue();
				double cur_y = 0.299*cur_c.getRed() + 0.587*cur_c.getGreen() + 0.114*cur_c.getBlue();
				double prv_y = 0.299*prv_c.getRed() + 0.587*prv_c.getGreen() + 0.114*prv_c.getBlue();
				subError += Math.abs(prv_y-cur_y);
			}
		return subError;
	}

	private void foregroundExtract(double[][][] motionVectors, BufferedImage[] imgs ){
		return;
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
				int frameLength = width * height * 3;
				long len = frameLength;
				byte[] bytes = new byte[(int) len];
				raf.read(bytes);
				raf.close();
				int ind = 0;
				for(int y = 0; y < height; y++)
					for(int x = 0; x < width; x++){
						int r = bytes[3*ind] & 0xff;
						int g = bytes[3*ind+1] & 0xff;
						int b = bytes[3*ind+2] & 0xff; 
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

	public void playVideo(){ 
		System.out.println("Start play in 24 FPS");
		for(int i=1; i < fFramesPath.size(); i++){
			try{
				showImg(inImgs[i]);
				Thread.sleep(41); // 1000ms/24fps = 41ms
			}catch (Exception e) {System.out.println(e);}
		}
		System.out.println("The end. To exit please press ctrl+c");
	}

	public void showImg(BufferedImage imgOne){
		// JFrame frame = new JFrame();
		// GridBagLayout gLayout = new GridBagLayout();
		// frame.getContentPane().setLayout(gLayout);
		// JLabel lbIm1 = new JLabel(new ImageIcon(imgOne));

		// GridBagConstraints c = new GridBagConstraints();
		// c.fill = GridBagConstraints.HORIZONTAL;
		// c.anchor = GridBagConstraints.CENTER;
		// c.weightx = 0.5;
		// c.gridx = 0;
		// c.gridy = 0;

		// c.fill = GridBagConstraints.HORIZONTAL;
		// c.gridx = 0;
		// c.gridy = 1;
		// frame.getContentPane().add(lbIm1, c);

		// frame.pack();
		// frame.setVisible(true);

		JLabel lbIm1 = new JLabel(new ImageIcon(imgOne));
		frame.getContentPane().add(lbIm1, c,0);
		// frame.getContentPane().remove(0);
		frame.pack();
		frame.setVisible(true);
	}

	public static void main(String[] args) {	
		String foreGroundDir;
		if(args.length == 0)
			foreGroundDir = "C:\\video_rgb\\SAL_490_270_437";
			// foreGroundDir = "C:\\video_rgb\\video2_240_424_383";
			// foreGroundDir = "C:\\hw2RGB\\subtraction\\background_subtraction_2_640_480_480";
		else
			foreGroundDir = args[0];
		foregroundSplit cvt = new foregroundSplit(foreGroundDir);
		cvt.playVideo();
	}

}
