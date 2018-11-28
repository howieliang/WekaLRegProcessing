//*********************************************
// Weka for Processing
// LinReg_Weka_basic
// Rong-Hao Liang: r.liang@tue.nl
//*********************************************
// Drag the cursor to make datapoint
// [T] to Save CSV, Train Model, and Test Data

import processing.serial.*;
Serial port; 

int[] rawData;
int sensorNum = 3; 
int dataNum = 500;

Table csvData;
String fileName = "data/testData.csv";
boolean b_saveCSV = false;
boolean b_train = false;
boolean b_test = false;

int label = 0;
int dataCnt = 0;

PGraphics pg;
int[] testFeatures;

void setup() {
  size(500, 500);

  //Initiate the dataList and set the header of table
  csvData = loadTable("testData.csv", "header");
  
  //Initiate the serial port
  rawData = new int[sensorNum];
  for (int i = 0; i < Serial.list().length; i++) println("[", i, "]:", Serial.list()[i]);
  String portName = Serial.list()[Serial.list().length-1];//MAC: check the printed list
  //String portName = Serial.list()[9];//WINDOWS: check the printed list
  port = new Serial(this, portName, 115200);
  port.bufferUntil('\n'); // arduino ends each data packet with a carriage return 
  port.clear();           // flush the Serial buffer

  pg = createGraphics(width, height);
  testFeatures = new int[sensorNum];
  
  try {
    initTrainingSet(csvData); // in Weka.pde
    lReg = (LinearRegression) weka.core.SerializationHelper.read(dataPath("lReg.model"));
    b_train = false;
    b_test = true;
    pg = get2DRegLine(pg, (Classifier)lReg, training);
  } 
  catch (Exception e) {
    e.printStackTrace();
  }
}

void draw() {
  background(255);
  
  if (b_saveCSV) {
    //Save the table to the file folder
    saveTable(csvData, fileName); //save table as CSV file
    println("Saved as: ", fileName);
    //reset b_saveCSV;
    b_saveCSV = false;
  }

  if (b_train) {
    //Save the table to the file folder
    try {
      initTrainingSet(csvData); // in Weka.pde
      lReg = new LinearRegression();
      lReg.buildClassifier(training);
      double slope1 = lReg.coefficients()[0];
      double slope2 = lReg.coefficients()[1];
      double slope3 = lReg.coefficients()[2];
      double intercept = lReg.coefficients()[lReg.coefficients().length-1];
      println(lReg);
      println("slope1:", slope1);
      println("slope2:", slope2);
      println("slope3:", slope3);
      println("intercept:", intercept);
      //println("ssr:", ssr);
      //println("r-Square:", rSquared);
      Evaluation eval = new Evaluation(training);
      eval.crossValidateModel(lReg, training, 10, new Random(1)); //10-fold cross validation
      println(eval.toSummaryString("\nResults\n======\n", false));
      
      weka.core.SerializationHelper.write(dataPath("lReg.model"), lReg);
      b_train = false;
      b_test = true;
    } 
    catch (Exception e) {
      e.printStackTrace();
    }
  }

  if (b_test) {
    //println(attributes.size());
    Instance inst = new DenseInstance(sensorNum+1);     
    inst.setValue(training.attribute(0), (float)testFeatures[0]); 
    inst.setValue(training.attribute(1), (float)testFeatures[1]);
    inst.setValue(training.attribute(2), (float)testFeatures[2]);
    // "instance" has to be associated with "Instances"
    Instances testData = new Instances("Test Data", attributes, 0);
    testData.add(inst);
    testData.setClassIndex(sensorNum);        

    double classification = -1;
    try {
      // have to get the data out of Instances
      classification = lReg.classifyInstance(testData.firstInstance());
    } 
    catch (Exception e) {
      e.printStackTrace();
    }
    String result = "y="+nf((float)classification,0,2)+", X=["+(float)testFeatures[0]+","+(float)testFeatures[1]+","+(float)testFeatures[2]+"]";
    pushStyle();
    fill(0);
    textSize(24);
    text(result, 20, 20);
    popStyle();
  }

  for (int i = 0; i < csvData.getRowCount(); i++) { 
    //read the values from the file
    TableRow row = csvData.getRow(i);
    float x = row.getFloat("x");
    float y = row.getFloat("y");
    float z = row.getFloat("z");
    // add more features here if you have

    //form a feature array
    float[] features = { x, y, z }; //form an array of input features

    //draw the data on the Canvas: 
    //Note: the row index is used as the label instead
    drawDataPoint1D(i, features);
  }
}

void serialEvent(Serial port) {   
  String inData = port.readStringUntil('\n');  // read the serial string until seeing a carriage return
  if (inData.charAt(0) == 'A') {  
    rawData[0] = int(trim(inData.substring(1)));
    return;
  }
  if (inData.charAt(0) == 'B') {  
    rawData[1] = int(trim(inData.substring(1)));
    return;
  }
  if (inData.charAt(0) == 'C') {  
    rawData[2] = int(trim(inData.substring(1)));
    //add a new row of data
    if (csvData.getRowCount() < dataCnt) {
      //add a row with new data 
      TableRow newRow = csvData.addRow();
      newRow.setFloat("x", rawData[0]);
      newRow.setFloat("y", rawData[1]);
      newRow.setFloat("z", rawData[2]);
      newRow.setFloat("index", label);
    }
    testFeatures[0] = rawData[0];
    testFeatures[1] = rawData[1];
    testFeatures[2] = rawData[2];
    return;
  }
}


void keyPressed() {
  if (key == 'S' || key == 's') {
    b_saveCSV = true;
  }
  if (key == 'T' || key == 't') {
    b_train = true;
    b_test = false;
    b_saveCSV = true;
  }
  if (key == 'D' || key == 'd') {
    b_train = true;
    b_test = false;
  }
  if (key == ' ') {
    csvData.clearRows();
    label = 0;
    dataCnt = 0;
  }
  if (key >= '0' && key <= '9') {
    label = key - '0';
    if (dataCnt<dataNum) dataCnt+=50;
    if (b_test) {
      dataNum+=50;
      dataCnt+=50;
    }
  }
}

//functions for drawing the data
void drawDataPoint1D(int _i, float[] _features) { 
  float pD = max(width/dataNum,1);
  float pX = map(((float)_i+0.5)/(float)dataNum, 0, 1, 0, width);
  float[] pY = new float[_features.length];
  for (int j = 0; j < _features.length; j++) pY[j] = map(_features[j], 0, 1024, 0, height) ; 
  pushStyle();
  for (int j = 0; j < _features.length; j++) {
    noStroke();
    if (j==0)fill(255, 0, 0);
    else if (j==1)fill(0, 255, 0);
    else if (j==2)fill(0, 0, 255);
    else if (j==3)fill(255, 0, 255);
    else fill(0);
    ellipse(pX, pY[j], pD, pD);
  }
  popStyle();
}