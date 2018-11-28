//*********************************************
// Weka for Processing
// LinReg_Weka_basic
// Rong-Hao Liang: r.liang@tue.nl
//*********************************************
// Drag the cursor to make datapoint
// [T] to Save CSV, Train Model, and Test Data

Table csvData;
String fileName = "data/testData.csv";
boolean b_saveCSV = false;
boolean b_train = false;
boolean b_test = false;

int label = 0;
PGraphics pg;

void setup() {
  size(500, 500);

  //Initiate the dataList and set the header of table
  csvData = loadTable("testData.csv", "header");

  pg = createGraphics(width, height);
}

void draw() {
  background(255);

  if (b_train) {
    //Save the table to the file folder
    try {
      initTrainingSet(csvData, 2); // in Weka.pde
      lReg = new LinearRegression();
      lReg.buildClassifier(training);
      double slope = lReg.coefficients()[0];
      double intercept = lReg.coefficients()[lReg.coefficients().length-1];
      double ssr = RegressionAnalysis.calculateSSR(training, attributes.get(0), slope, intercept);
      double rSquared = RegressionAnalysis.calculateRSquared(training, ssr);
      println(lReg);
      println("slope:", slope);
      println("intercept:", intercept);
      println("ssr:", ssr);
      println("r-Square:", rSquared);
      Evaluation eval = new Evaluation(training);
      eval.crossValidateModel(lReg, training, 10, new Random(1)); //10-fold cross validation
      println(eval.toSummaryString("\nResults\n======\n", false));
      //double rmse = eval.rootMeanSquaredError();

      weka.core.SerializationHelper.write(dataPath("lReg.model"), lReg);
      b_train = false;
      b_test = true;

      pg = get2DRegLine(pg, (Classifier)lReg, training);
    } 
    catch (Exception e) {
      e.printStackTrace();
    }
  }

  if (b_test) {

    //println(attributes.size());
    Instance inst = new DenseInstance(2);     
    inst.setValue(training.attribute(0), (float)mouseX); 
    // "instance" has to be associated with "Instances"
    Instances testData = new Instances("Test Data", attributes, 0);
    testData.add(inst);
    testData.setClassIndex(1);        

    double classification = -1;
    try {
      // have to get the data out of Instances
      classification = lReg.classifyInstance(testData.firstInstance());
    } 
    catch (Exception e) {
      e.printStackTrace();
    }
    image(pg, 0, 0);
    drawMouseMono(classification);
  } else {
    drawMouseMono(mouseY);
  }

  for (int i = 0; i < csvData.getRowCount(); i++) { 
    //read the values from the file
    TableRow row = csvData.getRow(i);
    float x = row.getFloat("x");
    int index = (int) row.getFloat("y");

    //form a feature array
    float[] features = {x};

    //draw the data on the Canvas
    drawDataMono(index, features);
  }

  if (b_saveCSV) {
    //Save the table to the file folder
    saveTable(csvData, fileName); //save table as CSV file
    println("Saved as: ", fileName);
    //reset b_saveCSV;
    b_saveCSV = false;
  }
}

void mousePressed() {
  if (mouseButton == RIGHT) {
    ++label;
    label %= 10;
  }
}

void mouseDragged() {
  //add a row with new data 
  TableRow newRow = csvData.addRow();
  newRow.setFloat("x", (float)mouseX);
  newRow.setFloat("y", (float)mouseY);
  newRow.setFloat("label", label);
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
  }
  if (key >= '0' && key <= '9') {
    label = key - '0';
  }
}

void drawMouseMono(double _cls) {
  pushStyle();
  textSize(12);
  textAlign(CENTER, CENTER);
  if (mousePressed) {
    stroke(0);
    fill(255);
  } else { 
    noStroke();
    fill(0);
  }
  ellipse(mouseX, mouseY, 40, 40);

  if (mousePressed) {
    noStroke();
    fill(0);
  } else { 
    fill(255);
  }
  text((int)_cls, mouseX, mouseY);

  popStyle();
}

void drawMouseCursor(int _index) {
  color colors[] = {
    color(155, 89, 182), color(63, 195, 128), color(214, 69, 65), 
    color(82, 179, 217), color(244, 208, 63), color(242, 121, 53), 
    color(0, 121, 53), color(128, 128, 0), color(52, 0, 128), 
    color(128, 52, 0), color(52, 128, 0), color(128, 52, 0)
  };
  pushStyle();
  textSize(12);
  textAlign(CENTER, CENTER);
  if (mousePressed) {
    stroke(0);
    fill(255);
  } else { 
    noStroke();
    fill(colors[_index]);
  }
  ellipse(mouseX, mouseY, 20, 20);

  if (mousePressed) {
    noStroke();
    fill(0);
  } else { 
    fill(255);
  }
  text(_index, mouseX, mouseY);

  popStyle();
}

void drawDataMono(int _index, float[] _features) {
  pushStyle();
  textSize(12);
  textAlign(CENTER, CENTER);

  stroke(0);
  fill(255);
  ellipse(_features[0], _index, 20, 20);

  noStroke();
  fill(0);
  text(_index, _features[0], _index);

  popStyle();
}

void drawDataPoint(int _index, float[] _features) {
  color colors[] = {
    color(155, 89, 182), color(63, 195, 128), color(214, 69, 65), 
    color(82, 179, 217), color(244, 208, 63), color(242, 121, 53), 
    color(0, 121, 53), color(128, 128, 0), color(52, 0, 128), 
    color(128, 52, 0), color(52, 128, 0), color(128, 52, 0)
  };
  pushStyle();
  textSize(12);
  textAlign(CENTER, CENTER);

  stroke(0);
  fill(colors[_index]);
  ellipse(_features[0], _features[1], 20, 20);

  noStroke();
  fill(0);
  text(_index, _features[0], _features[1]);

  popStyle();
}