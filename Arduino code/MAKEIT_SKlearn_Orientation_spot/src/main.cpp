 /*
  * MakeIT 4th tutorial
  * Title : Orientation spotting using SKlearn classifier model 
  * Author : Baptiste ZLOCH (MakeIT owner)
  * Description : This code will make classification of the orientation of the board using a machine learning model build with SciKit-learn.
  * Date : 03/08/2021
  * Tested with : ESP8266, ESP8266 WEMOS D1 mini lite, Arduino Uno.
  */
 
#include <Arduino.h>
#include "model.h" //the classifier model
#include <Wire.h> //I2C protocol
#include <Adafruit_MPU6050.h> //to use the MPU6050 accelerometer
#include <Adafruit_Sensor.h> //to use the adafruit sensors

Adafruit_MPU6050 accelerometer; //instanciate the accelerometer object 
Eloquent::ML::Port::SVM SVM_SKleanr_classifier;//instanciate the classifier object 

void setup()
{
  Serial.begin(115200);//begin the Serial communication
  Wire.begin(); //begin the I2C communication

  if (!accelerometer.begin()) //begin the MPU communication
  {
    Serial.println("Error with MPU");
  }

  Serial.println("Everything is setup !");
}

void loop()
{
  sensors_event_t a, g, temp; //instanciate the sensors's reading object 
  accelerometer.getEvent(&a, &g, &temp);//register the read values to the object declared before

  float features[] = {a.acceleration.x, a.acceleration.y, a.acceleration.z}; //create the array containing the read values size is : 1x3 same as defined in google colab
  String output_str = SVM_SKleanr_classifier.predictLabel(features); //run inference

  Serial.println(output_str);//print the resulting orientation
  delay(700);
}