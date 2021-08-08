# MakeIT-How-to-deploy-SK-learn-model-on-ESP8266

This repository is dedicated to another tutorial of my youtube channel MakeIT. This project is about tiny machine learning and how to classify orientation of an ESP8266 using a SciKit-learn machine learning model. It's available [here](https://youtu.be/2Rrz7orwJvU)

You can find below the pipeline to develop a TinyML project :

![Pipeline](https://github.com/BaptisteZloch/MakeIT-How-to-deploy-SK-learn-model-on-ESP8266/blob/master/Assets/pipeline.png?raw=true)


## [Python part](https://github.com/BaptisteZloch/MakeIT-How-to-deploy-SK-learn-model-on-ESP8266/tree/master/Python%20notebook)

This has been designed using SciKit-learn library for machine learning. This network has been trained on [dataset](https://github.com/BaptisteZloch/MakeIT-How-to-deploy-SK-learn-model-on-ESP8266/blob/master/Python%20notebook/Orientation.csv) created with [Datasets Builder](https://github.com/BaptisteZloch/MakeIT-How-to-build-your-own-datasets-with-Datasets-Builder) my software to create datasets.
Then this model has been exported into c code .h file. To convert the model I have used [micromlgen](https://github.com/eloquentarduino/micromlgen) python library.

## [Arduino part](https://github.com/BaptisteZloch/MakeIT-How-to-deploy-SK-learn-model-on-ESP8266/tree/master/Arduino%20code/MAKEIT_SKlearn_Orientation_spot)

In this part I have used VS code and platformIO IDE to deploy the model onto the [ESP8266 WEMOS D1 mini lite](https://www.ebay.fr/itm/NodeMCU-Lua-ESP8266-ESP-12-WeMos-D1-Mini-WIFI-4M-Bytes-Development-CP2104-Board-/173447615078). To do it we had to import our model, and the libraries to use I2C protocol and MPU6050 acccelerometer on the ESP8266.

Here is the schema of the wiring : 

![Wiring schema](https://github.com/BaptisteZloch/MakeIT-How-to-deploy-SK-learn-model-on-ESP8266/blob/master/Assets/D1WemosMini_MPU6050.png?raw=true)

To conclude by running the code with get an accuracy about 100% which is quite satisfiying.
