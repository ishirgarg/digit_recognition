# digit_recognition

A convolutional neural network that reads handwritten mathematical expressions and computes their value.
To build, run
```
python3 -m PyInstaller app.spec
```
This will build an executable called 'app' in the dist/app/ folder

Note that the executable may take a long time to startup; this is because it has to import the model parameters and setup the GUI