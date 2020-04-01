# KTH Group 6 - Activity Recognition
This repo contains the code used by KTH group 6 to recognize three activities
based on three second intervals of accelerometer data:
1. Walking
2. Standing
3. Running

The code uses the data contained in the `data` directory. This data was collected by Javad
using his android phone.

Our work is based on the work presented in

```
Siirtola, Pekka, and Juha Röning. "Recognizing human activities user-independently on
smartphones based on accelerometer data." IJIMAI 1.5 (2012): 38-45.
```

which can be found [here](https://dialnet.unirioja.es/descarga/articulo/3954593.pdf).

## Running the code
To run the code simply open Matlab and run `process_data.m`. By changing the value of
variable `plot_data` you can toggle whether the script plots or not. Examples of plots
are saved in the `media` folder.
