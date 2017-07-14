<!-- vim: set fileencoding=utf-8 fileformat=unix : -->
<!-- vim: set spell spelllang=en : -->
<!-- -*- coding: utf-8 -*- -->
<!-- vim: set ts=8 et sw=4 sts=4 sta : -->

# atmosounding
Atmospheric radio sounding analysis using a Savitzky-Golay low-pass filter working with irregularly sampled data.


## Summary

The [`sounding.py`](http://github.com/cvr/atmosounding) python script
reads data from an atmospheric radio-sounding
(in this case from file `example_data.txt`).
The data is a table in the ASCII format structured by several columns,
delimited by spaces.

The program does not assumes a specific number of columns,
instead it works by trying to read an header with the variables names.
This function will not work unless the following variable names are found:

    Time, Alt_AGL, Press, Temp, DP, VTemp, RelHum, Mix_Rat, WSpeed

The data is processed and a Savitzky-Golay low-pass filter is applied
to smooth some vertical profiles (wind speed and temperature) and to
extract both the smoothed profiles and its derivatives.


## Savitzky-Golay algorithm for irregularly sampled data

The [`savitzky_golay_lowpassfilter.py`](http://github.com/cvr/atmosounding)
python script contains functions that apply a Savitzky-Golay low-pass filter
to irregularly sampled data.



### Functions

```python
F, dF, ddF = savgol_irregular_brute (f, x, xwindow, order=4, deriv=2)
```
This function applies the Savitzky-Golay filter in a brute force way.
For each record in the input data, a polynomial is fitted, meaning
a matrix is pseudo-inverted to solve the least-squares problem.
    
The original technique assumes a signal with uniform sampling and only
one matrix is pseudo-inverted, independently of the number of records
in the input data.

```python
F, dF, ddF = savgol_irregular_interp (f, x, xwindow, order=4, deriv=2, dx=None)
```
This function applies the Savitzky-Golay filter, linearly interpolating
the input data to regularize it. Although a polynomial is fitted to
each record of the input data, the least-squares problem requires only
one coefficients matrix, thus the pseudo-invertion of the matrix is
performed only once. The exception is for recors at the boundaries,
where a polynomial is fitted for each reacord.
    
### References

A Savitzky, MJE Golay, (1964).
Smoothing and Differentiation of Data by Simplified Least Squares Procedures.
*Analytical Chemistry*, vol. 36(8):1627-1639.


## Example

The code runs with either python 2 and 3. Simply run:
```sh
./sounding.py
```

