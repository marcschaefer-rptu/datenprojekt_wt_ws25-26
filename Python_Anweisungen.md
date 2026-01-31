# Python Anweisungen - WT_Projekt_2026

## Imports
- import os
- import matplotlib.pyplot as plt
- import matplotlib.dates as mpl_dates
- import matplotlib.ticker as mpl_tick
- from statistics import mean, median, mode, multimode
- import numpy as np
- import csv
- from scipy.optimize import curve_fit

## Konstanten
- MONATE
- MONATE_ZAHLEN

## Funktionen
1. monthToInt(month)
2. intToMonth(month)
3. abweichungMedian(data)
4. quartile(data)
5. dezile(data)
6. variationsKoeffizient(data)
7. korrelationsKoeffizient(data, data2)
8. readFromDataToArray(data, array, dataSet)
9. clean(array)
10. outputToFile(fileName, array, beschreibung)
11. urlist(filePath, fileName, data)
12. ranglist(filePath, fileName, data)
13. boxWhiskerPlot(filePath, fileName, data, yLabel)
14. histogram(data, fileName)
15. sine_function(x, A, B, C, D)
16. _fit_sine_curve(zeit, werte, initial_guess)
17. scatterPlot(fileName, data, title, yLabel)
18. scatterPlotNoLine(fileName, data, title, yLabel)
19. stepsPerWeekday(data, fileName)
20. stepsPerMonth(data, fileName)
21. stepsPerDay(data, fileName)
22. stepsPerDayBoxPlot(data, fileName)
23. stepsPerMonthBoxPlot(data, fileName)

