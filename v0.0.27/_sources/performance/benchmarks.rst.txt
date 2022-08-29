Benchmarks
==========

The :obj:`galois` library comes with benchmarking tests. They are contained in the `benchmarks/` folder. They are `pytest <https://docs.pytest.org/en/7.0.x/>`_
tests using the `pytest-benchmark <https://pytest-benchmark.readthedocs.io/en/latest/>`_ extension.

Install dependencies
--------------------

First, `pytest` and `pytest-benchmark` must be installed on your system. Easily install them by installing the development dependencies.

.. code-block:: sh

   $ python3 -m pip install -r requirements-dev.txt

Create a benchmark
------------------

To create a benchmark, invoke `pytest` on the `benchmarks/` folder or a specific test set (e.g., `benchmarks/test_field_arithmetic.py`). It is also
advised to pass extra arguments to format the display `--benchmark-columns=min,max,mean,stddev,median` and `--benchmark-sort=name`.

.. code-block::

   $ python3 -m pytest benchmarks/test_field_arithmetic.py --benchmark-columns=min,max,mean,stddev,median --benchmark-sort=name
   ============================================================================== test session starts ===============================================================================platform linux -- Python 3.8.10, pytest-4.6.9, py-1.8.1, pluggy-0.13.0
   benchmark: 3.4.1 (defaults: timer=time.perf_counter disable_gc=False min_rounds=5 min_time=0.000005 max_time=1.0 calibration_precision=10 warmup=False warmup_iterations=100000)
   rootdir: /mnt/c/Users/matth/repos/galois, inifile: setup.cfg
   plugins: requests-mock-1.9.3, cov-3.0.0, benchmark-3.4.1, typeguard-2.13.3, anyio-3.5.0
   collected 56 items

   benchmarks/test_field_arithmetic.py ........................................................                                                                               [100%]

   ------------------- benchmark "GF(2) Array Arithmetic: shape=(100_000,), ufunc_mode='jit-calculate'": 8 tests -------------------
   Name (time in us)                    Min                 Max                Mean             StdDev              Median
   ---------------------------------------------------------------------------------------------------------------------------------
   test_add                         14.6070 (1.20)     246.2890 (1.42)      18.5947 (1.27)      8.1284 (1.18)      16.2800 (1.24)
   test_additive_inverse            12.2030 (1.0)      205.2510 (1.19)      14.6355 (1.0)       6.9034 (1.0)       13.1750 (1.0)
   test_divide                      31.0980 (2.55)     411.8500 (2.38)      37.0098 (2.53)     16.5464 (2.40)      32.5610 (2.47)
   test_multiplicative_inverse      20.9890 (1.72)     253.5650 (1.47)      28.9587 (1.98)     14.3950 (2.09)      22.5020 (1.71)
   test_multiply                    14.1260 (1.16)     305.0260 (1.76)      17.1140 (1.17)     11.2444 (1.63)      14.8480 (1.13)
   test_power                      144.7910 (11.87)    173.0440 (1.0)      156.4390 (10.69)    11.8957 (1.72)     153.3870 (11.64)
   test_scalar_multiply            521.6480 (42.75)    823.1500 (4.76)     551.7997 (37.70)    34.6098 (5.01)     545.2820 (41.39)
   test_subtract                    14.5470 (1.19)     564.8020 (3.26)      19.6030 (1.34)     14.3437 (2.08)      15.9500 (1.21)
   ---------------------------------------------------------------------------------------------------------------------------------

   ---------------------- benchmark "GF(257) Array Arithmetic: shape=(100_000,), ufunc_mode='jit-calculate'": 8 tests -----------------------
   Name (time in us)                      Min                   Max                  Mean              StdDev                Median
   ------------------------------------------------------------------------------------------------------------------------------------------
   test_add                           73.2680 (1.45)       161.8240 (1.0)         77.5348 (1.38)       6.5241 (1.0)         74.7200 (1.44)
   test_additive_inverse              50.6750 (1.0)        228.7990 (1.41)        56.0843 (1.0)       14.0921 (2.16)        52.0480 (1.0)
   test_divide                     3,110.9100 (61.39)    3,194.0960 (19.74)    3,150.1173 (56.17)     22.0701 (3.38)     3,148.5205 (60.49)
   test_multiplicative_inverse     3,129.6760 (61.76)    3,176.1430 (19.63)    3,155.8828 (56.27)     12.4384 (1.91)     3,153.8910 (60.60)
   test_multiply                     190.8880 (3.77)       211.8770 (1.31)       198.7725 (3.54)       7.5219 (1.15)       197.5500 (3.80)
   test_power                      3,093.3480 (61.04)    3,941.4670 (24.36)    3,360.6968 (59.92)    368.8893 (56.54)    3,150.8450 (60.54)
   test_scalar_multiply              503.8040 (9.94)       750.9060 (4.64)       521.2844 (9.29)      13.8751 (2.13)       520.4705 (10.00)
   test_subtract                      73.3680 (1.45)       294.1580 (1.82)        78.2542 (1.40)       8.8710 (1.36)        75.0700 (1.44)
   ------------------------------------------------------------------------------------------------------------------------------------------

   -------------------- benchmark "GF(257) Array Arithmetic: shape=(100_000,), ufunc_mode='jit-lookup'": 8 tests -------------------
   Name (time in us)                    Min                 Max                Mean             StdDev              Median
   ---------------------------------------------------------------------------------------------------------------------------------
   test_add                         74.2090 (1.44)      94.9680 (1.06)      81.2720 (1.46)      7.0786 (1.05)      78.1520 (1.48)
   test_additive_inverse            51.3860 (1.0)       89.6480 (1.0)       55.8487 (1.0)       8.6054 (1.28)      52.6985 (1.0)
   test_divide                     198.2020 (3.86)     252.7840 (2.82)     209.8239 (3.76)     12.5969 (1.87)     207.0735 (3.93)
   test_multiplicative_inverse     135.6840 (2.64)     172.2930 (1.92)     144.4076 (2.59)      9.8891 (1.47)     141.1450 (2.68)
   test_multiply                   187.7320 (3.65)     288.9800 (3.22)     200.3634 (3.59)     11.2736 (1.67)     198.4620 (3.77)
   test_power                      255.5790 (4.97)     295.4040 (3.30)     269.9318 (4.83)     10.7527 (1.60)     266.8300 (5.06)
   test_scalar_multiply            502.5610 (9.78)     757.1780 (8.45)     524.1787 (9.39)     13.9009 (2.06)     524.0170 (9.94)
   test_subtract                    74.7700 (1.46)      94.8770 (1.06)      79.6818 (1.43)      6.7367 (1.0)       75.6520 (1.44)
   ---------------------------------------------------------------------------------------------------------------------------------

   ------------------------- benchmark "GF(2^8) Array Arithmetic: shape=(100_000,), ufunc_mode='jit-calculate'": 8 tests --------------------------
   Name (time in us)                       Min                    Max                   Mean                StdDev                 Median
   ------------------------------------------------------------------------------------------------------------------------------------------------
   test_add                            14.1770 (1.16)         63.0380 (1.0)          15.8036 (1.04)         3.4018 (1.0)          14.8980 (1.13)
   test_additive_inverse               12.2530 (1.0)         231.6340 (3.67)         15.2299 (1.0)          5.7599 (1.69)         13.1950 (1.0)
   test_divide                     18,110.1600 (>1000.0)  23,570.4150 (373.91)   20,069.0754 (>1000.0)  2,438.1073 (716.71)   18,794.8630 (>1000.0)
   test_multiplicative_inverse     16,511.2540 (>1000.0)  16,933.0940 (268.62)   16,692.3108 (>1000.0)    175.2790 (51.53)    16,691.8020 (>1000.0)
   test_multiply                    1,092.8780 (89.19)     1,158.5510 (18.38)     1,126.8230 (73.99)       24.4832 (7.20)      1,136.4800 (86.13)
   test_power                      16,858.2620 (>1000.0)  18,860.9460 (299.20)   17,547.2790 (>1000.0)    906.1260 (266.37)   16,982.8170 (>1000.0)
   test_scalar_multiply               835.2460 (68.17)     1,118.3870 (17.74)       862.5502 (56.64)       14.4826 (4.26)        861.3440 (65.28)
   test_subtract                       14.2560 (1.16)        203.0710 (3.22)         16.1151 (1.06)         4.4977 (1.32)         15.0290 (1.14)
   ------------------------------------------------------------------------------------------------------------------------------------------------

   --------------------- benchmark "GF(2^8) Array Arithmetic: shape=(100_000,), ufunc_mode='jit-lookup'": 8 tests --------------------
   Name (time in us)                    Min                   Max                Mean             StdDev              Median
   -----------------------------------------------------------------------------------------------------------------------------------
   test_add                         14.1270 (1.15)       234.6490 (4.07)      16.0570 (1.18)      4.4345 (1.31)      14.9080 (1.17)
   test_additive_inverse            12.2530 (1.0)         90.4190 (1.57)      13.6181 (1.0)       3.3801 (1.0)       12.7340 (1.0)
   test_divide                     204.2830 (16.67)      251.3910 (4.36)     219.1132 (16.09)    16.4107 (4.86)     212.2180 (16.67)
   test_multiplicative_inverse     136.0750 (11.11)      170.1890 (2.95)     143.8333 (10.56)    10.7180 (3.17)     138.2690 (10.86)
   test_multiply                   189.9960 (15.51)      213.2000 (3.70)     200.5345 (14.73)     7.8811 (2.33)     201.4980 (15.82)
   test_power                      318.4670 (25.99)      346.3900 (6.01)     336.9693 (24.74)     8.8297 (2.61)     337.2825 (26.49)
   test_scalar_multiply            795.9620 (64.96)    1,107.1750 (19.21)    829.7765 (60.93)    28.8501 (8.54)     825.2110 (64.80)
   test_subtract                    14.3570 (1.17)        57.6480 (1.0)       16.2090 (1.19)      4.3581 (1.29)      15.0080 (1.18)
   -----------------------------------------------------------------------------------------------------------------------------------

   ------------------------- benchmark "GF(3^5) Array Arithmetic: shape=(1_000,), ufunc_mode='jit-calculate'": 8 tests -------------------------
   Name (time in us)                       Min                    Max                   Mean             StdDev                 Median
   ---------------------------------------------------------------------------------------------------------------------------------------------
   test_add                           267.9320 (1.61)        340.8490 (1.53)        290.3090 (1.62)     20.4666 (1.53)        281.4870 (1.66)
   test_additive_inverse              166.5620 (1.0)         222.3970 (1.0)         179.3101 (1.0)      15.4222 (1.16)        169.9880 (1.0)
   test_divide                     11,050.2190 (66.34)    11,139.6170 (50.09)    11,085.6634 (61.82)    36.7729 (2.76)     11,067.3310 (65.11)
   test_multiplicative_inverse     10,000.1510 (60.04)    10,081.4430 (45.33)    10,039.3364 (55.99)    36.0173 (2.70)     10,049.1930 (59.12)
   test_multiply                      861.3440 (5.17)        896.6900 (4.03)        875.0538 (4.88)     13.3343 (1.0)         874.5490 (5.14)
   test_power                       9,822.0570 (58.97)    10,007.8650 (45.00)     9,937.5374 (55.42)    69.6998 (5.23)      9,945.8890 (58.51)
   test_scalar_multiply               731.6210 (4.39)      1,445.9930 (6.50)        776.0280 (4.33)     67.8114 (5.09)        758.5320 (4.46)
   test_subtract                      265.7590 (1.60)        633.8150 (2.85)        301.0594 (1.68)     68.9719 (5.17)        282.0190 (1.66)
   ---------------------------------------------------------------------------------------------------------------------------------------------

   ------------------- benchmark "GF(3^5) Array Arithmetic: shape=(1_000,), ufunc_mode='jit-lookup'": 8 tests -------------------
   Name (time in us)                   Min                 Max               Mean             StdDev             Median
   ------------------------------------------------------------------------------------------------------------------------------
   test_add                        12.7240 (1.28)      33.4230 (1.13)     15.9908 (1.24)      5.6576 (1.50)     14.2765 (1.30)
   test_additive_inverse           10.1190 (1.01)      34.8550 (1.18)     12.9293 (1.0)       5.9502 (1.57)     11.2965 (1.03)
   test_divide                     10.9400 (1.10)      29.5050 (1.0)      13.0403 (1.01)      3.7818 (1.0)      12.1275 (1.10)
   test_multiplicative_inverse      9.9790 (1.0)      106.0190 (3.59)     15.9433 (1.23)     19.9987 (5.29)     10.9910 (1.0)
   test_multiply                   10.7700 (1.08)      34.5450 (1.17)     13.2007 (1.02)      5.5236 (1.46)     11.5815 (1.05)
   test_power                      14.0560 (1.41)      51.4260 (1.74)     18.7223 (1.45)      8.8523 (2.34)     15.5290 (1.41)
   test_scalar_multiply            17.5130 (1.75)     260.2270 (8.82)     20.3556 (1.57)      6.9250 (1.83)     18.9950 (1.73)
   test_subtract                   14.1060 (1.41)      33.1720 (1.12)     16.8235 (1.30)      5.8832 (1.56)     14.4920 (1.32)
   ------------------------------------------------------------------------------------------------------------------------------

   Legend:
   Outliers: 1 Standard Deviation from Mean; 1.5 IQR (InterQuartile Range) from 1st Quartile and 3rd Quartile.
   OPS: Operations Per Second, computed as 1 / Mean
   ==================================================================== 56 passed, 16 warnings in 16.14 seconds =====================================================================

Compare with a previous benchmark
---------------------------------

If you would like to compare the performance impacts of a branch, first run a benchmark on `master` using the `--benchmark-save` option.
This will save the file `.benchmarks/0001_master.json`.

.. code-block::

   $ git checkout master
   $ python3 -m pytest benchmarks/test_field_arithmetic.py --benchmark-save=master --benchmark-columns=min,max,mean,stddev,median --benchmark-sort=name

Next, checkout your branch and run another benchmark. This will save the file `.benchmarks/0001_branch.json`.

.. code-block::

   $ git checkout branch
   $ python3 -m pytest benchmarks/test_field_arithmetic.py --benchmark-save=branch --benchmark-columns=min,max,mean,stddev,median --benchmark-sort=name

And finally, compare the two benchmarks.

.. code-block::

   $ python3 -m pytest-benchmark compare 0001_master 0001_branch
