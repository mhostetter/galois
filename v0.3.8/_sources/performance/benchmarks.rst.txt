Benchmarks
==========

The :obj:`galois` library comes with benchmarking tests. They are contained in the `benchmarks/` folder. They are `pytest <https://docs.pytest.org/en/7.0.x/>`_
tests using the `pytest-benchmark <https://pytest-benchmark.readthedocs.io/en/latest/>`_ extension.

Install dependencies
--------------------

First, `pytest` and `pytest-benchmark` must be installed on your system. Easily install them by installing the development dependencies.

.. code-block:: console

   $ python3 -m pip install -r requirements-dev.txt

Create a benchmark
------------------

To create a benchmark, invoke `pytest` on the `benchmarks/` folder or a specific test set (e.g., `benchmarks/test_field_arithmetic.py`). It is also
advised to pass extra arguments to format the display `--benchmark-columns=min,max,mean,stddev,median` and `--benchmark-sort=name`.

.. code-block:: console

   $ python3 -m pytest benchmarks/test_field_arithmetic.py --benchmark-columns=min,max,mean,stddev,median --benchmark-sort=name
   ===================================================================== test session starts =====================================================================platform linux -- Python 3.8.10, pytest-4.6.9, py-1.8.1, pluggy-0.13.0
   benchmark: 3.4.1 (defaults: timer=time.perf_counter disable_gc=False min_rounds=5 min_time=0.000005 max_time=1.0 calibration_precision=10 warmup=False warmup_iterations=100000)
   rootdir: /mnt/c/Users/matth/repos/galois, inifile: setup.cfg
   plugins: requests-mock-1.9.3, cov-3.0.0, benchmark-3.4.1, typeguard-2.13.3, anyio-3.5.0
   collected 56 items

   benchmarks/test_field_arithmetic.py ........................................................                                                            [100%]

   -------------------- benchmark "GF(2) Array Arithmetic: shape=(100_000,), ufunc_mode='jit-calculate'": 8 tests --------------------
   Name (time in us)                    Min                   Max                Mean             StdDev              Median
   -----------------------------------------------------------------------------------------------------------------------------------
   test_add                         16.3810 (1.21)       218.3280 (1.22)      18.9455 (1.17)      5.4959 (1.07)      17.3620 (1.22)
   test_additive_inverse            13.5850 (1.0)        206.5360 (1.15)      16.1445 (1.0)       5.9249 (1.16)      14.2670 (1.0)
   test_divide                     132.0870 (9.72)       191.0680 (1.07)     149.6357 (9.27)     16.9537 (3.31)     145.4920 (10.20)
   test_multiplicative_inverse      91.4410 (6.73)       179.0050 (1.0)      102.6590 (6.36)     18.8467 (3.68)      94.4670 (6.62)
   test_multiply                    16.0400 (1.18)       229.4400 (1.28)      18.3296 (1.14)      5.1267 (1.0)       16.9010 (1.18)
   test_power                      150.2410 (11.06)      212.2870 (1.19)     168.8103 (10.46)    16.4850 (3.22)     166.2860 (11.66)
   test_scalar_multiply            543.3970 (40.00)      714.2870 (3.99)     562.2968 (34.83)    12.4125 (2.42)     559.1370 (39.19)
   test_subtract                    16.3110 (1.20)     2,233.8710 (12.48)     19.2938 (1.20)     23.4038 (4.57)      17.2520 (1.21)
   -----------------------------------------------------------------------------------------------------------------------------------

   ---------------------- benchmark "GF(257) Array Arithmetic: shape=(100_000,), ufunc_mode='jit-calculate'": 8 tests ----------------------
   Name (time in us)                      Min                   Max                  Mean             StdDev                Median
   -----------------------------------------------------------------------------------------------------------------------------------------
   test_add                           78.2860 (1.37)       311.2620 (1.40)        87.9984 (1.26)     12.1680 (1.04)        81.7530 (1.36)
   test_additive_inverse              57.0860 (1.0)        281.9070 (1.27)        69.7403 (1.0)      17.0927 (1.47)        60.0520 (1.0)
   test_divide                     3,274.0860 (57.35)    3,351.6220 (15.09)    3,309.5920 (47.46)    27.4510 (2.36)     3,307.3240 (55.07)
   test_multiplicative_inverse     3,245.1620 (56.85)    4,295.9590 (19.34)    3,350.8016 (48.05)    96.3332 (8.26)     3,321.1050 (55.30)
   test_multiply                     197.1090 (3.45)       305.5620 (1.38)       218.1805 (3.13)     20.9767 (1.80)       213.6600 (3.56)
   test_power                      3,270.7210 (57.29)    3,520.5480 (15.85)    3,349.1942 (48.02)    91.3962 (7.84)     3,329.6105 (55.45)
   test_scalar_multiply              544.0880 (9.53)     1,182.1140 (5.32)       575.6227 (8.25)     42.0059 (3.60)       562.4830 (9.37)
   test_subtract                      77.6160 (1.36)       222.1760 (1.0)         88.3242 (1.27)     11.6562 (1.0)         82.8905 (1.38)
   -----------------------------------------------------------------------------------------------------------------------------------------

   -------------------- benchmark "GF(257) Array Arithmetic: shape=(100_000,), ufunc_mode='jit-lookup'": 8 tests -------------------
   Name (time in us)                    Min                 Max                Mean             StdDev              Median
   ---------------------------------------------------------------------------------------------------------------------------------
   test_add                         79.0580 (1.37)     393.6670 (2.39)      86.7954 (1.26)     12.6945 (1.0)       81.4630 (1.34)
   test_additive_inverse            57.9080 (1.0)      164.6380 (1.0)       69.0218 (1.0)      21.7213 (1.71)      60.6330 (1.0)
   test_divide                     228.7890 (3.95)     280.8050 (1.71)     243.1431 (3.52)     16.6688 (1.31)     241.0210 (3.98)
   test_multiplicative_inverse     263.8140 (4.56)     348.4620 (2.12)     290.6663 (4.21)     20.8113 (1.64)     284.3620 (4.69)
   test_multiply                   193.5820 (3.34)     475.2490 (2.89)     216.4317 (3.14)     24.6557 (1.94)     212.2370 (3.50)
   test_power                      311.6030 (5.38)     389.2180 (2.36)     328.9333 (4.77)     18.9217 (1.49)     326.1145 (5.38)
   test_scalar_multiply            539.7710 (9.32)     973.1410 (5.91)     573.4538 (8.31)     49.0047 (3.86)     557.7030 (9.20)
   test_subtract                    80.3500 (1.39)     270.0450 (1.64)      97.6062 (1.41)     37.3127 (2.94)      89.1270 (1.47)
   ---------------------------------------------------------------------------------------------------------------------------------

   ------------------------ benchmark "GF(2^8) Array Arithmetic: shape=(100_000,), ufunc_mode='jit-calculate'": 8 tests -------------------------
   Name (time in us)                       Min                    Max                   Mean              StdDev                 Median
   ----------------------------------------------------------------------------------------------------------------------------------------------
   test_add                            16.6110 (1.21)        218.1990 (1.09)         19.4288 (1.21)       5.8745 (1.11)         17.4830 (1.22)
   test_additive_inverse               13.6750 (1.0)         200.7150 (1.0)          16.0465 (1.0)        5.2959 (1.0)          14.3070 (1.0)
   test_divide                     13,280.4310 (971.15)   13,367.6440 (66.60)    13,340.0968 (831.34)    36.5738 (6.91)     13,354.6500 (933.43)
   test_multiplicative_inverse     11,842.1600 (865.97)   15,404.4870 (76.75)    12,129.1417 (755.88)   529.9702 (100.07)   12,015.3740 (839.82)
   test_multiply                    1,079.0300 (78.91)     1,137.0780 (5.67)      1,098.1473 (68.44)     18.6741 (3.53)      1,092.5140 (76.36)
   test_power                      12,832.8340 (938.41)   13,115.7640 (65.35)    12,942.1951 (806.54)    92.9381 (17.55)    12,928.9640 (903.68)
   test_scalar_multiply               883.2930 (64.59)     1,192.1310 (5.94)        928.3991 (57.86)     44.9582 (8.49)        912.0860 (63.75)
   test_subtract                       16.6210 (1.22)      1,334.7780 (6.65)         19.7528 (1.23)      15.2536 (2.88)         17.4330 (1.22)
   ----------------------------------------------------------------------------------------------------------------------------------------------

   --------------------- benchmark "GF(2^8) Array Arithmetic: shape=(100_000,), ufunc_mode='jit-lookup'": 8 tests --------------------
   Name (time in us)                    Min                   Max                Mean             StdDev              Median
   -----------------------------------------------------------------------------------------------------------------------------------
   test_add                         16.0900 (1.23)       277.5990 (3.89)      18.8739 (1.24)      5.6347 (1.35)      17.1720 (1.24)
   test_additive_inverse            13.1050 (1.0)         71.3340 (1.0)       15.1649 (1.0)       4.1860 (1.0)       13.8860 (1.0)
   test_divide                     215.6730 (16.46)      271.6490 (3.81)     233.7595 (15.41)    16.0094 (3.82)     229.9500 (16.56)
   test_multiplicative_inverse     152.3150 (11.62)      207.4480 (2.91)     167.0589 (11.02)    12.9483 (3.09)     166.4220 (11.98)
   test_multiply                   199.3430 (15.21)      250.2580 (3.51)     220.8079 (14.56)    17.1620 (4.10)     216.1740 (15.57)
   test_power                      331.7910 (25.32)      401.3410 (5.63)     348.8168 (23.00)    17.4759 (4.17)     348.8730 (25.12)
   test_scalar_multiply            850.2810 (64.88)    1,128.3010 (15.82)    884.6499 (58.34)    29.6705 (7.09)     876.5800 (63.13)
   test_subtract                    16.0400 (1.22)        83.5460 (1.17)      18.2685 (1.20)      4.4904 (1.07)      16.8610 (1.21)
   -----------------------------------------------------------------------------------------------------------------------------------

   --------------------- benchmark "GF(3^5) Array Arithmetic: shape=(100_000,), ufunc_mode='jit-lookup'": 8 tests --------------------
   Name (time in us)                    Min                   Max                Mean             StdDev              Median
   -----------------------------------------------------------------------------------------------------------------------------------
   test_add                        313.4770 (2.04)       358.2300 (1.58)     328.4561 (1.85)     12.1327 (1.0)      326.4100 (1.87)
   test_additive_inverse           153.6980 (1.0)        226.6550 (1.0)      177.9128 (1.0)      19.6890 (1.62)     174.6160 (1.0)
   test_divide                     222.3460 (1.45)       284.7130 (1.26)     235.5486 (1.32)     15.5184 (1.28)     232.1795 (1.33)
   test_multiplicative_inverse     165.4600 (1.08)       241.2010 (1.06)     186.5927 (1.05)     23.5185 (1.94)     178.2130 (1.02)
   test_multiply                   202.1690 (1.32)       327.1620 (1.44)     231.3098 (1.30)     30.2870 (2.50)     219.7315 (1.26)
   test_power                      361.5260 (2.35)       447.0060 (1.97)     385.7585 (2.17)     28.6975 (2.37)     375.4475 (2.15)
   test_scalar_multiply            756.5460 (4.92)     1,014.9590 (4.48)     792.1778 (4.45)     29.3465 (2.42)     786.1765 (4.50)
   test_subtract                   383.7790 (2.50)       461.3640 (2.04)     411.7450 (2.31)     26.7056 (2.20)     403.7260 (2.31)
   -----------------------------------------------------------------------------------------------------------------------------------

   -------------------------- benchmark "GF(3^5) Array Arithmetic: shape=(10_000,), ufunc_mode='jit-calculate'": 8 tests --------------------------
   Name (time in us)                       Min                    Max                   Mean                StdDev                 Median
   ------------------------------------------------------------------------------------------------------------------------------------------------
   test_add                           876.9310 (1.57)      1,635.8940 (1.52)        936.2487 (1.48)        76.1260 (3.84)        915.1175 (1.58)
   test_additive_inverse              557.6440 (1.0)       1,945.0700 (1.81)        632.3527 (1.0)        257.9239 (13.01)       578.4425 (1.0)
   test_divide                     90,022.6490 (161.43)   96,282.8560 (89.50)    92,257.7516 (145.90)   2,808.8230 (141.69)   90,481.3870 (156.42)
   test_multiplicative_inverse     82,011.9590 (147.07)   83,817.2670 (77.91)    82,897.2702 (131.09)     471.2330 (23.77)    82,992.5040 (143.48)
   test_multiply                    6,847.6130 (12.28)     6,894.3920 (6.41)      6,872.3102 (10.87)       19.8231 (1.0)       6,876.2980 (11.89)
   test_power                      77,322.3730 (138.66)   78,040.5270 (72.54)    77,650.6814 (122.80)     267.5041 (13.49)    77,693.8380 (134.32)
   test_scalar_multiply             6,049.4100 (10.85)     7,260.1360 (6.75)      6,184.4565 (9.78)       146.6458 (7.40)      6,153.1895 (10.64)
   test_subtract                      888.4720 (1.59)      1,075.8030 (1.0)         944.4420 (1.49)        47.1406 (2.38)        936.5830 (1.62)
   ------------------------------------------------------------------------------------------------------------------------------------------------

   Legend:
     Outliers: 1 Standard Deviation from Mean; 1.5 IQR (InterQuartile Range) from 1st Quartile and 3rd Quartile.
     OPS: Operations Per Second, computed as 1 / Mean
   =========================================================== 56 passed, 16 warnings in 19.54 seconds ===========================================================

Compare with a previous benchmark
---------------------------------

If you would like to compare the performance impact of a branch, first run a benchmark on `main` using the `--benchmark-save` option.
This will save the file `.benchmarks/0001_master.json`.

.. code-block:: console

   $ git checkout main
   $ python3 -m pytest benchmarks/test_field_arithmetic.py --benchmark-save=main --benchmark-columns=min,max,mean,stddev,median --benchmark-sort=name

Next, run a benchmark on the branch under test while comparing against the benchmark from `main`.

.. code-block:: console

   $ git checkout branch
   $ python3 -m pytest benchmarks/test_field_arithmetic.py --benchmark-compare=0001_master --benchmark-columns=min,max,mean,stddev,median --benchmark-sort=name

Or, save a benchmark run from `branch` and compare it explicitly against the one from `main`. This benchmark run will save the file `.benchmarks/0001_branch.json`.

.. code-block:: console

   $ git checkout branch
   $ python3 -m pytest benchmarks/test_field_arithmetic.py --benchmark-save=branch --benchmark-columns=min,max,mean,stddev,median --benchmark-sort=name
   $ python3 -m pytest-benchmark compare 0001_master 0001_branch
