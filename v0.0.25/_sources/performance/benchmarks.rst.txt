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
   =============================================================================== test session starts ===============================================================================platform linux -- Python 3.8.10, pytest-4.6.9, py-1.8.1, pluggy-0.13.0
   benchmark: 3.4.1 (defaults: timer=time.perf_counter disable_gc=False min_rounds=5 min_time=0.000005 max_time=1.0 calibration_precision=10 warmup=False warmup_iterations=100000)
   rootdir: /mnt/c/Users/matth/repos/galois, inifile: setup.cfg
   plugins: requests-mock-1.9.3, cov-3.0.0, benchmark-3.4.1, anyio-3.5.0
   collected 56 items

   benchmarks/test_field_arithmetic.py ........................................................                                                                                [100%]

   ------------------- benchmark "GF(2) Array Arithmetic: shape=(100_000,), ufunc_mode='jit-calculate'": 8 tests -------------------
   Name (time in us)                    Min                 Max                Mean             StdDev              Median
   ---------------------------------------------------------------------------------------------------------------------------------
   test_add                         34.3540 (1.09)     326.2900 (1.67)      37.6302 (1.04)      7.8535 (1.0)       35.6970 (1.08)
   test_additive_inverse            31.4490 (1.0)      407.0170 (2.08)      36.0906 (1.0)      10.9850 (1.40)      32.9320 (1.0)
   test_divide                     250.4890 (7.96)     561.1350 (2.87)     273.2708 (7.57)     27.9056 (3.55)     259.8170 (7.89)
   test_multiplicative_inverse     244.7790 (7.78)     544.8090 (2.79)     265.2462 (7.35)     26.1761 (3.33)     252.0770 (7.65)
   test_multiply                    33.7330 (1.07)     278.8380 (1.43)      40.2364 (1.11)     14.8305 (1.89)      35.5870 (1.08)
   test_power                      155.9320 (4.96)     195.5260 (1.0)      167.5988 (4.64)     15.6836 (2.00)     160.6160 (4.88)
   test_scalar_multiply            547.2450 (17.40)    949.3950 (4.86)     592.0535 (16.40)    42.1333 (5.36)     586.4980 (17.81)
   test_subtract                    34.2440 (1.09)     286.8660 (1.47)      39.9417 (1.11)     10.2580 (1.31)      35.8870 (1.09)
   ---------------------------------------------------------------------------------------------------------------------------------

   ---------------------- benchmark "GF(257) Array Arithmetic: shape=(100_000,), ufunc_mode='jit-calculate'": 8 tests ----------------------
   Name (time in us)                      Min                   Max                  Mean             StdDev                Median
   -----------------------------------------------------------------------------------------------------------------------------------------
   test_add                           97.2320 (1.24)       187.9820 (1.00)       104.8390 (1.26)      9.9335 (1.21)       100.0770 (1.25)
   test_additive_inverse              78.2460 (1.0)        187.5010 (1.0)         83.3856 (1.0)       8.2153 (1.0)         79.9845 (1.0)
   test_divide                     3,157.2540 (40.35)    3,246.9410 (17.32)    3,192.1388 (38.28)    24.4100 (2.97)     3,193.6465 (39.93)
   test_multiplicative_inverse     3,151.1620 (40.27)    3,197.0380 (17.05)    3,172.3083 (38.04)    16.6883 (2.03)     3,165.2480 (39.57)
   test_multiply                     211.6760 (2.71)       253.5550 (1.35)       222.6776 (2.67)     12.4609 (1.52)       216.7160 (2.71)
   test_power                      3,110.7570 (39.76)    3,153.5060 (16.82)    3,140.2459 (37.66)    14.3064 (1.74)     3,147.0140 (39.35)
   test_scalar_multiply              527.4980 (6.74)       625.8330 (3.34)       549.4393 (6.59)     12.8193 (1.56)       551.7140 (6.90)
   test_subtract                      98.3940 (1.26)       298.4900 (1.59)       105.8258 (1.27)     14.5217 (1.77)       100.4880 (1.26)
   -----------------------------------------------------------------------------------------------------------------------------------------

   -------------------- benchmark "GF(257) Array Arithmetic: shape=(100_000,), ufunc_mode='jit-lookup'": 8 tests -------------------
   Name (time in us)                    Min                 Max                Mean             StdDev              Median
   ---------------------------------------------------------------------------------------------------------------------------------
   test_add                         99.7770 (1.28)     126.3470 (1.18)     105.6933 (1.30)      9.3289 (1.25)     100.9590 (1.28)
   test_additive_inverse            77.6750 (1.0)      107.1700 (1.0)       81.6013 (1.0)       7.4726 (1.0)       78.8470 (1.0)
   test_divide                     233.4670 (3.01)     278.1910 (2.60)     249.1636 (3.05)     14.3489 (1.92)     242.0540 (3.07)
   test_multiplicative_inverse     158.2560 (2.04)     223.6990 (2.09)     170.7402 (2.09)     16.0821 (2.15)     164.9190 (2.09)
   test_multiply                   211.3160 (2.72)     409.9580 (3.83)     228.5190 (2.80)     18.5487 (2.48)     220.8040 (2.80)
   test_power                      280.7560 (3.61)     335.9490 (3.13)     297.4570 (3.65)     18.4990 (2.48)     289.4320 (3.67)
   test_scalar_multiply            530.2030 (6.83)     764.5500 (7.13)     554.8452 (6.80)     19.3715 (2.59)     556.6580 (7.06)
   test_subtract                   100.3380 (1.29)     157.8560 (1.47)     109.3698 (1.34)     14.7642 (1.98)     101.6355 (1.29)
   ---------------------------------------------------------------------------------------------------------------------------------

   ------------------------ benchmark "GF(2^8) Array Arithmetic: shape=(100_000,), ufunc_mode='jit-calculate'": 8 tests -------------------------
   Name (time in us)                       Min                    Max                   Mean              StdDev                 Median
   ----------------------------------------------------------------------------------------------------------------------------------------------
   test_add                            41.4180 (1.08)        642.5490 (3.67)         49.2204 (1.17)      14.0472 (2.04)         43.4420 (1.09)
   test_additive_inverse               38.3320 (1.0)         250.6000 (1.43)         42.1670 (1.0)        7.0472 (1.03)         39.8150 (1.0)
   test_divide                     17,594.1940 (459.00)   18,964.6810 (108.43)   17,935.5072 (425.34)   583.3312 (84.92)    17,655.0580 (443.43)
   test_multiplicative_inverse     16,486.8900 (430.11)   16,866.1610 (96.43)    16,687.3492 (395.74)   136.7594 (19.91)    16,700.1245 (419.44)
   test_multiply                    1,117.6840 (29.16)     1,165.6630 (6.66)      1,134.4561 (26.90)     15.9309 (2.32)      1,132.0800 (28.43)
   test_power                      15,916.0410 (415.22)   16,131.0530 (92.23)    16,006.6844 (379.60)   112.1577 (16.33)    15,934.3250 (400.21)
   test_scalar_multiply               868.4760 (22.66)     1,063.7620 (6.08)        903.3674 (21.42)     17.4971 (2.55)        901.1280 (22.63)
   test_subtract                       40.9370 (1.07)        174.8980 (1.0)          44.8113 (1.06)       6.8691 (1.0)          42.4490 (1.07)
   ----------------------------------------------------------------------------------------------------------------------------------------------

   --------------------- benchmark "GF(2^8) Array Arithmetic: shape=(100_000,), ufunc_mode='jit-lookup'": 8 tests --------------------
   Name (time in us)                    Min                   Max                Mean             StdDev              Median
   -----------------------------------------------------------------------------------------------------------------------------------
   test_add                         40.9270 (1.06)       589.8250 (2.90)      46.2227 (1.08)     13.8633 (1.82)      42.5290 (1.05)
   test_additive_inverse            38.7730 (1.0)        203.3380 (1.0)       42.6781 (1.0)       7.6301 (1.0)       40.3260 (1.0)
   test_divide                     228.9490 (5.90)       274.8450 (1.35)     246.6170 (5.78)     16.2190 (2.13)     243.7515 (6.04)
   test_multiplicative_inverse     174.1560 (4.49)       238.0960 (1.17)     191.3213 (4.48)     19.8284 (2.60)     184.0150 (4.56)
   test_multiply                   221.7660 (5.72)       266.3990 (1.31)     235.2228 (5.51)     17.4305 (2.28)     227.4660 (5.64)
   test_power                      362.6090 (9.35)       409.8170 (2.02)     382.3277 (8.96)     16.0054 (2.10)     384.2700 (9.53)
   test_scalar_multiply            823.3820 (21.24)    1,120.2880 (5.51)     867.6072 (20.33)    30.0068 (3.93)     861.3540 (21.36)
   test_subtract                    41.3880 (1.07)       718.4240 (3.53)      47.6819 (1.12)     15.5575 (2.04)      42.9000 (1.06)
   -----------------------------------------------------------------------------------------------------------------------------------

   ------------------------- benchmark "GF(3^5) Array Arithmetic: shape=(1_000,), ufunc_mode='jit-calculate'": 8 tests --------------------------
   Name (time in us)                       Min                    Max                   Mean              StdDev                 Median
   ----------------------------------------------------------------------------------------------------------------------------------------------
   test_add                           286.9570 (1.46)        376.3550 (1.0)         305.7452 (1.38)      21.9760 (1.37)        294.2010 (1.44)
   test_additive_inverse              196.6080 (1.0)         425.6750 (1.13)        221.3740 (1.0)       42.2878 (2.63)        204.4580 (1.0)
   test_divide                     11,875.8930 (60.40)    11,970.4800 (31.81)    11,914.6582 (53.82)     35.7418 (2.22)     11,916.3200 (58.28)
   test_multiplicative_inverse     10,993.7020 (55.92)    12,051.7220 (32.02)    11,250.7244 (50.82)    451.9707 (28.13)    11,040.1180 (54.00)
   test_multiply                      943.6080 (4.80)        986.9380 (2.62)        965.4164 (4.36)      16.0692 (1.0)         968.4540 (4.74)
   test_power                      10,423.0930 (53.01)    10,560.8100 (28.06)    10,480.1514 (47.34)     53.9692 (3.36)     10,484.2870 (51.28)
   test_scalar_multiply               806.9220 (4.10)      1,069.2530 (2.84)        846.3491 (3.82)      22.7568 (1.42)        844.1710 (4.13)
   test_subtract                      287.6280 (1.46)        419.9370 (1.12)        313.0968 (1.41)      27.8982 (1.74)        302.2270 (1.48)
   ----------------------------------------------------------------------------------------------------------------------------------------------

   ------------------- benchmark "GF(3^5) Array Arithmetic: shape=(1_000,), ufunc_mode='jit-lookup'": 8 tests -------------------
   Name (time in us)                   Min                 Max               Mean             StdDev             Median
   ------------------------------------------------------------------------------------------------------------------------------
   test_add                        22.0210 (1.20)      46.1970 (1.21)     24.8062 (1.18)      6.8689 (1.73)     22.4220 (1.17)
   test_additive_inverse           18.5550 (1.01)      40.8070 (1.07)     20.9531 (1.0)       5.3813 (1.35)     19.2510 (1.01)
   test_divide                     19.6070 (1.07)      38.2020 (1.0)      21.3653 (1.02)      3.9815 (1.0)      20.2480 (1.06)
   test_multiplicative_inverse     18.2950 (1.0)       38.2220 (1.00)     21.1822 (1.01)      5.5094 (1.38)     19.1160 (1.0)
   test_multiply                   19.4370 (1.06)      43.8820 (1.15)     21.9691 (1.05)      5.9414 (1.49)     19.9730 (1.04)
   test_power                      23.7550 (1.30)      47.5690 (1.25)     25.7413 (1.23)      5.1572 (1.30)     24.0850 (1.26)
   test_scalar_multiply            26.7100 (1.46)     236.3210 (6.19)     29.4050 (1.40)      6.4931 (1.63)     27.8220 (1.46)
   test_subtract                   22.8230 (1.25)      64.1200 (1.68)     28.5587 (1.36)     12.9444 (3.25)     23.2485 (1.22)
   ------------------------------------------------------------------------------------------------------------------------------

   Legend:
   Outliers: 1 Standard Deviation from Mean; 1.5 IQR (InterQuartile Range) from 1st Quartile and 3rd Quartile.
   OPS: Operations Per Second, computed as 1 / Mean
   ===================================================================== 56 passed, 1 warnings in 18.94 seconds ======================================================================

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
