### Task 5b: Estimating Parallel Fraction using Amdahl's Law

To estimate the parallel fraction, we first analyze the timings from our scaling benchmark:
* **Time taken with 1 worker ($T(1)$):** 1352.41 seconds
* **Time taken with 4 workers ($T(4)$):** 469.03 seconds

**1. Calculate Speedup ($S$)**
The speedup is calculated by dividing the sequential time by the parallel time:
$$S(4) = \frac{T(1)}{T(4)} = \frac{1352.41}{469.03} \approx 2.883$$
This means that using 4 cores, the simulation runs almost 3 times faster.

**2. Calculate Parallel Fraction ($F$)**
Using Amdahl's Law from our lecture materials:
$$S(p) = \frac{1}{(1 - F) + \frac{F}{p}}$$

Where:
* $S(p)$ is the Speedup (`2.883`)
* $p$ is the number of processors (`4`)
* $F$ is the fraction of parallelized code

Substituting our values into the formula:
$$2.883 = \frac{1}{(1 - F) + \frac{F}{4}}$$

To solve for $F$, we perform the inverse operations:
1. Divide 1 by `2.883`:  $1/2.883 = 0.346$
2. Set up the equation: $0.346 = 1 - F + 0.25F$
3. Combine $F$ terms: $0.346 = 1 - 0.75F$
4. Isolate $F$: $0.75F = 1 - 0.346 = 0.654$
5. Solve for $F$: $F = 0.654 / 0.75 = 0.872$

**Conclusion:**
The parallel fraction ($F$) is **87.2%**. This indicates that roughly 87% of the Python simulation code is successfully parallelized across the CPU cores, while the remaining ~13% consists of sequential overhead (such as file I/O operations and multiprocessing setup).

### Task 5c: Maximum Theoretical Speedup

**Theoretical Maximum Speedup**
According to Amdahl's Law, the theoretical maximum speedup ($S_{max}$) assumes an infinite number of processors, reducing the parallelizable time to 0. It is defined as:
$$S_{max} = \frac{1}{1 - F}$$

Using our estimated parallel fraction ($F = 0.872$):
$$S_{max} = \frac{1}{1 - 0.872} = \frac{1}{0.128} \approx 7.81$$
The mathematical limit for speedup on this program is **7.81x**. It can never exceed this threshold due to the 12.8% sequential overhead bottleneck.

**Actual Achievements**
Based on our scaling benchmark, the highest actual speedup recorded was:
* **Cores Used:** 16
* **Execution Time:** 236.64 seconds
* **Actual Speedup:** $\frac{1352.41}{236.64} \approx 5.71$

We achieved a maximum speedup of **5.71**, which represents roughly **73%** of the theoretical maximum limit ($5.71 / 7.81$). It took **16 CPU cores** to reach this point, and adding further cores would yield strictly diminishing returns as it flattens toward the 7.81x asymptote.

### Task 5d: Total Dataset Processing Estimation

**Methodology**
To estimate the execution time for the full dataset, we extrapolate from the fastest parallel performance achieved during our benchmarking.

**Calculations**
* **Fastest Run:** 16 Cores
* **Test Size:** 100 floorplans
* **Execution Time:** 236.64 seconds
* **Processing Rate:** $236.64 / 100 = 2.366$ seconds per floorplan

The full Swiss Dwellings dataset contains 4,571 floorplans. 
* **Estimated Total Time:** $4,571 \times 2.366 \text{ s} \approx 10,816.8 \text{ seconds}$

**Conclusion**
It would take approximately **3.00 hours** ($10,816.8 / 3600$) to process the entire full dataset of 4,571 floorplans using our fastest 16-core parallel map solution.
