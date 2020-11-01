# IQtoMorse

IQtoMorse is a simple **signal analytical morse decoder** which decodes **raw iq samples** based on **probabilistic signal analysis** using python3, numpy, scipy and matplotlib.

![IQtoMorse.png](https://github.com/eikeviehmann/IQtoMorse/blob/main/IQtoMorse.png?raw=true)
```
[[(0, 2), (3, 5), (6, 9)], [(10, 12)]]
[['.-.', '-.-', '....'], ['...']]
[['R', 'K', 'H'], ['S']]
RKH S
0:00:00.062408s (62ms)
```
## Usage
```console
python3 IQtoMorse.py iqdata fs -plot
```
###### Arguments:
1. **iqdata** - path to raw iq data (sequentially stored float32 I and Q components) 
2. **fs** - sample frequency in hertz
3. **fs** - cutoff frequency in hertz [optional]
4. **-plot** - enables plot [optional]

The cutoff frequency is taken from frequency automatically in case no optional cutoff frequency has given.

###### Example:
```console
python3 IQtoMorse.py data/partae 67000 -plot
```

