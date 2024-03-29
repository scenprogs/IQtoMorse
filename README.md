# IQtoMorse

IQtoMorse is a simple **signal analytical morse decoder** which decodes **raw iq samples** based on **statistical signal analysis** using python3, numpy, scipy and matplotlib.

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
1. **iqdata** - path to file with raw iq data  
2. **fs** - sample frequency in hertz
3. **-fc** - cutoff frequency in hertz **[optional]**
4. **-plot** - enables plot **[optional]**

File with raw iq data should hold **sequentially stored float32 I and Q components**.<br/>
The cutoff frequency is taken **from frequency domain automatically** in case no optional cutoff frequency has given.

###### Examples:
```console
python3 IQtoMorse.py data/partae 67000 -plot
```
```console
python3 IQtoMorse.py data/partae 67000 -plot -fc 1000
```


