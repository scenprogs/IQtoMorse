# IQtoMorse

IQtoMorse is a simple signal analytical morse decoder which decodes raw iq samples based on probabilistic signal analysis.

![IQtoMorse.png](https://github.com/eikeviehmann/IQtoMorse/blob/main/IQtoMorse.png?raw=true)

```console
Identified frequency of morse keying at 37.96729945499092 Hz
Peaks:
	13 peaks found
	Minimum peak length is 2884 (43.04ms)
	Average peak length is 4590.46 (68.51ms)
	Maximum peak length is 8996 (134.27ms)
	Short break range is (0, 4625.0)
	Medium break range is (4626.0, 9250.0)
Translation:
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
Arguments:
1. **iqdata** - path to raw iq data (sequentially stored float32 I and Q components) 
2. **fs** sample frequency in hertz
3. **-plot** [optional]

Example:
```
python3 IQtoMorse.py data/partae 67000 -plot
```

