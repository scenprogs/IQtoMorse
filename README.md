# IQtoMorse

IQtoMorse is a simple signal analytical morse decoder which decodes raw iq samples based on probabilistic signal analysis.

![IQtoMorse.png](https://github.com/eikeviehmann/IQtoMorse/blob/main/IQtoMorse.png?raw=true)

## Usage
```
python3 IQtoMorse.py iqdata fs -plot
```
Arguments:
1. **iqdata** - path to raw iq data (sequentially stored float32 I and Q components) 
2. **fs** sample frequency in hertz
3. -plot [optional]

## Example
```
python3 IQtoMorse.py data/partae 67000 -plot
```

