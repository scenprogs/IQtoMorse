# IQtoMorse

IQtoMorse is a simple signal analytical morse decoder which decodes raw iq samples based on probabilistic signal analysis.

![IQtoMorse.png](https://github.com/eikeviehmann/IQtoMorse/blob/main/IQtoMorse.png?raw=true)

## Usage
```
python3 IQtoMorse.py iqdata samplefrequency -plot
```
## Terminal parameters
1. iqdata - path to raw iq data []
2. samplefrequency in hertz []
3. -plot [optional]

## Example
```
python3 IQtoMorse.py data/partae 67000 -plot
```

