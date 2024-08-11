## Multi-task CNN Model for Attribute Prediction
this unfficial paper implmenatation [Multi-task CNN Model for Attribute Prediction](https://arxiv.org/pdf/1601.00400)  



<div align="center">    

<div align="center">
    <img src="./assets/figure.png"/></br>
    <figcaption>CNN-MTL </figcaption>
</div>
 
### Multi-task CNN Model for Attribute Prediction

[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://academic.oup.com/bib/article/23/1/bbab545/6489100)
[![Made With Love](https://img.shields.io/badge/Made%20With-Love-orange.svg)](https://github.com/PaperWeek/Multi-task--APCNN)
[![GitHub issues](https://img.shields.io/github/issues/PaperWeek/Multi-task--APCNN)](https://github.com/PaperWeek/Multi-task--APCNN/issues) 
[![GitHub forks](https://img.shields.io/github/forks/PaperWeek/Multi-task--APCNN)](https://github.com/PaperWeek/Multi-task--APCNN/network) 
[![GitHub stars](https://img.shields.io/github/stars/PaperWeek/Multi-task--APCNN)](https://github.com/PaperWeek/Multi-task--APCNN/stargazers)
[![GitHub license](https://img.shields.io/github/license/youness-elbrag/AdapterLoRa)](https://github.com/PaperWeek/Multi-task--APCNN/blob/master/LICENSE)
<!--
ARXIV   
[![Paper](http://img.shields.io/badge/arxiv-math.co:1480.1111-B31B1B.svg)](https://www.nature.com/articles/nature14539)
-->
<!--  
Conference   
-->   
</div>

### Instalation 

1. clone the repo following the command 

```bash

git clone https://github.com/deep-matter/Multi-task--APCNN && Multi-task--APCNN
```

2. install packages from repo 

```bash 
pip install -e
```

## Convolution Neural Network 

**Notation** : each CNN has Trained of Single attributes which later on used the Feature extraction representation into the Multi-task leaning Framework if the dealing with large attributes **need a Fine-tune of CNN**


```python 
from src.models.cnn import BaseCNN
import yaml 
import os

if __name__ == "__main__":
    
    with open("config.yml", 'r') as file:
        yaml.dump(dummy_config, file)



    model = BaseCNN(config_path="config.yml")

    # input tensor (e.g., batch size of 8, 3 channels, 64x64 images)
    sample_input = torch.randn(8, 3, 64, 64)
    output = model(sample_input)
    print("Output shape:", output.shape)
```  

### Citation   
```
@article{abdulnabi2015multi,
  title={Multi-task CNN model for attribute prediction},
  author={Abdulnabi, Abrar H and Wang, Gang and Lu, Jiwen and Jia, Kui},
  journal={IEEE Transactions on Multimedia},
  volume={17},
  number={11},
  pages={1949--1959},
  year={2015},
  publisher={IEEE}
}
```   

# get in touch

Libraries utilisées :
<p align="center">
  <a href="https://github.com/deep-matter" class="fancybox" ><img src="https://user-images.githubusercontent.com/63207451/97302854-e484da80-1859-11eb-9374-5b319ca51197.png" title="GitHub" width="40" height="40"></a>
  <a href="https://www.linkedin.com/in/youness-el-brag-b13628203/" class="fancybox" ><img src="https://user-images.githubusercontent.com/63207451/97303444-b2c04380-185a-11eb-8cfc-864c33a64e4b.png" title="LinkedIn" width="40" height="40"></a>
  <a href="mailto:younsselbrag@gmail.com" class="fancybox" ><img src="https://user-images.githubusercontent.com/63207451/97303543-cec3e500-185a-11eb-8adc-c1364e2054a9.png" title="Mail" width="40" height="40"></a>
</p>
