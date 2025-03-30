---
title: Gaussian mixture models as a proxy for interacting language models
keywords: 
    - Gaussian mixture models
    - Large language models
---
+++ {"part": "abstract"}
Large language models (LLMs) are a powerful tool with the ability to match human capabilities and behavior in many tasks. The use of retrieval-augmented generation further allows LLMs to generate diverse output depending on the contents of the database they have access to. This motivates their use in the social sciences to study human social behavior between individuals where large-scale experiments are infeasible. However, LLMs are also computationally expensive and difficult to understand. In this paper, we introduce a computational framework using interacting Gaussian mixture models (GMMs) as an alternative to similar frameworks using LLMs. We compare this simplified model on select experiments from the paper \textit{Investigating social alignment via mirroring in a system of interacting language models} that utilize language models at the core of their simulations. We find that our framework is able to capture many of the findings discovered by the LLM, and address the differences between the two models. Finally, we discuss the benefits of the GMM model, future directions, and potential modifications to the framework to make it more useful in social science studies.  
+++
# Introduction

Large language models (LLMs) have made incredible progress in their abilities, approaching and sometimes exceeding human capabilities in text generation and language processing. Of particular interest is the fact that LLMs are able to replicate human tendencies and characteristics demonstrated in previous cognitive science, linguistic, and social science literature [@aher2023using; @cai2023does]. Additionally, retrieval augmented generation (RAG) allows LLMs to retrieve relevant information from a database of information as additional context when completing a task, resulting in more diverse and accurate responses [@lewis2020retrieval]. This database of information is akin to the knowledge and memories of a person. Thus, LLMs utilizing RAG are a promising tool when building computational frameworks for modeling social interactions between individuals with diverse perspectives. 

This is extremely useful in sociology, for example, where large-scale experiments on social phenomena is often impractical. In McGuinness et al., this approach was applied to understand how mirroring, the act of copying the behaviors of others, affects social alignment, the property where multiple individuals share a common perspective [@mcguinness2024investigating]. Their approach yielded conclusions matching those found in prior literature and extended new results on how various parameters of social networks affects development of social alignment. 

Despite the immense utility of modern language models in social science research, their inner workings are largely a black box. The lack of understanding about the behavior of LLM output may call into question the reliability of conclusions arising from complex systems of interacting LLMs. Furthermore, LLMs are extremely complex, which translates to heavy energy, hardware, and time costs for training and text generation[@jiang2024preventing]. These concerns motivate us to consider whether we can develop alternative computational models that are simpler, understandable, and computationally inexpensive, but also capture enough of the complexity of interacting LLM systems to be useful.

In this paper, we propose a computational framework utilizing multiple Gaussian mixture models (GMMs) as agents each with an associated set of vectors to act as a RAG database. Gaussian mixture models are well studied in the literature and their training mechanism through the EM algorithm can be interpreted as updating the associated agent's belief or perspective. As a result, they are a reasonable choice of model that can potentially capture complexities of interacting agents, while also being interpretable and easy to work with. In this paper, we implement a system of interacting GMMs, running similar experiments as discussed in McGuinness et al., comparing the results obtained from their RAG LLM computational framework and our GMM framework.

# Unstable silo behavior
Words, words, words see[](#unstable_silo_comparison)

Code block incoming see [](#gmm-code)

```{code} python
:label: gmm-code
:caption: Simulating interacting agents using GMMs

import numpy as np
import pandas as pd
import scipy
from scipy.stats import norm
from sklearn import mixture
import random

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

#import itertools
#from concurrent.futures import ThreadPoolExecutor, as_completed
from joblib import delayed, Parallel
from tqdm import tqdm
import pickle
```

Collapsible code block see [](#gmm-code-collapsible)

:::{dropdown} Code
```{code} python
:label: gmm-code-collapsible
:caption: Simulating interacting agents using GMMs

import numpy as np
import pandas as pd
import scipy
from scipy.stats import norm
from sklearn import mixture
import random

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

#import itertools
#from concurrent.futures import ThreadPoolExecutor, as_completed
from joblib import delayed, Parallel
from tqdm import tqdm
import pickle
```
:::

```{figure} images/unstable_silo_comparison.png
:name: unstable_silo_comparison
Comparison of unstable silo behavior for our GMM simulation and the LLM simulation from McGuinness et al. $\textbf{Left.}$ Unstable silo behavior for the GMM model described in this paper. We set the global parameters to be $p=0.4$, $k=29$, $r=5$, and $T=400$. $\textbf{Right.}$ Unstable silo behavior for the LLM model in McGuiness et al. $\textbf{Top.}$ An example unstable silo system where each line represents an agent. $\textbf{Middle.}$ The evolution of the number of agents in each possible silo where each line represents a silo. $\textbf{Bottom.}$ Comparison of the stability metric for stable and unstable silos.
```

