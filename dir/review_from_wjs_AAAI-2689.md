
- Quality: 3: good
- Clarity: 3：good
- Originality: 3: good
- Significance: 3: good

## summary：
This paper present FOrLoRA to tackles the problem of aggregation noise in federated learning with Low-rank Adaptation. The method is well-motivated by a clear analysis of the problem's root cause and offers a sophisticated solution involving rank-wise concatenation, clustering, and linear projection. The extensive experimental validation across NLU, NLG, and vision tasks is compelling and demonstrates significant improvements over strong baselines.


## pros：
1. The authors provide a complete and formally sound derivation of the Noise-Free Alignment Objective, including the closed-form solutions for the projection matrices using the Moore-Penrose pseudoinverse.
2. The authors evaluate their proposed FOrLoRA framework across a diverse suite of tasks spanning NLU, NLG, and Computer Vision.


## cons：
1. The study lacks a systematic computational complexity analysis of solving Eqn 12. While noting direct solution of Eqn 10 is costly and proposing Eqn 12 as an alternative, the authors don’t analyze the time complexity of solving this surrogate objective.

2. Rank-space compression depends on the number of clusters $R$. The paper does not adequately analyze the impact of the projection rank R on the accuracy of FOrLoRA.

3. The paper does not provide a clear definition of \( \Delta\mathbf{W}^{+} \), nor does it offer a detailed explanation of the relationships among $\( \Delta\mathbf{W}^{+} \)$, $\( \Delta\mathbf{W}^{-} \)$, and $\( \Delta\mathbf{W} \)$.
4. The paper proposes a method for noise-reduced aggregation via linear projection (see Eqn. 17 and 18). However, this process requires the server to transmit the projected matrices back to the clients. The authors fail to provide quantitative experiments or analysis on the additional communication overhead introduced by these matrices. This omission undermines the paper’s core claim of being “communication-efficient,” as sufficient empirical evidence is lacking to support this assertion.
5. In vision datasets (e.g., Cars, DTD), FOrLoRA underperforms FlexLoRA.   The authors dismiss this as “minimal gap,” but further discussion is needed.
6. The reproducibility checklist claims that all code/datasets are provided, but this is not provided in the paper itself.


2. There is no clear discussion on the setting of some parameters in the experiment, such as $R$ clusters. And how is the clustering performance when calculating similarity across different ranks under heterogeneous settings?



## Questions：
1. Why not use SVD to directly decompose the extended $P$ diagonal matrix? What is the reason for using $P'$ instead of $P$? $BP'A$ does not seem to be equivalent to $BPA$, which also leads to lossy aggregation without implementing the author's Noise-Free Alignment Objective.

 How to select the number of clusters during clustering? If the number of clusters is not much different from $r_{sum}$, it does not seem to reduce significant communication overhead? Will different number of clusters lead to different gains of the aggregation effect?

 To ensure that the client successfully updates the parameters, $R$ seems to have to be greater than or equal to the largest client rank, but this does not seem to be discussed by the author?

 Table 8 seems to provide an unfair comparison, for example, in Communication Overhead, FLoRA calculates the total communication overhead of $N$ clients, while other algorithms only compare the communication overhead of one client. And the computation overhead of FLoRA at aggregation time is not listed.

 In addition, like all other methods, FOrLoRA does not provide privacy protection, what is the basis for FOrLoRA privacy protection mentioned in Table 8?